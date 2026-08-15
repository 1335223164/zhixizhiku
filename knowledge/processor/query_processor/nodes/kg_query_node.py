import os, json, re, logging
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Set
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, \
    fetch_chunks_by_chunk_ids
from knowledge.utils.neo4j_utils import get_neo4j_driver

MAX_ENTITY_NAME_LENGTH: int = 15
DEFAULT_ENTITY_NAME_ALIGN = 0.5

ALLOWED_ENTITY_LABELS_CN: str = (
    "设备(Device)、部件(Part)、操作(Operation)、步骤(Step)、"
    "警告(Warning)、条件(Condition)、工具(Tool)"
)

_ENTITY_EXTRACT_SYSTEM_PROMPT = f"""
你是一个知识图谱问答系统的"实体识别"模块。
请从用户问题中抽取用于查询图数据库(Neo4j)的实体名称。

【图谱中存在的实体类型】
{ALLOWED_ENTITY_LABELS_CN}

【约束】
1) 优先抽取上述类型的名词短语（设备名、部件名、操作名、工具名、步骤名、条件、警告等）
2) 每个实体名称不超过 {MAX_ENTITY_NAME_LENGTH} 个字符，超过请截取核心部分
3) 不要输出完整句子，只输出实体关键词
4) 输出必须是严格 JSON，只含一个字段 entities（字符串数组）

【输出示例】
{{"entities": ["电池安装", "螺丝刀", "表笔"]}}
"""

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# neo4j 的语句
_CYPHER_EXACt_SEEDS = """
MATCH (n:Entity)
WHERE n.item_name = $item_name AND n.name = $entity_name
RETURN n.item_name as item_name,n.name as name
LIMIT 1
"""

_CYPHER_FUZZY_SEEDS = """
MATCH (n:Entity)
WHERE toLower(n.name) CONTAINS toLower($entity_name)
        AND n.item_name = $item_name
RETURN n.name as name, n.item_name as item_name
LIMIT $limit
"""

# 查询一跳关系
_CYPHER_ONE_HOP_RELATIONS = """
MATCH (seed:Entity {name:$name,item_name:$item_name})-[:r]-(nbr:Entity)
WHERE type(r) <> 'MENTIONED_IN' AND nbr.item_name=$item_name
RETURN 
    CASE WHEN startNode(r)=seed THEN seed.name ELSE nbr.name END AS head,
    type(r) as rel,
    CASE WHEN startNode(r)=seed THEN nbr.name ELSE seed.name END AS tail
LIMIT $limit
"""

_CYPHER_LOOKUP_CHUNK = """
UNWIND $weight_nodes as n
MATCH (e:Entity {name:n.entity_name, item_name:n.item_name}) - [r:MENTIONED_IN] -> (c:chunk {item_name:n.item_name})
WITH c, sum(n.weight) as score, count(e) as cnt
RETURN c.id as chunk_id, c.item_name as item_name, score, cnt
ORDER BY score DESC, cnt DESC, c.chunk_id DESC
LIMIT $limit
"""


# ---------------------------------------
# 工具函数
def _clean_parse_llm_content(llm_response_content):
    """
    清洗以及解析llm的输出
    :param llm_response_content:
    :return:
    """

    # 1. 判断llm内容
    if not llm_response_content:
        return []

    # 2. 清洗json围栏
    cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
    content = re.sub(r"\s*```$", "", cleaned.strip())

    # 3. 反序列化
    try:
        deserialized = json.loads(content)
    except JSONDecodeError as e:
        logger.error(f"LLM输出的json反序列化失败, 原因{str(e)}")
        return []

    # 4. 获取提取的实体名
    entities_names = deserialized.get('entities', [])

    # 5. 判断
    if not entities_names or not isinstance(entities_names, list):
        return []

    # 遍历
    seen = set()
    final_entities_name = []
    for entities_name in entities_names:
        # 判断是否为空
        if not entities_name:
            continue

        # 判断是否过长
        if entities_name > MAX_ENTITY_NAME_LENGTH:
            entities_name = entities_name[:MAX_ENTITY_NAME_LENGTH].strip()

        # 去重保序
        if entities_name not in seen:
            seen.add(entities_name)
            final_entities_name.append(entities_name)

    return final_entities_name


def _item_name_filter_expr(item_names):
    quoted = "，".join(f"{item_name}" for item_name in item_names)

    return f"item_name in [{quoted}]"


def _build_item_entity_pair(aligned_entities_info):
    """
    从对齐后的实体详情中获取商品名加实体名的pair对
    去重
    :param aligned_entities_info:
    :return:
    """

    # 1. 判断对齐后的实体详情是否存在
    if not aligned_entities_info:
        return []

    # 2. 遍历对齐后的实体详情
    seen = set()
    item_entity_pairs = []
    for aligned_entity_info in aligned_entities_info:
        # 2.1 获取商品名
        item_name = aligned_entity_info.get("item_name").strip()
        # 2.2 获取对齐后的
        aligned_entity_name = aligned_entity_info.get("aligned", '').strip()
        # 2.3 判断实体名和商品名是否都存在
        if not item_name and not aligned_entity_name:
            continue
        # 2.4 去重
        key = (item_name, aligned_entity_name)
        if key not in seen:
            seen.add(key)
            item_entity_pairs.append({
                "item_name": item_name,
                "entity_name": aligned_entity_name
            })

    return item_entity_pairs


def _clean_seed_rows(rows):
    """
    清洗查询种子节点的数据
    :param rows:
    :return:
    """
    if not rows:
        return []

    # 1. 遍历
    clean_seeds_result = []
    for row in rows:
        # 1.1 获取item_name
        item_name = row.get("item_name", '').strip()
        # 1.2 获取实体名称
        entity_name = row.get("name", '').strip()
        if not item_name or not entity_name:
            continue
        # 1.3 构建一下
        clean_seeds_result.append({
            "item_name": item_name,
            "entity_name": entity_name
        })
    return clean_seeds_result


def _one_hop_triples_to_texts(triples):
    """
           将一跳三元组转为大模型易理解的文本描述。
           上游 find_one_hop_relations 已去重，此处直接转文本。
           格式示例：[商品A] 电池 -(REQUIRED)-> 9V

           Args:
               triples: 去重后的一跳三元组列表，每条包含 head/rel/tail/item_name

           Returns:
               关系文本描述列表，供下游组装 prompt 使用
           """
    if not triples:
        return []
    docs: List[str] = []
    for tr in triples:
        it = (tr.get("item_name") or "").strip()
        h = (tr.get("head") or "").strip()
        r = (tr.get("rel") or "").strip()
        t = (tr.get("tail") or "").strip()
        if not (h and r and t):
            continue
        docs.append(f"[{it}] {h} -({r})-> {t}" if it else f"{h} -({r})-> {t}")
    return docs

# ---------------------------------------


class EntityExtractor:
    """
    实体提取器
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(self, user_query: str) -> Dict[List[str]]:
        """
        根据用户问题提取实体名
        :param user_query:
        :return:
        """

        # 1. 获取llm客户端
        llm_client = AIClients.get_llm_client()

        # 2. 判断
        if llm_client is None:
            return []

        # 3. 获取提示词
        entities_name_extract_system_prompt = _ENTITY_EXTRACT_SYSTEM_PROMPT.format(
            ALLOWED_ENTITY_LABELS_CN=ALLOWED_ENTITY_LABELS_CN,
            MAX_ENTITY_NAME_LENGTH=MAX_ENTITY_NAME_LENGTH)

        # 4. 调用大模型
        try:
            # 4.1 发送请求
            llm_response = llm_client.invoke([
                SystemMessage(entities_name_extract_system_prompt),
                HumanMessage(f"用户问题:{user_query}")
            ])

            # 4.2 获取模型的结果
            llm_response_content = getattr(llm_response, 'content', '').strip()

            # 4.3 解析和清洗llm结果
            entities_names = _clean_parse_llm_content(llm_response_content)

            return entities_names


        except Exception as e:
            self.logger.error(f"LLM调用失败,原因:{str(e)}")
            return []


class EntityAligner:
    """
    实体对齐器
    根据llm提取到的实体名查询milvus,获取真正的实体名
    """

    def __init__(self, collection_name):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_name = collection_name

    def align(self, entity_names, item_names):
        """
        llm识别到的实体名进行milvus检索,并且对齐后返回结果
        :param entity_names:
        :param item_names:
        :return:
        """

        fallback_result = {"entities_aligned": [], "entities_elements": []}

        # 1. 判断实体是否有
        if not entity_names:
            return fallback_result

        # 2. 获取嵌入模型
        bge_ef_model = AIClients.get_bge_m3_client()

        if not bge_ef_model:
            self.logger.error(f"嵌入模型不存在")
            return fallback_result

        # 3. 获取Milvus客户端
        milvus_client = StorageClients.get_milvus_client()
        if not milvus_client:
            self.logger.error(f"milvus客户端获取失败")
            return fallback_result

        # 4. 向量化实体名
        embedding_result = generate_bge_m3_hybrid_vectors(bge_ef_model, entity_names)

        # 5. 校验向量结果
        if embedding_result is None:
            self.logger.error(f"向量实体名失败")
            return fallback_result
        dense = embedding_result.get('dense')
        sparse = embedding_result.get('sparse')

        # 6. 获取item_name的表达式
        item_name_filter_expr = _item_name_filter_expr(item_names)

        # 7. 遍历所有实体名
        aligned_entities_names = []  # 存放所有实体名
        aligned_entities_elements = []  # 存放实体详细信息
        for i, entity_name in enumerate(entity_names):
            # 7.1 对齐某一个实体的名字
            align_one_result = self._align_one(milvus_client, entity_name, item_name_filter_expr, dense[i], sparse[i],
                                               self.collection_name, item_names)

            # 7.2 构建所有实体名字
            aligned_entity_name = align_one_result.get('aligned')
            aligned_entities_names.extend(aligned_entity_name)

            # 7.3 构建对齐后的实体详细信息
            aligned_entities_elements.extend(align_one_result)

        return {"entities_aligned": aligned_entities_names, "entities_elements": aligned_entities_elements}

    def _align_one(self, milvus_client, entity_name, item_name_filter_expr, dense, sparse, collection_name, item_names):
        """
        对齐指定的实体名
        :param milvus_client:
        :param entity_name:
        :param item_name_filter_expr:
        :param param:
        :param param1:
        :return:
        """

        # 1. 校验混合向量
        if not dense or not sparse:
            return {"original_entity_name": entity_name, "aligned": "", "reason": "vector values is not exist"}

        # 2. 创建混合搜索请求
        reqs = create_hybrid_search_requests(dense_vector=dense, sparse_vector=sparse, expr=item_name_filter_expr)

        # 3. 获取向量检索结果
        reps = execute_hybrid_search_query(milvus_client=milvus_client, collection_name=collection_name,
                                           search_requests=reqs,
                                           output_fields=["source_chunk_id", "item_name", "context", "entity_name"])

        # 4. 解析结果
        if not reps or not reps[0]:
            return {"original_entity_name": entity_name, "aligned": "", "reason": "vector values is not exist"}

        # 5. 获取结果
        best_entity = self._pick_best_entity_name(reps[0])
        if best_entity is None:
            return {"original_entity_name": entity_name, "aligned": "", "reason": "vector values is not exist"}

        return {"original_entity_name": entity_name,
                "aligned": best_entity['entity_name'],
                "source_chunk_id": best_entity['source_chunk_id'],
                "item_name": best_entity['item_name'],
                "context": best_entity['context'],
                "reason": "top1"
                }

    def _pick_best_entity_name(self, search_entity_name):
        """
        从返回的五个实体名中只留下一个
        :param reps:
        :return:
        """

        # 1. 判断是否检索到
        if not search_entity_name:
            return None

        # 2. 获取第一个
        first_entity = search_entity_name[0]
        if not first_entity:
            return None

        # 3. 获取实体名分数
        first_entity_score = first_entity.get('distance')

        # 4. 判断是否超过阈值
        if not first_entity_score:
            return None

        if first_entity_score < DEFAULT_ENTITY_NAME_ALIGN:
            return None

        # 返回实体名的是第一个且分数超过阈值
        return first_entity


class Neo4jGraphReader:
    """
    所有对neo4j的操作都放到这个类
    种子节点查询
    查询种子节点一跳关系 双向
    查询实体节点关联的chunk节点
    返回全部chunk_Id,查询milvus
    """

    def __init__(self, database):
        super().__init__()
        self.database = database
        self.logger = logging.getLogger(self.__class__.__name__)

    def _session(self):
        neo4j_driver = get_neo4j_driver()
        if neo4j_driver:
            raise RuntimeError("neo4j驱动获取失败")
        return neo4j_driver.session(self.database)

    def find_seed_nodes(self, item_entity_pairs):
        """
        查询种子节点
        精确查询1条    模糊查询2-3条
        :param item_entity_pairs:
        :return:
        """

        # 1. 判断 pair对是否有
        if not item_entity_pairs:
            return []

        # 2. 遍历所有pair对
        final_seeds_result = []
        for pair in item_entity_pairs:
            # 2.1 获取item_name
            item_name = pair.get('item_name')
            # 2.2 获取entity_name
            entity_name = pair.get('entity_name')
            # 2.3 如果item_name entity_name没有就下一个
            if not item_name or not entity_name:
                continue

            # 2.4 执行查询语句
            try:
                with self._session() as session:
                    # 1. 执行查询种子节点语句
                    candidates_seed_nodes = self._execute_seed_nodes(session, item_name, entity_name)

                    # 2. 添加
                    final_seeds_result.extend(candidates_seed_nodes)

                    # 3. 判断
                    if len(final_seeds_result) > 30:
                        final_seeds_result = final_seeds_result[:30]
                        break

            except Exception as e:
                self.logger.error(f"获取种子节点失败,原因:{str(e)}")
                return []

        return final_seeds_result

    def _execute_seed_nodes(self, session, item_name, entity_name):
        # 1. 精确查询
        exact_rows = session.execute_read(
            lambda tx: tx.run(_CYPHER_EXACt_SEEDS, item_name=item_name, name=entity_name).data
        )
        # 2. 结果简单清洗
        if exact_rows:
            seeds_result = _clean_seed_rows(exact_rows)
            return seeds_result

        # 3. 模型查询
        fuzzy_rows = session.execute_read(
            lambda tx: tx.run(_CYPHER_FUZZY_SEEDS, item_name=item_name, entity_name=entity_name, limit=3).data
        )
        seeds_result = _clean_seed_rows(fuzzy_rows)
        return seeds_result

    def find_one_hop_relations(self, seed_nodes):
        """
        根据种子查询一跳的关系 除了chunk_id的关系
        :param seed_nodes:
        :return:
        """

        # 1. 判断种子节点
        if not seed_nodes:
            return []

        # 2. 遍历所有种子节点
        seen = set()
        final_result = []
        for seed_node in seed_nodes:
            # 2.1 获取item_name
            item_name = seed_node.get('item_name', '')
            # 2.2 提取实体名
            entity_name = seed_node.get('entity_name', '')
            # 2.3 判断是否存在
            if not item_name or not entity_name:
                continue
            # 2.4 执行Cypher语句
            try:
                with self._session() as session:
                    # 查询种子节点的一跳关系
                    seed_one_hop_relations = self._execute_one_hop_relations(session, item_name, entity_name)

                    if not seed_one_hop_relations:
                        continue

                    # 遍历所有的关系
                    for seed_one_hop_relation in seed_one_hop_relations:
                        # 获取头
                        head = seed_one_hop_relation.get('head')
                        # 获取rel
                        rel = seed_one_hop_relation.get('rel')
                        # 获取tail
                        tail = seed_one_hop_relation.get('tail')
                        # 获取item_name
                        item_name = seed_one_hop_relation.get('item_name')

                        # 去重
                        key = (head, rel, tail, item_name)

                        if key not in seen:
                            seen.add(key)
                            final_result.append(seed_one_hop_relation)

                    # 截取
                    if len(final_result) > 50:
                        final_result = final_result[:50]
                        break

            except Exception as e:
                self.logger.error(f"查询{entity_name}一跳关系节点失败,原因:{str(e)}")
                return []

        # 返回
        return final_result

    def _execute_one_hop_relations(self, session, item_name, entity_name):
        """

        :param session:
        :param item_name:
        :param entity_name:
        :return:
        """

        # 1. 执行查询方法
        one_hop_result = session.execute_read(
            lambda tx: tx.run(_CYPHER_ONE_HOP_RELATIONS, item_name=item_name, name=entity_name, limit=30).data
        )

        # 2. 解析结构
        if not one_hop_result:
            return []

        # 3. 遍历所有的一跳关系
        one_hop_relations_result = []
        for one_hop_relation in one_hop_result:
            # 3.1 提取head
            head = one_hop_relation.get('head', '').strip()
            # 3.2 获取rel
            rel = one_hop_relation.get('rel', '').strip()
            # 3.3 获取tail
            tail = one_hop_relation.get('tail', '').strip()

            # 3.4 判断
            if not head or not rel or not tail:
                continue

            # 3.5 将关系添加到最终结果中
            one_hop_relations_result.append({
                "head": head,
                "rel": rel,
                "tail": tail,
                "item_name": item_name
            })

        return one_hop_relations_result

    def by_weight(self, seed_nodes, one_hop_relations):
        """
        为种子节点设置权重 高,为邻居节点设置权重 低
        :param seed_nodes:
        :param one_hop_relations:
        :return:
        """

        # 1. 判断种子节点是否存在
        if not seed_nodes:
            return []

        # 2. 判断一跳关系是否存在
        if not one_hop_relations:
            return []

        # 3. 遍历所有种子节点
        weight_map = {}  # 存储所有节点 和 对应权重
        seen = set()
        for seed_node in seed_nodes:
            # 3.1 获取item_name
            item_name = seed_node.get('item_name')
            # 3.2 获取节点名
            entity_name = seed_node.get('entity_name')

            key = (item_name, entity_name)

            if key not in seen:
                seen.add(key)
                weight_map[key] = 2.0

        # 4. 遍历一跳关系节点
        for one_hop_relation in one_hop_relations:
            # 4.1 提取head
            head = one_hop_relation.get('head', '')
            # 4.2 获取item_name
            item_name = one_hop_relation.get('item_name')
            # 4.3 获取tail
            tail = one_hop_relation.get('tail', '')

            # 4.4 判断
            if (item_name, head) and (item_name, head) in weight_map:
                continue
            if (item_name, tail) and (item_name, tail) in weight_map:
                continue

            # 4.5
            weight_map[(item_name, head)] = 1.0
            weight_map[(item_name, tail)] = 1.0

        return [{"item_name": it, "entity_name": en, "weight": w} for (it, en), w in weight_map]

    def find_chunk_nodes(self, weight_nodes):
        """
        查询chunk文档节点
        :param weight_nodes:
        :return:
        """

        # 1. 执行Cypher语句
        try:
            with self._session() as session:
                sored_chunk_id_node = session.execute_read(
                    lambda tx: tx.run(_CYPHER_LOOKUP_CHUNK, weight_nodes=weight_nodes, limit=50).data)

        except Exception as e:
            self.logger.error(f"反查chunk节点失败,原因:{str(e)}")
            return []

        # 2. 出力结果
        hits = []
        for chunk_row in sored_chunk_id_node:
            chunk_id = chunk_row.get('chunk_id', '').strip()
            item_name = chunk_row.get('item_name', '').strip()
            score = chunk_row.get('score', '')

            if chunk_id and item_name:
                hits.append({
                    "id": None,
                    "distance": float(score or 0.0),
                    "entity": {"chunk_id": chunk_id, "item_name": str(item_name)}
                })
        return hits


class _ChunkBackFiller:
    """
    获取chunk_id
    根据chunk_id查询milvus的文档
    """

    def __init__(self, collection_name):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.collection_name = collection_name

    def back_file(self, chunk_nodes):
        """
        获取所有chunk_id
        查询milvus
        :param chunk_nodes:
        :return:
        """

        # 1. 判断chunk_nodes是否存在
        if not chunk_nodes:
            return []

        # 2. 获取chunk_ids
        chunk_ids = self._collect_chunk_ids(chunk_nodes)

        # 3. 查询milvus
        try:
            chunks = fetch_chunks_by_chunk_ids(collection_name=self.collection_name, chunk_ids=chunk_ids, batch_size=30)

            if not chunks:
                return []

        except Exception as e:
            self.logger.error(f"根据chunk_id批量查询chunk对象失败")
            return []

        # 4. 构建chunk_id和chunk映射表
        chunk_id_map = {str(chunk.get('chunk_id')): chunk for chunk in chunks if chunk.get("chunk_id") is not None}

        # 5. 根据真实chunk_id顺序查询有序的chunk对象
        return [{chunk_id_map[str(chunk_id)] for chunk_id in chunk_ids}]

    def _collect_chunk_ids(self, chunk_nodes):

        chunk_ids = []
        for chunk in chunk_nodes:
            # 判断chunk是否存在
            if not chunk:
                continue

            entity = chunk.get('entity', '')
            if not entity:
                continue

            chunk_id = entity.get('chunk_id', '')
            if not chunk_id:
                continue

            # chunk_id转换
            chunk_id_str = str(chunk_id)

            try:
                chunk_id_int = int(chunk_id_str)
                chunk_ids.append(chunk_id_int)

            except Exception as e:
                chunk_ids.append(chunk_id_str)

        return chunk_ids


class KnowLedeGraphSearchNode(BaseNode):
    name = "kg_query_node"

    @staticmethod
    def _parse_input(state: Dict[str, Any]) -> Tuple[str, List[str]]:
        question = state.get("rewritten_query")
        item_names = state.get("item_names")
        pattern = "|".join(re.escape(name) for name in item_names)
        user_query = re.sub(pattern, "", question).strip()
        return user_query, item_names

    def process(self, state: QueryGraphState) -> QueryGraphState:

        # 1. 参数校验
        validate_query, validate_item_names = self._validate_inputs(state)

        # 2. 执行流水线
        result = self._run_pipline(validate_query, validate_item_names)

        # 3. 返回state
        return {
            "kg_chunks": result.get("kg_chunks"),
            "kg_triples": result.get("kg_triples")
        }

    def _validate_inputs(self, state):
        # 1. 获取参数
        rewritten_query = state.get('rewritten_query')
        item_names = state.get('item_names')

        # 2. 判断是否存在
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name="rewritten_query", expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name="item_names", expected_type=list)

        # 3. 清洗问题 去除问题中的商品名
        rewritten_query, item_names = self._parse_input(state)

        # 4. 返回
        return rewritten_query, item_names

    def _run_pipline(self, validate_query, validate_item_names):

        # 1. 初始化组件
        entity_extractor = EntityExtractor()
        entity_aligner = EntityAligner(self.config.entity_name_collection)
        neo4j_graph_reader = Neo4jGraphReader("neo4j")
        chunk_back_filler = _ChunkBackFiller(self.config.chunks_collection)

        # 2. 利用提取器提取实体
        entity_name = entity_extractor.extract(user_query=validate_query)

        # 3. 通过llm识别实体名进行milvus进行向量检索和对齐
        entities_names_aligned = entity_aligner.align(entity_name, validate_item_names)

        # 4.构建商品名加实体名的pair对 ()
        aligned_entities_names = entities_names_aligned.get('entities_aligned', [])
        aligned_entities_info = entities_names_aligned.get('entities_elements', [])
        item_entity_pairs = _build_item_entity_pair(aligned_entities_info)

        # 5. 根据pair对 查询种子节点
        seed_nodes = neo4j_graph_reader.find_seed_nodes(item_entity_pairs)

        # 6. 根据种子节点查询一跳的关系
        one_hop_relations = neo4j_graph_reader.find_one_hop_relations(seed_nodes)

        # 7. 为每个节点设置权重
        weight_nodes = neo4j_graph_reader.by_weight(seed_nodes, one_hop_relations)

        # 8. 根据带权重节点反查chunk,基于权重/次数/id排序
        chunk_nodes = neo4j_graph_reader.find_chunk_nodes(weight_nodes)

        # 9. Milvus检索
        kg_chunks = chunk_back_filler.back_file(chunk_nodes)

        triples_docs = _one_hop_triples_to_texts(one_hop_relations)

        return {
            "kg_chunks": kg_chunks,
            "kg_triples": triples_docs,
        }
