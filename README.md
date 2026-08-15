

# 智析智库知识库系统（RAG智能问答）
## 项目简介
**掌柜问数知识库系统** 是面向商品知识库场景、基于标准RAG（检索增强生成）架构构建的企业级智能问答系统。
系统完整实现PDF文档批量导入、多路并行检索、HyDE查询增强、混合稠密+稀疏向量检索、结果重排序、SSE流式答案输出、MCP协议联网补充信息等核心能力，全流程采用节点化LangGraph工作流编排，专门适配商品领域知识库问答业务，整套架构、算法逻辑与服务代码均独立设计实现。

## 核心能力
1. 文档处理：PDF转Markdown、图片自动抽取&对象存储、语义切片、商品实体识别
2. 向量化引擎：BGE-M3混合稠密/稀疏向量生成，Milvus混合向量检索
3. 检索增强：HyDE假设文档生成、多路并行检索、RRF倒数排名融合、Reranker重排、断崖动态TopK过滤
4. 对话服务：历史对话持久化、LLM流式SSE输出、MCP协议联网补充信息
5. 工程架构：模块化节点设计、全异步FastAPI服务、Docker一键部署、完善异常与重试机制

## 技术栈与第三方中间件
### 1. 底层存储中间件
| 组件    | 官网                    | 核心用途                                |
| ------- | ----------------------- | --------------------------------------- |
| Milvus  | https://milvus.io       | 向量数据库，存储稠密/稀疏向量，混合检索 |
| MongoDB | https://www.mongodb.com | 文档数据库，持久化对话历史记录          |
| MinIO   | https://min.io          | 对象存储，保存原始PDF、文档内图片资源   |

### 2. AI大模型与框架
| 组件               | 官网                                           | 核心用途                               |
| ------------------ | ---------------------------------------------- | -------------------------------------- |
| LangChain          | https://python.langchain.com                   | LLM统一调用封装、提示词管理            |
| LangGraph          | https://langchain-ai.github.io/langgraph       | DAG工作流编排，导入/查询全流程节点调度 |
| BGE-M3             | https://huggingface.co/BAAI/bge-m3             | 混合嵌入模型，一次输出稠密+稀疏向量    |
| BGE-Reranker-Large | https://huggingface.co/BAAI/bge-reranker-large | 交叉编码器，检索结果精准重排序         |
| FlagEmbedding      | https://github.com/FlagOpen/FlagEmbedding      | 嵌入模型工具集，封装向量化、重排逻辑   |
| marker-pdf         | https://github.com/VikParuchuri/marker         | PDF高精度转Markdown，保留图文结构      |

### 3. Web服务与通信协议
| 组件     | 官网                            | 核心用途                             |
| -------- | ------------------------------- | ------------------------------------ |
| FastAPI  | https://fastapi.tiangolo.com    | 高性能异步Web后端，提供导入/查询API  |
| Pydantic | https://docs.pydantic.dev       | 请求响应数据校验、序列化             |
| Uvicorn  | https://uvicorn.org             | ASGI异步服务启动器                   |
| SSE      | MDN Web Docs                    | 服务端事件推送，实现流式打字机输出   |
| MCP      | https://modelcontextprotocol.io | 模型上下文协议，对接外部网络搜索节点 |

### 4. Python核心依赖库
- pymilvus：Milvus向量库Python SDK
- pymongo：MongoDB客户端
- minio-py：MinIO对象存储SDK
- python-dotenv：环境变量统一管理
- asyncio：全流程异步IO支持

## 项目整体业务流程
项目分为两大核心业务链路：**文档导入处理链路**、**用户查询问答链路**，所有流程均自研节点化拆分，解耦易维护。
### 一、文档导入链路流程
1. 文件入口校验：接收PDF文件，调用marker-pdf完成PDF转Markdown文本
2. 图片资源处理：提取文档内图片，上传至MinIO对象存储，替换原文图片链接为可访问URL
3. 语义切片：基于Token长度与语义边界对长文档进行分段切分
4. 商品实体识别：通过LLM提取切片内商品名称，完成实体与文本向量对齐
5. 混合向量编码：使用BGE-M3对切片文本生成稠密向量、稀疏词权重向量
6. 向量批量入库：将文本、图片地址、双份向量批量写入Milvus向量库

### 二、用户查询问答链路流程
1. 查询预处理：识别用户问题中商品指代，完成指代消解与查询重写优化
2. 多路并行检索分支
   - 基础向量检索：使用原始问题向量在Milvus执行稠密+稀疏混合检索
   - HyDE增强检索：LLM生成假设性标准答案，以假设文本二次检索补充召回
3. RRF多路结果融合：对两条检索分支返回结果执行倒数排名融合，合并统一候选列表
4. Rerank精准重排：BGE-Reranker对全部候选文档打分，通过自研断崖检测算法动态截取高相关片段
5. 答案生成：结合筛选后的上下文与领域定制提示词，调用LLM生成回答
6. 流式输出/联网补充：通过SSE协议逐字推送回答；问题信息不足时走MCP节点联网补充外部资料

## 核心技术亮点
### 1. BGE-M3混合向量检索
单模型一次编码同时输出稠密语义向量 + 稀疏词权重向量，Milvus加权混合检索平衡语义相似度与关键词精准匹配
```python
# 生成混合向量
dense_vectors = model.encode(texts)['dense_vecs']
sparse_vectors = model.encode(texts)['lexical_weights']

# Milvus混合搜索
search_requests = [
    AnnSearchRequest(dense_vectors, "dense_vector", ...),
    AnnSearchRequest(sparse_vectors, "sparse_vector", ...),
]
results = collection.hybrid_search(
    search_requests,
    rerank=WeightedRanker(0.7, 0.3) # 稠密0.7，稀疏0.3权重可调
)
```

### 2. 多路并行检索架构
基于LangGraph DAG自主设计多检索分支并行执行（向量检索 + HyDE增强检索），降低单次问答整体耗时，检索结果通过自研RRF融合逻辑统一排序。

### 3. HyDE假设性文档增强
自主实现HyDE检索增强逻辑：通过LLM基于用户问题生成假设性标准答案，使用假设文档做向量检索，弥补短问句语义缺失问题，大幅提升小样本商品知识库召回率。

### 4. 自研断崖检测动态TopK
针对Reranker打分序列设计分值断崖识别算法，自动截断低相关片段，摒弃固定TopK方案，有效减少无关上下文噪声，显著提升LLM回答准确度。

### 5. SSE流式实时输出
基于SSE协议封装问答接口，逐字分段推送生成内容，前端实现打字机流式效果，同步支持联网搜索内容实时穿插返回。

## 部署指南
### 开发环境 Docker Compose
```yaml
version: '3.8'
services:
  milvus:
    image: milvusdb/milvus:v2.3.0
    ports:
      - "19530:19530"
    volumes:
      - ./volumes/milvus:/var/lib/milvus
  mongodb:
    image: mongo:6.0
    ports:
      - "27017:27017"
  minio:
    image: minio/minio:latest
    ports:
      - "9000:9000"
      - "9001:9001"
    command: server /data --console-address ":9001"
```
1. 启动依赖存储服务：`docker-compose -f docker-compose.dev.yml up -d`
2. 配置`.env`环境变量：向量库地址、MinIO账号、LLM接口地址、模型路径
3. 启动Web服务：`uvicorn main:app --host 0.0.0.0 --port 8000`

### 生产环境部署建议
1. Milvus集群化部署，开启HNSW高性能向量索引
2. MongoDB配置副本集保障对话数据高可用
3. MinIO配置分布式对象存储，开启文件分片备份
4. 前端+后端Nginx反向代理，SSE长连接单独优化
5. 增加服务健康检查、熔断、重试、日志采集监控

## 整体架构总结
1. **分层解耦架构**：API接入层 → 业务服务层 → LangGraph流程编排层 → 工具能力层 → 底层存储基础设施
2. **自研节点化设计**：文档导入、查询问答每个步骤独立封装为LangGraph节点，可单独调试、替换、扩展
3. **并行调度能力**：多路检索并行执行，缩短单次问答耗时
4. **流式交互体验**：原生SSE实现实时打字机输出，支持边检索边生成边推送
5. **高容错设计**：单节点执行失败不阻断整体流程，降级返回可用结果



