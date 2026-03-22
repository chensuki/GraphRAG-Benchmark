# Framework 参数映射（Framework Parameter Map）

## 共享公平锁

同一比较组内，优先锁定：
- dataset subset
- 当前阶段 question sample count
- retrieval top-k
- llm model / provider family
- embedding type / model
- evaluation script

---

## LinearRAG

### 主要参数
- `max_iterations`：最大迭代次数（默认 3）
- `iteration_threshold`：迭代收敛阈值（默认 0.4）
- `top_k_sentence`：每个实体检索的 top-k 句子（默认 3）
- `spacy_model`：NER 模型（默认 en_core_web_trf）
- `use_vectorized`：是否向量化检索（默认 false）
- `skip_build`：是否跳过索引构建（默认 false）
- `chunk_token_size`：分块大小（默认 1200）
- `chunk_overlap_token_size`：分块重叠（默认 100）

### 运行阶段
- passage encoding
- NER
- sentence / entity graph build
- query prediction

### 操作建议
- 先小 sample 验证，再放大 sample，并切换 `skip_build: true`
- 调整 `iteration_threshold` 可以优化检索精度 vs 速度的平衡

---

## LightRAG

### 主要参数
- `mode`：LLM 调用模式（API / ollama）
- `embed_type`：embedding 类型（local / api）
- `embed_provider`：embedding 提供商（local / openai / ollama）
- `chunk_token_size`：分块大小（默认 1200）
- `chunk_overlap_token_size`：分块重叠（默认 100）
- `skip_build`：是否跳过索引构建（默认 false）

### 运行阶段
- chunking
- entity / relationship extraction
- graph indexing
- vector indexing
- hybrid query

### 操作建议
- query mode 在 adapter 中默认是 `hybrid`，当前 Runner 没有开放成 YAML 参数
- 提取格式的 warning 不一定致命，但要记录到 diagnosis
- 优先本地 embedding，遇到兼容性问题再切 API

---

## ClearRAG

### 主要参数
- `activation_mode`：激活模式（semantic_propagation / vector_search）
- `passage_retrieval_mode`：段落检索模式（vector / pagerank / none）
- `fast`：快速模式（禁用 Reflexion）
- `max_tool_calls`：Agent 工具调用上限（3~10）
- `max_tokens`：LLM 输出 token 上限（默认 8192）
- `similarity_threshold`：实体去重相似度阈值（默认 0.95）
- `vector_search_top_k`：向量搜索召回数量（默认 100）
- `activation_threshold`：向量激活阈值（默认 0.8）
- `max_context_length`：上下文最大字符数（默认 10000）
- `max_passage_content_length`：单段落最大字符数（默认 2000）
- `chunk_size`：分块大小（默认 1200）
- `chunk_overlap`：分块重叠（默认 100）
- `skip_build`：是否跳过索引构建（默认 false）

### Neo4j 数据库命名
- 例子：`musique-100-001`
- 需要手动在 Neo4j 中创建数据库

### 操作经验
- 如果本地 Neo4j 需要，先手动创建 database
- 图写入 / relation merge 往往是最重阶段
- `cartesian product` warning 多半说明性能问题，不一定是致命失败
- 依旧建议先小 sample，再放大 sample，并切换 `skip_build: true`

---

## HippoRAG2

### 主要参数
- `mode`：LLM 调用模式（API / ollama）
- `embed_model_path`：本地 embedding 模型路径（默认 BAAI/bge-large-en-v1.5）
- `chunk_token_size`：分块 token 大小（默认 256）
- `chunk_overlap`：分块重叠（默认 32）
- `force_index_from_scratch`：强制从头构建索引（默认 true）
- `rerank_dspy_file_path`：rerank 配置文件路径
- `max_qa_steps`：QA 最大步骤数（默认 3）
- `graph_type`：图类型（默认 facts_and_sim_passage_node_unidirectional）
- `embedding_batch_size`：embedding batch size（默认 8）
- `openie_mode`：OpenIE 模式（online / offline，默认 online）

### 特殊说明
- HippoRAG2 **仅支持本地 embedding 模型**，不支持 API embedding
- HippoRAG2 依赖环境变量 `OPENAI_API_KEY`，不能从 YAML 直接读取
- 如果配置了 `llm.api_key`，需要额外设置：
  ```bash
  export OPENAI_API_KEY=$LLM_API_KEY
  ```

### 操作建议
- 优先本地 embedding（HippoRAG2 原生支持）
- 确保环境变量设置正确
- `graph_type` 可以影响多跳推理效果

---

## Fast-GraphRAG

### 主要参数
- `mode`：LLM 调用模式（API / ollama）
- `domain`：自定义分析域（空则使用默认）
- `entity_types`：自定义实体类型（空则使用默认）
- `example_queries`：示例查询（用于优化检索）

### 操作建议
- 更轻量，适合快速做 graph-style baseline
- 若使用本地 embedding，需要依赖对上游包的自定义适配
- 建议先 P0/P2，而不是一开始就纳入无人值守长跑

---

## DigiMON

### 主要参数
- `mode`：LLM 调用模式（API / ollama）
- `config_path`：DIGIMON 配置文件路径（默认 ./config.yml）

### 状态说明
- 当前 adapter 在，但主链路接入价值还不高
- 更适合作为"后续接入项"，不适合当前自动化主链路
- `abatch_query()` 按设计应为 `List[question_dict]`，需验证实际实现

---

## 框架稳定性评估（基于历史实验）

### 🟢 高稳定性（建议作为 P0 主力）

| 框架 | 实验成功次数 | 失败次数 | 主要问题 | 评估 |
|------|-------------|---------|---------|------|
| **linearrag** | 多次 | 无 | 答案格式问题（可修复） | ✅ 可靠 |
| **lightrag** | 多次 | 无 | 提取格式 warning（非致命） | ✅ 可靠 |
| **clearrag** | 多次 | 无 | 需要手动创建 Neo4j 数据库 | ✅ 可靠 |

### 🟡 中等稳定性（可作为 P1 扩展）

| 框架 | 实验成功次数 | 失败次数 | 主要问题 | 评估 |
|------|-------------|---------|---------|------|
| **hipporag2** | 有记录 | 少量 | 需要环境变量设置 | ⚠️ 需谨慎 |

### 🔴 低稳定性或未验证（不建议作为 P0）

| 框架 | 实验成功次数 | 失败次数 | 主要问题 | 评估 |
|------|-------------|---------|---------|------|
| **fast-graphrag** | 无 | 未知 | 无历史记录 | ❌ 未验证 |
| **digimon** | 无 | 未知 | 无历史记录，接口可能不一致 | ❌ 未验证 |

---

## 参数锁定层级

### Layer A（严格锁定，不能漂移）
- 数据集子集：`dataset.subset`
- 问题采样：`dataset.sample`
- 语料采样：`dataset.corpus_sample`
- 评估脚本版本
- question file / corpus file 版本
- 输出 schema 版本
- 代码 commit / working tree 状态

### Layer B（同一比较组内必须锁死）
- LLM 模型和 endpoint（`llm.model`, `llm.base_url`）
- embedding 配置（`embedding.type`, `embedding.provider`, `embedding.model`）
- 检索参数（`retrieval.top_k`）
- chunk 策略（size / overlap）
- 是否 `skip_build`
- 索引是否复用
- workspace 隔离策略

### Layer C（允许调，但一次只改一个主变量）
- LightRAG：`chunk_token_size`, `chunk_overlap_token_size`
- ClearRAG：`activation_mode`, `passage_retrieval_mode`, `fast`
- LinearRAG：`max_iterations`, `iteration_threshold`, `top_k_sentence`, `use_vectorized`
- HippoRAG2：`graph_type`, `max_qa_steps`, `openie_mode`

**核心规则**：一组实验里一次只改一个主变量。
