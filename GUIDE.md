# GraphRAG-Benchmark 运行指南
---

## 框架概览

| 框架 | 状态 | 适配器 | 特点 |
|------|------|--------|------|
| **LightRAG** | ✅ 可用 | `lightrag_adapter.py` | API/Ollama双模式，多种嵌入提供商 |
| **ClearRAG** | ✅ 可用 | `clearrag_adapter.py` | Neo4j图数据库，语义传播激活 |
| **LinearRAG** | ✅ 可用 | `linearrag_adapter.py` | 迭代检索，SpaCy NER |
| **Fast-GraphRAG** | ✅ 可用 | `fast_graphrag_adapter.py` | 轻量级图RAG |
| **HippoRAG2** | ✅ 可用 | `hipporag2_adapter.py` | 海马体记忆模拟 |
| **DigiMON** | ✅ 可用 | `digimon_adapter.py` | 数字化知识图谱 |

### 支持的数据集

| 数据集 | 子集名 | 说明 |
|--------|--------|------|
| Sample | `sample` | 样本数据集（快速测试） |
| Medical | `medical` | 医疗领域数据集 |
| Medical_100 | `medical_100` | 医疗数据集100条平衡采样 |
| Novel | `novel` | 小说数据集（20个语料） |
| HotpotQA Distractor | `hotpotqa_distractor` | 多跳问答（预填充文档） |
| HotpotQA Fullwiki | `hotpotqa_fullwiki` | 多跳问答（完整Wikipedia） |
| 2WikiMultihop | `2wikimultihop` | 两跳维基问答 |
| MuSiQue | `musique` | 多跳问答数据集 |

---

## 环境安装

### 统一虚拟环境（推荐）

```powershell
# 1. 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip

# 2. 安装主依赖
pip install -r requirements.txt

# 3. 安装各框架（可编辑模式）
pip install -e ./clearrag
pip install -e ./LightRAG
pip install -e ./linearrag

# 4. 验证安装
pip check
```

### 额外依赖

```powershell
# LinearRAG 需要的 SpaCy 模型
python -m spacy download en_core_web_trf

# Fast-GraphRAG（如需使用）
pip install fast-graphrag
# 本地 embedding 支持（可选）
pip install transformers torch

# HippoRAG2（如需使用）
git clone https://github.com/OSU-NLP-Group/HippoRAG.git
pip install -e ./HippoRAG
# 下载本地 embedding 模型（必需，HippoRAG 不支持 API embedding）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# DigiMON（如需使用）
git clone https://github.com/JayLZhou/GraphRAG.git
cd GraphRAG
conda env create -f experiment.yml -n digimon
conda activate digimon
# 配置 LLM：编辑 config.yaml，设置 api_type/model/api_key
# 运行：python main.py -opt Option/Method/LightRAG.yaml -dataset_name your_dataset
```

---

## 配置文件说明

### 首次使用：创建配置文件

```powershell
# 从模板复制配置文件
cp configs/template.yaml configs/experiment.yaml

# 编辑配置，填入你的 API 密钥
notepad configs/experiment.yaml
```


### 配置优先级

1. **框架配置** > **全局配置**（如 `frameworks.lightrag.embed_provider` 覆盖 `embedding.provider`）
2. **环境变量** 作为默认值（如 `LLM_API_KEY`）
3. **路径模板** 支持 `{run_id}`、`{framework}` 变量

---

## 运行命令

### 推荐方式：统一 YAML 配置

```powershell
# 预览配置（不执行）
python Examples/run_from_yaml.py --config configs/experiment.yaml --dry-run

# 运行配置中指定的框架（run.framework）
python Examples/run_from_yaml.py --config configs/experiment.yaml

# 强制运行单个框架
python Examples/run_from_yaml.py --framework lightrag

# 运行所有启用的框架（enabled: true）
python Examples/run_from_yaml.py --framework all
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `configs/experiment.yaml` |
| `--framework` | 框架名称（auto/all/lightrag/clearrag/...） | `auto` |
| `--dry-run` | 仅预览配置，不执行 | `false` |

---

## 框架详细配置

### LightRAG

```yaml
frameworks:
  lightrag:
    enabled: true
    workspace_dir: ./workspace/lightrag_workspace
    mode: API                           # API 或 ollama
    embed_provider: zhipu               # 覆盖全局 embedding.provider
    corpus_concurrency: 1               # 语料库并发数
    skip_build: false                   # 跳过索引构建
    index_only: false                   # 仅构建索引
    chunk_token_size: 1200              # 分块大小
    chunk_overlap_token_size: 100       # 分块重叠
```

**特点**：
- 支持 API 和 Ollama 两种模式
- 支持多种嵌入提供商：`zhipu`、`openai`、`api`、`hf`（本地）
- 使用 `hybrid` 检索模式（向量 + 图遍历）

### ClearRAG

```yaml
frameworks:
  clearrag:
    enabled: true
    workspace_dir: ./workspace/clearrag_workspace
    activation_mode: semantic_propagation   # vector_search / semantic_propagation
    passage_retrieval_mode: pagerank        # vector / pagerank / none
    fast: false                             # 快速模式（禁用 Reflexion）
    max_concurrency: 5                      # 构建并发数
    corpus_concurrency: 2                   # 语料库并发数
```

**检索模式说明**：

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `semantic_propagation` + `pagerank` | 高精度，需Linear索引 | 生产环境 |
| `vector_search` + `none` | 快速，无需额外文件 | 快速测试 |
| `--fast` | 禁用自反思 | API限流场景 |

**依赖**：需要运行 Neo4j 数据库

### LinearRAG

```yaml
frameworks:
  linearrag:
    enabled: true
    workspace_dir: ./workspace/linearrag_workspace
    max_iterations: 3                 # 最大迭代次数
    iteration_threshold: 0.4          # 迭代阈值
    top_k_sentence: 3                 # 每个实体的 top-k 句子数
    use_vectorized: false             # 向量化检索
    spacy_model: en_core_web_trf      # NER 模型
    max_workers: 8                    # NER 并发数
    corpus_concurrency: 1
```

**特点**：
- 使用 SpaCy 进行命名实体识别
- 支持迭代检索，逐步扩展上下文
- 支持向量化检索路径

### Fast-GraphRAG

```yaml
frameworks:
  fast-graphrag:
    enabled: true
    workspace_dir: ./workspace/fast-graphrag_workspace
    mode: API
    embed_provider: api               # api 或 local
    domain: ""                        # 自定义分析域
    entity_types: []                  # 自定义实体类型
```

### HippoRAG2

```yaml
frameworks:
  hipporag2:
    enabled: true
    workspace_dir: ./workspace/hipporag2_workspace
    mode: API
    embed_model_path: "BAAI/bge-large-en-v1.5"  # 本地 embedding 模型
    chunk_token_size: 256
    chunk_overlap: 32
    force_index_from_scratch: true
    max_qa_steps: 3
    graph_type: "facts_and_sim_passage_node_unidirectional"
    embedding_batch_size: 8
    openie_mode: "online"             # online / offline
```

---

## 输出结果

### 路径结构

```
results/
├── {framework}/
│   └── {subset}/
│       └── {run_id}/
│           ├── predictions_{corpus_name}.json    # 预测结果
│           └── workspace/                         # 索引文件
```

### 结果格式

```json
[
  {
    "id": "q1",
    "question": "问题内容",
    "source": "corpus_name",
    "context": ["检索到的上下文1", "上下文2"],
    "generated_answer": "模型生成的答案",
    "ground_truth": "标准答案",
    "question_type": "Fact Retrieval"
  }
]
```

---

## 评估

### 统一评估（推荐）

使用 `unified_eval.py` 进行一站式评估，自动检测数据集格式并计算所有适用指标：

```powershell
python -m Evaluation.unified_eval `
  --data_file ./results/clearrag/hotpotqa_500/20260317-125800/predictions_hotpotqa_500.json `
  --output_file ./results/evaluations/clearrag_hotpotqa_eval.json `
  --report
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--data_file` | 预测结果文件路径（必需） |
| `--output_file` | 评估结果输出路径（必需） |
| `--detailed` | 输出详细结果（包含每条样本得分） |
| `--report` | 打印格式化报告 |

**评估指标**：

| 指标类别 | 具体指标 | 说明 |
|----------|----------|------|
| 答案质量 | `answer_em`, `answer_f1`, `rouge_score` | 精确匹配、F1、ROUGE |
| 检索质量 | `evidence_coverage` | 证据覆盖度 |
| 支持事实 | `sf_em`, `sf_f1` | 支持事实召回（HotpotQA/MuSiQue） |
| 联合指标 | `joint_em`, `joint_f1` | 答案+支持事实联合评估 |
| 三元组 | `triple_f1` | 知识三元组评估（2Wiki） |
| 推理步骤 | `step_accuracy` | 推理步骤评估（MuSiQue） |
| Hop分层 | `hop_stratified` | 按跳数分层评估（MuSiQue） |

**支持的格式**：
- 自动检测 HotpotQA / MuSiQue / 2WikiMultihop / Medical 格式
- 根据数据集字段自动启用对应评估

### 检索评估

评估上下文相关性和证据召回率：

```powershell
python -m Evaluation.retrieval_eval `
  --mode API `
  --model deepseek-chat `
  --base_url https://api.deepseek.com/v1 `
  --data_file ./results/lightrag/medical/exp-001/predictions_medical.json `
  --output_file ./results/retrieval_eval.json `
  --num_samples 10
```

**指标**：
- `context_relevancy`: 上下文相关性
- `evidence_recall`: 证据召回率

### 生成评估

评估答案质量：

```powershell
python -m Evaluation.generation_eval `
  --mode API `
  --model deepseek-chat `
  --base_url https://api.deepseek.com/v1 `
  --embedding_model BAAI/bge-large-en-v1.5 `
  --data_file ./results/lightrag/medical/exp-001/predictions_medical.json `
  --output_file ./results/generation_eval.json
```

**指标**：
- `rouge_score`: ROUGE 分数
- `answer_correctness`: 答案正确性
- `faithfulness`: 忠实度
- `coverage_score`: 覆盖度

### 索引评估

评估图结构质量：

```powershell
# LightRAG / Fast-GraphRAG
python -m Evaluation.indexing_eval `
  --framework lightrag `
  --base_path ./workspace/lightrag_workspace `
  --output ./results/indexing_metrics.txt

# HippoRAG2
python -m Evaluation.indexing_eval `
  --framework hipporag2 `
  --base_path ./workspace/hipporag2_workspace `
  --folder_name Medical
```

**指标**：节点数、边数、平均度、聚类系数、连通分量数等

---

## 环境变量

### 必需环境变量

```powershell
# LLM API
$env:LLM_API_KEY="your-llm-api-key"

# Embedding API（可选，也可在 YAML 中配置）
$env:ZHIPUAI_API_KEY="your-zhipu-key"
```

### ClearRAG 额外环境变量

```powershell
$env:NEO4J_URI="bolt://localhost:7687"
$env:NEO4J_USER="neo4j"
$env:NEO4J_PASSWORD="your-password"
$env:NEO4J_DATABASE="neo4j"
```

---

## 快速测试流程

```powershell
# 1. 激活环境
.\.venv\Scripts\Activate.ps1

# 2. 创建配置文件（首次使用）
cp configs/template.yaml configs/experiment.yaml

# 3. 编辑配置（填入 API 密钥）
notepad configs/experiment.yaml

# 4. 预览配置
python Examples/run_from_yaml.py --dry-run

# 5. 运行测试
python Examples/run_from_yaml.py --framework lightrag

# 6. 查看结果
cat ./results/lightrag/sample/exp-001/predictions_sample_dataset.json
```

---

## 常见问题

### 1. 模块导入错误

```powershell
# 确保已安装框架
pip install -e ./LightRAG
pip install -e ./clearrag
pip install -e ./linearrag
```

### 2. Neo4j 连接失败（ClearRAG）

```powershell
# 检查 Neo4j 状态
docker ps | findstr neo4j

# 启动 Neo4j
docker start neo4j-clearrag
```

### 3. 查询结果为空

**原因**：索引未完成或被中断

**解决**：
1. 检查 `workspace/{framework}/{corpus}/` 目录是否有完整索引文件
2. 设置 `skip_build: false` 重新构建索引
3. 降低 `corpus_concurrency` 避免并发冲突

### 4. API 限流

**解决**：
- 降低 `corpus_concurrency` 至 1
- 设置 `fast: true`（ClearRAG）
- 减少 `sample` 数量

### 5. 内存不足

**解决**：
- 降低 `max_concurrency` 参数
- 减少 `sample` 数量
- 使用 `index_only: true` 分阶段执行

---

## 高级用法

### 分阶段执行（索引 + 查询分离）

```yaml
# 第一阶段：仅构建索引
frameworks:
  lightrag:
    skip_build: false
    index_only: true

# 第二阶段：仅查询
frameworks:
  lightrag:
    skip_build: true
    index_only: false
```

### 并行运行多框架

```yaml
run:
  framework: all  # 运行所有 enabled: true 的框架
  continue_on_error: true  # 单个框架失败不影响其他
```

### 自定义输出路径

```yaml
frameworks:
  lightrag:
    workspace_dir: ./custom/path/{run_id}
```

### 接入openclaw

#### 输入认证token以及会话id


