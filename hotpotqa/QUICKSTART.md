# HotpotQA 快速开始指南

## 🚀 快速开始（3步）

### 步骤 1: 安装依赖

```bash
pip install datasets pandas pyarrow
```

### 步骤 2: 下载和转换数据集

```bash
cd hotpotqa
python download_and_convert.py
```

这将：
- 自动下载 HotpotQA validation 集
- 转换为评估框架格式
- 控制规模为 50 个文档，200 个问题（比 Medical 小）

### 步骤 3: 运行评估

#### 使用 ClearRAG

```bash
# 构建知识图谱并回答问题
python Examples/run_clearrag.py \
  --subset hotpotqa \
  --config_path ClearRAG/config/config.yaml

# 如果已构建过，跳过构建阶段
python Examples/run_clearrag.py \
  --subset hotpotqa \
  --config_path ClearRAG/config/config.yaml \
  --skip-build
```

#### 使用 LightRAG

```bash
python Examples/run_lightrag.py \
  --subset hotpotqa \
  --mode API \
  --model_name deepseek-chat \
  --llm_base_url https://api.deepseek.com/v1 \
  --embed_provider zhipu \
  --embed_model embedding-3 \
  --retrieve_topk 5 \
  --base_dir ./lightrag_workspace
```

## 📊 预期输出

转换完成后，你将得到：

```
Datasets/
├── Corpus/
│   └── hotpotqa.parquet          # 50 个维基百科文档
└── Questions/
    └── hotpotqa_questions.json   # 200 个问题
```

## ⚙️ 自定义规模

如果想调整数据集规模，编辑 `download_and_convert.py`：

```python
# 修改这些常量
TARGET_CORPUS_SIZE = 50      # 语料库文档数
TARGET_QUESTIONS_SIZE = 200  # 问题数
```

## 🔍 验证数据

转换完成后，可以验证数据：

```python
import pandas as pd
import json

# 检查语料库
corpus = pd.read_parquet('Datasets/Corpus/hotpotqa.parquet')
print(f"语料库文档数: {len(corpus)}")
print(f"平均文档长度: {corpus['context'].str.len().mean():.0f} 字符")

# 检查问题
with open('Datasets/Questions/hotpotqa_questions.json', 'r', encoding='utf-8') as f:
    questions = json.load(f)
print(f"问题数: {len(questions)}")
print(f"示例问题: {questions[0]}")
```

## 📈 与 Medical 数据集对比

| 特性 | HotpotQA | Medical |
|------|----------|---------|
| 文档数 | 50 | ~100+ |
| 问题数 | 200 | ~500+ |
| 问题类型 | Complex Reasoning | 4 种类型 |
| 推理需求 | 多跳推理 | 单跳+多跳 |
| 语料库 | 维基百科 | 医疗文档 |

## ⚠️ 常见问题

### Q1: 下载失败怎么办？

**A**: 可以手动下载：
```python
from datasets import load_dataset
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
```

### Q2: 内存不足？

**A**: 减少目标规模：
```python
TARGET_CORPUS_SIZE = 30
TARGET_QUESTIONS_SIZE = 100
```

### Q3: 如何只使用部分问题？

**A**: 在运行脚本时使用 `--sample` 参数：
```bash
python Examples/run_clearrag.py \
  --subset hotpotqa \
  --config_path ClearRAG/config/config.yaml \
  --sample 50  # 只处理前 50 个问题
```

## 📚 更多信息

- 详细文档: `hotpotqa/README.md`
- 数据集分析: `数据集适配性分析.md`


