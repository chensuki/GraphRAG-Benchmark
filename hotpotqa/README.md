# HotpotQA 数据集下载和使用指南

## 📋 简介

HotpotQA 是一个多跳问答数据集，需要跨多个文档进行推理。本工具将 HotpotQA 转换为评估框架所需格式，并控制规模比 Medical 数据集小。

## 🎯 目标规模

- **语料库文档数**: 50 个（比 Medical 小）
- **问题数**: 200 个（比 Medical 小）

## 📥 下载和转换

### 方法 1: 使用脚本（推荐）

```bash
# 1. 确保已安装依赖
pip install datasets pandas pyarrow

# 2. 运行下载和转换脚本
cd hotpotqa
python download_and_convert.py
```

### 方法 2: 手动下载

如果自动下载失败，可以手动下载：

```python
from datasets import load_dataset

# 下载 validation 集（比 train 集小）
dataset = load_dataset("hotpot_qa", "fullwiki", split="validation")
```

## 📊 输出文件

转换后的文件将保存在：

- **语料库**: `Datasets/Corpus/hotpotqa.parquet`
- **问题集**: `Datasets/Questions/hotpotqa_questions.json`

## 🔧 使用转换后的数据集

### 1. 更新运行脚本

需要在 `Examples/run_clearrag.py` 中添加 hotpotqa 配置：

```python
SUBSET_PATHS = {
    "medical": {...},
    "novel": {...},
    "hotpotqa": {
        "corpus": "./Datasets/Corpus/hotpotqa.parquet",
        "questions": "./Datasets/Questions/hotpotqa_questions.json"
    }
}
```

### 2. 运行评估

```bash
# 使用 ClearRAG
python Examples/run_clearrag.py \
  --subset hotpotqa \
  --config_path ClearRAG/config/config.yaml \
  --skip-build  # 如果已构建过

# 使用 LightRAG
python Examples/run_lightrag.py \
  --subset hotpotqa \
  --mode API \
  --model_name deepseek-chat \
  --llm_base_url https://api.deepseek.com/v1
```

## 📈 数据格式

### 语料库格式

```json
{
  "corpus_name": "HotpotQA",
  "context": "完整的维基百科文章文本..."
}
```

### 问题格式

```json
{
  "id": "HotpotQA-xxx",
  "source": "HotpotQA",
  "question": "问题文本",
  "answer": "答案文本",
  "question_type": "Complex Reasoning",
  "evidence": "支持事实的句子..."
}
```

## ⚠️ 注意事项

1. **首次下载**: 数据集较大，首次下载可能需要一些时间
2. **内存使用**: 处理完整数据集需要足够内存
3. **问题类型**: HotpotQA 的问题主要标记为 "Complex Reasoning"（多跳推理）

## 🔍 数据特点

- ✅ **完整语料库**: 维基百科完整文章
- ✅ **多跳推理**: 需要跨文档推理
- ✅ **精确证据**: 句子级别的证据标注
- ✅ **自然答案**: 开放式文本答案

## 📚 参考

- HotpotQA 官网: https://hotpotqa.github.io/
- HuggingFace: https://huggingface.co/datasets/hotpot_qa


