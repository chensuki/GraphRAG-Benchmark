# Datasets

## 支持的数据集

| 数据集 | 描述 | 来源 |
|--------|------|------|
| hotpotqa_distractor | HotpotQA 预填充文档版本 | hotpotqa/hotpot_qa |
| hotpotqa_fullwiki | HotpotQA 完整 Wikipedia 版本 | hotpotqa/hotpot_qa |
| 2wikimultihop | 2WikiMultihopQA 两跳问答 | framolfese/2WikiMultihopQA |
| musique | MuSiQue 多跳问答（可回答样本） | bdsaglam/musique |
| ultradomain | UltraDomain 长文本问答 | TommyChien/UltraDomain |

## 使用方法

### 1. 下载原始数据

```bash
# 创建目录
mkdir -p Datasets/raw

# 下载 HotpotQA
cd Datasets/raw
git clone https://huggingface.co/datasets/hotpotqa/hotpot_qa hotpot_qa

# 下载 2WikiMultihopQA
git clone https://huggingface.co/datasets/framolfese/2WikiMultihopQA 2wikimultihop

# 下载 MuSiQue
git clone https://huggingface.co/datasets/bdsaglam/musique musique

# 下载 UltraDomain
git clone https://huggingface.co/datasets/TommyChien/UltraDomain ultradomain
```

### 2. 转换数据集

```bash
# 转换单个数据集
python Datasets/download_datasets.py --datasets hotpotqa_distractor

# 转换所有数据集
python Datasets/download_datasets.py --datasets all

# 强制重新转换
python Datasets/download_datasets.py --datasets hotpotqa_distractor --force

# 查看所有支持的数据集
python Datasets/download_datasets.py --list
```

### 3. 输出结构

```
Datasets/
├── raw/                        # 原始下载数据
│   ├── hotpot_qa/
│   ├── 2wikimultihop/
│   ├── musique/
│   └── ultradomain/
├── Corpus/                     # 解析后语料库（框架使用）
│   ├── hotpotqa_distractor.parquet
│   ├── hotpotqa_fullwiki.parquet
│   ├── 2wikimultihop.parquet
│   ├── musique.parquet
│   └── ultradomain.parquet
└── Questions/                  # 解析后问题集（评估使用）
    ├── hotpotqa_distractor_questions.parquet
    ├── hotpotqa_fullwiki_questions.parquet
    ├── 2wikimultihop_questions.parquet
    ├── musique_questions.parquet
    └── ultradomain_questions.parquet
```

## 数据集字段

### Questions 格式

| 字段 | 说明 |
|------|------|
| id | 问题唯一标识 |
| source | 数据集来源 |
| question | 问题文本 |
| answer | 标准答案 |
| evidence | 证据字符串 |
| question_type | 问题类型（Complex Reasoning） |
| type | 原始类型（可选） |
| supporting_facts | 支持事实（多跳数据集） |
| evidences | 结构化证据（2Wiki） |

### Corpus 格式

| 字段 | 说明 |
|------|------|
| corpus_name | 语料名称 |
| context | 合并后的文本内容 |