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

### 3. 截取数据集子集

截取指定规模的子集，按原始问题类型比例均衡抽取，确保问题依据完整且语料库精简。

```bash
# 截取单个数据集（100个问题）
python Datasets/subset_datasets.py --dataset musique --num_questions 100
python Datasets/subset_datasets.py --dataset hotpotqa --num_questions 500
python Datasets/subset_datasets.py --dataset 2wikimultihop --num_questions 200

# 截取所有数据集
python Datasets/subset_datasets.py --dataset all --num_questions 100
```

**特性**：
- 按原始数据集的问题类型比例进行均衡抽取
- 确保问题的依据（evidence）在语料库中存在
- 过滤语料库，只保留问题涉及的文章
- **自动发现注册**：无需修改代码即可使用

**输出文件**：
```
Datasets/
├── Corpus/
│   ├── {dataset}_subset_{num}.json       # 截取后的语料库
│   └── ...
└── Questions/
    ├── {dataset}_questions_subset_{num}.json  # 截取后的问题集
    └── ...
```

**自动发现注册机制**：

`Examples/subset_registry.py` 会在导入时自动扫描并注册所有子集：
```bash
# 查看所有可用的子集
python Examples/subset_registry.py
```

输出示例：
```
============================================================
可用的数据集子集
============================================================

基础数据集:
  - sample
  - medical
  - hotpotqa_distractor
  - musique
  ...

自动发现的子集:
  - hotpotqa_500
  - musique_500
============================================================
```

**命名规则**：
- Corpus: `{dataset}_subset_{num}.json`
- Questions: `{dataset}_questions_subset_{num}.json`

只需将文件放入对应目录，即可被自动发现和使用。

**问题类型分布**：
| 数据集 | 类型字段 | 类型值 |
|--------|----------|--------|
| MuSiQue | question_type | 2hop, 3hop, 4hop |
| HotpotQA | question_type | bridge, comparison |
| 2WikiMultihop | question_type | compositional, comparison, inference, bridge_comparison |

### 4. 输出结构

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