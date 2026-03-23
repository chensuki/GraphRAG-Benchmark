# 数据集 Schema（Dataset Schema）

定义 GraphRAG-Benchmark 支持的数据集格式。

---

## 语料库格式（Corpus）

**路径**：`Datasets/Corpus/{dataset}.json`

```json
[
  {
    "corpus_name": "sample_dataset",
    "context": "完整的文本内容...",
    "title": "文档标题（可选）"
  }
]
```

**字段说明**：
- `corpus_name`: 语料库名称（唯一标识）
- `context`: 完整文本内容
- `title`: 文档标题（可选）

---

## 问题集格式（Questions）

**路径**：`Datasets/Questions/{dataset}_questions.json`

```json
[
  {
    "id": "q1",
    "question": "问题内容",
    "answer": "标准答案",
    "question_type": "Fact Retrieval",
    "source": "corpus_name",
    "supporting_facts": [
      {"title": "文档名", "sent_id": 0}
    ]
  }
]
```

**字段说明**：
- `id`: 问题唯一标识
- `question`: 问题内容
- `answer`: 标准答案（ground truth）
- `question_type`: 问题类型（可选）
- `source`: 对应的语料库名称
- `supporting_facts`: 支持事实（可选，用于多跳推理评估）

---

## 预测结果格式（Predictions）

**路径**：`results/{framework}/{subset}/{run_id}/predictions_{corpus_name}.json`

```json
[
  {
    "id": "q1",
    "question": "问题内容",
    "source": "corpus_name",
    "context": [
      {"type": "chunk", "content": "检索到的文本块"},
      {"type": "entity", "content": "实体名 (类型): 描述"}
    ],
    "generated_answer": "模型生成的答案",
    "ground_truth": "标准答案"
  }
]
```

**Context 类型**：
- `chunk`: 文本块
- `entity`: 实体
- `relationship`: 关系

---

## 可用数据集

| 数据集 | Corpus | Questions | 特点 |
|--------|--------|-----------|------|
| sample | sample.json | sample_questions.json | 测试用小样本 |
| medical | medical.json | medical_questions.json | 医疗领域 |
| medical_100 | medical_100.json | medical_100_questions.json | 医疗子集 |
| novel | novel.json | novel_questions.json | 小说领域 |
| hotpotqa_distractor | hotpotqa_distractor.json | hotpotqa_distractor_questions.json | 多跳推理 |
| hotpotqa_fullwiki | hotpotqa_fullwiki.json | hotpotqa_fullwiki_questions.json | 全量维基 |
| 2wikimultihop | 2wikimultihop.json | 2wikimultihop_questions.json | 多跳推理 |
| musique | musique.json | musique_questions.json | 多跳推理 |

---

## 子集创建

```bash
python Datasets/subset_datasets.py --dataset hotpotqa --num_questions 100
```

**输出**：
- `Corpus/{dataset}_subset_{num}.json`
- `Questions/{dataset}_questions_subset_{num}.json`

子集自动被 `Examples/subset_registry.py` 发现，无需手动注册。