# 结果命名与完成规则（Result Spec）

该文件定义 GraphRAG-Benchmark 实验的结果命名规范和完成判定规则。

## 1. Run ID 命名规范

建议使用以下格式：

```
{framework}_{dataset}_{sample}_{timestamp}
```

示例：
```
linearrag_sample_5_20260322-220500
lightrag_medical_100_null_20260322-221500
clearrag_hotpotqa_100_10_20260322-230000
```

字段说明：
- `framework`: 框架名称（linearrag/lightrag/clearrag/...）
- `dataset`: 数据集子集（sample/medical_100/hotpotqa_100/...）
- `sample`: 采样数（5/10/null/...）
- `timestamp`: 时间戳（YYYYMMDD-HHMMSS）

---

## 2. 结果目录结构

```
results/
  {framework}/
    {subset}/
      {run_id}/
        predictions_{corpus_name}.json
        config_effective.json           # [重要] 实际生效的配置
        eval_unified_{corpus_name}.json
        retrieval_eval.json              # 可选
        generation_eval.json             # 可选
        stdout.log                      # 可选
        stderr.log                      # 可选
```

示例：
```
results/
  linearrag/
    sample/
      linearrag_sample_5_20260322-220500/
        predictions_sample_dataset.json
        config_effective.json
        eval_unified_sample_dataset.json
```

---

## 3. 每个 run 最少需要的输出

### 3.1 仓库原生输出（必需）

- `results/<framework>/<subset>/<run_id>/predictions_*.json`
  - 包含所有问题的预测结果
  - 格式：`{"id": ..., "question": ..., "generated_answer": ..., "context": [...]}`

- `results/<framework>/<subset>/<run_id>/config_effective.json` **[重要]**
  - 包含实际生效的配置
  - 格式：
    ```json
    {
      "timestamp": "2026-03-22T14:51:00",
      "dataset": {
        "subset": "sample",
        "sample": 5,
        "corpus_sample": null
      },
      "llm": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1"
      },
      "embedding": {
        "type": "local",
        "provider": "local",
        "model": "BAAI/bge-large-en-v1.5"
      },
      "retrieval": {
        "top_k": 5
      },
      "framework_config": {...}
    }
    ```

### 3.2 归一化 / 汇总输出（必需）

- `results/<framework>/<subset>/<run_id>/eval_unified_{corpus_name}.json`
  - 统一评估结果
  - 格式：
    ```json
    {
      "total_samples": 5,
      "valid_samples": 5,
      "fairness_metrics": {
        "success_rate": 1.0,
        "empty_answer_rate": 0.0,
        "empty_context_rate": 0.0
      },
      "average_scores": {
        "answer_em": 0.6,
        "answer_f1": 0.6,
        "rouge_score": 0.4,
        "evidence_coverage": 1.0
      }
    }
    ```

### 3.3 可选输出

- `stdout.log`：标准输出日志
- `stderr.log`：标准错误日志
- `workspace/`：索引产物（用于诊断）

---

## 4. 结果完成判定

### 4.1 必须满足的条件

一个 run 只有同时满足以下条件才算**完成**：

1. ✅ `predictions_*.json` 已落盘
   - 文件存在且可读取
   - 包含所有问题的预测结果

2. ✅ `config_effective.json` 已落盘
   - 记录了实际使用的参数
   - 用于复现和追溯

3. ✅ evaluation 文件存在或明确标记为 pending/skipped
   - `eval_unified_*.json` 存在
   - 或在 manifest 中标记为 `pending`/`skipped`

### 4.2 进程退出码 ≠ 完成判定

**错误示例**：
- ❌ 进程活着 = 完成
- ❌ 退出码为 0 = 完成

**正确判定**：
- ✅ 必须检查 predictions 文件是否存在
- ✅ 必须检查 evaluation 文件是否存在

---

## 5. Workspace 隔离规则

### 5.1 推荐：带 run_id 的独立 workspace

```
workspace/
  {framework}/
    {run_id}/
      {corpus_name}/
        {索引文件}
```

示例：
```
workspace/
  linearrag/
    linearrag_sample_5_20260322-220500/
      sample_dataset/
        {linearrag 索引文件}
```

### 5.2 不推荐：固定 workspace 路径

```
workspace/
  linearrag_workspace/  # ❌ 不同 run 可能复用同一索引
```

**问题**：
- 旧索引可能混入新 run
- skip_build 更容易误用
- 公平性失真

### 5.3 索引复用规范

如果需要复用索引：
1. 必须在 `config_effective.json` 中记录索引来源
2. 明确标注是否为复用索引

示例：
```json
{
  "index_info": {
    "is_reused": true,
    "source_run_id": "linearrag_sample_5_20260322-220500",
    "reused_at": "2026-03-22T14:55:00"
  }
}
```

---

## 6. Neo4j 数据库命名规范（ClearRAG）

### 6.1 命名规则

```
{dataset}-{序号}
```

示例：
```
sample-001
medical_100-001
hotpotqa_100-002
```

字段说明：
- `dataset`: 数据集名称（sample/medical_100/hotpotqa_100/...）
- `序号`: 该数据集的实验序号（001/002/003/...）

### 6.2 使用规范

1. 每次使用新数据集时，序号重置为 001
2. 同一数据集的后续实验，序号递增
3. 如果复用索引，数据库名称保持一致
4. 需要手动在 Neo4j 中创建数据库

---

## 7. 归一化 / 汇总输出

### 7.1 Diagnosis Summary

```json
{
  "diagnosis_id": "diag-001",
  "run_id": "linearrag_sample_5_20260322-220500",
  "status": "COMPLETED",
  "failure_type": null,
  "root_cause_hypothesis": null,
  "evidence_logs": [],
  "evidence_code_paths": []
}
```

### 7.2 Recommendation Summary

```json
{
  "recommendation_id": "rec-001",
  "target_param": "answer_normalization",
  "suggested_value": "enable_post_processing",
  "expected_effect": "提升 EM 分数到 0.8-1.0",
  "risk": "需要修改 adapter 代码",
  "confidence": 0.85
}
```
