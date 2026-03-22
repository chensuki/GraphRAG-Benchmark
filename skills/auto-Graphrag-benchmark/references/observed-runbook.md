# 真实运行观察（Observed Runbook，2026-03-17）

## 实验设置
- Repo: `GraphRAG-Benchmark`
- Dataset: `musique_100`
- Sample: `5`
- Evaluation: `Evaluation.unified_eval`

## 实测结果
- LinearRAG
  - answer_em: 0.2000
  - answer_f1: 0.3333
  - rouge_score: 0.3333
  - evidence_coverage: 0.8952
- LightRAG
  - answer_em: 0.0000
  - answer_f1: 0.1333
  - rouge_score: 0.1000
  - evidence_coverage: 0.6750
- ClearRAG
  - answer_em: 0.2000
  - answer_f1: 0.2000
  - rouge_score: 0.2000
  - evidence_coverage: 0.3053

## 运行观察
- LightRAG 在这轮 smoke run 中虽然出现 extraction-format warnings，但最终仍能成功落盘 predictions。
- ClearRAG 也能完成，但绝大部分 wall time 花在图构建 / Neo4j-heavy 阶段。
- ClearRAG 出现了 `cartesian product` 这类 Cypher 性能 warning，这类信号应纳入 diagnosis，但不应直接视为 fatal。

## 汇报经验
- 长实验的汇报不该只重复日志片段，而应围绕阶段切换和文件落盘来讲。
- run 完成必须以 `predictions_*.json` 已存在为准。
- predictions 落盘后，应立即补跑统一评估。

## 指标解释的注意事项
- 三个 framework 在 unified eval 里 supporting-facts 指标都接近 0，并不一定说明检索彻底失败。
- 更可能的原因是 prediction 输出里没有补齐统一 supporting-facts 字段。
- 因此，在这一轮实验中，更可靠的比较轴是：
  - answer_em
  - answer_f1
  - rouge_score
  - evidence_coverage
