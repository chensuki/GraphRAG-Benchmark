# 公平性策略（Fairness Policy）

## 参数锁定层级

- Layer A（严格锁定）：dataset split、当前阶段 sample 数、评估指标、输出 schema、预算规则
- Layer B（组内锁定）：embedding/reranker 模型、top-k、context budget、chunk 策略
- Layer C（允许调节）：图构建强度、query mode、framework 自带搜索参数

## Guardrails

- 每个比较组一次只改一个主变量。
- 每次实验都保存完整有效配置。
- 不要比较锁定字段不一致的 runs。
- 如果 framework 缺少 supporting-facts 输出字段，要把答案指标和 supporting-fact 指标分开解释。
- 因 schema 缺失导致的 supporting-fact 指标失真，应标记为 `UNDERREPORTED`，而不是直接视为失败。

## Smoke-run 策略

在任何大样本 / 全量实验之前：
- 必须先 dry-run
- 必须先跑小样本 smoke test
- 只有 smoke test 合格后，才能扩展到更大 sample 或 `null`
- 同一 dataset/workspace 可以通过 `skip_build: true` 复用索引

## 小样本规则

对于 `sample=5` 或 `sample=10` 这种 smoke runs：
- 只能用于可行性验证和粗排序
- 不能直接用来做最终架构决策
- 如果某个 framework 在小样本上领先，应在更大样本上复验后再定结论
