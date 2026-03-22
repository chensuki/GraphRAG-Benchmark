# 实验阶段推进策略（Experiment Pipeline）

强制实验顺序，避免盲目实验。

---

## 阶段定义

### Stage 1: Baseline Reproduction

**目标**：复现基线结果

**Gate 条件**：
- Metrics 在 reported 值的 2% 以内
- 或 variance 在合理范围

**预算**：≤20 次尝试

---

### Stage 2: Hyperparameter Tuning

**目标**：找到稳定配置

**Gate 条件**：
- 配置稳定
- Variance < 5% 跨 3 次独立运行

**预算**：≤12 次尝试

---

### Stage 3: Method Validation

**目标**：验证方法有效性

**Gate 条件**：
- 优于 tuned baseline 的 primary metric

**预算**：≤12 次尝试

---

### Stage 4: Ablation Study

**目标**：验证各贡献点

**Gate 条件**：
- 每个声称的贡献都有对照实验

**预算**：≤18 次尝试

---

## 禁止行为

- ❌ 不要跳过 baseline（Stage 1）
- ❌ 不要跳过 hyperparam tuning（Stage 2）
- ❌ 不要跳过方法验证（Stage 3）
- ❌ 不要跳过 ablation（Stage 4）

---

## 预算控制原则

**为什么需要预算**：
- 防止"跑更多实验，希望碰运气"
- 强制系统性思考，而不是盲目迭代
- 如果预算耗尽仍未通过，说明问题可能在更深处

**预算耗尽后**：
1. 停止当前阶段
2. 回到上一阶段检查问题
3. 重新评估假设