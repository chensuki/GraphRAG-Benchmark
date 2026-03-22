---
name: auto-Graphrag-benchmark
description: 面向 GraphRAG-Benchmark 风格仓库的 GraphRAG benchmark 编排技能。用于运行、比较、归一化、诊断和总结 linearrag、lightrag、clearrag 等框架实验；制定公平对比参数；遵循分阶段 benchmark 流程；以及根据真实实验结果更新 benchmark 运行规则。
---

# Auto-GraphRAG-benchmark Skill

使用这个 skill 时，不要把 benchmark 当成"随便跑一条命令"的任务，而要按**稳定的分阶段流程**执行。

## 核心操作原则

按下面顺序执行：
1. `config`
2. `dry-run`
3. `small-sample smoke run`
4. `full prediction run`
5. `evaluation`
6. `analysis + diagnosis + recommendations`

不要从配置直接跳到大样本 / 全量长实验。

## 用户操作规则

- 需要安装新的依赖时，必须先获得用户批准。
- 遇到兼容性问题时，先评估多种方案，再选择**入侵性最小**的可行方案。
- 要尽早发现问题并及时汇报，不要等整个 run 失败后再补报。

---

# GraphRAG Benchmark 实验流程

## 1. 数据准备（./Datasets 已有可忽略）

**数据集应该已存在 `./Datasets/`。如需提取子集：**

- 使用 `Datasets/subset_datasets.py`
- 子集自动被 `Examples/subset_registry.py` 发现

**可用数据集**：
- sample, medical, medical_100, novel
- hotpotqa_distractor, hotpotqa_fullwiki
- 2wikimultihop, musique

**注意**：详细命令参考：`references/dataset-schema.md`

---

## 2. 配置实验

直接复制已有配置文件 `configs/experiment.yaml` 到 runs 目录下，命名方式：`{framework}_{dataset}_{sample}.yaml`，然后修改对应的配置。

**关键配置规则**：

### 数据集问题采样数 `sample`
- 第一次索引需要使用一个比较小的问题采样数，如 10
- 避免索引出现问题导致后续的查询预测不准确
- 可以修改的前提是：索引成功后没有报错，再修改为较大的问题采样数，比如 NULL

### 嵌入模型
- 优先使用本地模型，如 `BAAI/bge-large-en-v1.5`
- 中间遇到兼容性问题再去切换 api 方式的 embedding

### 关于 ClearRAG 的 Neo4j 设置
- 只需要修改的是：`database`
- 这里的名称有一个规范，就是取用一个和当前数据集相关的名称
- 例子：使用的是 musique_100，数据库名称就是：musique-100-001，001 表示的是第一次运行该数据集
- 以此类推，并且需要手动去 neo4j (localhost:7474) 中创建一个数据库
- 名称为：musique-100-001，然后在 configs/experiment.yaml 中修改 database 的值为：musique-100-001

### `skip_build`
- 先检查一遍对应的目录是否有已经构建好的索引
- 第一次索引需要设置为 false
- 后续相同的数据集可以设置为 true

### ClearRAG 框架的参数可以修改
- `max_tool_calls`：agent 的最大调用工具次数（3~10）
- `passage_retrieval_mode`：段落检索模式（vector / pagerank / none），通常是 pagerank 和 none

---

## 3. 运行实验

### 3.1 脚本 Examples/run_from_yaml.py 支持的 CLI 参数
```bash
python Examples/run_from_yaml.py --help
```

**建议运行方式**：复制 yaml 文件到 runs 目录下，并修改对应的参数，然后运行通过 `--config` 参数运行脚本。

### 3.2 先运行以下命令检查配置是否正确
```bash
python Examples/run_from_yaml.py --framework clearrag --dry-run
```

### 3.3 正式运行
```bash
python Examples/run_from_yaml.py --framework clearrag
```

### 3.4 样本升级

由于先是设置了 sample，所以这里需要根据上面的运行结果进行分析：
- 如果上面没有报错
- 并且查询样本没有出现明显的问题和报错
- 就需要把 sample 再修改为 NULL
- 同时需要修改 skip_build 这个参数为 true

### 3.5 运行预测
```bash
python Examples/run_from_yaml.py --framework clearrag
```

### 3.6 运行流程

1. 加载配置 → 解析框架和数据集
2. 获取数据集路径（corpus + questions）
3. 初始化框架适配器
4. 索引构建：切分文本 → 提取实体/关系 → 构建图谱/向量索引
5. 查询回答：对每个问题检索相关上下文 → LLM 生成答案
6. 保存结果到 `results/{framework}/{subset}/{run_id}/`

---

## 4. 评估结果

### 4.1 统一评估（推荐）

**先进行环境变量的设置**：
```bash
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_API_KEY="可以从 configs/experiment.yaml 里面找"
export LLM_MODEL="deepseek-chat"
```

```bash
python -m Evaluation.unified_eval \
  --data_file results/{run}/predictions.json \
  --output_file results/{run}/eval_unified.json \
  --report
```

**参数说明**：
- `--data_file`：预测结果文件（必需）
- `--output_file`：评估结果输出路径（必需）
- `--detailed`：输出详细结果（包含每个样本）
- `--report`：打印格式化报告

**输出指标**：
- 答案指标：`answer_em`（精确匹配），`answer_f1`（F1 分数），`rouge_score`（ROUGE 分数）
- 检索质量：`evidence_coverage`（证据覆盖率）
- 支持事实：`sf_em`，`sf_f1`（如框架输出 supporting facts）
- 三元组/推理步骤（如适用）

### 4.2 检索评估（可选）

详细命令参考：`references/dataset-schema.md`

### 4.3 生成评估（可选）

详细命令参考：`references/dataset-schema.md`

### 4.4 索引评估（可选，比较少用）

详细命令参考：`references/dataset-schema.md`

---

## 5. 支持的框架

### 5.1 比较稳定可运行的框架
- linearrag
- lightrag
- clearrag

### 5.2 需要进一步测试的框架
- fast-graphrag
- hipporag2
- digimon

---

## 6. 问题诊断与调试

**当实验失败或结果不如预期时，使用 5 步诊断流程**（来自 experiment-craft）：

### 6.1 收集失败案例
- 收集具体的失败或糟糕结果示例
- 查看实际输出，不只是聚合指标

### 6.2 找到工作版本
- 简化任务、减少复杂度、切换数据
- 从工作版本开始，逐因素添加
- 找到那个单一因素导致失败

### 6.3 桥接差距
- 从工作版本开始，渐进添加复杂性
- 找到单一因素，确认根本原因

### 6.4 假设与验证
- 列出可能的解释并按可能性排序
- 设计针对性实验验证或消除每个假设
- 通过实验实际确认原因

### 6.5 提出与实施
- 搜索解决该具体原因的技术
- 设计针对性修复并验证

**详细诊断流程参考**：`references/experiment-craft.md`

---

## 7. 谨慎推进策略

**强制实验顺序，避免盲目实验**（来自 experiment-pipeline）：

### 7.1 禁止跳过阶段
- ❌ 不要跳过 baseline（Stage 1）
- ❌ 不要跳过 hyperparam tuning（Stage 2）
- ❌ 不要跳过方法验证（Stage 3）
- ❌ 不要跳过 ablation（Stage 4）

### 7.2 每次只进一阶
- 只有当前 stage 通过 gate 条件后，才进入下一 stage
- Gate 条件：
  - Stage 1：Metrics 在 reported 值的 2% 以内（或 variance 在合理范围）
  - Stage 2：配置稳定，variance < 5% 跨 3 次独立运行
  - Stage 3：优于 tuned baseline 的 primary metric
  - Stage 4：每个声称的贡献都有对照实验

### 7.3 预算控制
- Stage 1：≤20 次尝试
- Stage 2：≤12 次尝试
- Stage 3：≤12 次尝试
- Stage 4：≤18 次尝试

**为什么需要预算**：
- 防止"跑更多实验，希望碰运气"
- 强制系统性思考，而不是盲目迭代
- 如果预算耗尽仍未通过，说明问题可能在更深处

---

## 8. 实验日志结构化

**每次运行必须记录 5 个部分**（来自 experiment-pipeline 的 Code Trajectory Logging）：

### 8.1 Purpose（目的）
- 运行这个实验的目标；你期望学到什么？

### 8.2 Setting（配置）
- 实际使用的配置、框架、数据集、参数

### 8.3 Results（结果）
- 量化指标（EM/F1/ROUGE）+ 定性观察

### 8.4 Analysis（分析）
- 结果是否匹配预期？为什么？

### 8.5 Next Steps（下一步）
- 基于分析的下一步行动

---

## 9. 框架稳定性

**高稳定性（P0 主力）**：
- linearrag：检索质量优秀，稳定
- lightrag：工程化成熟，生产就绪
- clearrag：稳定，需要 Neo4j 设置

**中等稳定性（P1 扩展）**：
- hipporag2：可用，需要环境变量设置

**低/未知（P2 后续）**：
- fast-graphrag：未验证
- digimon：未验证，接口可能不一致

---

## 10. 每次完整的实验运行的必需输出

**最低要求**：
- `predictions_*.json`：查询结果
- `config_effective.json`：实际配置
- `eval_unified_*.json`：评估指标

**推荐**：
- 完整的实验日志（按 5 部分结构）
- 诊断分析
- 调优建议

---

## 11. 按需读取的参考文件

- 数据集格式与可用列表：`references/dataset-schema.md`
- 问题诊断流程：`references/experiment-craft.md`
- 阶段推进策略：`references/experiment-pipeline.md`
- 公平性与参数锁定策略：`references/fairness-policy.md`
- 结果命名与完成规则：`references/result-spec.md`
- 各 framework 参数映射：`references/framework-parameter-map.md`
- 真实运行经验沉淀：`references/observed-runbook.md`
