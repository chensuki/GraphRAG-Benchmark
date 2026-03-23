<div align="center">

# When to use Graphs in RAG: A Comprehensive Benchmark and Analysis for Graph Retrieval-Augmented Generation

[![Static Badge](https://img.shields.io/badge/arxiv-2506.05690-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2506.05690)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://graphrag-bench.github.io/)  [![Static Badge](https://img.shields.io/badge/license-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/LICENSE)

<p>
    <a href="#关于本项目" style="text-decoration: none; font-weight: bold;">📌关于本项目</a> •
    <a href="GUIDE.md" style="text-decoration: none; font-weight: bold;">📘运行指南</a> •
    <a href="#project-changes" style="text-decoration: none; font-weight: bold;">🔄项目变更</a> •
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉News</a> •
    <a href="#about" style="text-decoration: none; font-weight: bold;">📖About</a> •
    <a href="#leaderboards" style="text-decoration: none; font-weight: bold;">🏆Leaderboards</a>
</p>
<p>
    <a href="#task-examples" style="text-decoration: none; font-weight: bold;">🧩Task Examples</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🔧Getting Started</a> •
    <a href="#contribution--contact" style="text-decoration: none; font-weight: bold;">📬Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝Citation</a>
</p>
</div>

<h2 id="关于本项目">📌 关于本项目</h2>

本项目基于 [GraphRAG-Bench/GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) 进行了重构和优化，主要改进如下：

### 🔧 统一化改进

**1. 统一的适配器模式**
- 引入 `BaseFrameworkAdapter` 抽象基类，标准化所有框架接口
- 支持 6 种 GraphRAG 框架：LightRAG、ClearRAG、LinearRAG、Fast-GraphRAG、HippoRAG2、DigiMON
- 使用注册表模式实现框架的动态发现和延迟加载

**2. YAML 配置简化**
- 单一配置文件 `configs/template.yaml` 管理所有框架参数
- 支持全局配置与框架特定配置的层级覆盖
- 简化命令行参数，通过配置文件控制实验

**3. 统一的运行流程**
- `run_from_yaml.py` 作为唯一入口，支持多框架批量运行
- `FrameworkRunner` 提供标准化的索引构建、查询、结果保存流程
- 异步并发处理，支持语料库级别的并发控制

<h2 id="project-changes">🔄 项目变更（对比原始仓库）</h2>

### 🎯 核心功能改进

| 功能 | 原项目 | 本项目 |
|------|--------|--------|
| 框架运行 | 各框架独立脚本 | 统一入口 + 适配器模式 |
| 配置方式 | 分散在各脚本 | 集中式 YAML 配置 |
| 框架集成 | 直接调用框架 API | 适配器抽象层隔离 |
| 评估脚本 | 分离的检索/生成评估 | 新增统一评估入口 |
| 数据集管理 | 手动下载 | 自动下载 + 子集截取 |
| 扩展性 | 需修改主流程 | 只需实现新适配器 |

### 🚀 新增功能

**1. 框架适配器系统**
- 6 种框架统一接口：LightRAG、ClearRAG、LinearRAG、Fast-GraphRAG、HippoRAG2、DigiMON
- 注册表模式支持动态发现和延迟加载

**2. 统一评估脚本**
- `unified_eval.py` 一键完成答案、检索、支持事实等多维度评估
- 支持 HotpotQA、MuSiQue、2WikiMultihop 等多数据集格式

**3. 数据集管理工具**
- `download_datasets.py` 自动下载原始数据集
- `subset_datasets.py` 快速截取实验子集

**4. 实验通知集成**
- `openclaw_notifier.py` 支持openclaw自动化实验

### 🚀 快速开始

```bash
# 1. 复制配置文件（首次使用）
cp configs/template.yaml configs/experiment.yaml

# 2. 运行单个框架
python Examples/run_from_yaml.py --framework lightrag

# 运行所有启用的框架
python Examples/run_from_yaml.py --framework all

# 干运行（查看配置）
python Examples/run_from_yaml.py --dry-run
```

详细配置说明请参考 [📘 GUIDE.md](GUIDE.md) 运行指南。

---

<h2 id="news">🎉 News</h2>

- **[2026-01-26]** Our [GraphRAG Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark) is accepted by **ICLR'26**.
- **[2026-01-26]** Our [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG) is accepted by **ICLR'26**.
- **[2025-10-27]** We release [LinearRAG](https://github.com/DEEP-PolyU/LinearRAG), a relation-free method for efficient GraphRAG.
- **[2025-08-24]** We support [DIGIMON](https://github.com/JayLZhou/GraphRAG) for flexible benchmarking across GraphRAG models.
- **[2025-05-25]** We release the [GraphRAG Benchmark](https://graphrag-bench.github.io) for evaluating GraphRAG models.
- **[2025-01-21]** We release the [GraphRAG survey](https://github.com/DEEP-PolyU/Awesome-GraphRAG).

📑 Please [cite our paper](https://arxiv.org/abs/2506.05690) if you find our survey or repository helpful!

📬 **Contact us via emails:** {xiangzhishang,wuchuanjie}@stu.xmu.edu.cn, qinggang.zhang@polyu.edu.hk

<h2 id="about">📖 About</h2>

- Introduces Graph Retrieval-Augmented Generation (GraphRAG) concept
- Compares traditional RAG vs GraphRAG approach
- Explains research objective: Identify scenarios where GraphRAG outperforms traditional RAG
- Visual comparison diagram of RAG vs GraphRAG

![overview](./RAGvsGraphRAG.jpg)

<details>
<summary>
  More Details
</summary>
Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning. Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models on both hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, covering fact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph construction and knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application.
</details>

<h2 id="leaderboards">🏆 Leaderboards</h2>

Two domain-specific leaderboards with comprehensive metrics:

**1. GraphRAG-Bench (Novel)**

- Evaluates models on literary/fictional content

**2. GraphRAG-Bench (Medical)**

- Evaluates models on medical/healthcare content

**Evaluation Dimensions:**

- Fact Retrieval (Accuracy, ROUGE-L)
- Complex Reasoning (Accuracy, ROUGE-L)
- Contextual Summarization (Accuracy, Coverage)
- Creative Generation (Accuracy, Factual Score, Coverage)

<h2 id="task-examples">🧩 Task Examples</h2>
Four difficulty levels with representative examples:

**Level 1: Fact Retrieval**
*Example: "Which region of France is Mont St. Michel located?"*

**Level 2: Complex Reasoning**
*Example: "How did Hinze's agreement with Felicia relate to the perception of England's rulers?"*

**Level 3: Contextual Summarization**
*Example: "What role does John Curgenven play as a Cornish boatman for visitors exploring this region?"*

**Level 4: Creative Generation**
*Example: "Retell King Arthur's comparison to John Curgenven as a newspaper article."*

<h2 id="getting-started">🔧 Getting Started</h2>

详细安装和运行说明请参考 [📘 GUIDE.md](GUIDE.md)。

### 快速安装

```bash
pip install -r requirements.txt
pip install -e ./LightRAG  # 或其他框架
```

### 支持的数据集

| 数据集 | 子集名 | 说明 |
|--------|--------|------|
| Sample | `sample` | 样本数据集（快速测试） |
| Medical | `medical` | 医疗领域数据集 |
| Novel | `novel` | 小说数据集（20个语料） |
| HotpotQA | `hotpotqa_distractor` | 多跳问答 |
| MuSiQue | `musique` | 多跳问答数据集 |

更多框架和数据集说明，请查看 [Examples README](Examples/README.md) 和 [Evaluation README](Evaluation/README.md)。

<h2 id="contribution--contact">📬 Contribution & Contact</h2>

Contributions to improve the benchmark website are welcome. Please contact the project team via <a href="mailto:GraphRAG@hotmail.com">GraphRAG@hotmail.com </a>.

<h2 id="citation">📝 Citation</h2>

If you find this benchmark helpful, please cite our paper:

```
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```
