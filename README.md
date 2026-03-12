<div align="center">

# When to use Graphs in RAG: A Comprehensive Benchmark and Analysis for Graph Retrieval-Augmented Generation

[![Static Badge](https://img.shields.io/badge/arxiv-2506.05690-ff0000?style=for-the-badge&labelColor=000)](https://arxiv.org/abs/2506.05690)  [![Static Badge](https://img.shields.io/badge/huggingface-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/datasets/GraphRAG-Bench/GraphRAG-Bench)  [![Static Badge](https://img.shields.io/badge/leaderboard-steelblue?style=for-the-badge&logo=googlechrome&logoColor=ffffff)](https://graphrag-bench.github.io/)  [![Static Badge](https://img.shields.io/badge/license-mit-teal?style=for-the-badge&labelColor=000)](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/LICENSE)

<p>
    <a href="#关于本项目" style="text-decoration: none; font-weight: bold;">📌关于本项目</a> •
    <a href="#news" style="text-decoration: none; font-weight: bold;">🎉News</a> •
    <a href="#about" style="text-decoration: none; font-weight: bold;">📖About</a> •
    <a href="#leaderboards" style="text-decoration: none; font-weight: bold;">🏆Leaderboards</a> •
    <a href="#task-examples" style="text-decoration: none; font-weight: bold;">🧩Task Examples</a>

</p>
  <p>
  <a href="#getting-started" style="text-decoration: none; font-weight: bold;">🔧Getting Started</a> •
    <a href="#contribution--contact" style="text-decoration: none; font-weight: bold;">📬Contact</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">📝Citation</a> •
    <a href="#stars" style="text-decoration: none; font-weight: bold;">✨Stars History</a>
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

### 📝 主要变化

| 模块 | 原项目 | 本项目 |
|------|--------|--------|
| 框架运行 | 各框架独立脚本 | 统一 `runner.py` + 适配器模式 |
| 配置方式 | 分散在各脚本中 | 集中式 YAML 配置 |
| 框架集成 | 直接调用框架API | 适配器抽象层隔离差异 |
| 参数管理 | 命令行参数为主 | YAML配置优先 |
| 扩展性 | 需修改主流程 | 只需实现新适配器 |

### 🚀 快速开始

```bash
# 运行单个框架
python Examples/run_from_yaml.py --framework lightrag

# 运行所有启用的框架
python Examples/run_from_yaml.py --framework all

# 干运行（查看配置）
python Examples/run_from_yaml.py --dry-run
```

详细配置说明请参考 `configs/template.yaml`。

---

If you find this benchmark helpful, please cite our paper:

```
@article{xiang2025use,
  title={When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation},
  author={Xiang, Zhishang and Wu, Chuanjie and Zhang, Qinggang and Chen, Shengyuan and Hong, Zijin and Huang, Xiao and Su, Jinsong},
  journal={arXiv preprint arXiv:2506.05690},
  year={2025}
}
```

This repository is for the GraphRAG-Bench project, a comprehensive benchmark for evaluating Graph Retrieval-Augmented Generation models.
![pipeline](./pipeline.jpg)

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

First, install the necessary dependencies for GraphRAG-Bench.

```bash
pip install -r requirements.txt
```

## 🛠 Installation Guide

**To prevent dependency conflicts, we strongly recommend using separate Conda environments for each framework:**

We use the installation of LightRAG as an example. For other frameworks, please refer to their respective installation instructions.

```bash
# Create and activate environment (example for LightRAG)
conda create -n lightrag python=3.10 -y
conda activate lightrag

# Install LightRAG
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
pip install -e .

```

## 🚀 Running Examples

We provide detailed instructions on how to use GraphRAG-Bench to evaluate each framework. 

Specifically, we introduce how to perform index construction and batch inference for each framework in the `Examples` folder with instructions in the [Examples README](Examples/README.md).

Note that the evaluation code is standardized across all frameworks to ensure fair comparison. Please refer to the `Evaluation` folder and the  [Evaluation README](Evaluation/README.md) for detailed instructions on the evaluation.





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

<h2 id="stars">✨ Stars History</h2>

[![Star History Chart](https://api.star-history.com/svg?repos=GraphRAG-Bench/GraphRAG-Benchmark&type=Date)](https://www.star-history.com/#GraphRAG-Bench/GraphRAG-Benchmark&Date)
