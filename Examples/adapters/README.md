# GraphRAG 框架适配器

统一的框架适配器接口，支持多种 GraphRAG 框架接入 Benchmark。

## 架构设计

```
Examples/adapters/
├── __init__.py          # 模块导出
├── base.py              # 抽象基类和数据结构
├── registry.py          # 适配器注册表
├── lightrag_adapter.py  # LightRAG 适配器
├── clearrag_adapter.py  # ClearRAG 适配器
├── linearrag_adapter.py # LinearRAG 适配器
├── hipporag2_adapter.py # HippoRAG2 适配器（占位）
├── fast_graphrag_adapter.py # Fast-GraphRAG 适配器（占位）
└── digimon_adapter.py   # DIGIMON 适配器（占位）
```

## 设计原则（SOLID）

| 原则 | 实现 |
|------|------|
| **S** 单一职责 | 每个适配器仅负责框架的索引和查询 |
| **O** 开闭原则 | 通过继承扩展，不修改基类 |
| **L** 里氏替换 | 所有适配器可互换使用 |
| **I** 接口隔离 | 接口最小化（仅 2 个抽象方法） |
| **D** 依赖倒置 | 依赖抽象（BaseFrameworkAdapter） |

## 快速开始

### 新增框架接入

仅需两步：

**步骤 1：创建适配器**

```python
# Examples/adapters/new_framework_adapter.py
from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

@register_adapter("new_framework")  # 注册框架
class NewFrameworkAdapter(BaseFrameworkAdapter):

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        # 框架特定的索引逻辑
        pass

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        # 框架特定的查询逻辑
        return BenchmarkQueryResult(answer="...", context=["..."])
```

**步骤 2：添加配置**

```yaml
# configs/experiment.yaml
frameworks:
  new_framework:
    enabled: true
    subset: medical
    # 框架特定参数...
```

**完成！** 不再需要：
- ❌ 创建 `run_new_framework.py`
- ❌ 重复数据加载代码
- ❌ 修改 `run_from_yaml.py`

### 使用统一运行器

```python
from Examples.runner import FrameworkRunner, FrameworkConfig

config = FrameworkConfig(
    llm_model="deepseek-chat",
    llm_base_url="https://api.deepseek.com/v1",
    llm_api_key="sk-xxx",
    embed_model="embedding-3",
    embed_provider="zhipu",
)

runner = FrameworkRunner(
    framework="lightrag",
    config=config,
    subset="medical",
    corpus_path="./Datasets/Corpus/medical.parquet",
    questions_path="./Datasets/Questions/medical_questions.parquet",
    output_dir="./results",
)

await runner.run()
```

## API 参考

### BaseFrameworkAdapter

```python
class BaseFrameworkAdapter(ABC):

    def __init__(self, working_dir: str, config: FrameworkConfig):
        """初始化适配器"""

    @abstractmethod
    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""

    @abstractmethod
    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""

    async def aclose(self) -> None:
        """清理资源（可选）"""

    def load_index(self, corpus_name: str) -> None:
        """加载已有索引（可选）"""
```

### BenchmarkQueryResult

```python
@dataclass
class BenchmarkQueryResult:
    answer: str
    context: List[str]

    def get_context_list(self) -> List[str]:
        """确保返回字符串列表"""
```

### FrameworkConfig

```python
@dataclass
class FrameworkConfig:
    # LLM 配置
    llm_model: str
    llm_base_url: str
    llm_api_key: str

    # 嵌入配置
    embed_model: str
    embed_provider: str  # api, zhipu, openai, local
    embed_api_key: Optional[str]
    embed_base_url: Optional[str]

    # 检索配置
    top_k: int = 5

    # 并发配置
    max_concurrency: int = 5

    # 框架特定配置
    extra: Dict[str, Any] = field(default_factory=dict)
```

## 运行模式

### 模式 1：统一运行器（推荐）

```bash
python Examples/run_from_yaml.py --framework lightrag
```

### 模式 2：直接使用适配器

```python
from Examples.adapters import create_adapter, FrameworkConfig

config = FrameworkConfig(...)
adapter = create_adapter("lightrag", working_dir, config)

await adapter.abuild_index(corpus_text)
result = await adapter.aquery("问题?")
await adapter.aclose()
```

## 迁移指南

从旧脚本迁移到新适配器：

| 旧代码 | 新代码 |
|--------|--------|
| `run_lightrag.py` 独立脚本 | `run_from_yaml.py --framework lightrag` |
| `run_clearrag.py` 独立脚本 | `run_from_yaml.py --framework clearrag` |
| 各脚本独立的数据加载 | `FrameworkRunner` 自动处理 |
| 各脚本独立的并发控制 | `FrameworkRunner` 自动处理 |

## 已实现适配器

| 框架 | 状态 | 说明 |
|------|------|------|
| LinearRAG | ✅ 完整 | 支持向量化/迭代检索 |
| LightRAG | ✅ 完整 | 支持 API/Ollama 模式 |
| ClearRAG | ✅ 完整 | 使用现有 Benchmark 适配器 |
| HippoRAG2 | ✅ 完整 | 支持本地 BGE 嵌入 |
| Fast-GraphRAG | ✅ 完整 | 支持 HF/API 嵌入 |
| DIGIMON | ✅ 完整 | 统一框架配置 |