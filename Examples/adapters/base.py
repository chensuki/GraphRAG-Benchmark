"""
GraphRAG 框架适配器基类
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 默认值常量
DEFAULT_EMBED_TYPE = "api"
DEFAULT_EMBED_PROVIDER = "zhipu"
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONCURRENCY = 5


def format_entity_content(name: str, entity_type: str, description: str) -> str:
    """格式化实体 content: "实体名 (类型): 描述"""
    if not name and not description:
        return ""

    parts = []
    if name:
        parts.append(f"{name} ({entity_type})" if entity_type else name)
    if description:
        parts.append(description)

    return ": ".join(parts)


def format_relationship_content(src: str, tgt: str, description: str) -> str:
    """格式化关系 content: "src->tgt: description"""
    if not src or not tgt:
        return ""
    return f"{src}->{tgt}: {description}" if description else f"{src}->{tgt}"


@dataclass
class BenchmarkQueryResult:
    """标准查询结果"""
    answer: str
    context: List[Dict[str, Any]] = field(default_factory=list)

    def get_context_list(self) -> List[str]:
        """获取纯文本 context 列表"""
        if not self.context:
            return []
        result = []
        for item in self.context:
            if isinstance(item, dict):
                result.append(item.get("content", ""))
        return [r for r in result if r]

    def get_chunks(self) -> List[Dict[str, Any]]:
        """获取所有 chunk 类型的 context"""
        return [item for item in self.context if isinstance(item, dict) and item.get("type") == "chunk"]

    def get_entities(self) -> List[Dict[str, Any]]:
        """获取所有 entity 类型的 context"""
        return [item for item in self.context if isinstance(item, dict) and item.get("type") == "entity"]

    def get_relationships(self) -> List[Dict[str, Any]]:
        """获取所有 relationship 类型的 context"""
        return [item for item in self.context if isinstance(item, dict) and item.get("type") == "relationship"]


@dataclass
class FrameworkConfig:
    """
    框架配置

    嵌入配置参数：
    - embed_type: api | local
    - embed_provider: zhipu | openai | custom
    - embed_model: 模型名称
    - embed_base_url: API 地址
    - embed_dimensions: 向量维度（None=API默认）
    - embed_batch_size: 批处理大小
    """
    # LLM 配置
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""

    # 嵌入配置
    embed_model: str = ""
    embed_type: str = DEFAULT_EMBED_TYPE
    embed_provider: str = DEFAULT_EMBED_PROVIDER
    embed_api_key: Optional[str] = None
    embed_base_url: Optional[str] = None
    embed_dimensions: Optional[int] = None  # None = API 默认
    embed_batch_size: int = 64

    # 检索配置
    top_k: int = DEFAULT_TOP_K

    # 并发配置
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    # 框架特定配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_extra(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)

    def get_effective_embed_base_url(self) -> str:
        """获取有效的 base_url（必须通过 YAML 配置）"""
        if self.embed_base_url:
            return self.embed_base_url
        if self.embed_type == "local":
            return ""
        raise ValueError(f"embed_base_url is required for provider={self.embed_provider}")


class BaseFrameworkAdapter(ABC):
    """框架适配器抽象基类"""

    def __init__(self, working_dir: str, config: FrameworkConfig):
        self.working_dir = working_dir
        self.config = config

    @abstractmethod
    async def abuild_index(self, content: str, **kwargs) -> None:
        pass

    @abstractmethod
    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        pass

    async def aclose(self) -> None:
        pass

    async def abatch_query(self, questions: List[Dict[str, Any]], top_k: int = 5, **kwargs) -> List[BenchmarkQueryResult]:
        return [await self.aquery(q["question"], top_k=top_k, **kwargs) for q in questions]

    def load_index(self, corpus_name: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support loading index.")

    def _ensure_content_string(self, content: Any) -> str:
        if content is None:
            raise ValueError("Content cannot be None")
        if isinstance(content, str):
            if not content.strip():
                raise ValueError("Content cannot be empty")
            return content
        if isinstance(content, list):
            return "\n\n".join(str(item) for item in content if item)
        raise ValueError(f"Content must be str, got {type(content)}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(working_dir={self.working_dir!r})"