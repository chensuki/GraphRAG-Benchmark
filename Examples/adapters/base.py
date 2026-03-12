"""
GraphRAG 框架适配器基类

定义统一的框架接口，所有 GraphRAG 框架适配器必须继承此基类。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================
# 统一默认值常量
# ============================================================

DEFAULT_EMBED_PROVIDER = "api"
DEFAULT_TOP_K = 5
DEFAULT_MAX_CONCURRENCY = 5


@dataclass
class BenchmarkQueryResult:
    """标准查询结果"""
    answer: str
    context: List[str] = field(default_factory=list)

    def get_context_list(self) -> List[str]:
        """确保返回字符串列表"""
        if self.context is None:
            return []
        if isinstance(self.context, str):
            return [self.context] if self.context else []
        if isinstance(self.context, list):
            return [str(item) for item in self.context if item]
        return []


@dataclass
class FrameworkConfig:
    """
    框架配置

    封装所有框架共享的配置项。
    """
    # LLM 配置
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""

    # 嵌入配置
    embed_model: str = ""
    embed_provider: str = DEFAULT_EMBED_PROVIDER
    embed_api_key: Optional[str] = None
    embed_base_url: Optional[str] = None

    # 检索配置
    top_k: int = DEFAULT_TOP_K

    # 并发配置
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY

    # 框架特定配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def get_extra(self, key: str, default: Any = None) -> Any:
        """获取额外配置项"""
        return self.extra.get(key, default)


class BaseFrameworkAdapter(ABC):
    """
    框架适配器抽象基类

    生命周期：
    1. __init__(): 初始化配置
    2. abuild_index(): 构建索引
    3. aquery(): 执行查询
    4. aclose(): 清理资源
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        self.working_dir = working_dir
        self.config = config

    @abstractmethod
    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        pass

    @abstractmethod
    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        pass

    async def aclose(self) -> None:
        """清理资源"""
        pass

    async def abatch_query(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs
    ) -> List[BenchmarkQueryResult]:
        '''批量查询（默认实现：循环调用 aquery）'''
        results = []
        for q in questions:
            result = await self.aquery(q["question"], top_k=top_k, **kwargs)
            results.append(result)
        return results

    def load_index(self, corpus_name: str) -> None:
        """加载已有索引"""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support loading index."
        )

    def _ensure_content_string(self, content: Any) -> str:
        """确保内容是字符串"""
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