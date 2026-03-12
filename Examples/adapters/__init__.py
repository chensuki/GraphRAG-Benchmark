"""
GraphRAG 框架适配器模块

提供统一的框架适配器接口，支持多种 GraphRAG 框架接入 Benchmark。
"""
from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import (
    register_adapter,
    get_adapter,
    create_adapter,
    list_adapters,
    has_adapter,
)

__all__ = [
    # 基类
    "BaseFrameworkAdapter",
    "BenchmarkQueryResult",
    "FrameworkConfig",
    # 注册表
    "register_adapter",
    "get_adapter",
    "create_adapter",
    "list_adapters",
    "has_adapter",
]