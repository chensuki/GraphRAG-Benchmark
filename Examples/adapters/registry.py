"""
框架适配器注册表

使用注册表模式管理框架适配器，支持动态注册和发现。

特点：
- 装饰器注册：使用 @register_adapter 简化注册
- 延迟加载：适配器类在首次使用时才导入
- 类型安全：注册时验证适配器类型
"""
from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Dict, Optional, Type

from .base import BaseFrameworkAdapter, FrameworkConfig

logger = logging.getLogger(__name__)

# 适配器注册表
_ADAPTER_REGISTRY: Dict[str, Type[BaseFrameworkAdapter]] = {}

# 延迟加载映射（框架名 -> 模块路径）
# 使用相对于 Examples 目录的导入路径
_LAZY_LOAD_MAP: Dict[str, str] = {
    "lightrag": "adapters.lightrag_adapter",
    "clearrag": "adapters.clearrag_adapter",
    "linearrag": "adapters.linearrag_adapter",
    "hipporag2": "adapters.hipporag2_adapter",
    "fast-graphrag": "adapters.fast_graphrag_adapter",
    "digimon": "adapters.digimon_adapter",
}


def register_adapter(name: str) -> Callable[[Type[BaseFrameworkAdapter]], Type[BaseFrameworkAdapter]]:
    """
    装饰器：注册框架适配器

    使用示例：
    ```python
    @register_adapter("lightrag")
    class LightRAGAdapter(BaseFrameworkAdapter):
        ...
    ```

    Args:
        name: 框架名称（唯一标识）

    Returns:
        装饰器函数
    """
    def decorator(cls: Type[BaseFrameworkAdapter]) -> Type[BaseFrameworkAdapter]:
        if not issubclass(cls, BaseFrameworkAdapter):
            raise TypeError(
                f"{cls.__name__} must be a subclass of BaseFrameworkAdapter"
            )
        if name in _ADAPTER_REGISTRY:
            logger.warning(f"Overwriting existing adapter: {name}")
        _ADAPTER_REGISTRY[name] = cls
        logger.debug(f"Registered adapter: {name} -> {cls.__name__}")
        return cls
    return decorator


def get_adapter(name: str) -> Type[BaseFrameworkAdapter]:
    """
    获取适配器类

    如果适配器未注册但支持延迟加载，会自动导入对应模块。

    Args:
        name: 框架名称

    Returns:
        适配器类

    Raises:
        ValueError: 适配器不存在
    """
    # 已注册则直接返回
    if name in _ADAPTER_REGISTRY:
        return _ADAPTER_REGISTRY[name]

    # 尝试延迟加载
    if name in _LAZY_LOAD_MAP:
        module_path = _LAZY_LOAD_MAP[name]
        try:
            importlib.import_module(module_path)
            if name in _ADAPTER_REGISTRY:
                logger.info(f"Lazy loaded adapter: {name}")
                return _ADAPTER_REGISTRY[name]
        except ImportError as e:
            raise ValueError(
                f"Failed to lazy load adapter '{name}' from '{module_path}': {e}"
            )

    raise ValueError(
        f"Unknown adapter: '{name}'. "
        f"Available: {list(_ADAPTER_REGISTRY.keys())}. "
        f"Lazy-loadable: {list(_LAZY_LOAD_MAP.keys())}"
    )


def create_adapter(
    name: str,
    working_dir: str,
    config: FrameworkConfig
) -> BaseFrameworkAdapter:
    """
    工厂方法：创建适配器实例

    Args:
        name: 框架名称
        working_dir: 工作目录
        config: 框架配置

    Returns:
        适配器实例
    """
    adapter_cls = get_adapter(name)
    return adapter_cls(working_dir, config)


def list_adapters() -> list:
    """
    列出所有已注册的适配器

    Returns:
        适配器名称列表
    """
    # 合并已注册和可延迟加载的
    all_adapters = set(_ADAPTER_REGISTRY.keys()) | set(_LAZY_LOAD_MAP.keys())
    return sorted(all_adapters)


def has_adapter(name: str) -> bool:
    """
    检查适配器是否存在

    Args:
        name: 框架名称

    Returns:
        True 如果适配器已注册或可延迟加载
    """
    return name in _ADAPTER_REGISTRY or name in _LAZY_LOAD_MAP


def register_lazy_load(name: str, module_path: str) -> None:
    """
    注册延迟加载映射

    Args:
        name: 框架名称
        module_path: 模块路径
    """
    _LAZY_LOAD_MAP[name] = module_path
    logger.debug(f"Registered lazy load: {name} -> {module_path}")


def clear_registry() -> None:
    """清空注册表（仅用于测试）"""
    _ADAPTER_REGISTRY.clear()