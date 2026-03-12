"""
DIGIMON 框架适配器

将 DIGIMON 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import logging
from typing import List

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("digimon")
class DigimonAdapter(BaseFrameworkAdapter):
    """
    DIGIMON 框架适配器

    特点：
    - 使用 YAML 配置
    - 支持多种检索模式
    """

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)

        # TODO: 实现 DIGIMON 索引构建
        raise NotImplementedError(
            "DIGIMON adapter is not yet implemented. "
            "Please use Examples/run_digimon.py directly for now."
        )

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        # TODO: 实现 DIGIMON 查询
        raise NotImplementedError(
            "DIGIMON adapter is not yet implemented. "
            "Please use Examples/run_digimon.py directly for now."
        )