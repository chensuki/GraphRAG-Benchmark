"""
DIGIMON 框架适配器

将 DIGIMON 适配到 Benchmark 标准格式。

特点：
- 使用 YAML 配置文件
- 支持多种检索模式
- 异步接口
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter("digimon")
class DigimonAdapter(BaseFrameworkAdapter):
    """
    DIGIMON 框架适配器

    使用 YAML 配置文件初始化，支持 API 和 Ollama 模式。
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._rag = None
        self._opt = None
        self._corpus_name = "default"

        # 框架特定配置
        self._config_path = config.get_extra("config_path", "./config.yml")
        self._mode = config.get_extra("mode", "config")

    def _ensure_context_list(self, context: Any) -> List[str]:
        """确保 context 是字符串列表"""
        if context is None:
            return []
        if isinstance(context, str):
            return [context] if context else []
        if isinstance(context, list):
            return [str(item) if not isinstance(item, str) else item for item in context]
        return [str(context)]

    async def _init_rag(self, corpus_name: str = "default") -> None:
        """初始化 GraphRAG 实例"""
        if self._rag is not None:
            return

        from Core.GraphRAG import GraphRAG
        from Option.Config2 import Config

        self._corpus_name = corpus_name

        # 解析配置
        config_path = Path(self._config_path)
        if not config_path.exists():
            # 尝试在工作目录中查找
            config_path = Path(self.working_dir) / "config.yml"

        if config_path.exists():
            self._opt = Config.parse(config_path, dataset_name=corpus_name)
        else:
            # 使用默认配置
            self._opt = self._create_default_config(corpus_name)

        # 覆盖 LLM 配置
        if self._mode == "ollama":
            if hasattr(self._opt, 'llm_config'):
                self._opt.llm_config.model_name = self.config.llm_model
                self._opt.llm_config.base_url = self.config.llm_base_url
                self._opt.llm_config.api_key = self.config.llm_api_key
                self._opt.llm_config.mode = "ollama"
            logger.info(f"Using Ollama mode: {self.config.llm_model} at {self.config.llm_base_url}")
        else:
            if hasattr(self._opt, 'llm_config'):
                self._opt.llm_config.model_name = self.config.llm_model
                self._opt.llm_config.base_url = self.config.llm_base_url
                self._opt.llm_config.api_key = self.config.llm_api_key
            logger.info(f"Using API mode: {self.config.llm_model} at {self.config.llm_base_url}")

        # 创建 GraphRAG 实例
        self._rag = GraphRAG(config=self._opt)
        logger.info(f"GraphRAG initialized for {corpus_name}")

    def _create_default_config(self, corpus_name: str) -> Any:
        """创建默认配置"""
        from Option.Config2 import Config

        # 尝试使用默认配置路径
        default_config = Path("./config.yml")
        if default_config.exists():
            return Config.parse(default_config, dataset_name=corpus_name)

        # 如果没有配置文件，创建最小配置
        logger.warning(f"No config file found, using minimal config")
        return Config.get_default_config(dataset_name=corpus_name)

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)

        # 从 kwargs 获取 corpus_name 或使用默认值
        corpus_name = kwargs.get("corpus_name", self._corpus_name)

        await self._init_rag(corpus_name)

        # 构建语料库格式
        corpus = [{
            "title": corpus_name,
            "content": content,
            "doc_id": 0,
        }]

        logger.info(f"Indexing corpus: {corpus_name} ({len(content.split())} words)")

        await self._rag.insert(corpus)

        logger.info(f"Indexed corpus: {corpus_name}")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        await self._init_rag()

        try:
            response, context = await self._rag.query(question)

            context_list = self._ensure_context_list(context)

            return BenchmarkQueryResult(
                answer=response if response else "",
                context=context_list
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"Query failed: {error_msg}")
            return BenchmarkQueryResult(answer=f"Query failed: {error_msg}", context=[])

    async def abatch_query(self, questions: List[str], top_k: int = 5, **kwargs) -> List[BenchmarkQueryResult]:
        """批量查询"""
        results = []
        for question in questions:
            result = await self.aquery(question, top_k=top_k, **kwargs)
            results.append(result)
        return results

    async def aclose(self) -> None:
        """清理资源"""
        self._rag = None
        self._opt = None