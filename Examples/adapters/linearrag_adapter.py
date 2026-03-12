"""
LinearRAG 框架适配器

将 LinearRAG 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)


def _ensure_linearrag_in_path() -> None:
    """确保 LinearRAG 模块在路径中"""
    project_root = Path(__file__).parent.parent.parent
    linearrag_dir = project_root / "linearrag"
    if str(linearrag_dir) not in sys.path:
        sys.path.insert(0, str(linearrag_dir))


@register_adapter("linearrag")
class LinearRAGAdapter(BaseFrameworkAdapter):
    """
    LinearRAG 框架适配器

    特点：
    - 支持向量化检索和迭代检索
    - 使用 SpaCy 进行 NER
    - 通过环境变量配置 LLM
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        _ensure_linearrag_in_path()

        self._rag = None
        self._embedding_model = None
        self._corpus_name = None

        # 设置 LLM 环境变量（LinearRAG 通过环境变量读取）
        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

    def _create_embedding_model(self) -> Any:
        """创建嵌入模型"""
        if self._embedding_model is not None:
            return self._embedding_model

        provider = self.config.embed_provider
        model_name = self.config.embed_model

        if provider in ["zhipu", "api"]:
            from src.zhipu_embedding import create_zhipu_embedding_model
            api_key = self.config.embed_api_key or self.config.extra.get("zhipuai_api_key")
            self._embedding_model = create_zhipu_embedding_model(
                api_key=api_key,
                model_name=model_name,
                batch_size=128
            )
        else:
            # 使用本地模型
            try:
                from sentence_transformers import SentenceTransformer
                device = "cuda" if self._has_cuda() else "cpu"
                self._embedding_model = SentenceTransformer(model_name, device=device)
            except ImportError:
                logger.warning(f"sentence_transformers not available, using zhipu")
                return self._create_embedding_model()

        return self._embedding_model

    @staticmethod
    def _has_cuda() -> bool:
        """检查是否有 CUDA"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    def _create_config(self, corpus_name: str) -> Any:
        """创建 LinearRAG 配置"""
        from src.config import LinearRAGConfig
        from src.utils import LLM_Model

        llm_model = LLM_Model(self.config.llm_model)
        embedding_model = self._create_embedding_model()

        return LinearRAGConfig(
            dataset_name=corpus_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            working_dir=self.working_dir,
            retrieval_top_k=self.config.top_k,
            max_iterations=self.config.get_extra("max_iterations", 3),
            iteration_threshold=self.config.get_extra("iteration_threshold", 0.4),
            top_k_sentence=self.config.get_extra("top_k_sentence", 3),
            use_vectorized_retrieval=self.config.get_extra("use_vectorized", False),
            max_workers=self.config.get_extra("max_workers", 8),
            spacy_model=self.config.get_extra("spacy_model", "en_core_web_trf"),
            passage_ratio=self.config.get_extra("passage_ratio", 1.5),
            damping=self.config.get_extra("damping", 0.5),
        )

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        from src.LinearRAG import LinearRAG

        content = self._ensure_content_string(content)
        corpus_name = kwargs.get("corpus_name", "default")
        self._corpus_name = corpus_name

        # 准备 passages（LinearRAG 格式：["0:段落1", "1:段落2", ...]）
        passages = self._prepare_passages(content)

        # 创建配置和 RAG 实例
        config = self._create_config(corpus_name)
        self._rag = LinearRAG(global_config=config)

        # 构建索引
        self._rag.index(passages)
        logger.info(f"Index built for {corpus_name}: {len(passages)} passages")

    def _prepare_passages(self, content: str) -> List[str]:
        """
        准备 LinearRAG 格式的 passages

        LinearRAG 期望格式：["0:段落1", "1:段落2", ...]
        """
        # 按换行或段落分割
        if "\n\n" in content:
            chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        elif "\n" in content:
            chunks = [chunk.strip() for chunk in content.split("\n") if chunk.strip()]
        else:
            chunks = [content.strip()]

        # 添加编号前缀
        return [f"{i}:{chunk}" for i, chunk in enumerate(chunks)]

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行单条查询"""
        results = await self.abatch_query(
            [{"question": question, "answer": ""}],
            top_k=top_k,
            **kwargs
        )
        return results[0] if results else BenchmarkQueryResult(answer="", context=[])

    async def abatch_query(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 5,
        **kwargs
    ) -> List[BenchmarkQueryResult]:
        import asyncio

        if self._rag is None:
            raise RuntimeError("Index not built. Call abuild_index() first.")

        linear_questions = [
            {"question": q["question"], "answer": q.get("answer", "")}
            for q in questions
        ]

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            self._rag.qa,
            linear_questions
        )

        benchmark_results = []
        for result in results:
            answer = result.get("pred_answer", "")
            raw_context = result.get("sorted_passage", [])
            context = self._normalize_context(raw_context)
            benchmark_results.append(BenchmarkQueryResult(answer=answer, context=context))

        return benchmark_results

    def _normalize_context(self, raw_context: List[str]) -> List[str]:
        """移除 LinearRAG 的 passage 编号前缀"""
        normalized = []
        seen = set()

        for ctx in raw_context:
            cleaned = ctx
            if ":" in ctx and ctx.split(":")[0].isdigit():
                cleaned = ":".join(ctx.split(":")[1:]).strip()

            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)

        return normalized

    def load_index(self, corpus_name: str) -> None:
        """加载已有索引"""
        from src.LinearRAG import LinearRAG

        self._corpus_name = corpus_name
        config = self._create_config(corpus_name)
        self._rag = LinearRAG(global_config=config)
        self._rag.load_index()
        logger.info(f"Index loaded for {corpus_name}")