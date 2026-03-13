"""
LinearRAG 框架适配器
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)


def _ensure_linearrag_in_path() -> None:
    project_root = Path(__file__).parent.parent.parent
    linearrag_dir = project_root / "linearrag"
    if str(linearrag_dir) not in sys.path:
        sys.path.insert(0, str(linearrag_dir))


@register_adapter("linearrag")
class LinearRAGAdapter(BaseFrameworkAdapter):
    """LinearRAG 适配器：支持迭代检索和 SpaCy NER"""

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        _ensure_linearrag_in_path()

        self._rag = None
        self._embedding_model = None

        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

    def _create_embedding_model(self) -> Any:
        """创建嵌入模型"""
        if self._embedding_model is not None:
            return self._embedding_model

        if self.config.embed_type == "local":
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            self._embedding_model = SentenceTransformer(self.config.embed_model, device=device)
            logger.info(f"Using local embedding: {self.config.embed_model}")
        else:
            from src.openai_embedding import create_openai_embedding_model
            self._embedding_model = create_openai_embedding_model(
                api_key=self.config.embed_api_key,
                model_name=self.config.embed_model,
                base_url=self.config.get_effective_embed_base_url(),
                batch_size=self.config.embed_batch_size,
                dimensions=self.config.embed_dimensions
            )
            logger.info(f"Using API embedding ({self.config.embed_provider}): {self.config.embed_model}")

        return self._embedding_model

    def _create_config(self, corpus_name: str) -> Any:
        from src.config import LinearRAGConfig
        from src.utils import LLM_Model

        return LinearRAGConfig(
            dataset_name=corpus_name,
            embedding_model=self._create_embedding_model(),
            llm_model=LLM_Model(self.config.llm_model),
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
        from src.LinearRAG import LinearRAG

        content = self._ensure_content_string(content)
        corpus_name = kwargs.get("corpus_name", "default")

        passages = self._prepare_passages(content)
        config = self._create_config(corpus_name)
        self._rag = LinearRAG(global_config=config)
        self._rag.index(passages)
        logger.info(f"Index built for {corpus_name}: {len(passages)} passages")

    def _prepare_passages(self, content: str) -> List[str]:
        if "\n\n" in content:
            chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        elif "\n" in content:
            chunks = [c.strip() for c in content.split("\n") if c.strip()]
        else:
            chunks = [content.strip()]
        return [f"{i}:{c}" for i, c in enumerate(chunks)]

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        results = await self.abatch_query([{"question": question, "answer": ""}], top_k=top_k)
        return results[0] if results else BenchmarkQueryResult(answer="", context=[])

    async def abatch_query(self, questions: List[Dict[str, Any]], top_k: int = 5, **kwargs) -> List[BenchmarkQueryResult]:
        if self._rag is None:
            raise RuntimeError("Index not built")

        linear_questions = [{"question": q["question"], "answer": q.get("answer", "")} for q in questions]
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, self._rag.qa, linear_questions)

        return [
            BenchmarkQueryResult(
                answer=r.get("pred_answer", ""),
                context=self._normalize_context(r.get("sorted_passage", []))
            )
            for r in results
        ]

    def _normalize_context(self, raw_context: List[str]) -> List[Dict[str, Any]]:
        """构建统一格式的 context"""
        seen = set()
        result = []
        for ctx in raw_context:
            if ":" in ctx and ctx.split(":")[0].isdigit():
                ctx = ":".join(ctx.split(":")[1:]).strip()
            if ctx and ctx not in seen:
                seen.add(ctx)
                result.append({"type": "chunk", "content": ctx})
        return result

    def load_index(self, corpus_name: str) -> None:
        from src.LinearRAG import LinearRAG

        config = self._create_config(corpus_name)
        self._rag = LinearRAG(global_config=config)
        self._rag.load_index()
        logger.info(f"Index loaded for {corpus_name}")