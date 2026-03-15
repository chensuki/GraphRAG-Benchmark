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

import tiktoken

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
    """LinearRAG 适配器：支持迭代检索和 SpaCy NER

    - 纯 Token 切分（chunk_size=1200, overlap=100）
    - 使用 tiktoken cl100k_base 编码
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        _ensure_linearrag_in_path()

        self._rag = None
        self._embedding_model = None
        self._tokenizer = None

        os.environ["OPENAI_API_KEY"] = config.llm_api_key
        os.environ["OPENAI_BASE_URL"] = config.llm_base_url

    def _get_tokenizer(self) -> tiktoken.Encoding:
        """获取 tiktoken 编码器（延迟初始化）"""
        if self._tokenizer is None:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        return self._tokenizer

    def _chunking_by_token_size(
        self,
        content: str,
        chunk_token_size: int = 1200,
        chunk_overlap_token_size: int = 100,
    ) -> List[str]:
        """

        步长 = chunk_token_size - chunk_overlap_token_size
        每次取 chunk_token_size 个 token
        """
        tokenizer = self._get_tokenizer()
        tokens = tokenizer.encode(content)

        if len(tokens) <= chunk_token_size:
            return [content]

        chunks = []
        step = chunk_token_size - chunk_overlap_token_size

        for start in range(0, len(tokens), step):
            chunk_tokens = tokens[start:start + chunk_token_size]
            chunk_content = tokenizer.decode(chunk_tokens)
            chunks.append(chunk_content.strip())

            if start + chunk_token_size >= len(tokens):
                break

        return chunks

    def _prepare_passages(self, content: str) -> List[str]:
        """为索引准备段落（带编号前缀）

        格式：["0:passage1", "1:passage2", ...]
        """
        chunk_token_size = self.config.get_extra("chunk_token_size", 1200)
        chunk_overlap_token_size = self.config.get_extra("chunk_overlap_token_size", 100)

        chunks = self._chunking_by_token_size(
            content,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
        )

        return [f"{i}:{chunk}" for i, chunk in enumerate(chunks)]

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
            dataset_name=".",
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
        """构建索引

        一次性插入完整语料库内容，内部使用与 LightRAG 相同的 Token 切分。
        """
        from src.LinearRAG import LinearRAG

        corpus_name = kwargs.get("corpus_name", "default")
        content = self._ensure_content_string(content)

        logger.info(f"Indexing: {corpus_name} ({len(content)} chars)")

        passages = self._prepare_passages(content)

        if not passages:
            raise ValueError("No passages generated for indexing")

        linearrag_config = self._create_config(corpus_name)
        self._rag = LinearRAG(global_config=linearrag_config)
        self._rag.index(passages)
        logger.info(f"Index built: {corpus_name} ({len(passages)} passages)")

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
            # 去除编号前缀 "0:", "1:", etc.
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