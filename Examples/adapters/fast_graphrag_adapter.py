"""
Fast-GraphRAG 框架适配器
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

import numpy as np

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
DEFAULT_EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
DEFAULT_ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]


class OpenAICompatibleEmbeddingService:
    """OpenAI 兼容 Embedding Service"""

    def __init__(self, model: str, base_url: str, api_key: str, dimensions: Optional[int] = None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.dimensions = dimensions

    async def encode(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        import httpx

        payload = {"model": model or self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            resp.raise_for_status()
            return np.array([item["embedding"] for item in resp.json()["data"]])


@register_adapter("fast-graphrag")
class FastGraphRAGAdapter(BaseFrameworkAdapter):
    """Fast-GraphRAG 适配器"""

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._grag = None
        self._mode = config.get_extra("mode", "API")
        self._domain = config.get_extra("domain") or DEFAULT_DOMAIN
        self._entity_types = config.get_extra("entity_types") or DEFAULT_ENTITY_TYPES
        self._example_queries = config.get_extra("example_queries") or DEFAULT_EXAMPLE_QUERIES

    def _create_embedding_service(self) -> Any:
        """创建 Embedding Service"""
        model = self.config.embed_model
        base_url = self.config.get_effective_embed_base_url()
        api_key = self.config.embed_api_key
        dimensions = self.config.embed_dimensions

        if self.config.embed_type == "local":
            from transformers import AutoTokenizer, AutoModel
            from fast_graphrag._llm import HuggingFaceEmbeddingService as HFEmbedding
            logger.info(f"Using local embedding: {model}")
            return HFEmbedding(
                model=AutoModel.from_pretrained(model),
                tokenizer=AutoTokenizer.from_pretrained(model),
                embedding_dim=dimensions or 1024,
                max_token_size=8192
            )

        logger.info(f"Using API embedding: {model}")
        return OpenAICompatibleEmbeddingService(model=model, base_url=base_url, api_key=api_key, dimensions=dimensions)

    def _create_llm_service(self) -> Any:
        """创建 LLM Service"""
        if self._mode == "ollama":
            from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper
            return OllamaWrapper(OllamaClient(base_url=self.config.llm_base_url), self.config.llm_model)

        from fast_graphrag._llm import OpenAILLMService
        return OpenAILLMService(
            model=self.config.llm_model,
            base_url=self.config.llm_base_url,
            api_key=self.config.llm_api_key
        )

    async def _init_grag(self) -> None:
        if self._grag is not None:
            return

        from fast_graphrag import GraphRAG
        self._grag = GraphRAG(
            working_dir=self.working_dir,
            domain=self._domain,
            example_queries="\n".join(self._example_queries),
            entity_types=self._entity_types,
            config=GraphRAG.Config(
                llm_service=self._create_llm_service(),
                embedding_service=self._create_embedding_service()
            )
        )

    async def abuild_index(self, content: str, **kwargs) -> None:
        await self._init_grag()
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._grag.insert, self._ensure_content_string(content))
        logger.info(f"Index built: {self.working_dir}")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        await self._init_grag()
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, self._grag.query, question)

            context = []
            try:
                chunks = resp.to_dict().get('context', {}).get('chunks', [])
                context = [c[0]["content"] for c in chunks if c and c[0]]
            except Exception:
                pass

            return BenchmarkQueryResult(answer=resp.response if resp else "", context=context)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return BenchmarkQueryResult(answer=f"Query failed: {e}", context=[])

    async def aclose(self) -> None:
        self._grag = None