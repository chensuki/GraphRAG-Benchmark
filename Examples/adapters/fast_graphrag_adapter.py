"""
Fast-GraphRAG 框架适配器

将 Fast-GraphRAG 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

import numpy as np

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)

# 默认配置
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
    """
    OpenAI 兼容的 Embedding Service

    支持 Zhipu、DeepSeek 等 OpenAI 兼容 API。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        embedding_dim: int = 1024,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_dim = embedding_dim

    async def encode(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        """异步编码文本"""
        import httpx

        model_to_use = model or self.model

        async with httpx.AsyncClient(timeout=60.0) as client:
            request_body = {
                "model": model_to_use,
                "input": texts
            }

            # Zhipu embedding-3 支持 dimensions 参数
            if "embedding-3" in model_to_use:
                request_body["dimensions"] = self.embedding_dim

            response = await client.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=request_body
            )

            response.raise_for_status()
            data = response.json()

            embeddings = [item["embedding"] for item in data["data"]]
            return np.array(embeddings)


@register_adapter("fast-graphrag")
class FastGraphRAGAdapter(BaseFrameworkAdapter):
    """
    Fast-GraphRAG 框架适配器

    支持 API 和 Local 两种 embedding 模式，
    支持 API 和 Ollama 两种 LLM 模式。
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._grag = None
        self._mode = config.get_extra("mode", "API")
        self._embed_provider = config.get_extra("embed_provider", "api")

        # 框架特定配置
        self._domain = config.get_extra("domain") or DEFAULT_DOMAIN
        self._entity_types = config.get_extra("entity_types") or DEFAULT_ENTITY_TYPES
        self._example_queries = config.get_extra("example_queries") or DEFAULT_EXAMPLE_QUERIES

    def _create_embedding_service(self) -> Any:
        """创建 Embedding Service"""
        provider = self._embed_provider
        model = self.config.embed_model
        api_key = self.config.embed_api_key
        base_url = self.config.embed_base_url

        if provider == "api":
            # embedding-3 维度为 2048，其他为 1024
            embedding_dim = 2048 if "embedding-3" in model else 1024
            logger.info(f"Using API embedding service: {model} at {base_url}")
            return OpenAICompatibleEmbeddingService(
                model=model,
                base_url=base_url,
                api_key=api_key,
                embedding_dim=embedding_dim,
            )
        else:
            # Local 模式使用 HuggingFace
            from transformers import AutoTokenizer, AutoModel
            from fast_graphrag._llm import HuggingFaceEmbeddingService as HFEmbedding

            embedding_tokenizer = AutoTokenizer.from_pretrained(model)
            embedding_model = AutoModel.from_pretrained(model)
            logger.info(f"Loaded local embedding model: {model}")
            return HFEmbedding(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                embedding_dim=1024,
                max_token_size=8192,
            )

    def _create_llm_service(self) -> Any:
        """创建 LLM Service"""
        if self._mode == "ollama":
            from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper
            ollama_client = OllamaClient(base_url=self.config.llm_base_url)
            logger.info(f"Using Ollama LLM service: {self.config.llm_model} at {self.config.llm_base_url}")
            return OllamaWrapper(ollama_client, self.config.llm_model)
        else:
            from fast_graphrag._llm import OpenAILLMService
            logger.info(f"Using OpenAI-compatible LLM service: {self.config.llm_model} at {self.config.llm_base_url}")
            return OpenAILLMService(
                model=self.config.llm_model,
                base_url=self.config.llm_base_url,
                api_key=self.config.llm_api_key,
            )

    async def _init_grag(self) -> None:
        """延迟初始化 GraphRAG 实例"""
        if self._grag is not None:
            return

        from fast_graphrag import GraphRAG

        embedding_service = self._create_embedding_service()
        llm_service = self._create_llm_service()

        self._grag = GraphRAG(
            working_dir=self.working_dir,
            domain=self._domain,
            example_queries="\n".join(self._example_queries),
            entity_types=self._entity_types,
            config=GraphRAG.Config(
                llm_service=llm_service,
                embedding_service=embedding_service,
            ),
        )

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)

        await self._init_grag()

        logger.info(f"Starting indexing")

        # Fast-GraphRAG 的 insert 是同步方法，需要异步包装
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._grag.insert, content)

        logger.info(f"Indexed corpus: {len(content.split())} words")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        await self._init_grag()

        try:
            # Fast-GraphRAG 的 query 是同步方法，需要异步包装
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._grag.query, question)

            answer = response.response if response else ""

            # 提取上下文
            context = []
            try:
                response_dict = response.to_dict()
                if 'context' in response_dict and 'chunks' in response_dict['context']:
                    context_chunks = response_dict['context']['chunks']
                    context = [item[0]["content"] for item in context_chunks if item] if context_chunks else []
            except Exception as e:
                logger.warning(f"Failed to extract context: {e}")

            return BenchmarkQueryResult(answer=answer, context=context)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return BenchmarkQueryResult(answer=f"Query failed: {e}", context=[])

    async def aclose(self) -> None:
        """清理资源"""
        self._grag = None