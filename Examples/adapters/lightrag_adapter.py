"""
LightRAG 框架适配器

将 LightRAG 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Knowledge Base---
{context_data}
"""


@register_adapter("lightrag")
class LightRAGAdapter(BaseFrameworkAdapter):
    """
    LightRAG 框架适配器

    特点：
    - 支持 API 和 Ollama 模式
    - 支持多种嵌入提供商（zhipu, openai, api, hf）
    - 使用 QueryParam 控制查询参数
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._rag = None
        self._mode = config.get_extra("mode", "API")
        self._corpus_name = os.path.basename(working_dir)

    def _create_embedding_func(self) -> Any:
        """创建嵌入函数"""
        from lightrag.utils import EmbeddingFunc

        provider = self.config.embed_provider
        model_name = self.config.embed_model
        api_key = self.config.embed_api_key
        base_url = self.config.embed_base_url or self.config.llm_base_url

        if self._mode == "API":
            if provider == "zhipu":
                from lightrag.llm.zhipu import zhipu_embedding
                return EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: zhipu_embedding(
                        texts, model=model_name, api_key=api_key
                    ),
                )
            elif provider in ("openai", "api"):
                from lightrag.llm.openai import openai_embed
                return EmbeddingFunc(
                    embedding_dim=1536,
                    max_token_size=8192,
                    func=lambda texts: openai_embed(
                        texts, model=model_name, base_url=base_url, api_key=api_key
                    ),
                )
            else:
                from lightrag.llm.hf import hf_embed
                from transformers import AutoModel, AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name)
                return EmbeddingFunc(
                    embedding_dim=1024,
                    max_token_size=8192,
                    func=lambda texts: hf_embed(texts, tokenizer, model),
                )
        elif self._mode == "ollama":
            from lightrag.llm.ollama import ollama_embed
            return EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: ollama_embed(
                    texts, embed_model=model_name, host=base_url
                ),
            )
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

    def _create_llm_func(self) -> Callable:
        """创建 LLM 函数"""
        model_name = self.config.llm_model
        base_url = self.config.llm_base_url
        api_key = self.config.llm_api_key

        if self._mode == "API":
            from lightrag.llm.openai import openai_complete_if_cache

            async def llm_func(
                prompt: str,
                system_prompt: str = None,
                history_messages: list = None,
                keyword_extraction: bool = False,
                **kwargs
            ) -> str:
                final_model_name = kwargs.pop("model_name", model_name)
                final_base_url = kwargs.pop("base_url", base_url)
                final_api_key = kwargs.pop("api_key", api_key)

                if not final_api_key:
                    final_api_key = os.getenv("LLM_API_KEY", "")

                return await openai_complete_if_cache(
                    final_model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    base_url=final_base_url,
                    api_key=final_api_key,
                    keyword_extraction=keyword_extraction,
                    **kwargs
                )
            return llm_func

        elif self._mode == "ollama":
            from lightrag.llm.ollama import ollama_model_complete

            async def llm_func(prompt: str, **kwargs) -> str:
                return await ollama_model_complete(
                    prompt,
                    host=base_url,
                    options={"num_ctx": 32768},
                    **kwargs
                )
            return llm_func

        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

    async def _init_rag(self) -> None:
        """延迟初始化 RAG 实例"""
        if self._rag is not None:
            return

        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status

        embedding_func = self._create_embedding_func()
        llm_func = self._create_llm_func()

        llm_kwargs = {
            "model_name": self.config.llm_model,
            "base_url": self.config.llm_base_url,
            "api_key": self.config.llm_api_key
        }

        # 从配置读取分块参数
        chunk_token_size = self.config.get_extra("chunk_token_size", 1200)
        chunk_overlap_token_size = self.config.get_extra("chunk_overlap_token_size", 100)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            workspace=self._corpus_name,
            llm_model_func=llm_func,
            llm_model_name=self.config.llm_model,
            llm_model_max_async=4,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            embedding_func=embedding_func,
            llm_model_kwargs=llm_kwargs
        )

        await self._rag.initialize_storages()
        await initialize_pipeline_status()

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)

        await self._init_rag()
        await self._rag.ainsert(content)
        logger.info(f"Index built: {len(content.split())} words")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        await self._init_rag()

        from lightrag import QueryParam

        query_param = QueryParam(
            mode=kwargs.get("mode", "hybrid"),
            top_k=top_k,
            max_entity_tokens=4000,
            max_relation_tokens=4000,
            max_total_tokens=4000
        )

        try:
            result = await self._rag.aquery(
                question,
                param=query_param,
                system_prompt=SYSTEM_PROMPT
            )

            answer = str(result) if result else ""
            context = await self._extract_context(question, query_param)

            return BenchmarkQueryResult(answer=answer, context=context)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return BenchmarkQueryResult(answer=f"Query failed: {e}", context=[])

    async def _extract_context(self, question: str, query_param) -> List[str]:
        """从查询结果提取上下文"""
        context = []

        try:
            query_data = await self._rag.aquery_data(question, param=query_param)

            if not query_data or not isinstance(query_data, dict):
                return context

            status = query_data.get("status", "unknown")
            if status == "failure":
                message = query_data.get("message", "")
                logger.warning(f"aquery_data failed: {message}")
                return context

            data = query_data.get("data", {})
            if not data:
                return context

            chunks = data.get("chunks", [])
            if chunks:
                for chunk in chunks:
                    if isinstance(chunk, dict) and "content" in chunk:
                        context.append(chunk["content"])
                    elif isinstance(chunk, str):
                        context.append(chunk)

            if not context:
                entities = data.get("entities", [])
                relationships = data.get("relationships", [])

                for entity in entities:
                    if isinstance(entity, dict):
                        desc = entity.get("description", "")
                        if desc:
                            context.append(desc)

                for rel in relationships:
                    if isinstance(rel, dict):
                        desc = rel.get("description", "")
                        if desc:
                            context.append(desc)

        except Exception as e:
            logger.warning(f"Failed to extract context: {e}")

        return context

    async def aclose(self) -> None:
        """清理资源"""
        self._rag = None