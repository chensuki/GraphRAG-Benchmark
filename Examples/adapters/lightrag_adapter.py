"""
LightRAG 框架适配器
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable, List

from .base import (
    BaseFrameworkAdapter,
    BenchmarkQueryResult,
    FrameworkConfig,
    format_entity_content,
    format_relationship_content,
)
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
    """LightRAG 适配器"""

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._rag = None
        self._mode = config.get_extra("mode", "API")
        self._corpus_name = os.path.basename(working_dir)

    def _create_embedding_func(self) -> Any:
        """创建嵌入函数"""
        from lightrag.utils import EmbeddingFunc

        model = self.config.embed_model
        api_key = self.config.embed_api_key
        base_url = self.config.get_effective_embed_base_url()
        dimensions = self.config.embed_dimensions

        if self.config.embed_type == "local":
            from lightrag.llm.hf import hf_embed
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_obj = AutoModel.from_pretrained(model)
            return EmbeddingFunc(
                embedding_dim=dimensions or 1024, max_token_size=8192,
                func=lambda texts: hf_embed(texts, tokenizer, model_obj)
            )

        if self._mode == "API":
            # 使用 LightRAG 内置的 zhipu 实现（如果 provider 是 zhipu）
            if self.config.embed_provider == "zhipu":
                from lightrag.llm.zhipu import zhipu_embedding
                return EmbeddingFunc(
                    embedding_dim=dimensions or 1024, max_token_size=8192,
                    func=lambda texts: zhipu_embedding(texts, model=model, api_key=api_key, embedding_dim=dimensions)
                )
            # 其他 provider 使用 OpenAI 兼容接口
            from lightrag.llm.openai import openai_embed
            return EmbeddingFunc(
                embedding_dim=dimensions or 1024, max_token_size=8192,
                func=lambda texts: openai_embed(texts, model=model, base_url=base_url, api_key=api_key, embedding_dim=dimensions)
            )

        from lightrag.llm.ollama import ollama_embed
        return EmbeddingFunc(
            embedding_dim=dimensions or 1024, max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model=model, host=base_url)
        )

    def _create_llm_func(self) -> Callable:
        """创建 LLM 函数"""
        model = self.config.llm_model
        base_url = self.config.llm_base_url
        api_key = self.config.llm_api_key

        if self._mode == "API":
            from lightrag.llm.openai import openai_complete_if_cache

            async def llm_func(prompt: str, system_prompt: str = None, history_messages: list = None,
                               keyword_extraction: bool = False, **kwargs) -> str:
                return await openai_complete_if_cache(
                    kwargs.pop("model_name", model), prompt,
                    system_prompt=system_prompt, history_messages=history_messages or [],
                    base_url=kwargs.pop("base_url", base_url),
                    api_key=kwargs.pop("api_key", api_key) or os.getenv("LLM_API_KEY", ""),
                    keyword_extraction=keyword_extraction, **kwargs
                )
            return llm_func

        from lightrag.llm.ollama import ollama_model_complete

        async def llm_func(prompt: str, **kwargs) -> str:
            return await ollama_model_complete(prompt, host=base_url, options={"num_ctx": 32768}, **kwargs)
        return llm_func

    async def _init_rag(self) -> None:
        """延迟初始化"""
        if self._rag is not None:
            return

        from lightrag import LightRAG
        from lightrag.kg.shared_storage import initialize_pipeline_status

        # 获取超时配置（默认 300 秒）
        llm_timeout = self.config.get_extra("llm_timeout", 300)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            workspace="",
            llm_model_func=self._create_llm_func(),
            llm_model_name=self.config.llm_model,
            llm_model_max_async=4,
            chunk_token_size=self.config.get_extra("chunk_token_size", 1200),
            chunk_overlap_token_size=self.config.get_extra("chunk_overlap_token_size", 100),
            embedding_func=self._create_embedding_func(),
            default_llm_timeout=llm_timeout,
            llm_model_kwargs={
                "model_name": self.config.llm_model,
                "base_url": self.config.llm_base_url,
                "api_key": self.config.llm_api_key,
                "timeout": llm_timeout,
            }
        )
        await self._rag.initialize_storages()
        await initialize_pipeline_status()

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        await self._init_rag()
        await self._rag.ainsert(self._ensure_content_string(content))
        logger.info(f"Index built: {self._corpus_name}")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        await self._init_rag()

        from lightrag import QueryParam
        query_param = QueryParam(
            mode=kwargs.get("mode", "hybrid"), top_k=top_k,
            max_entity_tokens=4000, max_relation_tokens=4000, max_total_tokens=4000
        )

        try:
            # 使用 aquery_llm 一次性获取答案和结构化数据
            result = await self._rag.aquery_llm(question, param=query_param, system_prompt=SYSTEM_PROMPT)

            # 提取 LLM 响应
            llm_response = result.get("llm_response", {})
            answer = llm_response.get("content", "") if not llm_response.get("is_streaming") else ""

            # 构建统一格式的 context
            context = self._build_unified_context(result)

            return BenchmarkQueryResult(answer=answer, context=context)
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return BenchmarkQueryResult(answer=f"Query failed: {e}", context=[])

    def _build_unified_context(self, result: dict) -> List[dict]:
        """构建统一格式的 context"""
        context = []

        if not result:
            return context

        status = result.get("status", "unknown")
        if status == "failure":
            logger.warning(f"Query failed: {result.get('message', '')}")
            return context

        data = result.get("data", {})
        if not data:
            return context

        chunks = data.get("chunks", [])
        for chunk in chunks:
            if isinstance(chunk, dict):
                content = chunk.get("content", "")
                if content:
                    context.append({"type": "chunk", "content": content})
            elif isinstance(chunk, str):
                context.append({"type": "chunk", "content": chunk})

        entities = data.get("entities", [])
        for entity in entities:
            if isinstance(entity, dict):
                entity_name = entity.get("entity_name", "")
                entity_type = entity.get("entity_type", "")
                description = entity.get("description", "")
                content = format_entity_content(entity_name, entity_type, description)
                if content:
                    context.append({
                        "type": "entity",
                        "content": content,
                        "entity_name": entity_name,
                        "entity_type": entity_type,
                        "description": description
                    })

        relationships = data.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict):
                src = rel.get("src_id", "")
                tgt = rel.get("tgt_id", "")
                desc = rel.get("description", "")
                content = format_relationship_content(src, tgt, desc)
                if content:
                    context.append({
                        "type": "relationship",
                        "content": content,
                        "src_id": src,
                        "tgt_id": tgt,
                        "description": desc,
                        "keywords": rel.get("keywords", ""),
                        "weight": rel.get("weight", 1.0)
                    })

        return context

    async def aclose(self) -> None:
        """清理资源"""
        self._rag = None