"""
ClearRAG 框架适配器

将 ClearRAG 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    BaseFrameworkAdapter,
    BenchmarkQueryResult,
    FrameworkConfig,
    format_entity_content,
    format_relationship_content,
)
from .registry import register_adapter

logger = logging.getLogger(__name__)


def _ensure_clearrag_in_path() -> None:
    """确保 ClearRAG 模块在路径中"""
    project_root = Path(__file__).parent.parent.parent
    clearrag_dir = project_root / "clearrag"

    if str(clearrag_dir) not in sys.path:
        sys.path.insert(0, str(clearrag_dir))

    # 添加适配器路径
    adapter_path = clearrag_dir / "Examples"
    if str(adapter_path) not in sys.path:
        sys.path.insert(0, str(adapter_path))


@register_adapter("clearrag")
class ClearRAGAdapter(BaseFrameworkAdapter):
    """
    ClearRAG 框架适配器

    特点：
    - 使用 Neo4j 作为图数据库
    - 支持多种激活模式（semantic_propagation, vector_search）
    - 支持段落检索模式（pagerank, vector, none）
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        _ensure_clearrag_in_path()

        self._adapter = None
        self._corpus_name = None

        # 从配置获取 Neo4j 参数
        self._neo4j_uri = config.get_extra("neo4j_uri") or os.getenv("NEO4J_URI")
        self._neo4j_user = config.get_extra("neo4j_user") or os.getenv("NEO4J_USER")
        self._neo4j_password = config.get_extra("neo4j_password") or os.getenv("NEO4J_PASSWORD")
        self._neo4j_database = config.get_extra("neo4j_database") or os.getenv("NEO4J_DATABASE", "neo4j")

        # ClearRAG 特定配置
        self._activation_mode = config.get_extra("activation_mode", "semantic_propagation")
        self._passage_retrieval_mode = config.get_extra("passage_retrieval_mode", "pagerank")
        self._fast_mode = config.get_extra("fast_mode", False)

    def _create_adapter(self, corpus_name: str) -> Any:
        """创建 ClearRAG Benchmark 适配器"""
        try:
            from clearrag_benchmark_adapter import ClearRAGBenchmarkAdapter
        except ImportError as e:
            raise ImportError(
                f"Failed to import ClearRAGBenchmarkAdapter: {e}. "
                "Make sure clearrag/Examples/clearrag_benchmark_adapter.py exists."
            )

        project_root = Path(__file__).parent.parent.parent
        clearrag_dir = project_root / "clearrag"

        # 获取有效的 embedding base_url
        effective_embed_base_url = self.config.get_effective_embed_base_url()

        # 切分参数
        chunk_size = self.config.get_extra("chunk_size", 1200)
        chunk_overlap = self.config.get_extra("chunk_overlap", 100)

        # 构建参数
        max_tool_calls = self.config.get_extra("max_tool_calls", 3)
        max_tokens = self.config.get_extra("max_tokens", 8192)
        similarity_threshold = self.config.get_extra("similarity_threshold", 0.95)

        # 查询参数
        vector_search_top_k = self.config.get_extra("vector_search_top_k", 100)
        activation_threshold = self.config.get_extra("activation_threshold", 0.8)
        passage_top_k = self.config.top_k  # 使用全局 top_k
        max_context_length = self.config.get_extra("max_context_length", 10000)
        max_passage_content_length = self.config.get_extra("max_passage_content_length", 2000)
        embedding_dimensions = self.config.embed_dimensions
        embedding_mode = self.config.embed_type if self.config.embed_type in ("api", "local") else "api"

        return ClearRAGBenchmarkAdapter(
            working_dir=self.working_dir,
            llm_model_name=self.config.llm_model,
            embedding_model_name=self.config.embed_model,
            llm_base_url=self.config.llm_base_url,
            llm_api_key=self.config.llm_api_key,
            embed_base_url=effective_embed_base_url,
            embed_api_key=self.config.embed_api_key,
            neo4j_uri=self._neo4j_uri,
            neo4j_user=self._neo4j_user,
            neo4j_password=self._neo4j_password,
            neo4j_database=self._neo4j_database,
            activation_mode=self._activation_mode,
            passage_retrieval_mode=self._passage_retrieval_mode,
            fast_mode=self._fast_mode,
            max_concurrency=self.config.max_concurrency,
            enable_checkpoint=True,
            skill_registry_path=str(clearrag_dir / "clearrag" / "skills"),
            # 切分参数
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 构建参数
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            similarity_threshold=similarity_threshold,
            # 查询参数
            vector_search_top_k=vector_search_top_k,
            activation_threshold=activation_threshold,
            passage_top_k=passage_top_k,
            max_context_length=max_context_length,
            max_passage_content_length=max_passage_content_length,
            embedding_dimensions=embedding_dimensions,
            embedding_mode=embedding_mode,
        )

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)
        corpus_name = kwargs.get("corpus_name", "default")
        self._corpus_name = corpus_name

        # 创建适配器
        self._adapter = self._create_adapter(corpus_name)

        # 构建索引
        await self._adapter.abuild_index(content=content, source_type="text")
        logger.info(f"Index built for {corpus_name}")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        if self._adapter is None:
            raise RuntimeError("Index not built. Call abuild_index() first.")

        result = await self._adapter.aquery(question=question, top_k=top_k)

        context = self._build_unified_context(result)

        return BenchmarkQueryResult(
            answer=result.answer,
            context=context
        )

    async def abatch_query(
        self,
        questions: List[Dict[str, Any]],
        top_k: int = 5,
        batch_concurrency: int = 5,
        **kwargs
    ) -> List[BenchmarkQueryResult]:
        """
        批量查询
        """
        if self._adapter is None:
            raise RuntimeError("Index not built. Call abuild_index() first.")

        # 提取问题文本列表
        question_texts = [q.get("question", "") if isinstance(q, dict) else str(q) for q in questions]

        # 调用批量查询
        results = await self._adapter.abatch_query(
            questions=question_texts,
            top_k=top_k,
            batch_concurrency=batch_concurrency,
            **kwargs
        )

        # 转换为标准格式
        return [
            BenchmarkQueryResult(
                answer=r.answer,
                context=self._build_unified_context(r)
            )
            for r in results
        ]

    def _build_unified_context(self, result: Any) -> List[Dict[str, Any]]:
        """构建统一格式的 context"""
        context = []

        if hasattr(result, 'passages') and result.passages:
            for passage in result.passages:
                if isinstance(passage, dict):
                    text = passage.get("content", "") or passage.get("text", "")
                    if text:
                        context.append({"type": "chunk", "content": text})
                elif isinstance(passage, str) and passage:
                    context.append({"type": "chunk", "content": passage})

        if hasattr(result, 'nodes') and result.nodes:
            for node in result.nodes:
                if isinstance(node, dict):
                    name = node.get("name", "")
                    desc = node.get("description", "")
                    entity_type = node.get("entity_type", node.get("type", ""))
                    content = format_entity_content(name, entity_type, desc)
                    if content:
                        context.append({
                            "type": "entity",
                            "content": content,
                            "name": name,
                            "entity_type": entity_type,
                            "description": desc
                        })

        if hasattr(result, 'relationships') and result.relationships:
            for rel in result.relationships:
                if isinstance(rel, dict):
                    source = rel.get("source_name", "") or rel.get("source", {}).get("name", "")
                    target = rel.get("target_name", "") or rel.get("target", {}).get("name", "")
                    desc = rel.get("description", "")
                    content = format_relationship_content(source, target, desc)
                    if content:
                        context.append({
                            "type": "relationship",
                            "content": content,
                            "source_name": source,
                            "target_name": target,
                            "description": desc
                        })

        if not context and hasattr(result, 'context') and result.context:
            for ctx in result.context:
                if isinstance(ctx, str) and ctx:
                    context.append({"type": "chunk", "content": ctx})

        return context

    def load_index(self, corpus_name: str) -> None:
        """加载已有索引"""
        self._corpus_name = corpus_name
        self._adapter = self._create_adapter(corpus_name)
        logger.info(f"Index loaded for {corpus_name}")

    async def aclose(self) -> None:
        """清理资源"""
        if self._adapter:
            try:
                await self._adapter.aclose()
                logger.info(f"Resources cleaned up for {self._corpus_name}")
            except Exception as e:
                logger.warning(f"Failed to close adapter: {e}")
            finally:
                self._adapter = None