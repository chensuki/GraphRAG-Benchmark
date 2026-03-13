"""
HippoRAG2 框架适配器

将 HippoRAG2 适配到 Benchmark 标准格式。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .base import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig
from .registry import register_adapter

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_CHUNK_TOKEN_SIZE = 256
DEFAULT_CHUNK_OVERLAP = 32


@register_adapter("hipporag2")
class HippoRAG2Adapter(BaseFrameworkAdapter):
    """
    HippoRAG2 框架适配器

    使用本地 embedding 模型，支持 tokenizer 分块和批量查询。
    """

    def __init__(self, working_dir: str, config: FrameworkConfig):
        super().__init__(working_dir, config)
        self._hipporag = None
        self._base_config = None
        self._tokenizer = None
        self._chunks = []

        # HippoRAG2 仅支持本地模型
        if config.embed_type == "api":
            logger.warning(
                "HippoRAG2 only supports local embedding models. "
                f"Ignoring embed_type=api, using local model: {config.embed_model}"
            )

        # 框架特定配置
        self._embed_model_path = config.get_extra("embed_model_path", config.embed_model)
        self._chunk_token_size = config.get_extra("chunk_token_size", DEFAULT_CHUNK_TOKEN_SIZE)
        self._chunk_overlap = config.get_extra("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self._force_index_from_scratch = config.get_extra("force_index_from_scratch", True)
        self._skip_build = config.get_extra("skip_build", False)

        # BaseConfig 参数
        self._rerank_dspy_file_path = config.get_extra(
            "rerank_dspy_file_path",
            "hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"
        )
        self._max_qa_steps = config.get_extra("max_qa_steps", 3)
        self._graph_type = config.get_extra("graph_type", "facts_and_sim_passage_node_unidirectional")
        self._embedding_batch_size = config.get_extra("embedding_batch_size", 8)
        self._openie_mode = config.get_extra("openie_mode", "online")

        logger.info(f"HippoRAG2 using local embedding model: {self._embed_model_path}")

    def _load_tokenizer(self) -> Any:
        """加载 tokenizer"""
        if self._tokenizer is not None:
            return self._tokenizer

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self._embed_model_path)
        logger.info(f"Loaded tokenizer: {self._embed_model_path}")
        return self._tokenizer

    def _split_text(self, text: str) -> List[str]:
        """将文本按 token 长度分割成 chunks"""
        tokenizer = self._load_tokenizer()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + self._chunk_token_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            chunks.append(chunk_text)
            if end == len(tokens):
                break
            start += self._chunk_token_size - self._chunk_overlap

        return chunks

    def _create_base_config(self, corpus_len: int) -> Any:
        """创建 HippoRAG BaseConfig"""
        from hipporag.utils.config_utils import BaseConfig

        top_k = self.config.top_k

        return BaseConfig(
            save_dir=self.working_dir,
            llm_base_url=self.config.llm_base_url,
            llm_name=self.config.llm_model,
            embedding_model_name=self._embed_model_path,
            force_index_from_scratch=self._force_index_from_scratch and not self._skip_build,
            force_openie_from_scratch=self._force_index_from_scratch and not self._skip_build,
            rerank_dspy_file_path=self._rerank_dspy_file_path,
            retrieval_top_k=top_k,
            linking_top_k=top_k,
            max_qa_steps=self._max_qa_steps,
            qa_top_k=top_k,
            graph_type=self._graph_type,
            embedding_batch_size=self._embedding_batch_size,
            max_new_tokens=None,
            corpus_len=corpus_len,
            openie_mode=self._openie_mode,
        )

    async def _init_hipporag(self, corpus_len: int = 0) -> None:
        """延迟初始化 HippoRAG 实例"""
        if self._hipporag is not None:
            return

        from hipporag.HippoRAG import HippoRAG

        self._base_config = self._create_base_config(corpus_len)

        mode = self.config.get_extra("mode", "API")
        if mode == "ollama":
            self._base_config.llm_mode = "ollama"
            logger.info(f"Using Ollama mode: {self.config.llm_model} at {self.config.llm_base_url}")
        else:
            self._base_config.llm_mode = "openai"
            logger.info(f"Using OpenAI mode: {self.config.llm_model} at {self.config.llm_base_url}")

        self._hipporag = HippoRAG(global_config=self._base_config)

    async def abuild_index(self, content: str, **kwargs) -> None:
        """构建索引"""
        content = self._ensure_content_string(content)

        self._chunks = self._split_text(content)
        logger.info(f"Split corpus into {len(self._chunks)} chunks")

        # HippoRAG 需要 ['idx:chunk_text', ...] 格式
        docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(self._chunks)]

        await self._init_hipporag(corpus_len=len(docs))

        if not self._skip_build:
            # HippoRAG 的 index 是同步方法，需要异步包装
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._hipporag.index, docs)
            logger.info(f"Indexed corpus: {len(self._chunks)} chunks")
        else:
            logger.info(f"Skipping indexing (assuming corpus is already indexed)")

    async def aquery(self, question: str, top_k: int = 5, **kwargs) -> BenchmarkQueryResult:
        """执行查询"""
        await self._init_hipporag()

        try:
            # HippoRAG 使用批量查询接口
            loop = asyncio.get_event_loop()
            queries_solutions, _, _, _, _ = await loop.run_in_executor(
                None,
                lambda: self._hipporag.rag_qa(
                    queries=[question],
                    gold_docs=None,
                    gold_answers=None,
                )
            )

            if not queries_solutions:
                return BenchmarkQueryResult(answer="No solution found", context=[])

            solution = queries_solutions[0].to_dict()

            answer = solution.get("answer", "")

            # 提取上下文
            context = []
            docs = solution.get("docs", [])
            if docs:
                context = self._ensure_context_list(docs)

            return BenchmarkQueryResult(answer=answer, context=context)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return BenchmarkQueryResult(answer=f"Query failed: {e}", context=[])

    def _ensure_context_list(self, context: Any) -> List[str]:
        """确保 context 是字符串列表"""
        if context is None:
            return []
        if isinstance(context, str):
            return [context] if context else []
        if isinstance(context, list):
            return [str(item) if not isinstance(item, str) else item for item in context]
        return [str(context)]

    async def abatch_query(self, questions: List[str], top_k: int = 5, **kwargs) -> List[BenchmarkQueryResult]:
        """批量查询（HippoRAG 原生支持，效率更高）"""
        await self._init_hipporag()

        try:
            gold_answers = [[q] for q in questions]  # 占位符

            loop = asyncio.get_event_loop()
            queries_solutions, _, _, _, _ = await loop.run_in_executor(
                None,
                lambda: self._hipporag.rag_qa(
                    queries=questions,
                    gold_docs=None,
                    gold_answers=gold_answers,
                )
            )

            solution_map = {sol.question: sol.to_dict() for sol in queries_solutions}

            results = []
            for q in questions:
                solution = solution_map.get(q)
                if solution:
                    answer = solution.get("answer", "")
                    docs = solution.get("docs", [])
                    context = self._ensure_context_list(docs)
                    results.append(BenchmarkQueryResult(answer=answer, context=context))
                else:
                    results.append(BenchmarkQueryResult(answer="No solution found", context=[]))

            return results

        except Exception as e:
            logger.error(f"Batch query failed: {e}")
            return [BenchmarkQueryResult(answer=f"Query failed: {e}", context=[]) for _ in questions]

    async def aclose(self) -> None:
        """清理资源"""
        self._hipporag = None
        self._chunks = []