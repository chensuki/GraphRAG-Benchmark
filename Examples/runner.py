"""
统一框架运行器

提供统一的基准测试运行流程。
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    from .adapters import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig, create_adapter
except ImportError:
    from adapters import BaseFrameworkAdapter, BenchmarkQueryResult, FrameworkConfig, create_adapter

try:
    from .common_benchmark import (
        build_error_result,
        load_corpus_records,
        load_question_records,
        merge_corpus_by_name,
        save_results_json,
        group_questions_by_source,
    )
except ImportError:
    from common_benchmark import (
        build_error_result,
        load_corpus_records,
        load_question_records,
        merge_corpus_by_name,
        save_results_json,
        group_questions_by_source,
    )

logger = logging.getLogger(__name__)


class FrameworkRunner:
    """
    统一的框架运行器

    路径结构：
    {workspace_dir}/{corpus_name}/  - 索引文件
    {predictions_dir}/predictions_{corpus_name}.json  - 预测结果
    """

    def __init__(
        self,
        framework: str,
        config: FrameworkConfig,
        subset: str,
        corpus_path: str,
        questions_path: str,
        workspace_dir: str,
        predictions_dir: str,
        sample: Optional[int] = None,
        corpus_sample: Optional[int] = None,
        corpus_concurrency: int = 1,
        skip_build: bool = False,
        index_only: bool = False,
    ):
        self.framework = framework
        self.config = config
        self.subset = subset
        self.workspace_dir = Path(workspace_dir)
        self.predictions_dir = Path(predictions_dir)
        self.sample = sample
        self.corpus_sample = corpus_sample
        self.corpus_concurrency = max(1, corpus_concurrency)
        self.skip_build = skip_build
        self.index_only = index_only

        # 加载数据
        logger.info(f"Loading corpus: {corpus_path}")
        raw_corpus = load_corpus_records(corpus_path)
        logger.info(f"Loaded {len(raw_corpus)} records")

        self.corpus_data = merge_corpus_by_name(raw_corpus)
        logger.info(f"Merged into {len(self.corpus_data)} corpora")

        if corpus_sample and corpus_sample < len(self.corpus_data):
            self.corpus_data = self.corpus_data[:corpus_sample]
            logger.info(f"Sampled {corpus_sample} corpora")

        logger.info(f"Loading questions: {questions_path}")
        raw_questions = load_question_records(questions_path)
        logger.info(f"Loaded {len(raw_questions)} questions")

        self.questions = group_questions_by_source(raw_questions)

        # 确保目录存在
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> Dict[str, any]:
        """运行基准测试"""
        logger.info(f"Starting: {self.framework}, corpora={len(self.corpus_data)}, concurrency={self.corpus_concurrency}")

        semaphore = asyncio.Semaphore(self.corpus_concurrency)
        results = {}

        async def process_with_limit(corpus_item: dict) -> tuple:
            async with semaphore:
                return await self._process_corpus(corpus_item)

        tasks = [process_with_limit(item) for item in self.corpus_data]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(task_results):
            corpus_name = self.corpus_data[i]["corpus_name"]
            if isinstance(result, Exception):
                error_msg = f"{type(result).__name__}: {result}" if str(result) else type(result).__name__
                logger.error(f"Failed: {corpus_name} - {error_msg}")
                results[corpus_name] = {"status": "error", "error": error_msg}
            else:
                results[corpus_name] = result

        success = sum(1 for r in results.values() if r.get("status") == "success")
        error = sum(1 for r in results.values() if r.get("status") == "error")

        logger.info(f"Completed: {success} success, {error} error")

        return {
            "framework": self.framework,
            "subset": self.subset,
            "total": len(self.corpus_data),
            "success": success,
            "error": error,
            "output": str(self.predictions_dir),
        }

    async def _process_corpus(self, corpus_item: dict) -> Dict[str, any]:
        """处理单个语料库"""
        corpus_name = corpus_item["corpus_name"]
        context = corpus_item["context"]

        logger.info(f"Processing: {corpus_name}")

        # 语料库工作目录
        working_dir = self.workspace_dir / corpus_name
        working_dir.mkdir(parents=True, exist_ok=True)

        # 输出文件
        output_path = self.predictions_dir / f"predictions_{corpus_name}.json"

        # 获取问题
        corpus_questions = self.questions.get(corpus_name, [])
        if not corpus_questions:
            logger.warning(f"No questions: {corpus_name}")
            return {"status": "skipped", "reason": "no_questions"}

        if self.sample and self.sample < len(corpus_questions):
            corpus_questions = corpus_questions[:self.sample]

        adapter: Optional[BaseFrameworkAdapter] = None

        try:
            adapter = create_adapter(self.framework, str(working_dir), self.config)

            # 构建索引
            if not self.skip_build:
                logger.info(f"Indexing: {corpus_name} ({len(context)} chars)")
                await adapter.abuild_index(context)
                logger.info(f"Indexed: {corpus_name}")
            else:
                try:
                    adapter.load_index(corpus_name)
                except NotImplementedError:
                    pass

            if self.index_only:
                return {"status": "success", "indexed": True}

            logger.info(f"Querying: {len(corpus_questions)} questions")
            query_results = await adapter.abatch_query(
                corpus_questions,
                top_k=self.config.top_k
            )

            results = []
            for q, result in zip(corpus_questions, query_results):
                # 保留所有原始字段，添加生成字段
                result_dict = dict(q)
                result_dict.update({
                    "source": corpus_name,
                    "context": result.context,
                    "generated_answer": result.answer,
                    "ground_truth": q.get("answer", "")
                })
                results.append(result_dict)

            save_results_json(output_path, results)
            logger.info(f"Saved: {output_path} ({len(results)} results)")

            return {"status": "success", "queries": len(results), "output": str(output_path)}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.error(f"Failed: {corpus_name} - {error_msg}")
            error_results = [build_error_result(q, corpus_name, e, message_prefix="Failed") for q in corpus_questions]
            save_results_json(output_path, error_results)
            return {"status": "error", "error": error_msg}

        finally:
            if adapter:
                try:
                    await adapter.aclose()
                except Exception as e:
                    logger.warning(f"Close failed: {e}")