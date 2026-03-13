"""
LinearRAG GraphRAG-Benchmark 集成脚本

将 LinearRAG 适配到 Benchmark 标准格式，支持统一的评估流程。
"""
import argparse
import asyncio
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# 添加 linearrag 路径
project_root = Path(__file__).parent.parent
linearrag_dir = project_root / "linearrag"
sys.path.insert(0, str(linearrag_dir))

from tqdm import tqdm
from common_benchmark import (
    build_output_path,
    build_error_result,
    ensure_context_list,
    group_questions_by_source,
    load_corpus_records,
    load_question_records,
    merge_corpus_by_name,
    save_results_json,
)
from subset_registry import get_subset_paths, get_supported_subsets

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def normalize_context_list(raw_context: List[str]) -> List[str]:
    """移除 LinearRAG 的 passage 编号前缀，并在输出前保序去重。"""
    normalized_context = []
    seen = set()

    for ctx in raw_context:
        cleaned = ctx
        if ":" in ctx and ctx.split(":")[0].isdigit():
            cleaned = ":".join(ctx.split(":")[1:]).strip()

        if cleaned not in seen:
            seen.add(cleaned)
            normalized_context.append(cleaned)

    return normalized_context


class LinearRAGBenchmarkAdapter:
    """LinearRAG 适配器，统一接口到 Benchmark 格式"""
    
    def __init__(
        self,
        working_dir: str,
        llm_model: str,
        llm_base_url: str,
        llm_api_key: str,
        embedding_model,
        retrieval_top_k: int = 5,
        max_iterations: int = 3,
        iteration_threshold: float = 0.4,
        top_k_sentence: int = 3,
        use_vectorized: bool = False,
        max_workers: int = 8,
        spacy_model: str = "en_core_web_trf",
        passage_ratio: float = 1.5,
        damping: float = 0.5,
    ):
        self.working_dir = working_dir
        self.retrieval_top_k = retrieval_top_k
        self.embedding_model = embedding_model
        
        # 设置 LLM 环境变量（LinearRAG 通过环境变量读取）
        os.environ["OPENAI_API_KEY"] = llm_api_key
        os.environ["OPENAI_BASE_URL"] = llm_base_url
        
        from src.config import LinearRAGConfig
        from src.LinearRAG import LinearRAG

        # 创建 LLM 模型包装
        from src.utils import LLM_Model
        self.llm_model = LLM_Model(llm_model)
        self.linear_rag_cls = LinearRAG
        
        # 初始化配置
        self.config = LinearRAGConfig(
            dataset_name="",  # 稍后设置
            embedding_model=embedding_model,
            llm_model=self.llm_model,
            working_dir=working_dir,
            retrieval_top_k=retrieval_top_k,
            max_iterations=max_iterations,
            iteration_threshold=iteration_threshold,
            top_k_sentence=top_k_sentence,
            use_vectorized_retrieval=use_vectorized,
            max_workers=max_workers,
            spacy_model=spacy_model,
            passage_ratio=passage_ratio,
            damping=damping,
        )
        
        self.rag: Optional[object] = None
        
    def build_index(self, corpus_name: str, passages: List[str]) -> None:
        """构建索引"""
        self.config.dataset_name = corpus_name
        self.rag = self.linear_rag_cls(global_config=self.config)
        self.rag.index(passages)
        logger.info(f"✅ 索引构建完成: {corpus_name}")

    def load_index(self, corpus_name: str) -> None:
        """加载已有索引"""
        self.config.dataset_name = corpus_name
        self.rag = self.linear_rag_cls(global_config=self.config)
        self.rag.load_index()
        logger.info(f"✅ 索引加载完成: {corpus_name}")
    
    def query(
        self,
        questions: List[dict],
        corpus_name: str
    ) -> List[dict]:
        """
        批量查询并返回标准格式结果
        
        Args:
            questions: 问题列表，每个包含 id, question, answer, evidence, question_type
            corpus_name: 语料库名称
            
        Returns:
            标准 Benchmark 格式结果列表
        """
        if self.rag is None:
            raise RuntimeError("请先调用 build_index() 构建索引")
        
        # 转换为 LinearRAG 格式
        linear_questions = [
            {
                "question": q["question"],
                "answer": q.get("answer", "")
            }
            for q in questions
        ]
        
        # 执行检索和问答
        results = self.rag.qa(linear_questions)
        
        # 转换为标准格式
        benchmark_results = []
        for result, orig_q in zip(results, questions):
            context = ensure_context_list(result.get("sorted_passage", []))
            cleaned_context = normalize_context_list(context)
            
            benchmark_results.append({
                "id": orig_q.get("id", ""),
                "question": orig_q["question"],
                "source": corpus_name,
                "context": cleaned_context,
                "evidence": orig_q.get("evidence", ""),
                "question_type": orig_q.get("question_type", ""),
                "generated_answer": result.get("pred_answer", ""),
                "ground_truth": orig_q.get("answer", "")
            })
        
        return benchmark_results


def create_embedding_model(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """创建嵌入模型（统一使用 OpenAI 兼容接口）"""
    from src.openai_embedding import create_openai_embedding_model

    # 智谱 batch_size 限制
    max_batch = 64 if provider == "zhipu" or (base_url and "bigmodel" in base_url) else None

    # embedding-3 维度配置
    dimensions = 1024 if "embedding-3" in model_name else None

    if provider == "local":
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name, device="cuda" if _has_cuda() else "cpu")

    return create_openai_embedding_model(
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        batch_size=64,
        dimensions=dimensions,
        max_batch_size=max_batch
    )


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    questions: Dict[str, List[dict]],
    sample: Optional[int],
    output_dir: str,
    llm_model: str,
    llm_base_url: str,
    llm_api_key: str,
    embed_provider: str,
    embed_model_name: str,
    embed_api_key: Optional[str],
    embed_base_url: Optional[str],
    top_k: int,
    skip_build: bool = False,
    index_only: bool = False,
    max_iterations: int = 3,
    iteration_threshold: float = 0.4,
    top_k_sentence: int = 3,
    use_vectorized: bool = False,
    max_workers: int = 8,
    spacy_model: str = "en_core_web_trf",
):
    """处理单个语料库"""
    logger.info(f"📚 处理语料库: {corpus_name}")
    
    # 准备输出路径
    output_path = build_output_path(output_dir, f"predictions_{corpus_name}.json")

    # 获取问题
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logger.warning(f"⚠️ 未找到语料库 {corpus_name} 的问题")
        return
    
    # 采样
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    # 准备 passages（按句子或段落分割）
    # LinearRAG 期望格式为 ["0:段落1", "1:段落2", ...]
    # 这里按换行分割或使用整段
    passages = []
    if "\n" in context and len(context.split("\n")) > 1:
        chunks = [chunk.strip() for chunk in context.split("\n") if chunk.strip()]
        passages = [f"{i}:{chunk}" for i, chunk in enumerate(chunks)]
    else:
        passages = [f"0:{context}"]
    
    try:
        # 创建嵌入模型
        embedding_model = create_embedding_model(
            provider=embed_provider,
            model_name=embed_model_name,
            api_key=embed_api_key,
            base_url=embed_base_url,
        )
        
        # 初始化适配器
        adapter = LinearRAGBenchmarkAdapter(
            working_dir=os.path.join(base_dir, corpus_name),
            llm_model=llm_model,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embedding_model=embedding_model,
            retrieval_top_k=top_k,
            max_iterations=max_iterations,
            iteration_threshold=iteration_threshold,
            top_k_sentence=top_k_sentence,
            use_vectorized=use_vectorized,
            max_workers=max_workers,
            spacy_model=spacy_model,
        )
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        error_results = [build_error_result(q, corpus_name, e, message_prefix="Init failed") for q in corpus_questions]
        save_results_json(output_path, error_results)
        logger.info(f"💾 已保存 {len(error_results)} 个失败结果到: {output_path}")
        return
    
    # 构建/加载索引
    try:
        if not skip_build:
            adapter.build_index(corpus_name, passages)
        else:
            logger.info(f"⏭️ 跳过索引构建")
            adapter.load_index(corpus_name)
    except Exception as e:
        logger.error(f"❌ 索引阶段失败: {e}")
        error_results = [build_error_result(q, corpus_name, e, message_prefix="Index failed") for q in corpus_questions]
        save_results_json(output_path, error_results)
        logger.info(f"💾 已保存 {len(error_results)} 个失败结果到: {output_path}")
        return
    
    logger.info(f"🔍 找到 {len(corpus_questions)} 个问题")

    if index_only:
        logger.info(f"⏭️ 仅构建索引模式，跳过查询: {corpus_name}")
        return
    
    # 批量查询
    try:
        results = adapter.query(corpus_questions, corpus_name)
        save_results_json(output_path, results)
        logger.info(f"💾 已保存 {len(results)} 个预测结果到: {output_path}")
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        error_results = [build_error_result(q, corpus_name, e) for q in corpus_questions]
        save_results_json(output_path, error_results)
        logger.info(f"💾 已保存 {len(error_results)} 个失败结果到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LinearRAG: 处理语料库并回答问题"
    )
    
    # 核心参数
    parser.add_argument(
        "--subset",
        required=True,
        choices=get_supported_subsets("linearrag"),
        help="要处理的数据子集"
    )
    
    # 工作目录
    parser.add_argument(
        "--base_dir",
        default="./linearrag_workspace",
        help="工作目录"
    )
    parser.add_argument(
        "--output_dir",
        default="./results/linearrag",
        help="输出目录"
    )
    
    # LLM 配置
    parser.add_argument(
        "--model_name",
        default="deepseek-chat",
        help="LLM 模型名称"
    )
    parser.add_argument(
        "--llm_base_url",
        default="https://api.deepseek.com/v1",
        help="LLM API 地址"
    )
    parser.add_argument(
        "--llm_api_key",
        default="",
        help="LLM API 密钥（优先级：参数 > LLM_API_KEY 环境变量）"
    )
    
    # 嵌入配置
    parser.add_argument(
        "--embed_provider",
        choices=["zhipu", "api", "local"],
        default="zhipu",
        help="嵌入模型提供商"
    )
    parser.add_argument(
        "--embed_model",
        default="embedding-3",
        help="嵌入模型名称/路径"
    )
    parser.add_argument(
        "--embed_api_key",
        default="",
        help="嵌入 API 密钥"
    )
    parser.add_argument(
        "--embed_base_url",
        default="",
        help="嵌入 API 地址（用于 OpenAI 兼容 API）"
    )
    
    # 检索配置
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="检索 top-k"
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=3,
        help="最大迭代次数"
    )
    parser.add_argument(
        "--iteration_threshold",
        type=float,
        default=0.4,
        help="迭代阈值"
    )
    parser.add_argument(
        "--top_k_sentence",
        type=int,
        default=3,
        help="每个实体的 top-k 句子数"
    )
    parser.add_argument(
        "--use_vectorized",
        action="store_true",
        help="使用向量化检索（更快）"
    )
    
    # NER 配置
    parser.add_argument(
        "--spacy_model",
        default="en_core_web_trf",
        help="SpaCy NER 模型"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="并发工作线程数"
    )
    
    # 其他
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="采样问题数量"
    )
    parser.add_argument(
        "--corpus_sample",
        type=int,
        default=None,
        help="采样语料库数量"
    )
    parser.add_argument(
        "--corpus-concurrency",
        type=int,
        default=1,
        help="语料库并发处理数（默认 1，推荐 1 保证稳定）"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过索引构建"
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="仅构建/加载索引，跳过查询"
    )
    
    args = parser.parse_args()
    
    # 获取数据路径
    try:
        corpus_path, questions_path = get_subset_paths(args.subset)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # 处理 API Key
    llm_api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    if not llm_api_key:
        logger.error("❌ 缺少 LLM API Key")
        return
    
    embed_api_key = args.embed_api_key or os.getenv("ZHIPUAI_API_KEY", "") or os.getenv("EMBED_API_KEY", "")
    if args.embed_provider in ["zhipu", "api"] and not embed_api_key:
        logger.warning("⚠️ 使用远程嵌入 API 但未提供 API Key")
    
    # 加载数据
    corpus_data = load_corpus_records(corpus_path)
    logger.info(f"📖 已加载 {len(corpus_data)} 个语料库文档")
    
    # Merge corpus by name (critical for multi-hop QA datasets)
    original_count = len(corpus_data)
    corpus_data = merge_corpus_by_name(corpus_data)
    logger.info(f"📖 合并 {original_count} 个文档为 {len(corpus_data)} 个语料库")
    
    question_data = load_question_records(questions_path)
    grouped_questions = group_questions_by_source(question_data)
    logger.info(f"❓ 已加载 {len(question_data)} 个问题")
    
    async def run_all() -> None:
        concurrency = max(1, args.corpus_concurrency)
        logger.info(f"📦 语料库并发数: {concurrency}")
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_limit(item: dict):
            async with semaphore:
                await asyncio.to_thread(
                    process_corpus,
                    corpus_name=item["corpus_name"],
                    context=item["context"],
                    base_dir=args.base_dir,
                    questions=grouped_questions,
                    sample=args.sample,
                    output_dir=args.output_dir,
                    llm_model=args.model_name,
                    llm_base_url=args.llm_base_url,
                    llm_api_key=llm_api_key,
                    embed_provider=args.embed_provider,
                    embed_model_name=args.embed_model,
                    embed_api_key=embed_api_key,
                    embed_base_url=args.embed_base_url,
                    top_k=args.top_k,
                    skip_build=args.skip_build,
                    index_only=args.index_only,
                    max_iterations=args.max_iterations,
                    iteration_threshold=args.iteration_threshold,
                    top_k_sentence=args.top_k_sentence,
                    use_vectorized=args.use_vectorized,
                    max_workers=args.max_workers,
                    spacy_model=args.spacy_model,
                )

        tasks = [process_with_limit(item) for item in corpus_data]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"❌ 语料库处理异常: {result}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
