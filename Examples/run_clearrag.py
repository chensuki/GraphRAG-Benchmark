"""
ClearRAG GraphRAG-Benchmark 集成脚本（使用适配器模式）

使用 ClearRAG Benchmark 适配器处理语料库并回答问题，输出符合 GraphRAG-Benchmark 标准格式。

功能：
1. 加载语料库数据（支持 Parquet 和 JSON 格式）
2. 使用 ClearRAGBenchmarkAdapter 索引语料库
3. 查询问题并生成答案
4. 输出标准格式的预测结果（用于评估）

依赖：
    - ClearRAG 框架（在 clearrag/ 目录下）
    - clearrag/Examples/clearrag_benchmark_adapter.py 适配器
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
clearrag_dir = project_root / "clearrag"

# 确保 clearrag 在路径中
if str(clearrag_dir) not in os.environ.get("PYTHONPATH", ""):
    os.environ["PYTHONPATH"] = str(clearrag_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")

# 确保适配器路径在 sys.path
import sys
sys.path.insert(0, str(clearrag_dir / "Examples"))

from datasets import load_dataset
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 导入 ClearRAG Benchmark 适配器
try:
    from clearrag_benchmark_adapter import ClearRAGBenchmarkAdapter
    logger.info("✅ 成功导入 ClearRAGBenchmarkAdapter")
except ImportError as e:
    logger.error(f"❌ 无法导入 ClearRAGBenchmarkAdapter: {e}")
    logger.error("请确保 clearrag/Examples/clearrag_benchmark_adapter.py 存在")
    sys.exit(1)


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """按语料库名称分组问题

    Args:
        question_list: 问题列表，每个问题包含 source 字段

    Returns:
        按语料库名称分组的问题字典
    """
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    questions: Dict[str, List[dict]],
    sample: Optional[int],
    output_dir: str,
    top_k: int = 10,
    skip_build: bool = False,
    activation_mode: str = "semantic_propagation",
    passage_retrieval_mode: str = "pagerank",
    fast_mode: bool = False,
    max_concurrency: int = 5,
):
    """
    处理单个语料库：索引并回答问题

    Args:
        corpus_name: 语料库名称
        context: 语料库文本内容（必须是字符串）
        base_dir: 工作目录
        questions: 按语料库名称分组的问题字典
        sample: 采样的问题数量
        output_dir: 输出目录
        top_k: 查询时返回的上下文数量
        skip_build: 是否跳过构建阶段
        activation_mode: 实体激活模式（"semantic_propagation" 或 "vector_search"）
        passage_retrieval_mode: 段落检索模式（"pagerank" 或 "simple"）
        fast_mode: 快速模式（禁用Reflexion）
        max_concurrency: 最大并发度
    """
    logger.info(f"📚 处理语料库: {corpus_name}")

    # 1. 准备输出路径
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    # 2. 初始化 ClearRAG 适配器
    try:
        rag = ClearRAGBenchmarkAdapter(
            working_dir=os.path.join(base_dir, corpus_name),
            llm_model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
            embedding_model_name=os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            embed_base_url=os.getenv("EMBED_BASE_URL"),
            embed_api_key=os.getenv("EMBED_API_KEY"),
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE"),
            activation_mode=activation_mode,
            passage_retrieval_mode=passage_retrieval_mode,
            fast_mode=fast_mode,
            max_concurrency=max_concurrency,
            enable_checkpoint=True,  # 启用断点续传
        )
        logger.info("✅ ClearRAG适配器初始化成功")
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return

    try:
        # 3. 构建索引（content 必须是字符串）
        if not skip_build:
            logger.info(f"🔍 正在索引语料库: {corpus_name} ({len(context.split())} 词, {len(context)} 字符)")

            # 适配器的 abuild_index 接受字符串参数
            await rag.abuild_index(content=context, source_type="text")

            logger.info(f"✅ 索引完成: {corpus_name}")
        else:
            logger.info(f"⏭️  跳过构建阶段（假设语料库 {corpus_name} 已构建）")

        # 4. 获取该语料库的问题
        corpus_questions = questions.get(corpus_name, [])
        if not corpus_questions:
            logger.warning(f"⚠️ 未找到语料库 {corpus_name} 的问题")
            return

        # 5. 采样问题（如果指定）
        if sample and sample < len(corpus_questions):
            corpus_questions = corpus_questions[:sample]
            logger.info(f"🔍 采样了 {sample} 个问题（共 {len(questions.get(corpus_name, []))} 个）")

        logger.info(f"🔍 找到 {len(corpus_questions)} 个问题")

        # 6. 处理问题
        results = []
        for q in tqdm(corpus_questions, desc=f"回答问题 - {corpus_name}"):
            try:
                # 使用适配器的 aquery 方法（返回 BenchmarkQueryResult）
                query_result = await rag.aquery(
                    question=q["question"],
                    top_k=top_k,
                )

                # BenchmarkQueryResult 自动处理 context 类型
                answer = query_result.answer
                context_list = query_result.get_context_list()  # 确保返回字符串列表

                # 构建标准格式结果
                result = {
                    "id": q["id"],
                    "question": q["question"],
                    "source": corpus_name,
                    "context": context_list,  # BenchmarkQueryResult 确保是列表
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": answer,
                    "ground_truth": q.get("answer", "")
                }
                results.append(result)

            except Exception as e:
                logger.error(f"❌ 处理问题 {q.get('id')} 失败: {e}")
                results.append({
                    "id": q["id"],
                    "question": q["question"],
                    "source": corpus_name,
                    "context": [],  # 错误时返回空列表
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": f"查询失败: {str(e)}",
                    "ground_truth": q.get("answer", "")
                })

        # 7. 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 已保存 {len(results)} 个预测结果到: {output_path}")

    except Exception as e:
        logger.error(f"❌ 处理语料库 {corpus_name} 时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        # 8. 清理资源
        try:
            await rag.aclose()
            logger.info(f"✅ 资源已清理: {corpus_name}")
        except Exception as e:
            logger.warning(f"清理资源时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ClearRAG: 使用适配器处理语料库并回答问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用 medical_100 数据集快速测试
  python Examples/run_clearrag.py --subset medical_100

  # 使用完整 medical 数据集
  python Examples/run_clearrag.py --subset medical

  # 使用向量搜索激活模式（快速，无需矩阵文件）
  python Examples/run_clearrag.py --subset medical --activation-mode vector_search

  # 跳过构建，直接查询（假设已构建）
  python Examples/run_clearrag.py --subset medical --skip-build

  # 使用采样进行测试
  python Examples/run_clearrag.py --subset medical_100 --sample 5

  # 使用快速模式（禁用Reflexion）
  python Examples/run_clearrag.py --subset medical --fast

环境变量:
  LLM_MODEL: LLM 模型名称（默认: deepseek-chat）
  LLM_BASE_URL: LLM API 地址
  LLM_API_KEY: LLM API 密钥
  EMBED_MODEL: Embedding 模型名称（默认: BAAI/bge-large-en-v1.5）
  EMBED_BASE_URL: Embedding API 地址
  EMBED_API_KEY: Embedding API 密钥
  NEO4J_URI: Neo4j 连接地址
  NEO4J_USER: Neo4j 用户名
  NEO4J_PASSWORD: Neo4j 密码
  NEO4J_DATABASE: Neo4j 数据库名（默认: neo4j）
        """
    )

    # 核心参数
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel", "medical_100", "hotpotqa"],
        help="要处理的数据子集"
    )

    # 工作目录
    parser.add_argument(
        "--base_dir",
        default="./clearrag_workspace",
        help="工作目录（默认: ./clearrag_workspace）"
    )

    # 采样和调试
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="每个语料库采样的问题数量（用于测试）"
    )

    # 查询配置
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="查询时返回的上下文数量（默认: 5）"
    )

    # 激活模式
    parser.add_argument(
        "--activation-mode",
        type=str,
        default="semantic_propagation",
        choices=["vector_search", "semantic_propagation"],
        help="实体激活模式：vector_search（快速）或 semantic_propagation（高精度，需要矩阵文件）"
    )

    # 段落检索模式
    parser.add_argument(
        "--passage-retrieval-mode",
        type=str,
        default="pagerank",
        choices=["pagerank", "simple"],
        help="段落检索模式：pagerank（高精度）或 simple（快速）"
    )

    # 快速模式
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速模式（禁用Reflexion）"
    )

    # 并发度
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="最大并发度（1-10，默认5）"
    )

    # 语料库处理并发度（多语料库场景）
    parser.add_argument(
        "--corpus-concurrency",
        type=int,
        default=2,
        help="同时处理的语料库数量（1-3，推荐2，避免API过载）"
    )

    # 输出配置
    parser.add_argument(
        "--output_dir",
        default="./results/clearrag",
        help="输出目录（默认: ./results/clearrag）"
    )

    # 构建/查询控制
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过构建阶段，直接进入查询阶段（假设语料库已构建）"
    )

    args = parser.parse_args()

    # 定义数据路径
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet"
        },
        "medical_100": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions_100_balanced.json"
        },
        "hotpotqa": {
            "corpus": "./Datasets/Corpus/hotpotqa.parquet",
            "questions": "./Datasets/Questions/hotpotqa_questions.json"
        }
    }

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    # 验证环境变量
    required_env_vars = ["LLM_API_KEY", "EMBED_API_KEY", "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ 缺少必需的环境变量: {', '.join(missing_vars)}")
        logger.error("请设置以下环境变量:")
        for var in required_env_vars:
            logger.error(f"  export {var}=your_value")
        return

    # 加载语料库数据（支持 Parquet 和 JSON 格式）
    try:
        if corpus_path.endswith('.json'):
            # 加载 JSON 格式
            with open(corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                # 单个语料项（字典格式）
                corpus_data = [{
                    "corpus_name": data.get("corpus_name", "Unknown"),
                    "context": data.get("context", "")
                }]
            elif isinstance(data, list):
                # 多个语料项（列表格式）
                corpus_data = [{
                    "corpus_name": item.get("corpus_name", "Unknown"),
                    "context": item.get("context", "")
                } for item in data]
            else:
                raise ValueError(f"不支持的 JSON 格式: {type(data)}")
        else:
            # 加载 Parquet 格式
            corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
            corpus_data = []
            for item in corpus_dataset:
                corpus_data.append({
                    "corpus_name": item["corpus_name"],
                    "context": item["context"]
                })
        logger.info(f"📖 已加载 {len(corpus_data)} 个语料库文档")
    except Exception as e:
        logger.error(f"❌ 加载语料库失败: {e}")
        return

    # 采样语料库（如果指定）
    if args.sample:
        corpus_data = corpus_data[:1]
        logger.info(f"🔍 采样了 1 个语料库（用于测试）")

    # 加载问题数据（支持 Parquet 和 JSON 格式）
    try:
        if questions_path.endswith('.json'):
            # 加载 JSON 格式
            with open(questions_path, 'r', encoding='utf-8') as f:
                question_data = json.load(f)
            if not isinstance(question_data, list):
                raise ValueError(f"问题数据必须是列表格式，当前格式: {type(question_data)}")
            # 确保所有必需字段存在
            question_data = [{
                "id": item.get("id", ""),
                "source": item.get("source", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "question_type": item.get("question_type", ""),
                "evidence": item.get("evidence", "")
            } for item in question_data]
        else:
            # 加载 Parquet 格式
            questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
            question_data = []
            for item in questions_dataset:
                question_data.append({
                    "id": item["id"],
                    "source": item["source"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "question_type": item["question_type"],
                    "evidence": item["evidence"]
                })
        grouped_questions = group_questions_by_source(question_data)
        logger.info(f"❓ 已加载 {len(question_data)} 个问题")
    except Exception as e:
        logger.error(f"❌ 加载问题失败: {e}")
        return

    # 处理每个语料库（带并发限制）
    async def process_all():
        # 使用信号量限制并发数，避免 API 过载
        # 特别注意：Novel 数据集有 20 个语料，必须限制并发
        max_concurrent = args.corpus_concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_limit(item):
            async with semaphore:
                await process_corpus(
                    corpus_name=item["corpus_name"],
                    context=item["context"],
                    base_dir=args.base_dir,
                    questions=grouped_questions,
                    sample=args.sample,
                    output_dir=args.output_dir,
                    top_k=args.top_k,
                    skip_build=args.skip_build,
                    activation_mode=args.activation_mode,
                    passage_retrieval_mode=args.passage_retrieval_mode,
                    fast_mode=args.fast,
                    max_concurrency=args.max_concurrency,
                )

        tasks = [process_with_limit(item) for item in corpus_data]
        await asyncio.gather(*tasks, return_exceptions=True)

    # 运行
    try:
        asyncio.run(process_all())
        logger.info("✅ 处理完成")
    except KeyboardInterrupt:
        logger.info("\n[INFO] 程序已被用户中断")
        sys.exit(130)


if __name__ == "__main__":
    main()
