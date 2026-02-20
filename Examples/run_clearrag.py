"""
ClearRAG 框架集成示例
使用 ClearRAG SDK 处理语料库并回答问题

功能：
1. 加载语料库数据
2. 使用 ClearRAG SDK 索引语料库
3. 查询问题并生成答案
4. 输出标准格式的预测结果（用于评估）
"""
import os
import sys
import asyncio
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

# 配置日志（在导入之前配置，避免后续代码使用未初始化的 logger）
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 添加项目根目录和 ClearRAG 源码路径到 Python 路径
project_root = Path(__file__).parent.parent
clearrag_dir = project_root / "ClearRAG"
clearrag_src_path = clearrag_dir / "src"
clearrag_venv_path = clearrag_dir / ".venv"

# 检测并使用 ClearRAG 的虚拟环境（如果存在）
if clearrag_venv_path.exists():
    # 获取虚拟环境中的 Python 可执行文件路径
    if sys.platform == "win32":
        python_exe = clearrag_venv_path / "Scripts" / "python.exe"
        site_packages = clearrag_venv_path / "Lib" / "site-packages"
    else:
        python_exe = clearrag_venv_path / "bin" / "python"
        site_packages = clearrag_venv_path / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    
    # 检查 Python 版本是否匹配
    if python_exe.exists():
        try:
            import subprocess
            result = subprocess.run(
                [str(python_exe), "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            venv_python_version = result.stdout.strip()
            current_python_version = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            if venv_python_version and venv_python_version != current_python_version:
                logger.warning(f"⚠️ Python 版本不匹配:")
                logger.warning(f"   当前 Python: {current_python_version}")
                logger.warning(f"   ClearRAG 虚拟环境 Python: {venv_python_version}")
                logger.warning(f"   这可能导致 C 扩展模块（如 pydantic_core）无法正常工作")
                logger.warning(f"   建议：使用 ClearRAG 虚拟环境中的 Python 运行脚本")
        except Exception as e:
            logger.debug(f"无法检查 ClearRAG 虚拟环境的 Python 版本: {e}")
    
    # 将虚拟环境的 site-packages 添加到路径
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))
        logger.info(f"✅ 检测到 ClearRAG 虚拟环境，使用: {clearrag_venv_path}")
        logger.info(f"   添加 site-packages: {site_packages}")
    else:
        logger.warning(f"⚠️ ClearRAG 虚拟环境存在但 site-packages 路径不存在: {site_packages}")

# 添加源码路径
if clearrag_src_path.exists():
    sys.path.insert(0, str(clearrag_src_path))
    os.environ['PYTHONPATH'] = str(clearrag_src_path)
    logger.debug(f"添加 ClearRAG 源码路径: {clearrag_src_path}")
else:
    # 如果源码路径不存在，尝试使用项目根目录（已安装包的情况）
    sys.path.insert(0, str(project_root))
    logger.warning(f"ClearRAG 源码路径不存在: {clearrag_src_path}，尝试使用已安装的包")

# 尝试导入 ClearRAG SDK
try:
    from clearrag import ClearRAG
    from clearrag.infrastructure.error_handling.exception_types import (
        ConfigurationError,
        ValidationError,
        LLMServiceError,
        DatabaseOperationError,
        DatabaseConnectionError,
    )
    logger.info(f"✅ 成功导入 ClearRAG SDK（从源码路径: {clearrag_src_path}）")
except ImportError as e:
    error_msg = str(e)
    logger.error(f"❌ 无法导入 ClearRAG SDK: {error_msg}")
    logger.error(f"已尝试的路径: {clearrag_src_path}")
    
    # 检查是否是依赖缺失问题
    if "sentence_spliter" in error_msg or "sentence-spliter" in error_msg:
        logger.error("\n缺少依赖包 'sentence-spliter'")
        logger.error("请安装 ClearRAG 的依赖（推荐使用框架内的虚拟环境）：")
        logger.error("  方法1（推荐）: cd ClearRAG && uv sync")
        logger.error("     - 这会在 ClearRAG/.venv 中安装所有依赖")
        logger.error("     - 然后使用 ClearRAG 虚拟环境中的 Python 运行脚本")
        logger.error("  方法2: 在项目根目录虚拟环境中安装")
        logger.error("     - pip install sentence-spliter>=2.1.8")
        logger.error("  方法3: 以开发模式安装 ClearRAG")
        logger.error("     - pip install -e ClearRAG/")
    elif "pydantic_core" in error_msg or "_pydantic_core" in error_msg:
        logger.error("\n⚠️ pydantic_core 模块错误（通常是 Python 版本不匹配导致的）")
        logger.error("原因：当前 Python 解释器与 ClearRAG 虚拟环境中的 Python 版本不匹配")
        logger.error("解决方案：")
        logger.error("  方法1（推荐）: 使用 ClearRAG 虚拟环境中的 Python 运行脚本")
        logger.error("    Windows: ClearRAG\\.venv\\Scripts\\python.exe Examples\\run_clearrag.py ...")
        logger.error("    Linux/Mac: ClearRAG/.venv/bin/python Examples/run_clearrag.py ...")
        logger.error("  方法2: 在项目根目录虚拟环境中重新安装 pydantic")
        logger.error("    pip install --force-reinstall pydantic pydantic-core")
        logger.error("  方法3: 确保两个虚拟环境使用相同的 Python 版本")
    elif "clearrag" in error_msg.lower():
        logger.error("请确保 ClearRAG 源码在 ClearRAG/src/ 目录下")
    else:
        logger.error("请确保已安装所有必需的依赖包")
        logger.error("安装方法（推荐使用框架内的虚拟环境）：")
        logger.error("  方法1（推荐）: cd ClearRAG && uv sync")
        logger.error("     - 这会在 ClearRAG/.venv 中安装所有依赖")
        logger.error("     - 然后使用 ClearRAG 虚拟环境中的 Python 运行脚本")
        logger.error("  方法2: 在项目根目录虚拟环境中安装")
        logger.error("     - pip install -e ClearRAG/")
    
    sys.exit(1)


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """按语料库名称分组问题"""
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
    config_path: Optional[str],
    init_kwargs: Dict,
    questions: Dict[str, List[dict]],
    sample: Optional[int],
    output_dir: str,
    top_k: int = 10,
    top_k_sentences: Optional[int] = None,
    top_k_passages: Optional[int] = None,
    enable_passage_retrieval: bool = True,
    skip_build: bool = False
):
    """
    处理单个语料库：索引并回答问题
    
    Args:
        corpus_name: 语料库名称
        context: 语料库文本内容（如果skip_build=True，此参数可忽略）
        config_path: 配置文件路径（可选）
        init_kwargs: 初始化关键字参数（可选）
        questions: 按语料库名称分组的问题字典
        sample: 采样的问题数量（可选）
        output_dir: 输出目录
        top_k: 查询时返回的上下文数量（默认值）
        top_k_sentences: 返回的句子数量（可选，如果未指定则使用top_k值）
        top_k_passages: 返回的段落数量（可选，如果未指定则使用top_k值）
        enable_passage_retrieval: 是否启用段落检索（默认启用）
        skip_build: 是否跳过构建阶段，直接进入查询阶段
    """
    logger.info(f"📚 处理语料库: {corpus_name}")
    
    # 初始化 ClearRAG 客户端
    try:
        if config_path and os.path.exists(config_path):
            logger.info(f"使用配置文件路径初始化: {config_path}")
            rag = await ClearRAG.create_async(config_path=config_path)
        elif init_kwargs:
            logger.info("使用关键字参数初始化")
            rag = await ClearRAG.create_async(**init_kwargs)
        else:
            logger.info("使用默认配置路径初始化")
            rag = await ClearRAG.create_async()
        
        logger.info("✅ ClearRAG 客户端初始化成功")
    except DatabaseConnectionError as e:
        logger.error(f"❌ Neo4j 连接失败: {e}")
        logger.error("\n请检查以下事项：")
        logger.error("  1. Neo4j 服务是否正在运行？")
        logger.error("     - Windows: 检查 Neo4j Desktop 是否启动")
        logger.error("     - 或运行: neo4j start")
        logger.error("  2. 连接配置是否正确？")
        logger.error("     - 检查 ClearRAG/.env 文件中的 NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")
        logger.error("     - 默认配置: bolt://localhost:7687, username=neo4j")
        logger.error("  3. 数据库是否存在？")
        logger.error("     - 检查配置中的 database 名称是否正确")
        logger.error("     - 如果数据库不存在，Neo4j 会自动创建")
        return
    except Exception as e:
        logger.error(f"❌ 初始化失败: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return
    
    try:
        # 1. 索引语料库（如果未跳过构建）
        if not skip_build:
            logger.info(f"🔍 正在索引语料库: {corpus_name} ({len(context.split())} 词, {len(context)} 字符)")
            await rag.abuild(
                content=context,
                source_type="text"
            )
            logger.info(f"✅ 索引完成: {corpus_name}")
        else:
            logger.info(f"⏭️  跳过构建阶段，直接进入查询阶段（假设语料库 {corpus_name} 已构建）")
        
        # 2. 获取该语料库的问题
        corpus_questions = questions.get(corpus_name, [])
        if not corpus_questions:
            logger.warning(f"⚠️ 未找到语料库 {corpus_name} 的问题")
            return
        
        # 3. 采样问题（如果指定）
        if sample and sample < len(corpus_questions):
            corpus_questions = corpus_questions[:sample]
            logger.info(f"🔍 采样了 {sample} 个问题（共 {len(questions.get(corpus_name, []))} 个）")
        
        logger.info(f"🔍 找到 {len(corpus_questions)} 个问题")
        
        # 4. 准备输出路径
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
        
        # 5. 处理问题
        results = []
        for q in tqdm(corpus_questions, desc=f"回答问题 - {corpus_name}"):
            try:
                # 查询框架 - 使用aquery_full获取完整结果
                query_result = await rag.aquery_full(
                    query=q["question"],
                    top_k_sentences=top_k_sentences if top_k_sentences is not None else top_k,
                    top_k_passages=top_k_passages if top_k_passages is not None else top_k,
                    enable_passage_retrieval=enable_passage_retrieval,
                    generate_answer=True
                )
                
                # 提取结果
                answer = query_result.natural_language_answer or ""
                context_list = query_result.extract_context_list(
                    top_k_sentences=top_k_sentences,
                    top_k_passages=top_k_passages
                )
                
                # 确保 context 是列表格式
                if isinstance(context_list, str):
                    context_list = [context_list] if context_list else []
                elif not isinstance(context_list, list):
                    context_list = []
                
                # 构建标准格式结果（context 包含所有检索结果：图数据、段落数据、实体数据、句子数据）
                result = {
                    "id": q["id"],
                    "question": q["question"],
                    "source": corpus_name,
                    "context": context_list,
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
                    "context": [],
                    "evidence": q.get("evidence", ""),
                    "question_type": q.get("question_type", ""),
                    "generated_answer": f"查询失败: {str(e)}",
                    "ground_truth": q.get("answer", "")
                })
        
        # 6. 保存结果
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 已保存 {len(results)} 个预测结果到: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ 处理语料库 {corpus_name} 时出错: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    finally:
        # 清理资源
        try:
            rag.close()
        except Exception as e:
            logger.warning(f"清理资源时出错: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ClearRAG 框架：处理语料库并回答问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用测试数据集（快速测试）
  python Examples/run_clearrag.py \\
    --subset medical_test \\
    --config_path ClearRAG/config/config.yaml
  
  # 使用完整数据集
  python Examples/run_clearrag.py \\
    --subset medical \\
    --config_path ClearRAG/config/config.yaml
  
  # 跳过构建，直接查询（假设已构建）
  python Examples/run_clearrag.py \\
    --subset medical_test \\
    --config_path ClearRAG/config/config.yaml \\
    --skip-build
  
  # 使用关键字参数
  python Examples/run_clearrag.py \\
    --subset medical \\
    --llm_model gpt-4o-mini \\
    --llm_api_key your_key \\
    --llm_base_url https://api.openai.com/v1 \\
    --embedding_model text-embedding-3-small \\
    --embedding_api_key your_key \\
    --embedding_base_url https://api.openai.com/v1 \\
    --embedding_dimensions 1536 \\
    --neo4j_uri bolt://localhost:7687 \\
    --neo4j_username neo4j \\
    --neo4j_password your_password \\
    --neo4j_database knowledge_graph
        """
    )
    
    # 核心参数
    parser.add_argument(
        "--subset",
        required=True,
        choices=["medical", "novel", "medical_100", "hotpotqa"],
        help="要处理的数据子集（medical、novel、medical_test 或 hotpotqa）"
    )
    
    # 配置文件路径
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="配置文件路径（如 config/config.yaml）"
    )
    
    # LLM 配置参数
    parser.add_argument(
        "--llm_model",
        type=str,
        default=None,
        help="LLM 模型名称（如 gpt-4o-mini）"
    )
    parser.add_argument(
        "--llm_api_key",
        type=str,
        default=None,
        help="LLM API 密钥（也可使用 LLM_API_KEY 环境变量）"
    )
    parser.add_argument(
        "--llm_base_url",
        type=str,
        default=None,
        help="LLM API 基础地址"
    )
    parser.add_argument(
        "--llm_temperature",
        type=float,
        default=None,
        help="LLM 温度参数"
    )
    parser.add_argument(
        "--llm_max_tokens",
        type=int,
        default=None,
        help="LLM 最大 token 数"
    )
    
    # Embedding 配置参数
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=None,
        help="Embedding 模型名称"
    )
    parser.add_argument(
        "--embedding_api_key",
        type=str,
        default=None,
        help="Embedding API 密钥"
    )
    parser.add_argument(
        "--embedding_base_url",
        type=str,
        default=None,
        help="Embedding API 基础地址"
    )
    parser.add_argument(
        "--embedding_dimensions",
        type=int,
        default=None,
        help="向量维度"
    )
    
    # Neo4j 配置参数
    parser.add_argument(
        "--neo4j_uri",
        type=str,
        default=None,
        help="Neo4j 连接 URI"
    )
    parser.add_argument(
        "--neo4j_username",
        type=str,
        default=None,
        help="Neo4j 用户名"
    )
    parser.add_argument(
        "--neo4j_password",
        type=str,
        default=None,
        help="Neo4j 密码"
    )
    parser.add_argument(
        "--neo4j_database",
        type=str,
        default=None,
        help="Neo4j 数据库名称"
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
        default=10,
        help="查询时返回的上下文数量（默认 10）"
    )
    
    parser.add_argument(
        "--top_k_sentences",
        type=int,
        default=None,
        help="返回的句子数量（可选，如果未指定则使用--top_k值）"
    )
    
    parser.add_argument(
        "--top_k_passages",
        type=int,
        default=None,
        help="返回的段落数量（可选，如果未指定则使用--top_k值）"
    )
    
    parser.add_argument(
        "--enable_passage_retrieval",
        action="store_true",
        default=True,
        help="是否启用段落检索（默认启用）"
    )
    
    parser.add_argument(
        "--disable_passage_retrieval",
        action="store_true",
        help="禁用段落检索（仅返回句子和图检索结果）"
    )
    
    # 输出配置
    parser.add_argument(
        "--output_dir",
        default="./results/clearrag",
        help="输出目录"
    )
    
    # 构建/查询控制
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="跳过构建阶段，直接进入查询阶段（假设语料库已构建）"
    )
    
    args = parser.parse_args()
    
    # 定义数据路径（支持 Parquet 和 JSON 格式）
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
    
    # 处理 API 密钥（支持环境变量）
    llm_api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    embedding_api_key = args.embedding_api_key or os.getenv("EMBEDDING_API_KEY", "") or llm_api_key
    
    # 构建关键字参数
    init_kwargs = {}
    if args.llm_model:
        init_kwargs["llm_model"] = args.llm_model
    if llm_api_key:
        init_kwargs["llm_api_key"] = llm_api_key
    if args.llm_base_url:
        init_kwargs["llm_base_url"] = args.llm_base_url
    if args.llm_temperature is not None:
        init_kwargs["llm_temperature"] = args.llm_temperature
    if args.llm_max_tokens is not None:
        init_kwargs["llm_max_tokens"] = args.llm_max_tokens
    if args.embedding_model:
        init_kwargs["embedding_model"] = args.embedding_model
    if embedding_api_key:
        init_kwargs["embedding_api_key"] = embedding_api_key
    if args.embedding_base_url:
        init_kwargs["embedding_base_url"] = args.embedding_base_url
    if args.embedding_dimensions is not None:
        init_kwargs["embedding_dimensions"] = args.embedding_dimensions
    if args.neo4j_uri:
        init_kwargs["neo4j_uri"] = args.neo4j_uri
    if args.neo4j_username:
        init_kwargs["neo4j_username"] = args.neo4j_username
    if args.neo4j_password:
        init_kwargs["neo4j_password"] = args.neo4j_password
    if args.neo4j_database:
        init_kwargs["neo4j_database"] = args.neo4j_database
    
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
    
    # 处理每个语料库
    async def process_all():
        tasks = []
        for item in corpus_data:
            tasks.append(
                process_corpus(
                    corpus_name=item["corpus_name"],
                    context=item["context"],
                    config_path=args.config_path,
                    init_kwargs=init_kwargs if init_kwargs else None,
                    questions=grouped_questions,
                    sample=args.sample,
                    output_dir=args.output_dir,
                    top_k=args.top_k,
                    top_k_sentences=args.top_k_sentences,
                    top_k_passages=args.top_k_passages,
                    enable_passage_retrieval=not args.disable_passage_retrieval,
                    skip_build=args.skip_build
                )
            )
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # 运行
    asyncio.run(process_all())
    logger.info("✅ 处理完成")


if __name__ == "__main__":
    main()
