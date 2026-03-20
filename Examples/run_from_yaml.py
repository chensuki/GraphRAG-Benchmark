"""
统一 YAML 配置运行器

配置格式：
```yaml
dataset:
  subset: medical

llm:
  model: deepseek-chat

embedding:
  type: api                    # api | local
  provider: zhipu              # zhipu | openai | custom
  model: embedding-3

output:
  root: ./results

frameworks:
  lightrag:
    enabled: true
    mode: API
    corpus_concurrency: 1

notification:
  enabled: true
  gateway_url: http://127.0.0.1:18789
  token: ${OPENCLAW_TOKEN}
  # sessionKey 格式: agent:{agentId}:{platform}:{type}:{id}
  session_key: agent:main:feishu:group:oc_xxx
```
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from subset_registry import FRAMEWORK_SUPPORTED_SUBSETS, get_subset_paths
from runner import FrameworkRunner, ProgressCallback
from adapters import FrameworkConfig, has_adapter
from adapters.base import DEFAULT_EMBED_TYPE, DEFAULT_EMBED_PROVIDER, DEFAULT_TOP_K, DEFAULT_MAX_CONCURRENCY

# 尝试导入通知客户端
try:
    from openclaw_notifier import OpenClawWebhookClient, ExperimentTracker
    NOTIFIER_AVAILABLE = True
except ImportError:
    NOTIFIER_AVAILABLE = False

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SUPPORTED_FRAMEWORKS = list(FRAMEWORK_SUPPORTED_SUBSETS.keys())


# ============ 通知辅助函数 ============

def safe_notify(notify_func, *args, **kwargs):
    """
    安全发送通知，失败不阻塞主流程

    Args:
        notify_func: 通知方法
        *args, **kwargs: 传递给通知方法的参数
    """
    try:
        notify_func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"🦞 Notification failed: {e}")


def create_notifier(config: Dict[str, Any]) -> Optional["OpenClawWebhookClient"]:
    """
    根据配置创建通知客户端

    Args:
        config: YAML 配置字典

    Returns:
        OpenClawWebhookClient 实例，或 None（如果未启用）
    """
    if not NOTIFIER_AVAILABLE:
        logger.warning("openclaw_notifier not available, notifications disabled")
        return None

    notify_cfg = config.get("notification", {})

    if not notify_cfg.get("enabled", False):
        return None

    # 获取配置值
    token = notify_cfg.get("token", "")
    # 支持环境变量展开
    if token.startswith("${") and token.endswith("}"):
        env_var = token[2:-1]
        token = os.environ.get(env_var, "")

    if not token:
        logger.warning("notification.enabled=true but no token provided")
        return None

    try:
        session_key = notify_cfg.get("session_key", "")
        gateway_url = notify_cfg.get("gateway_url", "http://127.0.0.1:18789")

        logger.info(f"🦞 Creating notification client")
        logger.info(f"🦞 gateway_url: {gateway_url}")
        logger.info(f"🦞 session_key: {session_key}")

        client = OpenClawWebhookClient(
            gateway_url=gateway_url,
            token=token,
            session_key=session_key,
        )
        logger.info("🦞 Notification client created successfully")
        return client
    except Exception as e:
        logger.warning(f"🦞 Failed to create notification client: {e}")
        return None


def create_progress_callback(
    notifier: Optional["OpenClawWebhookClient"],
    experiment_name: str,
    config: Dict[str, Any],
) -> Optional[Callable[[str, float, Optional[Dict]], None]]:
    """
    创建进度回调函数（仅处理语料库级别进度）

    职责划分：
    - 外层（run_framework）：实验级别的 started/completed/error 通知
    - 此回调：语料库级别的 indexing/querying 进度
    """
    if not notifier:
        return None

    notify_cfg = config.get("notification", {})
    if not notify_cfg.get("notify_on_progress", True):
        return None

    progress_interval = notify_cfg.get("progress_interval", 0.2)
    last_progress = [0.0]

    def callback(stage: str, progress: float, info: Optional[Dict] = None):
        """进度回调：检查间隔后发送通知"""
        if progress - last_progress[0] >= progress_interval:
            last_progress[0] = progress
            logger.info(f"🦞 notify_progress: {experiment_name} @ {progress:.1%}")
            safe_notify(notifier.notify_progress, experiment_name, progress, metrics=info)

    return callback


def resolve_frameworks(args_framework: str | None, config: Dict[str, Any]) -> List[str]:
    """解析要运行的框架"""
    def is_enabled(name: str) -> bool:
        return bool(config.get("frameworks", {}).get(name, {}).get("enabled", False))

    if args_framework and args_framework != "auto":
        if args_framework == "all":
            return [name for name in SUPPORTED_FRAMEWORKS if is_enabled(name)]
        return [args_framework]

    run_cfg = config.get("run", {})
    from_cfg = run_cfg.get("framework")
    if from_cfg == "all":
        return [name for name in SUPPORTED_FRAMEWORKS if is_enabled(name)]
    if isinstance(from_cfg, str) and from_cfg:
        if from_cfg not in SUPPORTED_FRAMEWORKS:
            raise ValueError(f"Unsupported framework: {from_cfg}")
        if not is_enabled(from_cfg):
            raise ValueError(f"Framework '{from_cfg}' is not enabled")
        return [from_cfg]
    return []


async def run_framework(
    framework: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    notifier: Optional["OpenClawWebhookClient"] = None,
) -> int:
    """
    运行单个框架

    Args:
        framework: 框架名称
        config: 配置字典
        dry_run: 是否为干跑
        notifier: 通知客户端

    Returns:
        退出码 (0=成功, 1=失败)
    """
    if not has_adapter(framework):
        logger.error(f"No adapter for '{framework}'")
        return 1

    fw_cfg = config.get("frameworks", {}).get(framework, {})

    # 解析配置
    run_cfg = config.get("run", {})
    dataset_cfg = config.get("dataset", {})
    llm_cfg = config.get("llm", {})
    embed_cfg = config.get("embedding", {})
    retrieval_cfg = config.get("retrieval", {})
    neo4j_cfg = config.get("neo4j", {})
    output_cfg = config.get("output", {})

    # 数据集
    subset = dataset_cfg.get("subset", "medical")
    sample = dataset_cfg.get("sample")
    corpus_sample = dataset_cfg.get("corpus_sample")

    # 运行ID
    run_id = run_cfg.get("run_id") or datetime.now().strftime("%Y%m%d-%H%M%S")

    # 实验名称（用于通知）
    experiment_name = f"{framework} - {subset}"

    # 嵌入配置
    embed_type = fw_cfg.get("embed_type") or embed_cfg.get("type", DEFAULT_EMBED_TYPE)
    embed_provider = fw_cfg.get("embed_provider") or embed_cfg.get("provider", DEFAULT_EMBED_PROVIDER)
    embed_model = fw_cfg.get("embed_model") or embed_cfg.get("model", "")
    embed_base_url = fw_cfg.get("embed_base_url") or embed_cfg.get("base_url")
    embed_api_key = fw_cfg.get("embed_api_key") or embed_cfg.get("api_key")
    embed_dimensions = fw_cfg.get("embed_dimensions") or embed_cfg.get("dimensions")
    embed_batch_size = fw_cfg.get("embed_batch_size") or embed_cfg.get("batch_size", 64)

    # 构建 FrameworkConfig
    framework_config = FrameworkConfig(
        llm_model=llm_cfg.get("model", ""),
        llm_base_url=llm_cfg.get("base_url", ""),
        llm_api_key=llm_cfg.get("api_key", ""),
        embed_model=embed_model,
        embed_type=embed_type,
        embed_provider=embed_provider,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
        embed_dimensions=embed_dimensions,
        embed_batch_size=embed_batch_size,
        top_k=retrieval_cfg.get("top_k", DEFAULT_TOP_K),
        max_concurrency=fw_cfg.get("max_concurrency", DEFAULT_MAX_CONCURRENCY),
        extra={
            # LightRAG 参数
            "mode": fw_cfg.get("mode", "API"),
            "chunk_token_size": fw_cfg.get("chunk_token_size", 1200),
            "chunk_overlap_token_size": fw_cfg.get("chunk_overlap_token_size", 100),
            # ClearRAG 参数
            "activation_mode": fw_cfg.get("activation_mode", "semantic_propagation"),
            "passage_retrieval_mode": fw_cfg.get("passage_retrieval_mode", "pagerank"),
            "fast_mode": fw_cfg.get("fast", False),
            # LinearRAG 参数
            "max_iterations": fw_cfg.get("max_iterations", 3),
            "iteration_threshold": fw_cfg.get("iteration_threshold", 0.4),
            "top_k_sentence": fw_cfg.get("top_k_sentence", 3),
            "use_vectorized": fw_cfg.get("use_vectorized", False),
            "spacy_model": fw_cfg.get("spacy_model", "en_core_web_trf"),
            "max_workers": fw_cfg.get("max_workers", 8),
            # Fast-GraphRAG 参数
            "domain": fw_cfg.get("domain", ""),
            "entity_types": fw_cfg.get("entity_types", []),
            "example_queries": fw_cfg.get("example_queries", []),
            # HippoRAG2 参数
            "embed_model_path": fw_cfg.get("embed_model_path", embed_model),
            "chunk_token_size": fw_cfg.get("chunk_token_size", 256),
            "chunk_overlap": fw_cfg.get("chunk_overlap", 32),
            "force_index_from_scratch": fw_cfg.get("force_index_from_scratch", True),
            "skip_build": fw_cfg.get("skip_build", False),
            "rerank_dspy_file_path": fw_cfg.get("rerank_dspy_file_path", "hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json"),
            "max_qa_steps": fw_cfg.get("max_qa_steps", 3),
            "graph_type": fw_cfg.get("graph_type", "facts_and_sim_passage_node_unidirectional"),
            "embedding_batch_size": fw_cfg.get("embedding_batch_size", 8),
            "openie_mode": fw_cfg.get("openie_mode", "online"),
            # Neo4j 参数
            "neo4j_uri": neo4j_cfg.get("uri"),
            "neo4j_user": neo4j_cfg.get("user"),
            "neo4j_password": neo4j_cfg.get("password"),
            "neo4j_database": neo4j_cfg.get("database"),
            # DIGIMON 参数
            "config_path": fw_cfg.get("config_path", "./config.yml"),
        },
    )

    # 数据集路径
    try:
        corpus_path, questions_path = get_subset_paths(subset)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # 输出路径
    output_root = output_cfg.get("root", "./results")
    workspace_dir = fw_cfg.get("workspace_dir") or str(Path(output_root) / framework / subset / run_id / "workspace")
    predictions_dir = str(Path(output_root) / framework / subset / run_id)

    # 打印完整配置
    _print_config(framework, config, fw_cfg, dataset_cfg, llm_cfg, embed_cfg,
                  retrieval_cfg, neo4j_cfg, run_id, workspace_dir, predictions_dir)

    if dry_run:
        logger.info(f"[{framework}] Dry run - skipping")
        return 0

    # 创建进度回调
    progress_callback = create_progress_callback(notifier, experiment_name, config)

    # 通知配置
    notify_cfg = config.get("notification", {})
    notify_on_start = notify_cfg.get("notify_on_start", True)
    notify_on_complete = notify_cfg.get("notify_on_complete", True)
    notify_on_error = notify_cfg.get("notify_on_error", True)

    # 运行
    try:
        # 发送开始通知
        if notifier and notify_on_start:
            logger.info(f"🦞 notify_start: {experiment_name}")
            safe_notify(
                notifier.notify_start,
                experiment_name,
                config={
                    "framework": framework,
                    "subset": subset,
                    "sample": sample,
                    "run_id": run_id,
                },
            )

        runner = FrameworkRunner(
            framework=framework,
            config=framework_config,
            subset=subset,
            corpus_path=corpus_path,
            questions_path=questions_path,
            workspace_dir=workspace_dir,
            predictions_dir=predictions_dir,
            sample=sample,
            corpus_sample=corpus_sample,
            corpus_concurrency=fw_cfg.get("corpus_concurrency", 1),
            skip_build=fw_cfg.get("skip_build", False),
            index_only=fw_cfg.get("index_only", False),
            progress_callback=progress_callback,
        )

        result = await runner.run()
        error_count = result.get("error", 0)

        # 发送结束通知
        if notifier:
            if error_count == 0 and notify_on_complete:
                logger.info(f"🦞 notify_complete: {experiment_name}")
                safe_notify(
                    notifier.notify_complete,
                    experiment_name,
                    results={
                        "success": result.get("success", 0),
                        "error": result.get("error", 0),
                        "output": predictions_dir,
                    },
                )
            elif error_count > 0 and notify_on_error:
                logger.info(f"🦞 notify_error: {experiment_name} ({error_count} failed)")
                safe_notify(
                    notifier.notify_error,
                    experiment_name,
                    Exception(f"{error_count} corpus failed"),
                    context=result,
                )

        return 0 if error_count == 0 else 1

    except Exception as e:
        logger.error(f"[{framework}] Failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())

        # 发送错误通知
        if notifier and notify_on_error:
            logger.info(f"🦞 notify_error: {experiment_name} (exception)")
            safe_notify(notifier.notify_error, experiment_name, e)

        return 1


def _print_config(
    framework: str,
    config: Dict[str, Any],
    fw_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    embed_cfg: Dict[str, Any],
    retrieval_cfg: Dict[str, Any],
    neo4j_cfg: Dict[str, Any],
    run_id: str,
    workspace_dir: str,
    predictions_dir: str,
) -> None:
    """打印完整配置"""
    logger.info(f"[{framework}] Running")

    # 数据集
    logger.info(f"  subset: {dataset_cfg.get('subset', 'medical')}")
    logger.info(f"  sample: {dataset_cfg.get('sample')}")
    logger.info(f"  corpus_sample: {dataset_cfg.get('corpus_sample')}")

    # LLM
    logger.info(f"  model: {llm_cfg.get('model')}")
    logger.info(f"  base_url: {llm_cfg.get('base_url')}")

    # Embedding
    logger.info(f"  type: {fw_cfg.get('embed_type') or embed_cfg.get('type', 'api')}")
    logger.info(f"  provider: {fw_cfg.get('embed_provider') or embed_cfg.get('provider', 'zhipu')}")
    logger.info(f"  model: {fw_cfg.get('embed_model') or embed_cfg.get('model')}")
    logger.info(f"  base_url: {fw_cfg.get('embed_base_url') or embed_cfg.get('base_url')}")

    # Retrieval
    logger.info(f"  top_k: {retrieval_cfg.get('top_k', 5)}")

    # Run
    logger.info(f"  run_id: {run_id}")
    logger.info(f"  corpus_concurrency: {fw_cfg.get('corpus_concurrency', 1)}")
    logger.info(f"  skip_build: {fw_cfg.get('skip_build', False)}")
    logger.info(f"  index_only: {fw_cfg.get('index_only', False)}")

    # Path
    logger.info(f"  workspace: {workspace_dir}")
    logger.info(f"  output: {predictions_dir}")

    # 框架特有参数
    if framework == "lightrag":
        logger.info(f"  mode: {fw_cfg.get('mode', 'API')}")
        logger.info(f"  chunk_token_size: {fw_cfg.get('chunk_token_size', 1200)}")
        logger.info(f"  chunk_overlap: {fw_cfg.get('chunk_overlap_token_size', 100)}")
    elif framework == "clearrag":
        logger.info(f"  activation_mode: {fw_cfg.get('activation_mode', 'semantic_propagation')}")
        logger.info(f"  passage_retrieval_mode: {fw_cfg.get('passage_retrieval_mode', 'none')}")
        logger.info(f"  fast: {fw_cfg.get('fast', False)}")
        logger.info(f"  max_concurrency: {fw_cfg.get('max_concurrency', 5)}")
        logger.info(f"  neo4j_uri: {neo4j_cfg.get('uri')}")
        logger.info(f"  neo4j_database: {neo4j_cfg.get('database')}")
    elif framework == "linearrag":
        logger.info(f"  max_iterations: {fw_cfg.get('max_iterations', 3)}")
        logger.info(f"  iteration_threshold: {fw_cfg.get('iteration_threshold', 0.4)}")
        logger.info(f"  top_k_sentence: {fw_cfg.get('top_k_sentence', 3)}")
        logger.info(f"  use_vectorized: {fw_cfg.get('use_vectorized', False)}")
        logger.info(f"  spacy_model: {fw_cfg.get('spacy_model', 'en_core_web_trf')}")
        logger.info(f"  max_workers: {fw_cfg.get('max_workers', 8)}")
    elif framework == "fast-graphrag":
        logger.info(f"  mode: {fw_cfg.get('mode', 'API')}")
        logger.info(f"  embed_type: {fw_cfg.get('embed_type', 'api')}")
    elif framework == "hipporag2":
        logger.info(f"  mode: {fw_cfg.get('mode', 'API')}")
        logger.info(f"  embed_type: {fw_cfg.get('embed_type', 'local')}")
        logger.info(f"  embed_model_path: {fw_cfg.get('embed_model_path', embed_cfg.get('model'))}")
        logger.info(f"  chunk_token_size: {fw_cfg.get('chunk_token_size', 256)}")
        logger.info(f"  chunk_overlap: {fw_cfg.get('chunk_overlap', 32)}")
    elif framework == "digimon":
        logger.info(f"  mode: {fw_cfg.get('mode', 'API')}")
        logger.info(f"  config_path: {fw_cfg.get('config_path', './config.yml')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run benchmark frameworks from YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Examples/run_from_yaml.py --dry-run
  python Examples/run_from_yaml.py --framework lightrag --subset medical --sample 10
  python Examples/run_from_yaml.py --framework all --subset hotpotqa_500
        """
    )
    # 配置文件
    parser.add_argument("--config", default="configs/experiment.yaml",
                        help="配置文件路径 (default: configs/experiment.yaml)")
    
    # 框架选择
    parser.add_argument(
        "--framework",
        default="auto",
        choices=["auto", "all"] + SUPPORTED_FRAMEWORKS,
        help="框架名称: auto=使用配置, all=所有启用的框架, 或具体框架名"
    )
    
    # 数据集临时覆盖
    parser.add_argument("--subset", default=None,
                        help="数据集子集名称 (覆盖配置文件)")
    parser.add_argument("--sample", type=int, default=None,
                        help="问题采样数 (覆盖配置文件)")
    # 运行控制
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印配置，不执行")
    parser.add_argument("--skip-build", action="store_true",
                        help="跳过索引构建")
    
    args = parser.parse_args()

    # 加载配置
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 应用命令行参数覆盖
    if args.subset:
        config.setdefault("dataset", {})["subset"] = args.subset
        logger.info(f"[override] subset={args.subset}")
    if args.sample is not None:
        config.setdefault("dataset", {})["sample"] = args.sample
        logger.info(f"[override] sample={args.sample}")
    if args.skip_build:
        for fw in config.get("frameworks", {}):
            config["frameworks"][fw]["skip_build"] = True
        logger.info(f"[override] skip_build=True (all frameworks)")

    # 确定框架
    frameworks = resolve_frameworks(args.framework, config)
    if not frameworks:
        logger.error("No frameworks selected")
        return 1

    logger.info(f"[run] frameworks={frameworks}")

    # 创建通知客户端
    notifier = create_notifier(config)

    # 运行
    exit_code = 0
    continue_on_error = config.get("run", {}).get("continue_on_error", False)

    for framework in frameworks:
        result = asyncio.run(run_framework(framework, config, args.dry_run, notifier))
        if result != 0:
            exit_code = result
            if not continue_on_error:
                break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())