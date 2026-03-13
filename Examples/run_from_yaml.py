"""
统一 YAML 配置运行器

配置格式：
```yaml
dataset:
  subset: medical

llm:
  model: deepseek-chat

embedding:
  provider: api
  model: embedding-3

output:
  root: ./results

frameworks:
  lightrag:
    enabled: true
    mode: API
    corpus_concurrency: 1
```
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from subset_registry import FRAMEWORK_SUPPORTED_SUBSETS, get_subset_paths
from runner import FrameworkRunner
from adapters import FrameworkConfig, has_adapter
from adapters.base import DEFAULT_EMBED_PROVIDER, DEFAULT_TOP_K, DEFAULT_MAX_CONCURRENCY

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

SUPPORTED_FRAMEWORKS = list(FRAMEWORK_SUPPORTED_SUBSETS.keys())


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


async def run_framework(framework: str, config: Dict[str, Any], dry_run: bool = False) -> int:
    """运行单个框架"""
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

    # 框架配置覆盖嵌入设置
    embed_provider = fw_cfg.get("embed_provider") or embed_cfg.get("provider", DEFAULT_EMBED_PROVIDER)
    embed_model = fw_cfg.get("embed_model") or embed_cfg.get("model", "")
    embed_base_url = fw_cfg.get("embed_base_url") or embed_cfg.get("base_url")
    embed_api_key = fw_cfg.get("embed_api_key") or embed_cfg.get("api_key")

    # 构建 FrameworkConfig
    framework_config = FrameworkConfig(
        llm_model=llm_cfg.get("model", ""),
        llm_base_url=llm_cfg.get("base_url", ""),
        llm_api_key=llm_cfg.get("api_key", ""),
        embed_model=embed_model,
        embed_provider=embed_provider,
        embed_api_key=embed_api_key,
        embed_base_url=embed_base_url,
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
            "embed_provider": fw_cfg.get("embed_provider", "api"),
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
    # workspace: 优先使用框架配置中的 workspace_dir，否则使用默认路径
    # predictions: {root}/{framework}/{subset}/{run_id}/
    output_root = output_cfg.get("root", "./results")
    workspace_dir = fw_cfg.get("workspace_dir") or str(Path(output_root) / framework / subset / run_id / "workspace")
    predictions_dir = str(Path(output_root) / framework / subset / run_id)

    # 打印完整配置
    _print_config(framework, config, fw_cfg, dataset_cfg, llm_cfg, embed_cfg,
                  retrieval_cfg, neo4j_cfg, run_id, workspace_dir, predictions_dir)

    if dry_run:
        logger.info(f"[{framework}] Dry run - skipping")
        return 0

    # 运行
    try:
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
        )

        result = await runner.run()
        return 0 if result.get("error_count", 0) == 0 else 1

    except Exception as e:
        logger.error(f"[{framework}] Failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
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
    logger.info(f"  provider: {fw_cfg.get('embed_provider') or embed_cfg.get('provider', 'api')}")
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
        logger.info(f"  embed_provider: {fw_cfg.get('embed_provider', 'api')}")
    elif framework == "hipporag2":
        logger.info(f"  mode: {fw_cfg.get('mode', 'API')}")
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
  python Examples/run_from_yaml.py --framework lightrag
  python Examples/run_from_yaml.py --framework all
        """
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument(
        "--framework",
        default="auto",
        choices=["auto", "all"] + SUPPORTED_FRAMEWORKS,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 加载配置
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        return 1

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # 确定框架
    frameworks = resolve_frameworks(args.framework, config)
    if not frameworks:
        logger.error("No frameworks selected")
        return 1

    logger.info(f"[run] frameworks={frameworks}")

    # 运行
    exit_code = 0
    continue_on_error = config.get("run", {}).get("continue_on_error", False)

    for framework in frameworks:
        result = asyncio.run(run_framework(framework, config, args.dry_run))
        if result != 0:
            exit_code = result
            if not continue_on_error:
                break

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())