import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from subset_registry import FRAMEWORK_SUPPORTED_SUBSETS

SUPPORTED_FRAMEWORKS = list(FRAMEWORK_SUPPORTED_SUBSETS.keys())

def is_empty(value: Any) -> bool:
    return value is None or value == ""


def merge_value(common_value: Any, framework_value: Any, enforce_common: bool) -> Any:
    if enforce_common and not is_empty(common_value):
        return common_value
    if not is_empty(framework_value):
        return framework_value
    return common_value


def add_arg(cmd: List[str], name: str, value: Any) -> None:
    if is_empty(value):
        return
    if isinstance(value, bool):
        if value:
            cmd.append(name)
        return
    cmd.extend([name, str(value)])


def get_nested(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_run_id(config: Dict[str, Any]) -> str:
    runtime_run_id = get_nested(config, ["_runtime", "run_id"])
    if not is_empty(runtime_run_id):
        return str(runtime_run_id)
    run_id = get_nested(config, ["run", "run_id"])
    if not is_empty(run_id):
        return str(run_id)
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def render_path_template(path_value: Any, *, framework: str, run_id: str) -> str:
    text = str(path_value)
    return text.replace("{framework}", framework).replace("{run_id}", run_id)


def resolve_framework_paths(
    framework: str, config: Dict[str, Any], fw: Dict[str, Any]
) -> Dict[str, str]:
    run_id = resolve_run_id(config)
    paths = config.get("paths", {})
    default_base_tpl = os.path.join(
        paths.get("workspace_root", "./workspaces"),
        framework,
        "{run_id}",
    )
    default_output_tpl = os.path.join(
        paths.get("prediction_root", "./results/predictions"),
        framework,
        "{run_id}",
    )
    base_tpl = fw.get("base_dir", default_base_tpl)
    output_tpl = fw.get("output_dir", default_output_tpl)
    return {
        "base_dir": render_path_template(base_tpl, framework=framework, run_id=run_id),
        "output_dir": render_path_template(output_tpl, framework=framework, run_id=run_id),
    }


def build_env(config: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    common = config.get("common", {})
    llm = common.get("llm", {})
    embed = common.get("embed", {})
    neo4j = common.get("neo4j", {})
    clearrag = get_nested(config, ["frameworks", "clearrag"], {})
    run_id = resolve_run_id(config)
    linear_data_dir_tpl = clearrag.get("linear_data_dir")
    if is_empty(linear_data_dir_tpl):
        clearrag_paths = resolve_framework_paths("clearrag", config, clearrag)
        linear_data_dir = os.path.join(clearrag_paths["base_dir"], "linear")
    else:
        linear_data_dir = render_path_template(
            linear_data_dir_tpl, framework="clearrag", run_id=run_id
        )

    mappings = {
        "LLM_MODEL": llm.get("model", ""),
        "LLM_BASE_URL": llm.get("base_url", ""),
        "LLM_API_KEY": llm.get("api_key", ""),
        "EMBED_MODEL": embed.get("name", ""),
        "EMBED_BASE_URL": embed.get("base_url", ""),
        "EMBED_API_KEY": embed.get("api_key", ""),
        "ZHIPUAI_API_KEY": embed.get("api_key", ""),
        "OPENAI_API_KEY": llm.get("api_key", ""),
        "NEO4J_URI": neo4j.get("uri", ""),
        "NEO4J_USER": neo4j.get("user", ""),
        "NEO4J_PASSWORD": neo4j.get("password", ""),
        "NEO4J_DATABASE": neo4j.get("database", ""),
        "LINEAR_DATA_DIR": linear_data_dir,
    }
    for k, v in mappings.items():
        if v not in (None, ""):
            env[k] = str(v)
    return env


def build_lightrag_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    llm = common.get("llm", {})
    embed = common.get("embed", {})
    retrieval = common.get("retrieval", {})
    cmd = [sys.executable, "Examples/run_lightrag.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--mode", fw.get("mode", llm.get("mode", "API")))
    add_arg(
        cmd,
        "--model_name",
        merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
    )
    add_arg(cmd, "--llm_base_url", fw.get("llm_base_url", llm.get("base_url")))
    add_arg(cmd, "--llm_api_key", fw.get("llm_api_key", llm.get("api_key")))
    add_arg(cmd, "--embed_provider", fw.get("embed_provider", embed.get("provider", "hf")))
    add_arg(cmd, "--embed_model", fw.get("embed_model", embed.get("name")))
    add_arg(cmd, "--embed_api_key", fw.get("embed_api_key", embed.get("api_key")))
    add_arg(
        cmd,
        "--retrieve_topk",
        merge_value(retrieval.get("top_k", 5), fw.get("retrieve_topk"), enforce_common),
    )
    add_arg(cmd, "--base_dir", fw.get("base_dir"))
    add_arg(cmd, "--output_dir", fw.get("output_dir"))
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    add_arg(
        cmd,
        "--corpus-concurrency",
        merge_value(common.get("corpus_concurrency"), fw.get("corpus_concurrency"), enforce_common),
    )
    add_arg(
        cmd,
        "--index-only",
        merge_value(common.get("index_only"), fw.get("index_only"), enforce_common),
    )
    add_arg(cmd, "--skip-build", fw.get("skip_build", False))
    return cmd


def build_clearrag_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    retrieval = common.get("retrieval", {})
    cmd = [sys.executable, "Examples/run_clearrag.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--base_dir", fw.get("base_dir", "./clearrag_workspace"))
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(
        cmd,
        "--top_k",
        merge_value(retrieval.get("top_k", 5), fw.get("top_k"), enforce_common),
    )
    add_arg(cmd, "--activation-mode", fw.get("activation_mode", "semantic_propagation"))
    add_arg(cmd, "--passage-retrieval-mode", fw.get("passage_retrieval_mode", "pagerank"))
    add_arg(cmd, "--fast", fw.get("fast", False))
    add_arg(cmd, "--max-concurrency", fw.get("max_concurrency", 5))
    add_arg(cmd, "--corpus-concurrency", fw.get("corpus_concurrency", 2))
    add_arg(cmd, "--output_dir", fw.get("output_dir", "./results/clearrag"))
    add_arg(cmd, "--skip-build", fw.get("skip_build", False))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    return cmd


def build_fast_graphrag_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    llm = common.get("llm", {})
    embed = common.get("embed", {})
    cmd = [sys.executable, "Examples/run_fast-graphrag.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--mode", fw.get("mode", llm.get("mode", "API")))
    add_arg(cmd, "--base_dir", fw.get("base_dir", "./fast-graphrag_workspace"))
    add_arg(
        cmd,
        "--model_name",
        merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
    )
    add_arg(cmd, "--llm_base_url", fw.get("llm_base_url", llm.get("base_url")))
    add_arg(cmd, "--llm_api_key", fw.get("llm_api_key", llm.get("api_key")))
    add_arg(cmd, "--embed_provider", fw.get("embed_provider", embed.get("provider", "api")))
    add_arg(cmd, "--embed_model", fw.get("embed_model", embed.get("name")))
    add_arg(cmd, "--embed_base_url", fw.get("embed_base_url", embed.get("base_url")))
    add_arg(cmd, "--embed_api_key", fw.get("embed_api_key", embed.get("api_key")))
    add_arg(cmd, "--output_dir", fw.get("output_dir"))
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    add_arg(cmd, "--skip-build", fw.get("skip_build", False))
    return cmd


def build_hipporag2_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    llm = common.get("llm", {})
    embed = common.get("embed", {})
    retrieval = common.get("retrieval", {})
    cmd = [sys.executable, "Examples/run_hipporag2.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--mode", fw.get("mode", llm.get("mode", "API")))
    add_arg(cmd, "--base_dir", fw.get("base_dir", "./hipporag2_workspace"))
    add_arg(
        cmd,
        "--model_name",
        merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
    )
    add_arg(cmd, "--embed_model_path", fw.get("embed_model_path", embed.get("name")))
    add_arg(cmd, "--llm_base_url", fw.get("llm_base_url", llm.get("base_url")))
    add_arg(cmd, "--llm_api_key", fw.get("llm_api_key", llm.get("api_key")))
    add_arg(cmd, "--output_dir", fw.get("output_dir"))
    add_arg(
        cmd,
        "--top_k",
        merge_value(retrieval.get("top_k", 5), fw.get("top_k"), enforce_common),
    )
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    add_arg(cmd, "--skip-build", fw.get("skip_build", False))
    return cmd


def build_digimon_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    llm = common.get("llm", {})
    cmd = [sys.executable, "Examples/run_digimon.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--config", fw.get("config", "./config.yml"))
    add_arg(cmd, "--output_dir", fw.get("output_dir", "./results/GraphRAG"))
    add_arg(cmd, "--mode", fw.get("mode", "config"))
    add_arg(
        cmd,
        "--model_name",
        merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
    )
    add_arg(cmd, "--llm_base_url", fw.get("llm_base_url", llm.get("base_url")))
    add_arg(cmd, "--llm_api_key", fw.get("llm_api_key", llm.get("api_key")))
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    return cmd


def build_linearrag_cmd(
    common: Dict[str, Any], fw: Dict[str, Any], enforce_common: bool
) -> List[str]:
    """构建 LinearRAG 命令"""
    llm = common.get("llm", {})
    embed = common.get("embed", {})
    retrieval = common.get("retrieval", {})
    cmd = [sys.executable, "Examples/run_linearrag.py"]

    add_arg(
        cmd,
        "--subset",
        merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
    )
    add_arg(cmd, "--base_dir", fw.get("base_dir", "./linearrag_workspace"))
    add_arg(
        cmd,
        "--model_name",
        merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
    )
    add_arg(cmd, "--llm_base_url", fw.get("llm_base_url", llm.get("base_url")))
    add_arg(cmd, "--llm_api_key", fw.get("llm_api_key", llm.get("api_key")))
    add_arg(cmd, "--embed_provider", fw.get("embed_provider", embed.get("provider", "zhipu")))
    add_arg(cmd, "--embed_model", fw.get("embed_model", embed.get("name")))
    add_arg(cmd, "--embed_api_key", fw.get("embed_api_key", embed.get("api_key")))
    add_arg(cmd, "--embed_base_url", fw.get("embed_base_url", embed.get("base_url")))
    add_arg(
        cmd,
        "--top_k",
        merge_value(retrieval.get("top_k", 5), fw.get("top_k"), enforce_common),
    )
    add_arg(cmd, "--max_iterations", fw.get("max_iterations", 3))
    add_arg(cmd, "--iteration_threshold", fw.get("iteration_threshold", 0.4))
    add_arg(cmd, "--top_k_sentence", fw.get("top_k_sentence", 3))
    add_arg(cmd, "--use_vectorized", fw.get("use_vectorized", False))
    add_arg(cmd, "--spacy_model", fw.get("spacy_model", "en_core_web_trf"))
    add_arg(cmd, "--max_workers", fw.get("max_workers", 8))
    add_arg(
        cmd,
        "--corpus-concurrency",
        merge_value(common.get("corpus_concurrency"), fw.get("corpus_concurrency"), enforce_common),
    )
    add_arg(
        cmd,
        "--index-only",
        merge_value(common.get("index_only"), fw.get("index_only"), enforce_common),
    )
    add_arg(cmd, "--output_dir", fw.get("output_dir"))
    add_arg(cmd, "--sample", merge_value(common.get("sample"), fw.get("sample"), enforce_common))
    add_arg(cmd, "--corpus_sample", merge_value(common.get("corpus_sample"), fw.get("corpus_sample"), enforce_common))
    add_arg(cmd, "--skip-build", fw.get("skip_build", False))
    return cmd


def build_command(framework: str, config: Dict[str, Any]) -> List[str]:
    common = config.get("common", {})
    fw_cfg = get_nested(config, ["frameworks", framework], {})
    fw = dict(fw_cfg)
    resolved_paths = resolve_framework_paths(framework, config, fw_cfg)
    fw["base_dir"] = resolved_paths["base_dir"]
    fw["output_dir"] = resolved_paths["output_dir"]
    run_cfg = config.get("run", {})
    enforce_common = run_cfg.get("enforce_common", True)
    subset = merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common)
    allowed = set(FRAMEWORK_SUPPORTED_SUBSETS.get(framework, ()))
    if allowed and subset not in allowed:
        raise ValueError(
            f"Framework '{framework}' does not support subset '{subset}'. "
            f"Supported: {sorted(allowed)}"
        )
    builders = {
        "lightrag": build_lightrag_cmd,
        "clearrag": build_clearrag_cmd,
        "fast-graphrag": build_fast_graphrag_cmd,
        "hipporag2": build_hipporag2_cmd,
        "digimon": build_digimon_cmd,
        "linearrag": build_linearrag_cmd,
    }
    if framework not in builders:
        raise ValueError(f"Unsupported framework: {framework}")
    return builders[framework](common, fw, enforce_common)


def build_consistency_view(framework: str, config: Dict[str, Any]) -> Dict[str, Any]:
    common = config.get("common", {})
    fw = get_nested(config, ["frameworks", framework], {})
    run_cfg = config.get("run", {})
    enforce_common = run_cfg.get("enforce_common", True)
    llm = common.get("llm", {})
    retrieval = common.get("retrieval", {})

    view: Dict[str, Any] = {
        "subset": merge_value(common.get("subset", "medical"), fw.get("subset"), enforce_common),
        "model": merge_value(llm.get("model"), fw.get("model_name"), enforce_common),
        "top_k": None,
        "enforce_common": enforce_common,
    }

    if framework == "lightrag":
        view["top_k"] = merge_value(retrieval.get("top_k", 5), fw.get("retrieve_topk"), enforce_common)
    elif framework in {"clearrag", "hipporag2", "linearrag"}:
        view["top_k"] = merge_value(retrieval.get("top_k", 5), fw.get("top_k"), enforce_common)
    return view


def resolve_frameworks(args_framework: str | None, config: Dict[str, Any]) -> List[str]:
    def is_enabled(name: str) -> bool:
        return bool(get_nested(config, ["frameworks", name, "enabled"], False))

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
            raise ValueError(f"Unsupported run.framework: {from_cfg}")
        if not is_enabled(from_cfg):
            raise ValueError(
                f"run.framework is '{from_cfg}', but frameworks.{from_cfg}.enabled=false. "
                "Set enabled=true, or switch run.framework / --framework."
            )
        return [from_cfg]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark frameworks from one YAML config")
    parser.add_argument("--config", default="configs/experiment.yaml", help="Path to YAML config")
    parser.add_argument(
        "--framework",
        default="auto",
        choices=["auto", "all"] + SUPPORTED_FRAMEWORKS,
        help="Framework to run (auto=use config run.framework)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    run_id = resolve_run_id(config)
    config.setdefault("_runtime", {})["run_id"] = run_id

    frameworks = resolve_frameworks(args.framework, config)
    if not frameworks:
        raise ValueError("No frameworks selected. Set run.framework or pass --framework.")

    root = Path(__file__).resolve().parents[1]
    env = build_env(config)
    continue_on_error = get_nested(config, ["run", "continue_on_error"], False)
    print(f"[run] run_id={run_id}")

    for framework in frameworks:
        consistency = build_consistency_view(framework, config)
        print(
            f"[{framework}] params: subset={consistency['subset']}, "
            f"model={consistency['model']}, top_k={consistency['top_k']}, "
            f"enforce_common={consistency['enforce_common']}"
        )
        cmd = build_command(framework, config)
        printable = " ".join(cmd)
        print(f"[{framework}] {printable}")
        if args.dry_run:
            continue

        result = subprocess.run(cmd, cwd=str(root), env=env)
        if result.returncode != 0 and not continue_on_error:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

