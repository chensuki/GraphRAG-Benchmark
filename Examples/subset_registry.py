from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


# 基础数据集（手动定义）
BASE_SUBSET_DATA_PATHS: Dict[str, Dict[str, str]] = {
    # 样本数据集
    "sample": {
        "corpus": "./Datasets/Corpus/sample_corpus.json",
        "questions": "./Datasets/Questions/sample_questions.json",
    },

    # 原有数据集
    "medical": {
        "corpus": "./Datasets/Corpus/medical.parquet",
        "questions": "./Datasets/Questions/medical_questions.parquet",
    },
    "medical_100": {
        "corpus": "./Datasets/Corpus/medical.parquet",
        "questions": "./Datasets/Questions/medical_questions_100_balanced.json",
    },
    "novel": {
        "corpus": "./Datasets/Corpus/novel.parquet",
        "questions": "./Datasets/Questions/novel_questions.parquet",
    },

    # HotpotQA: distractor 版本（预填充文档）
    "hotpotqa_distractor": {
        "corpus": "./Datasets/Corpus/hotpotqa_distractor.parquet",
        "questions": "./Datasets/Questions/hotpotqa_distractor_questions.parquet",
    },

    # HotpotQA: fullwiki 版本（完整 Wikipedia）
    "hotpotqa_fullwiki": {
        "corpus": "./Datasets/Corpus/hotpotqa_fullwiki.parquet",
        "questions": "./Datasets/Questions/hotpotqa_fullwiki_questions.parquet",
    },

    # 2WikiMultihopQA: 两跳维基百科问答
    "2wikimultihop": {
        "corpus": "./Datasets/Corpus/2wikimultihop.parquet",
        "questions": "./Datasets/Questions/2wikimultihop_questions.parquet",
    },

    # MuSiQue: 多跳问答（可回答样本）
    "musique": {
        "corpus": "./Datasets/Corpus/musique.parquet",
        "questions": "./Datasets/Questions/musique_questions.parquet",
    },
}

BASE_SUBSETS: Tuple[str, ...] = (
    "sample",
    "medical",
    "medical_100",
    "novel",
    "hotpotqa_distractor",
    "hotpotqa_fullwiki",
    "2wikimultihop",
    "musique",
)


# 自动发现子集数据集
def _discover_subset_datasets() -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """
    自动发现 Datasets 目录下的子集数据集
    
    命名模式：
    - Corpus: {dataset}_subset_{num}.json
    - Questions: {dataset}_questions_subset_{num}.json
    
    Returns:
        (子集路径映射, 子集名称列表)
    """
    # 获取项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent
    
    corpus_dir = project_root / "Datasets" / "Corpus"
    questions_dir = project_root / "Datasets" / "Questions"
    
    subset_paths: Dict[str, Dict[str, str]] = {}
    subset_names: List[str] = []
    
    # 扫描语料库目录，寻找 *_subset_*.json 文件
    if not corpus_dir.exists():
        return subset_paths, subset_names
    
    # 匹配模式: {dataset}_subset_{num}.json
    pattern = re.compile(r"^(.+)_subset_(\d+)\.json$")
    
    for corpus_file in corpus_dir.glob("*_subset_*.json"):
        match = pattern.match(corpus_file.name)
        if not match:
            continue
        
        dataset = match.group(1)
        num = match.group(2)
        subset_key = f"{dataset}_{num}"
        
        # 检查对应的问题集文件是否存在
        questions_file = questions_dir / f"{dataset}_questions_subset_{num}.json"
        if not questions_file.exists():
            continue
        
        # 添加到映射（使用相对路径）
        corpus_rel = f"./Datasets/Corpus/{corpus_file.name}"
        questions_rel = f"./Datasets/Questions/{questions_file.name}"
        
        subset_paths[subset_key] = {
            "corpus": corpus_rel,
            "questions": questions_rel,
        }
        subset_names.append(subset_key)
    
    # 按数据集名称和数字排序
    subset_names.sort(key=lambda x: (x.rsplit('_', 1)[0], int(x.rsplit('_', 1)[1])))
    
    return subset_paths, subset_names


def _get_all_subsets() -> Tuple[Dict[str, Dict[str, str]], Tuple[str, ...]]:
    """
    合并基础数据集和自动发现的子集数据集
    
    Returns:
        (完整路径映射, 完整子集名称元组)
    """
    # 发现子集
    discovered_paths, discovered_names = _discover_subset_datasets()
    
    # 合并路径映射
    all_paths = dict(BASE_SUBSET_DATA_PATHS)
    all_paths.update(discovered_paths)
    
    # 合并名称列表
    all_names = list(BASE_SUBSETS) + discovered_names
    
    return all_paths, tuple(all_names)


# ============================================================================
# 导出的数据结构
# ============================================================================
SUBSET_DATA_PATHS: Dict[str, Dict[str, str]] = {}
UNIFIED_SUBSETS: Tuple[str, ...] = ()


def _initialize():
    """初始化数据结构（延迟加载）"""
    global SUBSET_DATA_PATHS, UNIFIED_SUBSETS
    SUBSET_DATA_PATHS, UNIFIED_SUBSETS = _get_all_subsets()


# 模块加载时初始化
_initialize()


FRAMEWORK_SUPPORTED_SUBSETS: Dict[str, Tuple[str, ...]] = {
    "lightrag": UNIFIED_SUBSETS,
    "clearrag": UNIFIED_SUBSETS,
    "fast-graphrag": UNIFIED_SUBSETS,
    "hipporag2": UNIFIED_SUBSETS,
    "digimon": UNIFIED_SUBSETS,
    "linearrag": UNIFIED_SUBSETS,
}


def get_supported_subsets(framework: str) -> List[str]:
    if framework not in FRAMEWORK_SUPPORTED_SUBSETS:
        raise ValueError(f"Unsupported framework: {framework}")
    return list(FRAMEWORK_SUPPORTED_SUBSETS[framework])


def get_subset_paths(subset: str) -> Tuple[str, str]:
    if subset not in SUBSET_DATA_PATHS:
        raise ValueError(
            f"Unsupported subset: {subset}. Available: {sorted(SUBSET_DATA_PATHS.keys())}"
        )
    mapping = SUBSET_DATA_PATHS[subset]
    return mapping["corpus"], mapping["questions"]


def list_available_subsets() -> None:
    """打印所有可用的子集"""
    print("\n" + "=" * 60)
    print("可用的数据集子集")
    print("=" * 60)
    
    print("\n基础数据集:")
    for name in BASE_SUBSETS:
        print(f"  - {name}")
    
    print("\n自动发现的子集:")
    discovered_paths, discovered_names = _discover_subset_datasets()
    if discovered_names:
        for name in discovered_names:
            print(f"  - {name}")
    else:
        print("  (暂无)")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    list_available_subsets()