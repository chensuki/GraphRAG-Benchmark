from __future__ import annotations

from typing import Dict, List, Tuple


SUBSET_DATA_PATHS: Dict[str, Dict[str, str]] = {
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
    "hotpotqa": {
        "corpus": "./Datasets/Corpus/hotpotqa.parquet",
        "questions": "./Datasets/Questions/hotpotqa_questions.json",
    },
}


UNIFIED_SUBSETS: Tuple[str, ...] = ("medical", "medical_100", "novel", "hotpotqa")


FRAMEWORK_SUPPORTED_SUBSETS: Dict[str, Tuple[str, ...]] = {
    "lightrag": UNIFIED_SUBSETS,
    "clearrag": UNIFIED_SUBSETS,
    "fast-graphrag": UNIFIED_SUBSETS,
    "hipporag2": UNIFIED_SUBSETS,
    "digimon": UNIFIED_SUBSETS,
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
