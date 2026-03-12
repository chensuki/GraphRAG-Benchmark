from __future__ import annotations

from typing import Dict, List, Tuple


SUBSET_DATA_PATHS: Dict[str, Dict[str, str]] = {
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


UNIFIED_SUBSETS: Tuple[str, ...] = (
    "sample",
    "medical",
    "medical_100",
    "novel",
    "hotpotqa_distractor",
    "hotpotqa_fullwiki",
    "2wikimultihop",
    "musique",
)


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