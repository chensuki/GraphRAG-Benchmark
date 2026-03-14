"""评估指标模块"""
from __future__ import annotations

# LLM 依赖指标
from .context_relevance import compute_context_relevance
from .answer_accuracy import compute_answer_correctness
from .coverage import compute_coverage_score
from .evidence_recall import compute_evidence_recall
from .faithfulness import compute_faithfulness_score
from .rouge import compute_rouge_score
from .utils import JSONHandler
from .context_relevance_v2 import compute_context_relevance as compute_context_relevance_v2

# 答案指标
from .answer_metrics import (
    compute_answer_em,
    compute_answer_f1,
    compute_answer_accuracy,
    compute_answer_scores,
    compute_token_overlap,
    normalize_answer,
)

# 三元组指标
from .triple_recall import (
    normalize_text,
    normalize_triple_element,
    compute_triple_recall,
    compute_triple_precision,
    compute_triple_f1,
)

# 支持事实指标
from .sf_metrics import (
    normalize_title,
    compute_sf_em,
    compute_sf_f1,
    compute_joint_em,
    compute_joint_f1,
    compute_multihop_scores,
)

__all__ = [
    "compute_context_relevance",
    "compute_answer_correctness",
    "compute_coverage_score",
    "compute_evidence_recall",
    "compute_faithfulness_score",
    "compute_rouge_score",
    "JSONHandler",
    "compute_context_relevance_v2",
    "compute_answer_em",
    "compute_answer_accuracy",
    "compute_answer_f1",
    "compute_answer_scores",
    "compute_token_overlap",
    "normalize_answer",
    "normalize_text",
    "normalize_triple_element",
    "compute_triple_recall",
    "compute_triple_precision",
    "compute_triple_f1",
    "normalize_title",
    "compute_sf_em",
    "compute_sf_f1",
    "compute_joint_em",
    "compute_joint_f1",
    "compute_multihop_scores",
]