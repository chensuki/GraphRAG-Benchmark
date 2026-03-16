"""
评估指标模块

核心设计原则：
1. 公平性第一 - 所有评估使用统一算法
2. 正确性 - 遵循官方实现（HotpotQA, MuSiQue, 2WikiMultihop）
3. 简洁性 - DRY，单一事实来源

模块结构：
- core.py: 核心计算（唯一来源）
- llm_metrics/: LLM依赖指标
"""
from __future__ import annotations

# ============================================================================
# 核心计算函数（从 core.py 导出）
# ============================================================================
from .core import (
    # 标准化函数
    normalize_answer,
    normalize_title,
    normalize_text,
    normalize_triple_element,

    # 基础计算
    compute_f1_from_tokens,
    compute_set_f1,
    compute_em_single,
    compute_f1_single,
    metric_max_over_ground_truths,

    # 答案评估
    compute_answer_scores,

    # 支持事实评估
    compute_supporting_facts_scores,

    # 联合指标
    compute_joint_scores,

    # 三元组评估
    compute_triple_scores,

    # 推理步骤评估
    compute_reasoning_step_scores,

    # 综合评估
    compute_multihop_scores,
    compute_hop_stratified_scores,

    # 特殊答案常量
    SPECIAL_ANSWERS,
)

# ============================================================================
# LLM 依赖指标
# ============================================================================
from .context_relevance import compute_context_relevance
from .answer_accuracy import compute_answer_correctness
from .coverage import compute_coverage_score
from .evidence_recall import compute_evidence_recall
from .faithfulness import compute_faithfulness_score
from .rouge import compute_rouge_score
from .utils import JSONHandler
from .context_relevance_v2 import compute_context_relevance as compute_context_relevance_v2


__all__ = [
    # 标准化函数
    "normalize_answer",
    "normalize_title",
    "normalize_text",
    "normalize_triple_element",

    # 基础计算
    "compute_f1_from_tokens",
    "compute_set_f1",
    "compute_em_single",
    "compute_f1_single",
    "metric_max_over_ground_truths",

    # 答案评估
    "compute_answer_scores",

    # 支持事实评估
    "compute_supporting_facts_scores",

    # 联合指标
    "compute_joint_scores",

    # 三元组评估
    "compute_triple_scores",

    # 推理步骤评估
    "compute_reasoning_step_scores",

    # 综合评估
    "compute_multihop_scores",
    "compute_hop_stratified_scores",

    # 常量
    "SPECIAL_ANSWERS",

    # LLM 依赖指标
    "compute_context_relevance",
    "compute_answer_correctness",
    "compute_coverage_score",
    "compute_evidence_recall",
    "compute_faithfulness_score",
    "compute_rouge_score",
    "JSONHandler",
    "compute_context_relevance_v2",
]