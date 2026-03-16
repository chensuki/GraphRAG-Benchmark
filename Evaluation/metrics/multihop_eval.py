"""
多跳问答评估指标

支持 MuSiQue、HotpotQA、2WikiMultihop 等多跳问答数据集

核心特性：
1. 支持 answer_aliases 的 metric_max_over_ground_truths 方法
2. 支持段落级别的支持事实评估（MuSiQue 只有标题，没有句子索引）
3. 支持推理步骤评估

参考：
- MuSiQue 官方: https://github.com/StonyBrookNLP/musique
- HotpotQA 官方: https://github.com/hotpotqa/hotpot
- 2WikiMultihop: https://github.com/Alab-NII/2wikimultihop
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

T = TypeVar('T')


def normalize_answer(s: str) -> str:
    """
    标准化答案：去除冠词、标点、大小写

    与 HotpotQA/MuSiQue 官方评测脚本一致
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_title(title: str) -> str:
    """标准化标题（用于支持事实匹配）"""
    if not title:
        return ""
    title = unicodedata.normalize('NFKC', title)
    title = title.replace('_', ' ')
    title = title.lower()
    title = ' '.join(title.split())
    return title.strip()


# ============================================================================
# metric_max_over_ground_truths - 处理多答案/别名的核心函数
# ============================================================================

def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], T],
    prediction: str,
    ground_truths: List[str]
) -> T:
    """
    在所有标准答案上计算指标，返回最大值

    这是 MuSiQue/HotpotQA 官方处理 answer_aliases 的标准方法

    Args:
        metric_fn: 指标函数，接受 (prediction, ground_truth) 返回标量或元组
        prediction: 预测答案
        ground_truths: 标准答案列表（包含主答案和别名）

    Returns:
        所有标准答案上的最大指标值

    Example:
        >>> def em(pred, gold): return 1 if pred == gold else 0
        >>> metric_max_over_ground_truths(em, "NYC", ["New York City", "NYC", "The Big Apple"])
        1
    """
    if not ground_truths:
        # 返回默认值（零值）
        result = metric_fn("", "")
        if isinstance(result, tuple):
            return type(result)(0.0 for _ in result)
        return 0.0

    scores = []
    for gold in ground_truths:
        score = metric_fn(prediction, gold)
        scores.append(score)

    if not scores:
        result = metric_fn("", "")
        if isinstance(result, tuple):
            return type(result)(0.0 for _ in result)
        return 0.0

    # 处理元组返回值（如 F1 返回 (f1, precision, recall)）
    if isinstance(scores[0], tuple):
        # 取 F1 最大的一组
        best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
        return scores[best_idx]
    else:
        return max(scores)


# ============================================================================
# 答案评估指标
# ============================================================================

def compute_em_single(prediction: str, gold: str) -> float:
    """单答案 EM（用于 metric_max_over_ground_truths）"""
    if not prediction or not gold:
        return 0.0
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    return 1.0 if pred_norm == gold_norm else 0.0


def compute_f1_single(prediction: str, gold: str) -> Tuple[float, float, float]:
    """单答案 F1（用于 metric_max_over_ground_truths），返回 (F1, Precision, Recall)"""
    if not prediction or not gold:
        return 0.0, 0.0, 0.0

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    # 特殊处理 yes/no/noanswer
    pred_str = ' '.join(pred_tokens)
    gold_str = ' '.join(gold_tokens)
    if pred_str in ['yes', 'no', 'noanswer'] and pred_str != gold_str:
        return 0.0, 0.0, 0.0
    if gold_str in ['yes', 'no', 'noanswer'] and pred_str != gold_str:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def compute_answer_scores_with_aliases(
    prediction: str,
    answer: str,
    answer_aliases: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算答案评估指标（支持别名）

    使用 metric_max_over_ground_truths 方法处理别名

    Args:
        prediction: 预测答案
        answer: 主标准答案
        answer_aliases: 答案别名列表（可选）

    Returns:
        包含 em, f1, precision, recall 的字典
    """
    # 构建所有标准答案
    ground_truths = [answer] if answer else []
    if answer_aliases:
        ground_truths.extend([a for a in answer_aliases if a])

    if not ground_truths:
        return {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # EM
    em = metric_max_over_ground_truths(compute_em_single, prediction, ground_truths)

    # F1
    f1, precision, recall = metric_max_over_ground_truths(
        compute_f1_single, prediction, ground_truths
    )

    return {
        "em": em,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================================
# 支持事实评估（段落级别）
# ============================================================================

def compute_paragraph_em(
    pred_titles: List[str],
    gold_titles: List[str]
) -> float:
    """
    段落级别支持事实 EM

    适用于 MuSiQue（只有标题，没有句子索引）

    Args:
        pred_titles: 框架检索到的段落标题列表
        gold_titles: 标准支持段落标题列表

    Returns:
        1.0 如果完全匹配，否则 0.0
    """
    if not gold_titles:
        return 0.0

    pred_set = set(normalize_title(t) for t in pred_titles if t)
    gold_set = set(normalize_title(t) for t in gold_titles if t)

    if not gold_set:
        return 0.0

    return 1.0 if pred_set == gold_set else 0.0


def compute_paragraph_f1(
    pred_titles: List[str],
    gold_titles: List[str]
) -> Tuple[float, float, float]:
    """
    段落级别支持事实 F1

    适用于 MuSiQue（只有标题，没有句子索引）

    Args:
        pred_titles: 框架检索到的段落标题列表
        gold_titles: 标准支持段落标题列表

    Returns:
        (F1, Precision, Recall) 元组
    """
    if not gold_titles:
        return 0.0, 0.0, 0.0

    pred_set = set(normalize_title(t) for t in pred_titles if t)
    gold_set = set(normalize_title(t) for t in gold_titles if t)

    if not gold_set:
        return 0.0, 0.0, 0.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    matched = len(pred_set & gold_set)
    precision = matched / len(pred_set)
    recall = matched / len(gold_set)

    if matched == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compute_supporting_facts_scores(
    pred_titles: List[str],
    gold_titles: List[str]
) -> Dict[str, float]:
    """
    计算支持事实评估指标（段落级别）

    Args:
        pred_titles: 框架检索到的段落标题列表
        gold_titles: 标准支持段落标题列表

    Returns:
        包含 sf_em, sf_f1, sf_precision, sf_recall 的字典
    """
    em = compute_paragraph_em(pred_titles, gold_titles)
    f1, precision, recall = compute_paragraph_f1(pred_titles, gold_titles)

    return {
        "sf_em": em,
        "sf_f1": f1,
        "sf_precision": precision,
        "sf_recall": recall
    }


# ============================================================================
# 联合指标（答案 + 支持事实）
# ============================================================================

def compute_joint_scores(
    answer_em: float,
    answer_f1: float,
    answer_precision: float,
    answer_recall: float,
    sf_em: float,
    sf_f1: float,
    sf_precision: float,
    sf_recall: float
) -> Dict[str, float]:
    """
    计算联合指标

    公式（HotpotQA 官方）:
    - Joint EM = answer_em == 1 AND sf_em == 1
    - Joint F1 = 2 * (answer_precision * sf_precision) * (answer_recall * sf_recall)
                 / ((answer_precision * sf_precision) + (answer_recall * sf_recall))
    """
    # Joint EM
    joint_em = 1.0 if (answer_em == 1.0 and sf_em == 1.0) else 0.0

    # Joint F1
    joint_precision = answer_precision * sf_precision
    joint_recall = answer_recall * sf_recall

    if joint_precision <= 0 or joint_recall <= 0:
        joint_f1 = 0.0
    else:
        joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)

    return {
        "joint_em": joint_em,
        "joint_f1": joint_f1,
        "joint_precision": joint_precision,
        "joint_recall": joint_recall
    }


# ============================================================================
# 推理步骤评估（可选）
# ============================================================================

def compute_reasoning_step_scores(
    pred_steps: List[Dict[str, Any]],
    gold_steps: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    计算推理步骤评估指标

    评估中间答案的正确性

    Args:
        pred_steps: 预测的推理步骤 [{"step": 1, "answer": "..."}, ...]
        gold_steps: 标准推理步骤

    Returns:
        包含 step_accuracy, step_em, step_f1 的字典
    """
    if not gold_steps:
        return {"step_accuracy": 0.0, "step_em": 0.0, "step_f1": 0.0}

    if not pred_steps:
        return {"step_accuracy": 0.0, "step_em": 0.0, "step_f1": 0.0}

    # 按步骤编号对齐
    gold_by_step = {s.get("step", i): s for i, s in enumerate(gold_steps)}
    pred_by_step = {s.get("step", i): s for i, s in enumerate(pred_steps)}

    correct_steps = 0
    total_f1 = 0.0
    matched_steps = 0

    for step_num, gold in gold_by_step.items():
        gold_answer = gold.get("answer", "")
        if not gold_answer:
            continue

        pred = pred_by_step.get(step_num, {})
        pred_answer = pred.get("answer", "")

        # EM
        em = compute_em_single(pred_answer, gold_answer)
        if em == 1.0:
            correct_steps += 1

        # F1
        f1, _, _ = compute_f1_single(pred_answer, gold_answer)
        total_f1 += f1
        matched_steps += 1

    if matched_steps == 0:
        return {"step_accuracy": 0.0, "step_em": 0.0, "step_f1": 0.0}

    return {
        "step_accuracy": correct_steps / matched_steps,
        "step_em": correct_steps / matched_steps,
        "step_f1": total_f1 / matched_steps
    }


# ============================================================================
# 综合评估函数
# ============================================================================

def compute_musique_scores(
    pred_answer: str,
    gold_answer: str,
    answer_aliases: Optional[List[str]] = None,
    pred_supporting_titles: Optional[List[str]] = None,
    gold_supporting_titles: Optional[List[str]] = None,
    pred_reasoning_steps: Optional[List[Dict]] = None,
    gold_reasoning_steps: Optional[List[Dict]] = None
) -> Dict[str, float]:
    """
    MuSiQue 完整评估指标集

    计算答案、支持事实、联合、推理步骤等全部指标

    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        answer_aliases: 答案别名（可选）
        pred_supporting_titles: 框架检索的支持段落标题
        gold_supporting_titles: 标准支持段落标题
        pred_reasoning_steps: 预测的推理步骤
        gold_reasoning_steps: 标准推理步骤

    Returns:
        完整的评估指标字典
    """
    results = {}

    # 答案指标
    answer_scores = compute_answer_scores_with_aliases(
        pred_answer, gold_answer, answer_aliases
    )
    results.update({f"answer_{k}": v for k, v in answer_scores.items()})

    # 支持事实指标
    if gold_supporting_titles and pred_supporting_titles is not None:
        sf_scores = compute_supporting_facts_scores(
            pred_supporting_titles, gold_supporting_titles
        )
        results.update(sf_scores)

        # 联合指标
        joint_scores = compute_joint_scores(
            answer_scores["em"],
            answer_scores["f1"],
            answer_scores["precision"],
            answer_scores["recall"],
            sf_scores["sf_em"],
            sf_scores["sf_f1"],
            sf_scores["sf_precision"],
            sf_scores["sf_recall"]
        )
        results.update(joint_scores)
    else:
        results.update({
            "sf_em": None, "sf_f1": None, "sf_precision": None, "sf_recall": None,
            "joint_em": None, "joint_f1": None, "joint_precision": None, "joint_recall": None
        })

    # 推理步骤指标（可选）
    if gold_reasoning_steps and pred_reasoning_steps is not None:
        step_scores = compute_reasoning_step_scores(
            pred_reasoning_steps, gold_reasoning_steps
        )
        results.update(step_scores)
    else:
        results.update({
            "step_accuracy": None, "step_em": None, "step_f1": None
        })

    return results


# ============================================================================
# Hop 分层评估
# ============================================================================

def compute_hop_stratified_scores(
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    按 hop 数分层计算指标

    适用于分析不同复杂度问题的表现

    Args:
        results: 评估结果列表，每个结果需包含 "hop" 字段

    Returns:
        {"2hop": {...}, "3hop": {...}, "4hop": {...}}
    """
    hop_groups = {}

    for result in results:
        # 从 id 或 hop 字段提取 hop 数
        item_id = result.get("id", "")
        if "hop" in item_id:
            match = re.match(r"(\d+)hop", item_id)
            if match:
                hop = match.group(1)
            else:
                hop = "unknown"
        else:
            hop = str(result.get("hop", "unknown"))

        if hop not in hop_groups:
            hop_groups[hop] = []
        hop_groups[hop].append(result)

    metrics = ["answer_em", "answer_f1", "sf_em", "sf_f1", "joint_em", "joint_f1"]
    stratified = {}

    for hop, group in hop_groups.items():
        scores = {"count": len(group)}
        for metric in metrics:
            values = [r.get(metric) for r in group if r.get(metric) is not None]
            if values:
                scores[metric] = sum(values) / len(values)
        stratified[f"{hop}hop"] = scores

    return stratified