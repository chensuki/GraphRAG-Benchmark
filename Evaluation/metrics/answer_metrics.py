"""
答案评估指标（可计算，无需LLM）

参考 HotpotQA 官方评测脚本实现
参考 OTHER/评估/Evaluation.py
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Tuple, List


def normalize_answer(s: str) -> str:
    """
    标准化答案：去除冠词、标点、大小写

    参考 HotpotQA 官方评测脚本
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        # 与官方一致：使用 string.punctuation
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_answer_em(predicted: str, gold: str) -> float:
    """
    答案精确匹配

    Args:
        predicted: 框架生成的答案
        gold: 标准答案

    Returns:
        1.0 如果匹配，否则 0.0
    """
    if not predicted or not gold:
        return 0.0

    # 处理多答案情况（用 | 分隔）
    gold_answers = [a.strip() for a in gold.split('|') if a.strip()]
    if not gold_answers:
        return 0.0

    predicted_norm = normalize_answer(predicted)

    for gold_answer in gold_answers:
        if predicted_norm == normalize_answer(gold_answer):
            return 1.0

    return 0.0


def compute_answer_accuracy(predicted: str, gold: str) -> float:
    """
    答案包含匹配（Accuracy）

    与 EM 不同：只要标准答案被包含在预测答案中即可

    参考 HippoRAG 评估脚本: https://github.com/OSU-NLP-Group/HippoRAG

    Args:
        predicted: 框架生成的答案
        gold: 标准答案

    Returns:
        1.0 如果 gold 包含在 predicted 中，否则 0.0
    """
    if not predicted or not gold:
        return 0.0

    predicted_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    return 1.0 if gold_norm in predicted_norm else 0.0


def compute_answer_f1(predicted: str, gold: str) -> Tuple[float, float, float]:
    """
    答案词级 F1（参考 HotpotQA 官方实现）

    Args:
        predicted: 框架生成的答案
        gold: 标准答案

    Returns:
        (F1, Precision, Recall) 元组
    """
    if not predicted or not gold:
        return 0.0, 0.0, 0.0

    # 处理多答案情况（用 | 分隔），取最高分
    gold_answers = [a.strip() for a in gold.split('|') if a.strip()]
    if not gold_answers:
        return 0.0, 0.0, 0.0

    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for gold_answer in gold_answers:
        pred_tokens = normalize_answer(predicted).split()
        gold_tokens = normalize_answer(gold_answer).split()

        if not pred_tokens or not gold_tokens:
            continue

        # 特殊处理 yes/no/noanswer（参考官方）
        pred_str = ' '.join(pred_tokens)
        gold_str = ' '.join(gold_tokens)
        if pred_str in ['yes', 'no', 'noanswer'] and pred_str != gold_str:
            continue
        if gold_str in ['yes', 'no', 'noanswer'] and pred_str != gold_str:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())

        if overlap == 0:
            continue

        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    return best_f1, best_precision, best_recall


def compute_answer_scores(predicted: str, gold: str) -> dict:
    """
    计算完整的答案评估指标集

    Args:
        predicted: 框架生成的答案
        gold: 标准答案

    Returns:
        包含 em, accuracy, f1, precision, recall 的字典
    """
    em = compute_answer_em(predicted, gold)
    accuracy = compute_answer_accuracy(predicted, gold)
    f1, precision, recall = compute_answer_f1(predicted, gold)

    return {
        "em": em,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def compute_token_overlap(predicted: str, gold: str) -> dict:
    """
    计算详细的 token 重叠信息

    Args:
        predicted: 框架生成的答案
        gold: 标准答案

    Returns:
        包含 precision, recall, f1, overlap 的字典
    """
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "overlap": 0, "pred_len": 0, "gold_len": 0}

    pred_tokens = normalize_answer(predicted).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "overlap": 0, "pred_len": len(pred_tokens), "gold_len": len(gold_tokens)}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall) if overlap > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "overlap": overlap,
        "pred_len": len(pred_tokens),
        "gold_len": len(gold_tokens)
    }