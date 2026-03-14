"""支持事实评估指标"""
from __future__ import annotations

import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_title(title: str) -> str:
    """标准化标题"""
    if not title:
        return ""
    title = unicodedata.normalize('NFKC', title)
    title = title.replace('_', ' ')
    title = title.lower()
    title = ' '.join(title.split())
    return title.strip()


def _safe_sent_id(value: Any, default: int = 0) -> int:
    """安全转换 sent_id"""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _build_sf_set(titles: List[str], sent_ids: List[int]) -> Set[Tuple[str, int]]:
    """构建 (title, sent_id) 支持事实集合"""
    result = set()
    for title, sent_id in zip(titles, sent_ids):
        normalized = normalize_title(title)
        if normalized:
            result.add((normalized, _safe_sent_id(sent_id)))
    return result


def _extract_pred_sf_set(pred_sf: List[List]) -> Set[Tuple[str, int]]:
    """从预测列表提取 (title, sent_id) 集合"""
    result = set()
    for item in pred_sf:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = normalize_title(item[0])
        sent_id = _safe_sent_id(item[1])
        if title:
            result.add((title, sent_id))
    return result


def compute_sf_em(pred_sf: List[List], gold_sf: Dict[str, List]) -> Optional[float]:
    """支持事实精确匹配，无 gold 时返回 None"""
    if not gold_sf or not gold_sf.get("title"):
        return None

    gold_set = _build_sf_set(gold_sf["title"], gold_sf.get("sent_id", []))
    if not gold_set:
        return None

    pred_set = _extract_pred_sf_set(pred_sf)
    return 1.0 if pred_set == gold_set else 0.0


def compute_sf_f1(
    pred_sf: List[List],
    gold_sf: Dict[str, List]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """支持事实 F1，返回 (F1, Precision, Recall)"""
    if not gold_sf or not gold_sf.get("title"):
        return None, None, None

    gold_set = _build_sf_set(gold_sf["title"], gold_sf.get("sent_id", []))
    if not gold_set:
        return None, None, None

    pred_set = _extract_pred_sf_set(pred_sf)
    if not pred_set:
        return 0.0, 0.0, 0.0

    matched = len(pred_set & gold_set)
    precision = matched / len(pred_set)
    recall = matched / len(gold_set)

    if matched == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def compute_joint_em(answer_em: float, sf_em: float) -> float:
    """联合精确匹配: answer_em == 1 AND sf_em == 1"""
    return 1.0 if (answer_em == 1.0 and sf_em == 1.0) else 0.0


def compute_joint_f1(
    answer_precision: float,
    answer_recall: float,
    sf_precision: float,
    sf_recall: float
) -> Tuple[float, float, float]:
    """
    联合 F1

    公式:
    - Joint Precision = answer_precision * sf_precision
    - Joint Recall = answer_recall * sf_recall
    - Joint F1 = 2 * JP * JR / (JP + JR)
    """
    if answer_precision <= 0 or sf_precision <= 0:
        return 0.0, 0.0, 0.0
    if answer_recall <= 0 or sf_recall <= 0:
        return 0.0, 0.0, 0.0

    joint_precision = answer_precision * sf_precision
    joint_recall = answer_recall * sf_recall

    if joint_precision == 0 or joint_recall == 0:
        return 0.0, joint_precision, joint_recall

    joint_f1 = 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    return joint_f1, joint_precision, joint_recall


def compute_multihop_scores(
    pred_answer: str,
    gold_answer: str,
    pred_sf: Optional[List[List]],
    gold_sf: Optional[Dict[str, List]],
    pred_triples: Optional[List[List]],
    gold_triples: Optional[List[List]]
) -> Dict[str, Optional[float]]:
    """计算多跳 QA 完整指标集"""
    from .answer_metrics import compute_answer_em, compute_answer_f1
    from .triple_recall import compute_triple_f1

    answer_em = compute_answer_em(pred_answer, gold_answer)
    answer_f1, answer_prec, answer_rec = compute_answer_f1(pred_answer, gold_answer)

    if pred_sf is not None and gold_sf is not None and gold_sf.get("title"):
        sf_em = compute_sf_em(pred_sf, gold_sf)
        sf_f1, sf_prec, sf_rec = compute_sf_f1(pred_sf, gold_sf)
    else:
        sf_em = sf_f1 = sf_prec = sf_rec = None

    if sf_em is not None:
        joint_em = compute_joint_em(answer_em, sf_em)
        if sf_prec is not None and sf_rec is not None:
            joint_f1, joint_prec, joint_rec = compute_joint_f1(
                answer_prec, answer_rec, sf_prec, sf_rec
            )
        else:
            joint_f1 = joint_prec = joint_rec = None
    else:
        joint_em = joint_f1 = joint_prec = joint_rec = None

    if pred_triples is not None and gold_triples is not None and gold_triples:
        triple_f1, triple_prec, triple_rec = compute_triple_f1(pred_triples, gold_triples)
    else:
        triple_f1 = triple_prec = triple_rec = None

    return {
        "answer_em": answer_em,
        "answer_f1": answer_f1,
        "answer_precision": answer_prec,
        "answer_recall": answer_rec,
        "sf_em": sf_em,
        "sf_f1": sf_f1,
        "sf_precision": sf_prec,
        "sf_recall": sf_rec,
        "joint_em": joint_em,
        "joint_f1": joint_f1,
        "joint_precision": joint_prec,
        "joint_recall": joint_rec,
        "triple_f1": triple_f1,
        "triple_precision": triple_prec,
        "triple_recall": triple_rec,
    }