"""三元组评估指标"""
from __future__ import annotations

import unicodedata
from typing import Any, List, Optional, Set, Tuple


def normalize_text(text: str) -> str:
    """标准化文本"""
    if not text:
        return ""

    text = unicodedata.normalize('NFKC', text)

    # 统一中文标点为英文
    punctuation_map = {
        '，': ',', '。': '.', '、': ',', '：': ':', '；': ';',
        '？': '?', '！': '!', '"': '"', '"': '"', ''': "'",
        ''': "'", '（': '(', '）': ')', '【': '[', '】': ']',
        '—': '-', '…': '...',
    }
    for cn, en in punctuation_map.items():
        text = text.replace(cn, en)

    text = text.replace('_', ' ')
    text = text.lower()
    text = ' '.join(text.split())

    return text.strip()


def normalize_triple_element(elem: Any) -> str:
    """标准化三元组元素"""
    if elem is None:
        return ""
    return normalize_text(str(elem))


def _make_triple_set(triples: List[List]) -> Set[Tuple[str, str, str]]:
    """构建标准化三元组集合"""
    result = set()
    for triple in triples:
        if not isinstance(triple, list) or len(triple) < 3:
            continue
        s = normalize_triple_element(triple[0])
        r = normalize_triple_element(triple[1])
        o = normalize_triple_element(triple[2])
        if s and r and o:
            result.add((s, r, o))
    return result


def compute_triple_recall(pred_triples: List[List], gold_triples: List[List]) -> Optional[float]:
    """三元组召回率，无 gold 时返回 None"""
    if not gold_triples:
        return None
    if not pred_triples:
        return 0.0

    pred_set = _make_triple_set(pred_triples)
    gold_set = _make_triple_set(gold_triples)

    if not gold_set:
        return None
    if not pred_set:
        return 0.0

    return len(pred_set & gold_set) / len(gold_set)


def compute_triple_precision(pred_triples: List[List], gold_triples: List[List]) -> Optional[float]:
    """三元组精确率，无 pred 时返回 None"""
    if not pred_triples:
        return None
    if not gold_triples:
        return 0.0

    pred_set = _make_triple_set(pred_triples)
    gold_set = _make_triple_set(gold_triples)

    if not pred_set:
        return 0.0
    if not gold_set:
        return 0.0

    return len(pred_set & gold_set) / len(pred_set)


def compute_triple_f1(
    pred_triples: List[List],
    gold_triples: List[List]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """三元组 F1，返回 (F1, Precision, Recall)"""
    precision = compute_triple_precision(pred_triples, gold_triples)
    recall = compute_triple_recall(pred_triples, gold_triples)

    if precision is None or recall is None:
        return None, None, None
    if precision == 0 or recall == 0:
        return 0.0, precision, recall

    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall