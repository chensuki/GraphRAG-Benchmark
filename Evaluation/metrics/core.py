"""
核心评估计算模块

统一所有指标计算的基础函数，消除代码重复，确保一致性和公平性。

设计原则：
1. DRY - 所有标准化和基础计算只定义一次
2. 公平性第一 - 统一的算法，可配置的行为
3. 正确性 - 遵循官方实现（HotpotQA, MuSiQue, 2WikiMultihop）

参考：
- HotpotQA: https://github.com/hotpotqa/hotpot
- MuSiQue: https://github.com/StonyBrookNLP/musique
- 2WikiMultihop: https://github.com/Alab-NII/2wikimultihop
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

T = TypeVar('T')


# ============================================================================
# 标准化函数
# ============================================================================

def normalize_answer(text: str, language: str = "en") -> str:
    """
    标准化答案文本

    统一的处理流程：大小写 -> 冠词 -> 标点 -> 空白

    Args:
        text: 原始文本
        language: 语言代码（"en" 英文，"zh" 中文）

    Returns:
        标准化后的文本

    Example:
        >>> normalize_answer("The Eiffel Tower")
        'eiffel tower'
    """
    if not text:
        return ""

    # 统一 Unicode
    text = unicodedata.normalize('NFKC', str(text))

    # 转小写
    text = text.lower()

    # 去除冠词（仅英文）
    if language == "en":
        text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # 去除标点
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)

    # 修复空白
    text = ' '.join(text.split())

    return text.strip()


def normalize_title(title: str) -> str:
    """
    标准化标题（用于支持事实匹配）

    处理流程：Unicode -> 下划线 -> 大小写 -> 空白

    Args:
        title: 原始标题

    Returns:
        标准化后的标题

    Example:
        >>> normalize_title("Eiffel_Tower")
        'eiffel tower'
    """
    if not title:
        return ""

    # 统一 Unicode
    title = unicodedata.normalize('NFKC', str(title))

    # 下划线转空格（Wikipedia 标题风格）
    title = title.replace('_', ' ')

    # 转小写
    title = title.lower()

    # 修复空白
    title = ' '.join(title.split())

    return title.strip()


def normalize_text(text: str) -> str:
    """
    通用文本标准化（用于三元组等）

    处理流程：Unicode -> 中文标点 -> 下划线 -> 大小写 -> 空白

    Args:
        text: 原始文本

    Returns:
        标准化后的文本
    """
    if not text:
        return ""

    text = unicodedata.normalize('NFKC', str(text))

    # 中文标点转英文
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


# ============================================================================
# 基础计算函数
# ============================================================================

def compute_f1_from_tokens(
    pred_tokens: List[str],
    gold_tokens: List[str]
) -> Tuple[float, float, float]:
    """
    从 token 列表计算 F1

    使用 Counter 处理重复词，确保公平性

    Args:
        pred_tokens: 预测的 token 列表
        gold_tokens: 标准的 token 列表

    Returns:
        (F1, Precision, Recall) 元组

    Note:
        遵循 MuSiQue 官方实现，空答案处理：
        - 预测和标准都为空 → F1 = 1.0
        - 只有一个为空 → F1 = 0.0
    """
    # 空答案处理（MuSiQue 官方逻辑）
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0  # 空对空，完全匹配
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0  # 只有一个为空，不匹配

    # 使用 Counter 处理重复词（公平性关键）
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


def compute_set_f1(
    pred_set: Set[Any],
    gold_set: Set[Any]
) -> Tuple[float, float, float]:
    """
    从集合计算 F1

    Args:
        pred_set: 预测集合
        gold_set: 标准集合

    Returns:
        (F1, Precision, Recall) 元组
    """
    if not gold_set:
        return 0.0, 0.0, 0.0

    if not pred_set:
        return 0.0, 0.0, 0.0

    matched = len(pred_set & gold_set)

    if matched == 0:
        return 0.0, 0.0, 0.0

    precision = matched / len(pred_set)
    recall = matched / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall)

    return f1, precision, recall


# ============================================================================
# 单答案评估（用于 metric_max_over_ground_truths）
# ============================================================================

# 特殊答案类型（HotpotQA/MuSiQue 官方定义）
SPECIAL_ANSWERS = {'yes', 'no', 'noanswer', 'unknown'}


def compute_em_single(
    prediction: str,
    gold: str,
    normalize: bool = True
) -> float:
    """
    单答案精确匹配

    Args:
        prediction: 预测答案
        gold: 标准答案
        normalize: 是否标准化

    Returns:
        1.0 如果匹配，否则 0.0
    """
    if not prediction or not gold:
        return 0.0

    if normalize:
        pred_norm = normalize_answer(prediction)
        gold_norm = normalize_answer(gold)
    else:
        pred_norm = prediction.strip().lower()
        gold_norm = gold.strip().lower()

    return 1.0 if pred_norm == gold_norm else 0.0


def compute_f1_single(
    prediction: str,
    gold: str,
    handle_special: bool = True,
    normalize: bool = True
) -> Tuple[float, float, float]:
    """
    单答案 F1 计算

    Args:
        prediction: 预测答案
        gold: 标准答案
        handle_special: 是否特殊处理 yes/no/noanswer（公平性关键）
        normalize: 是否标准化

    Returns:
        (F1, Precision, Recall) 元组

    Note:
        遵循 MuSiQue 官方实现，空答案处理：
        - 预测和标准都为空 → F1 = 1.0
        - 只有一个为空 → F1 = 0.0
    """
    # 标准化处理
    if normalize:
        pred_norm = normalize_answer(prediction) if prediction else ""
        gold_norm = normalize_answer(gold) if gold else ""
    else:
        pred_norm = prediction.strip().lower() if prediction else ""
        gold_norm = gold.strip().lower() if gold else ""

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    # 注意：这里调用 compute_f1_from_tokens，它已正确处理空对空情况
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0  # 空对空，完全匹配
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0  # 只有一个为空，不匹配

    # 公平性关键：yes/no 与其他答案不应有部分匹配
    if handle_special:
        pred_str = ' '.join(pred_tokens)
        gold_str = ' '.join(gold_tokens)

        # 如果预测或标准是特殊答案，且两者不相同，返回 0
        if pred_str in SPECIAL_ANSWERS and pred_str != gold_str:
            return 0.0, 0.0, 0.0
        if gold_str in SPECIAL_ANSWERS and pred_str != gold_str:
            return 0.0, 0.0, 0.0

    return compute_f1_from_tokens(pred_tokens, gold_tokens)


# ============================================================================
# 别名处理（metric_max_over_ground_truths）
# ============================================================================

def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], T],
    prediction: str,
    ground_truths: List[str]
) -> T:
    """
    在所有标准答案上计算指标，返回最大值

    这是 MuSiQue/HotpotQA 官方处理 answer_aliases 的标准方法
    公平性关键：确保所有别名都被公平对待

    Args:
        metric_fn: 指标函数，接受 (prediction, ground_truth) 返回标量或元组
        prediction: 预测答案
        ground_truths: 标准答案列表（包含主答案和别名）

    Returns:
        所有标准答案上的最大指标值

    Example:
        >>> em = metric_max_over_ground_truths(compute_em_single, "NYC", ["New York City", "NYC"])
        >>> # 返回 1.0，因为 "NYC" 匹配了别名
    """
    # 处理空列表
    if not ground_truths:
        result = metric_fn("", "")
        if isinstance(result, tuple):
            return type(result)(0.0 for _ in result)
        return 0.0

    scores = []
    for gold in ground_truths:
        if gold:  # 跳过空字符串
            score = metric_fn(prediction, gold)
            scores.append(score)

    if not scores:
        result = metric_fn("", "")
        if isinstance(result, tuple):
            return type(result)(0.0 for _ in result)
        return 0.0

    # 处理元组返回值（如 F1 返回 (f1, precision, recall)）
    if isinstance(scores[0], tuple):
        # 取第一个元素（F1）最大的一组
        best_idx = max(range(len(scores)), key=lambda i: scores[i][0])
        return scores[best_idx]
    else:
        return max(scores)


# ============================================================================
# 统一的答案评估接口
# ============================================================================

def compute_answer_scores(
    prediction: str,
    answer: str,
    answer_aliases: Optional[List[str]] = None,
    handle_special: bool = True
) -> Dict[str, float]:
    """
    统一的答案评估接口

    计算答案的 EM 和 F1 指标，支持别名处理

    Args:
        prediction: 预测答案
        answer: 主标准答案
        answer_aliases: 答案别名列表（可选）
        handle_special: 是否特殊处理 yes/no/noanswer

    Returns:
        包含 em, f1, precision, recall 的字典

    Example:
        >>> compute_answer_scores("NYC", "New York City", ["NYC", "The Big Apple"])
        {'em': 1.0, 'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
    """
    # 构建所有标准答案
    ground_truths = [answer] if answer else []
    if answer_aliases:
        ground_truths.extend([a for a in answer_aliases if a])

    if not ground_truths:
        return {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # EM
    em = metric_max_over_ground_truths(
        lambda p, g: compute_em_single(p, g),
        prediction,
        ground_truths
    )

    # F1
    f1, precision, recall = metric_max_over_ground_truths(
        lambda p, g: compute_f1_single(p, g, handle_special=handle_special),
        prediction,
        ground_truths
    )

    return {
        "em": em,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================================
# 支持事实评估
# ============================================================================

def compute_supporting_facts_scores(
    pred_items: List[Any],
    gold_items: List[Any],
    level: str = "sentence"
) -> Dict[str, Optional[float]]:
    """
    统一的支持事实评估接口

    支持两种级别：
    - sentence: HotpotQA 格式，需要句子索引
    - paragraph: MuSiQue 格式，只有标题

    Args:
        pred_items: 预测的支持事实
            - sentence 级别: List[List[title, sent_id]]
            - paragraph 级别: List[str] (标题列表)
        gold_items: 标准支持事实
            - sentence 级别: Dict{"title": [...], "sent_id": [...]}
            - paragraph 级别: List[str] (标题列表)
        level: 评估级别 "sentence" 或 "paragraph"

    Returns:
        包含 em, f1, precision, recall 的字典
    """
    if level == "paragraph":
        return _compute_paragraph_sf_scores(pred_items, gold_items)
    else:
        return _compute_sentence_sf_scores(pred_items, gold_items)


def _compute_paragraph_sf_scores(
    pred_titles: List[str],
    gold_titles: List[str]
) -> Dict[str, Optional[float]]:
    """段落级别支持事实评估"""
    if not gold_titles:
        return {"em": None, "f1": None, "precision": None, "recall": None}

    pred_set = set(normalize_title(t) for t in pred_titles if t)
    gold_set = set(normalize_title(t) for t in gold_titles if t)

    if not gold_set:
        return {"em": None, "f1": None, "precision": None, "recall": None}

    # EM
    em = 1.0 if pred_set == gold_set else 0.0

    # F1
    f1, precision, recall = compute_set_f1(pred_set, gold_set)

    return {
        "em": em,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def _compute_sentence_sf_scores(
    pred_sf: List[List],
    gold_sf: Dict[str, List]
) -> Dict[str, Optional[float]]:
    """句子级别支持事实评估"""
    if not gold_sf or not gold_sf.get("title"):
        return {"em": None, "f1": None, "precision": None, "recall": None}

    # 构建标准集合
    gold_titles = gold_sf["title"]
    gold_sent_ids = gold_sf.get("sent_id", [])

    gold_set = set()
    for title, sent_id in zip(gold_titles, gold_sent_ids):
        normalized = normalize_title(title)
        if normalized:
            gold_set.add((normalized, int(sent_id) if sent_id is not None else 0))

    if not gold_set:
        return {"em": None, "f1": None, "precision": None, "recall": None}

    # 构建预测集合
    pred_set = set()
    for item in pred_sf:
        if not isinstance(item, list) or len(item) < 2:
            continue
        title = normalize_title(item[0])
        sent_id = int(item[1]) if len(item) > 1 and item[1] is not None else 0
        if title:
            pred_set.add((title, sent_id))

    # EM
    em = 1.0 if pred_set == gold_set else 0.0

    # F1
    f1, precision, recall = compute_set_f1(pred_set, gold_set)

    return {
        "em": em,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================================
# 联合指标
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
    计算联合指标（答案 + 支持事实）

    使用 HotpotQA 官方公式:
    - Joint EM = Answer EM == 1 AND SF EM == 1
    - Joint Precision = Answer_P * SF_P
    - Joint Recall = Answer_R * SF_R
    - Joint F1 = 2 * JP * JR / (JP + JR)

    公平性关键：乘法公式确保答案和支持事实的贡献均衡

    Args:
        answer_em/f1/precision/recall: 答案指标
        sf_em/f1/precision/recall: 支持事实指标

    Returns:
        包含 joint_em, joint_f1, joint_precision, joint_recall 的字典
    """
    # Joint EM
    joint_em = 1.0 if (answer_em == 1.0 and sf_em == 1.0) else 0.0

    # Joint Precision/Recall
    joint_precision = answer_precision * sf_precision
    joint_recall = answer_recall * sf_recall

    # Joint F1
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
# 三元组评估
# ============================================================================

def compute_triple_scores(
    pred_triples: List[List],
    gold_triples: List[List]
) -> Dict[str, Optional[float]]:
    """
    计算三元组评估指标

    Args:
        pred_triples: 预测的三元组 [[s, r, o], ...]
        gold_triples: 标准三元组 [[s, r, o], ...]

    Returns:
        包含 em, f1, precision, recall 的字典
    """
    if not gold_triples:
        return {"em": None, "f1": None, "precision": None, "recall": None}

    # 构建标准集合
    gold_set = set()
    for triple in gold_triples:
        if not isinstance(triple, list) or len(triple) < 3:
            continue
        s = normalize_triple_element(triple[0])
        r = normalize_triple_element(triple[1])
        o = normalize_triple_element(triple[2])
        if s and r and o:
            gold_set.add((s, r, o))

    if not gold_set:
        return {"em": None, "f1": None, "precision": None, "recall": None}

    # 构建预测集合
    pred_set = set()
    if pred_triples:
        for triple in pred_triples:
            if not isinstance(triple, list) or len(triple) < 3:
                continue
            s = normalize_triple_element(triple[0])
            r = normalize_triple_element(triple[1])
            o = normalize_triple_element(triple[2])
            if s and r and o:
                pred_set.add((s, r, o))

    # EM
    em = 1.0 if pred_set == gold_set else 0.0

    # F1
    f1, precision, recall = compute_set_f1(pred_set, gold_set)

    return {
        "em": em,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ============================================================================
# 推理步骤评估
# ============================================================================

def compute_reasoning_step_scores(
    pred_steps: List[Dict[str, Any]],
    gold_steps: List[Dict[str, Any]],
    handle_special: bool = True
) -> Dict[str, Optional[float]]:
    """
    计算推理步骤评估指标

    评估中间答案的正确性

    Args:
        pred_steps: 预测的推理步骤 [{"step": 1, "answer": "..."}, ...]
        gold_steps: 标准推理步骤
        handle_special: 是否特殊处理 yes/no

    Returns:
        包含 accuracy, f1 的字典
    """
    if not gold_steps:
        return {"accuracy": None, "f1": None}

    # 按步骤编号对齐
    gold_by_step = {s.get("step", i): s for i, s in enumerate(gold_steps)}
    pred_by_step = {s.get("step", i): s for i, s in enumerate(pred_steps or [])}

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
        f1, _, _ = compute_f1_single(pred_answer, gold_answer, handle_special=handle_special)
        total_f1 += f1
        matched_steps += 1

    if matched_steps == 0:
        return {"accuracy": None, "f1": None}

    return {
        "accuracy": correct_steps / matched_steps,
        "f1": total_f1 / matched_steps
    }


# ============================================================================
# 综合评估函数
# ============================================================================

def compute_multihop_scores(
    pred_answer: str,
    gold_answer: str,
    answer_aliases: Optional[List[str]] = None,
    pred_supporting: Optional[List[Any]] = None,
    gold_supporting: Optional[List[Any]] = None,
    supporting_level: str = "paragraph",
    pred_triples: Optional[List[List]] = None,
    gold_triples: Optional[List[List]] = None,
    pred_reasoning_steps: Optional[List[Dict]] = None,
    gold_reasoning_steps: Optional[List[Dict]] = None,
    handle_special: bool = True
) -> Dict[str, Optional[float]]:
    """
    多跳问答综合评估

    计算答案、支持事实、联合、三元组、推理步骤等全部指标

    Args:
        pred_answer: 预测答案
        gold_answer: 标准答案
        answer_aliases: 答案别名（可选）
        pred_supporting: 预测的支持事实
        gold_supporting: 标准支持事实
        supporting_level: 支持事实级别 "paragraph" 或 "sentence"
        pred_triples: 预测的三元组
        gold_triples: 标准三元组
        pred_reasoning_steps: 预测的推理步骤
        gold_reasoning_steps: 标准推理步骤
        handle_special: 是否特殊处理 yes/no

    Returns:
        完整的评估指标字典
    """
    results = {}

    # 答案指标
    answer_scores = compute_answer_scores(
        pred_answer, gold_answer, answer_aliases, handle_special
    )
    results.update({
        "answer_em": answer_scores["em"],
        "answer_f1": answer_scores["f1"],
        "answer_precision": answer_scores["precision"],
        "answer_recall": answer_scores["recall"]
    })

    # 支持事实指标
    if gold_supporting is not None and pred_supporting is not None:
        sf_scores = compute_supporting_facts_scores(
            pred_supporting, gold_supporting, supporting_level
        )
        results.update({
            "sf_em": sf_scores["em"],
            "sf_f1": sf_scores["f1"],
            "sf_precision": sf_scores["precision"],
            "sf_recall": sf_scores["recall"]
        })

        # 联合指标
        if sf_scores["em"] is not None:
            joint_scores = compute_joint_scores(
                answer_scores["em"],
                answer_scores["f1"],
                answer_scores["precision"],
                answer_scores["recall"],
                sf_scores["em"],
                sf_scores["f1"],
                sf_scores["precision"],
                sf_scores["recall"]
            )
            results.update(joint_scores)
        else:
            results.update({
                "joint_em": None, "joint_f1": None,
                "joint_precision": None, "joint_recall": None
            })
    else:
        results.update({
            "sf_em": None, "sf_f1": None,
            "sf_precision": None, "sf_recall": None,
            "joint_em": None, "joint_f1": None,
            "joint_precision": None, "joint_recall": None
        })

    # 三元组指标
    if gold_triples is not None:
        triple_scores = compute_triple_scores(pred_triples or [], gold_triples)
        results.update({
            "triple_em": triple_scores["em"],
            "triple_f1": triple_scores["f1"],
            "triple_precision": triple_scores["precision"],
            "triple_recall": triple_scores["recall"]
        })
    else:
        results.update({
            "triple_em": None, "triple_f1": None,
            "triple_precision": None, "triple_recall": None
        })

    # 推理步骤指标
    if gold_reasoning_steps is not None:
        step_scores = compute_reasoning_step_scores(
            pred_reasoning_steps or [], gold_reasoning_steps, handle_special
        )
        results.update({
            "step_accuracy": step_scores["accuracy"],
            "step_f1": step_scores["f1"]
        })
    else:
        results.update({
            "step_accuracy": None, "step_f1": None
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
        results: 评估结果列表，每个结果需包含 "question_type" 字段

    Returns:
        {"2hop": {...}, "3hop": {...}, "4hop": {...}}

    Note:
        MuSiQue 数据集中 question_type 字段值即为 "2hop", "3hop", "4hop"
        无需从 id 额外提取
    """
    hop_groups = {}

    for result in results:
        # 直接使用 question_type 作为 hop 分类
        question_type = result.get("question_type", "")

        # 检查是否为 hop 类型（2hop, 3hop, 4hop）
        import re
        match = re.match(r'(\d+hop)', question_type)
        if match:
            hop = match.group(1)
        else:
            # 非 hop 类型的问题，跳过 hop 分层
            continue

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
        stratified[f"{hop}"] = scores

    return stratified