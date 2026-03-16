"""
统一评估脚本

核心原则：
1. 公平性第一 - 使用统一的 core.py 计算函数
2. 正确性 - 遵循 HotpotQA/MuSiQue 官方评估方法
3. 支持多种数据集格式（HotpotQA, MuSiQue, 2WikiMultihop, Medical, Novel）
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# 核心评估函数（统一来源）
from Evaluation.metrics import (
    compute_answer_scores,
    compute_supporting_facts_scores,
    compute_joint_scores,
    compute_triple_scores,
    compute_reasoning_step_scores,
    compute_hop_stratified_scores,
    compute_rouge_score,
)


def is_empty(value: Any) -> bool:
    """检查值是否为空"""
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def extract_text_from_context(context: Any) -> str:
    """从 context 提取纯文本"""
    if not context:
        return ""
    if isinstance(context, str):
        return context
    if isinstance(context, list):
        texts = []
        for item in context:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(item.get("content", ""))
        return " ".join(texts)
    return ""


def compute_evidence_coverage(context: Any, evidence: str) -> Optional[float]:
    """
    计算证据覆盖度

    评估检索上下文对证据文本的覆盖程度
    """
    if not evidence:
        return None
    if is_empty(context):
        return 0.0

    context_text = extract_text_from_context(context).lower()
    if not context_text:
        return 0.0

    evidence_words = set(re.findall(r'\w+', evidence.lower()))
    stop_words = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'to', 'of',
        'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'but',
        'or', 'so', 'the', 'that', 'this', 'it', 'its',
    }
    evidence_words = evidence_words - stop_words

    if not evidence_words:
        return None

    covered = sum(1 for word in evidence_words if word in context_text)
    return covered / len(evidence_words)


def detect_dataset_format(item: Dict[str, Any]) -> Dict[str, bool]:
    """
    检测数据集格式

    Returns:
        {
            "is_musique": 是否为 MuSiQue 格式,
            "has_aliases": 是否有答案别名,
            "has_sentence_sf": 是否有句子级别支持事实,
            "has_paragraph_sf": 是否有段落级别支持事实,
            "has_triples": 是否有三元组,
            "has_reasoning_steps": 是否有推理步骤,
        }
    """
    return {
        "is_musique": bool(item.get("supporting_paragraph_titles")),
        "has_aliases": bool(item.get("answer_aliases")),
        "has_sentence_sf": bool(item.get("supporting_facts", {}).get("title")),
        "has_paragraph_sf": bool(item.get("supporting_paragraph_titles")),
        "has_triples": bool(item.get("evidences")),
        "has_reasoning_steps": bool(item.get("reasoning_steps")),
    }


def normalize_reasoning_steps(steps: Any) -> List[Dict[str, Any]]:
    """标准化推理步骤格式"""
    if not steps:
        return []

    if isinstance(steps, list):
        result = []
        for i, step in enumerate(steps):
            if isinstance(step, str):
                # 字符串格式 -> 转换为字典格式
                result.append({"step": i, "answer": step})
            elif isinstance(step, dict):
                # 字典格式 -> 保持不变
                result.append(step)
        return result

    return []


async def evaluate_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估单个样本

    使用统一的 core.py 计算函数，确保公平性
    """
    results = {}

    # 基础统计
    results["has_answer"] = not is_empty(item.get("generated_answer"))
    results["has_context"] = not is_empty(item.get("context"))
    results["has_retrieved_sf"] = bool(item.get("retrieved_supporting_facts") or item.get("retrieved_supporting_titles"))
    results["has_retrieved_triples"] = bool(item.get("retrieved_triples"))

    # 检测数据集格式
    fmt = detect_dataset_format(item)

    # 提取答案
    pred_answer = item.get("generated_answer", "") or ""
    gold_answer = item.get("ground_truth") or item.get("answer", "")
    answer_aliases = item.get("answer_aliases")

    # ========================================
    # 答案评估（统一使用 compute_answer_scores）
    # ========================================
    answer_scores = compute_answer_scores(
        prediction=pred_answer,
        answer=gold_answer,
        answer_aliases=answer_aliases,
        handle_special=True  # 统一处理 yes/no
    )
    results["answer_em"] = answer_scores["em"]
    results["answer_f1"] = answer_scores["f1"]
    results["answer_precision"] = answer_scores["precision"]
    results["answer_recall"] = answer_scores["recall"]

    # ROUGE 评估
    results["rouge_score"] = await compute_rouge_score(pred_answer, gold_answer)

    # ========================================
    # 支持事实评估
    # ========================================
    sf_evaluated = False

    # 段落级别（MuSiQue 格式）
    if fmt["has_paragraph_sf"]:
        gold_titles = item.get("supporting_paragraph_titles", [])
        pred_titles = item.get("retrieved_supporting_titles", [])

        if gold_titles and pred_titles is not None:
            sf_scores = compute_supporting_facts_scores(
                pred_titles, gold_titles, level="paragraph"
            )
            results["sf_em"] = sf_scores["em"]
            results["sf_f1"] = sf_scores["f1"]
            results["sf_precision"] = sf_scores["precision"]
            results["sf_recall"] = sf_scores["recall"]
            sf_evaluated = True

    # 句子级别（HotpotQA 格式）
    if not sf_evaluated and fmt["has_sentence_sf"]:
        gold_sf = item.get("supporting_facts")
        pred_sf = item.get("retrieved_supporting_facts")

        if gold_sf and gold_sf.get("title") and pred_sf:
            sf_scores = compute_supporting_facts_scores(
                pred_sf, gold_sf, level="sentence"
            )
            results["sf_em"] = sf_scores["em"]
            results["sf_f1"] = sf_scores["f1"]
            results["sf_precision"] = sf_scores["precision"]
            results["sf_recall"] = sf_scores["recall"]
            sf_evaluated = True

    # 如果没有支持事实数据，设置为 None
    if not sf_evaluated:
        results["sf_em"] = None
        results["sf_f1"] = None
        results["sf_precision"] = None
        results["sf_recall"] = None

    # ========================================
    # 联合指标
    # ========================================
    if sf_evaluated and results["sf_em"] is not None:
        joint = compute_joint_scores(
            answer_em=results["answer_em"],
            answer_f1=results["answer_f1"],
            answer_precision=results["answer_precision"],
            answer_recall=results["answer_recall"],
            sf_em=results["sf_em"],
            sf_f1=results["sf_f1"],
            sf_precision=results["sf_precision"],
            sf_recall=results["sf_recall"]
        )
        results["joint_em"] = joint["joint_em"]
        results["joint_f1"] = joint["joint_f1"]
        results["joint_precision"] = joint["joint_precision"]
        results["joint_recall"] = joint["joint_recall"]
    else:
        results["joint_em"] = None
        results["joint_f1"] = None
        results["joint_precision"] = None
        results["joint_recall"] = None

    # ========================================
    # 三元组评估
    # ========================================
    gold_triples = item.get("evidences")
    pred_triples = item.get("retrieved_triples")

    if gold_triples and pred_triples:
        triple_scores = compute_triple_scores(pred_triples, gold_triples)
        results["triple_em"] = triple_scores["em"]
        results["triple_f1"] = triple_scores["f1"]
        results["triple_precision"] = triple_scores["precision"]
        results["triple_recall"] = triple_scores["recall"]
    else:
        results["triple_em"] = None
        results["triple_f1"] = None
        results["triple_precision"] = None
        results["triple_recall"] = None

    # ========================================
    # 推理步骤评估
    # ========================================
    gold_steps = normalize_reasoning_steps(item.get("reasoning_steps"))
    pred_steps = normalize_reasoning_steps(item.get("retrieved_reasoning_steps"))

    if gold_steps and pred_steps:
        step_scores = compute_reasoning_step_scores(pred_steps, gold_steps)
        results["step_accuracy"] = step_scores["accuracy"]
        results["step_f1"] = step_scores["f1"]
    else:
        results["step_accuracy"] = None
        results["step_f1"] = None

    # ========================================
    # 检索质量评估
    # ========================================
    context = item.get("context")
    evidence = item.get("evidence", "") or item.get("evidences_text", "")
    if evidence:
        results["evidence_coverage"] = compute_evidence_coverage(context, evidence)

    return results


def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析数据集字段情况"""
    total = len(data)

    def count_ratio(key: str, check_func=None) -> float:
        if check_func:
            return sum(1 for item in data if check_func(item)) / total
        return sum(1 for item in data if item.get(key)) / total

    has_gold_sf = count_ratio("supporting_facts", lambda x: x.get("supporting_facts", {}).get("title"))
    has_gold_triples = count_ratio("evidences")
    has_evidence = count_ratio("evidence") + count_ratio("evidences_text")
    has_pred_sf = count_ratio("retrieved_supporting_facts") + count_ratio("retrieved_supporting_titles")
    has_pred_triples = count_ratio("retrieved_triples")

    # MuSiQue 特有字段
    has_supporting_titles = count_ratio("supporting_paragraph_titles")
    has_answer_aliases = count_ratio("answer_aliases")
    has_reasoning_steps = count_ratio("reasoning_steps")

    type_dist = defaultdict(int)
    for item in data:
        qt = item.get("question_type", "unknown")
        type_dist[qt] += 1

    # 判断数据集类型
    is_musique = has_supporting_titles > 0.5

    return {
        "gold_sf": has_gold_sf,
        "gold_triples": has_gold_triples,
        "gold_evidence": has_evidence,
        "pred_sf": has_pred_sf,
        "pred_triples": has_pred_triples,
        "question_types": dict(type_dist),
        "sf_evaluable": (has_gold_sf > 0 and has_pred_sf > 0) or has_supporting_titles > 0,
        "triple_evaluable": has_gold_triples > 0 and has_pred_triples > 0,
        # MuSiQue 特有
        "is_musique": is_musique,
        "supporting_titles": has_supporting_titles,
        "answer_aliases": has_answer_aliases,
        "reasoning_steps": has_reasoning_steps,
    }


def compute_fairness_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算公平性指标"""
    total = len(results)
    success = sum(1 for r in results if r.get("has_answer") and "error" not in r)
    empty_answer = sum(1 for r in results if not r.get("has_answer"))
    empty_context = sum(1 for r in results if not r.get("has_context"))

    return {
        "success_rate": success / total,
        "empty_answer_rate": empty_answer / total,
        "empty_context_rate": empty_context / total,
    }


def group_by_question_type(
    data: List[Dict[str, Any]],
    results: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """按问题类型分组计算指标"""
    groups = defaultdict(list)
    for item, result in zip(data, results):
        qt = item.get("question_type", "unknown")
        groups[qt].append(result)

    metrics = ["answer_em", "answer_f1", "rouge_score", "evidence_coverage"]
    grouped_results = {}
    for qt, items in groups.items():
        scores = {"count": len(items)}
        for metric in metrics:
            values = [r.get(metric) for r in items if r.get(metric) is not None]
            if values:
                scores[metric] = float(np.nanmean(values))
                scores[f"{metric}_n"] = len(values)
        grouped_results[qt] = scores

    return grouped_results


async def evaluate_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """评估整个数据集"""
    all_results = []
    analysis = analyze_dataset(data)

    print(f"\n开始评估 {len(data)} 个样本...")
    if analysis.get("is_musique"):
        print("  [MuSiQue 格式检测]")
        print(f"  supporting_paragraph_titles: {analysis['supporting_titles']*100:.0f}%")
        print(f"  answer_aliases: {analysis['answer_aliases']*100:.0f}%")
        print(f"  reasoning_steps: {analysis['reasoning_steps']*100:.0f}%")
    else:
        print(f"  数据集 supporting_facts: {analysis['gold_sf']*100:.0f}%")
        print(f"  数据集 evidences: {analysis['gold_triples']*100:.0f}%")
    print(f"  数据集 evidence 文本: {analysis['gold_evidence']*100:.0f}%")
    print(f"  框架 retrieved_supporting_facts: {analysis['pred_sf']*100:.0f}%")
    print(f"  框架 retrieved_triples: {analysis['pred_triples']*100:.0f}%")
    print(f"  问题类型: {list(analysis['question_types'].keys())}")
    print(f"\n  评估能力:")
    print(f"    支持事实评估: {'可用' if analysis['sf_evaluable'] else '不可用'}")
    print(f"    三元组评估: {'可用' if analysis['triple_evaluable'] else '不可用'}")

    for i, item in enumerate(data):
        try:
            sample_result = await evaluate_sample(item)
            sample_result["id"] = item.get("id", f"sample_{i}")
            sample_result["question_type"] = item.get("question_type", "unknown")
            all_results.append(sample_result)
            if (i + 1) % 10 == 0:
                print(f"  已完成: {i + 1}/{len(data)}")
        except Exception as e:
            print(f"  样本 {item.get('id', i)} 评估失败: {e}")
            all_results.append({
                "id": item.get("id", f"sample_{i}"),
                "question_type": item.get("question_type", "unknown"),
                "error": str(e)
            })

    fairness = compute_fairness_metrics(all_results)
    grouped = group_by_question_type(data, all_results)

    # Hop 分层评估（MuSiQue）
    hop_stratified = None
    if analysis.get("is_musique"):
        hop_stratified = compute_hop_stratified_scores(all_results)

    metrics = [
        "answer_em", "answer_f1", "answer_precision", "answer_recall",
        "rouge_score", "evidence_coverage",
        "sf_em", "sf_f1", "sf_precision", "sf_recall",
        "joint_em", "joint_f1", "joint_precision", "joint_recall",
        "triple_f1", "triple_precision", "triple_recall",
        "step_accuracy", "step_f1",
    ]

    avg_scores = {}
    metric_counts = {}
    for metric in metrics:
        values = [r.get(metric) for r in all_results if r.get(metric) is not None]
        if values:
            avg_scores[metric] = float(np.nanmean(values))
            metric_counts[metric] = len(values)

    return {
        "total_samples": len(data),
        "valid_samples": len([r for r in all_results if "error" not in r]),
        "fairness_metrics": fairness,
        "metric_counts": metric_counts,
        "average_scores": avg_scores,
        "grouped_scores": grouped,
        "hop_stratified": hop_stratified,
        "detailed_results": all_results,
        "field_analysis": analysis,
    }


def format_report(results: Dict[str, Any]) -> str:
    """格式化结果报告"""
    a = results.get("field_analysis", {})
    f = results.get("fairness_metrics", {})
    counts = results.get("metric_counts", {})

    lines = [
        "=" * 60,
        "GraphRAG-Benchmark 评估报告",
        "=" * 60,
        f"总样本数: {results['total_samples']}",
        f"有效样本数: {results['valid_samples']}",
    ]

    # MuSiQue 格式标记
    if a.get("is_musique"):
        lines.append("")
        lines.append("[MuSiQue 格式]")
        lines.append(f"  supporting_paragraph_titles: {a.get('supporting_titles', 0)*100:.0f}%")
        lines.append(f"  answer_aliases: {a.get('answer_aliases', 0)*100:.0f}%")
        lines.append(f"  reasoning_steps: {a.get('reasoning_steps', 0)*100:.0f}%")

    lines.extend([
        "",
        "公平性指标:",
        f"  success_rate:      {f.get('success_rate', 0)*100:.1f}%",
        f"  empty_answer_rate: {f.get('empty_answer_rate', 0)*100:.1f}%",
        f"  empty_context_rate:{f.get('empty_context_rate', 0)*100:.1f}%",
        "",
        "评估能力:",
        f"  supporting_facts: {'可用' if a.get('sf_evaluable') else '不可用'}",
        f"  triples:          {'可用' if a.get('triple_evaluable') else '不可用'}",
        "",
        "答案指标:",
        "-" * 40,
    ])

    for metric in ["answer_em", "answer_f1", "answer_precision", "answer_recall", "rouge_score"]:
        if metric in results.get("average_scores", {}):
            score = results["average_scores"][metric]
            n = counts.get(metric, 0)
            lines.append(f"  {metric:20s}: {score:.4f} (n={n})")

    if "evidence_coverage" in results.get("average_scores", {}):
        lines.append("")
        lines.append("检索质量指标:")
        lines.append("-" * 40)
        score = results["average_scores"]["evidence_coverage"]
        n = counts.get("evidence_coverage", 0)
        lines.append(f"  evidence_coverage: {score:.4f} (n={n})")

    sf_metrics = ["sf_em", "sf_f1", "joint_em", "joint_f1"]
    if any(m in results.get("average_scores", {}) for m in sf_metrics):
        lines.append("")
        lines.append("支持事实与联合指标:")
        lines.append("-" * 40)
        for metric in sf_metrics:
            if metric in results.get("average_scores", {}):
                score = results["average_scores"][metric]
                n = counts.get(metric, 0)
                lines.append(f"  {metric:20s}: {score:.4f} (n={n})")

    # 推理步骤指标
    step_metrics = ["step_accuracy", "step_f1"]
    if any(m in results.get("average_scores", {}) for m in step_metrics):
        lines.append("")
        lines.append("推理步骤指标:")
        lines.append("-" * 40)
        for metric in step_metrics:
            if metric in results.get("average_scores", {}):
                score = results["average_scores"][metric]
                n = counts.get(metric, 0)
                lines.append(f"  {metric:20s}: {score:.4f} (n={n})")

    triple_metrics = ["triple_f1", "triple_precision", "triple_recall"]
    if any(m in results.get("average_scores", {}) for m in triple_metrics):
        lines.append("")
        lines.append("三元组指标:")
        lines.append("-" * 40)
        for metric in triple_metrics:
            if metric in results.get("average_scores", {}):
                score = results["average_scores"][metric]
                n = counts.get(metric, 0)
                lines.append(f"  {metric:20s}: {score:.4f} (n={n})")

    # Hop 分层评估（MuSiQue）
    hop_stratified = results.get("hop_stratified")
    if hop_stratified:
        lines.append("")
        lines.append("Hop 分层评估 (MuSiQue):")
        lines.append("-" * 40)
        for hop_key in sorted(hop_stratified.keys()):
            hop_data = hop_stratified[hop_key]
            lines.append(f"  [{hop_key}] (n={hop_data.get('count', 0)})")
            for metric in ["answer_em", "answer_f1", "sf_f1", "joint_f1"]:
                if metric in hop_data:
                    lines.append(f"    {metric}: {hop_data[metric]:.4f}")

    grouped = results.get("grouped_scores", {})
    if grouped:
        lines.append("")
        lines.append("按问题类型分组:")
        lines.append("-" * 40)
        for qt, scores in sorted(grouped.items(), key=lambda x: -x[1].get("count", 0)):
            lines.append(f"  [{qt}] (n={scores.get('count', 0)})")
            for metric in ["answer_em", "answer_f1", "rouge_score"]:
                if metric in scores:
                    lines.append(f"    {metric}: {scores[metric]:.4f}")

    lines.append("=" * 60)
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="GraphRAG-Benchmark 统一评估脚本")
    parser.add_argument("--data_file", required=True, help="预测结果文件路径")
    parser.add_argument("--output_file", required=True, help="评估结果输出路径")
    parser.add_argument("--detailed", action="store_true", help="输出详细结果")
    parser.add_argument("--report", action="store_true", help="打印格式化报告")
    args = parser.parse_args()

    if not os.path.exists(args.data_file):
        print(f"错误: 文件不存在 - {args.data_file}")
        return

    print(f"加载数据: {args.data_file}")
    with open(args.data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("错误: 数据格式不正确，应为 JSON 数组")
        return

    results = await evaluate_dataset(data)
    results["source_file"] = args.data_file
    results["evaluated_at"] = datetime.now().isoformat()

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if not args.detailed:
        results.pop("detailed_results", None)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n评估完成: {args.output_file}")

    if args.report:
        print("\n" + format_report(results))
    else:
        print("\n公平性指标:")
        for k, v in results.get("fairness_metrics", {}).items():
            print(f"  {k}: {v*100:.1f}%")
        print("\n平均指标:")
        for metric, score in results.get("average_scores", {}).items():
            print(f"  {metric}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(main())