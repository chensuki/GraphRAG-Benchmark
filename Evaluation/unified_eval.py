"""统一评估脚本"""
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

from Evaluation.metrics import (
    compute_answer_em,
    compute_answer_f1,
    compute_answer_accuracy,
    compute_rouge_score,
    compute_triple_f1,
    compute_sf_em,
    compute_sf_f1,
    compute_joint_em,
    compute_joint_f1,
    # 多跳问答评估
    compute_answer_scores_with_aliases,
    compute_supporting_facts_scores,
    compute_musique_scores,
    compute_hop_stratified_scores,
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
    """计算证据覆盖度"""
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


async def evaluate_sample(item: Dict[str, Any]) -> Dict[str, Any]:
    """评估单个样本"""
    results = {}

    results["has_answer"] = not is_empty(item.get("generated_answer"))
    results["has_context"] = not is_empty(item.get("context"))
    results["has_retrieved_sf"] = bool(item.get("retrieved_supporting_facts"))
    results["has_retrieved_triples"] = bool(item.get("retrieved_triples"))

    # 检测数据集类型
    is_musique = bool(item.get("supporting_paragraph_titles"))
    has_aliases = bool(item.get("answer_aliases"))

    pred_answer = item.get("generated_answer", "") or ""
    gold_answer = item.get("ground_truth") or item.get("answer", "")

    # 使用多跳问答评估（支持别名和段落级别支持事实）
    if is_musique or has_aliases:
        # MuSiQue 格式评估
        answer_aliases = item.get("answer_aliases")
        gold_supporting_titles = item.get("supporting_paragraph_titles", [])
        pred_supporting_titles = item.get("retrieved_supporting_titles", [])
        gold_reasoning_steps = item.get("reasoning_steps", [])
        pred_reasoning_steps = item.get("retrieved_reasoning_steps", [])

        musique_scores = compute_musique_scores(
            pred_answer=pred_answer,
            gold_answer=gold_answer,
            answer_aliases=answer_aliases,
            pred_supporting_titles=pred_supporting_titles if pred_supporting_titles else None,
            gold_supporting_titles=gold_supporting_titles,
            pred_reasoning_steps=pred_reasoning_steps if pred_reasoning_steps else None,
            gold_reasoning_steps=gold_reasoning_steps
        )
        results.update(musique_scores)
    else:
        # 传统评估
        results["answer_em"] = compute_answer_em(pred_answer, gold_answer)
        results["answer_accuracy"] = compute_answer_accuracy(pred_answer, gold_answer)
        f1, precision, recall = compute_answer_f1(pred_answer, gold_answer)
        results["answer_f1"] = f1
        results["answer_precision"] = precision
        results["answer_recall"] = recall

    results["rouge_score"] = await compute_rouge_score(pred_answer, gold_answer)

    # 检索质量
    context = item.get("context")
    evidence = item.get("evidence", "") or item.get("evidences_text", "")
    if evidence:
        results["evidence_coverage"] = compute_evidence_coverage(context, evidence)

    # 传统支持事实指标（HotpotQA 格式：需要 sent_id）
    if not is_musique:
        gold_sf = item.get("supporting_facts")
        pred_sf = item.get("retrieved_supporting_facts")

        if gold_sf and gold_sf.get("title") and pred_sf:
            results["sf_em"] = compute_sf_em(pred_sf, gold_sf)
            sf_f1, sf_prec, sf_rec = compute_sf_f1(pred_sf, gold_sf)
            results["sf_f1"] = sf_f1
            results["sf_precision"] = sf_prec
            results["sf_recall"] = sf_rec
            if results.get("sf_em") is not None:
                results["joint_em"] = compute_joint_em(results["answer_em"], results["sf_em"])
            if sf_prec is not None and sf_rec is not None:
                jf1, jp, jr = compute_joint_f1(
                    results.get("answer_precision", 0),
                    results.get("answer_recall", 0),
                    sf_prec, sf_rec
                )
                results["joint_f1"] = jf1
                results["joint_precision"] = jp
                results["joint_recall"] = jr

    # 三元组指标
    gold_triples = item.get("evidences")
    pred_triples = item.get("retrieved_triples")

    if gold_triples and pred_triples:
        triple_f1, triple_prec, triple_rec = compute_triple_f1(pred_triples, gold_triples)
        results["triple_f1"] = triple_f1
        results["triple_precision"] = triple_prec
        results["triple_recall"] = triple_rec

    return results


def analyze_dataset(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析数据集字段情况"""
    total = len(data)

    has_gold_sf = sum(1 for item in data if item.get("supporting_facts")) / total
    has_gold_triples = sum(1 for item in data if item.get("evidences")) / total
    has_evidence = sum(1 for item in data if item.get("evidence")) / total
    has_pred_sf = sum(1 for item in data if item.get("retrieved_supporting_facts")) / total
    has_pred_triples = sum(1 for item in data if item.get("retrieved_triples")) / total

    # MuSiQue 特有字段
    has_supporting_titles = sum(1 for item in data if item.get("supporting_paragraph_titles")) / total
    has_answer_aliases = sum(1 for item in data if item.get("answer_aliases")) / total
    has_reasoning_steps = sum(1 for item in data if item.get("reasoning_steps")) / total

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
        "sf_evaluable": has_gold_sf > 0 and has_pred_sf > 0,
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
    print(f"    支持事实评估: {'可用' if analysis['sf_evaluable'] or analysis.get('is_musique') else '不可用（框架未输出 retrieved_supporting_facts/titles）'}")
    print(f"    三元组评估: {'可用' if analysis['triple_evaluable'] else '不可用（框架未输出 retrieved_triples）'}")

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
        "answer_em", "answer_accuracy", "answer_f1", "answer_precision",
        "answer_recall", "rouge_score", "evidence_coverage",
        "sf_em", "sf_f1", "sf_precision", "sf_recall",
        "joint_em", "joint_f1", "joint_precision", "joint_recall",
        "triple_f1", "triple_precision", "triple_recall",
        # MuSiQue 特有
        "step_accuracy", "step_em", "step_f1",
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
        f"  supporting_facts: {'可用' if a.get('sf_evaluable') or a.get('is_musique') else '不可用'}",
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

    sf_metrics = ["sf_em", "sf_f1", "joint_em"]
    if any(m in results.get("average_scores", {}) for m in sf_metrics):
        lines.append("")
        lines.append("支持事实指标:")
        lines.append("-" * 40)
        for metric in sf_metrics:
            if metric in results.get("average_scores", {}):
                score = results["average_scores"][metric]
                n = counts.get(metric, 0)
                lines.append(f"  {metric:20s}: {score:.4f} (n={n})")

    # MuSiQue 推理步骤指标
    step_metrics = ["step_accuracy", "step_em", "step_f1"]
    if any(m in results.get("average_scores", {}) for m in step_metrics):
        lines.append("")
        lines.append("推理步骤指标 (MuSiQue):")
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