"""
评估结果可视化脚本
用于解析 JSON 评估结果并生成论文实验图

Usage:
    # 单个文件
    python scripts/plot_evaluation_results.py --input results/evaluations/clearrag/clear_retrieval_medical_100_纯图检索.json

    # 多个文件对比
    python scripts/plot_evaluation_results.py --input results/evaluations/clearrag/clear_retrieval_medical_100_纯图检索.json results/evaluations/linearrag/retrieval/retrieval_Medical_完整数据集.json

    # 指定输出目录和图表类型
    python scripts/plot_evaluation_results.py --input <files...> --output figures/ --type bar
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 定义颜色方案
COLORS = {
    # GraphRAG 框架
    'clearrag': '#2196F3',      # 蓝色
    'linearrag': '#4CAF50',     # 绿色
    'lightrag': '#FF9800',      # 橙色
    'graphrag': '#9C27B0',      # 紫色
    'hipporag': '#E91E63',      # 粉色
    'hipporag2': '#E91E63',     # 粉色
    'fast-graphrag': '#00BCD4', # 青色
    'digimon': '#795548',       # 棕色
    # 别名映射（文件名中可能出现的变体）
    'clear': '#2196F3',         # clearrag 别名
    'linear': '#4CAF50',        # linearrag 别名
    'light': '#FF9800',         # lightrag 别名
    'hippo': '#E91E63',         # hipporag 别名
    'fast': '#00BCD4',          # fast-graphrag 别名
    # 默认
    'default': '#607D8B',       # 灰色
}

# 核心指标分组（用于智能筛选）
CORE_METRICS = {
    'unified': ['answer_f1', 'rouge_score', 'evidence_coverage'],
    'retrieval': ['context_relevancy', 'evidence_recall'],
    'generation': ['rouge_score', 'answer_correctness'],
}

# 指标显示名称
METRIC_NAMES = {
    # 检索指标
    'context_relevancy': 'Context Relevancy',
    'evidence_recall': 'Evidence Recall',
    # 生成指标
    'rouge_score': 'ROUGE-L',
    'answer_correctness': 'Answer Correctness',
    'coverage_score': 'Coverage Score',
    'faithfulness': 'Faithfulness',
    # 统一评估指标
    'answer_em': 'Answer EM',
    'answer_f1': 'Answer F1',
    'answer_precision': 'Answer Precision',
    'answer_recall': 'Answer Recall',
    'evidence_coverage': 'Evidence Coverage',
    'sf_em': 'SF EM',
    'sf_f1': 'SF F1',
    'joint_em': 'Joint EM',
    'joint_f1': 'Joint F1',
    'triple_f1': 'Triple F1',
    'step_accuracy': 'Step Accuracy',
    # 通用
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 Score',
    'accuracy': 'Accuracy',
}

# 问题类型显示名称
QUESTION_TYPE_NAMES = {
    # 标准类型
    'Complex Reasoning': 'Complex\nReasoning',
    'Contextual Summarize': 'Contextual\nSummarize',
    'Creative Generation': 'Creative\nGeneration',
    'Fact Retrieval': 'Fact\nRetrieval',
    # HotpotQA 原始类型
    'bridge': 'Bridge',
    'comparison': 'Comparison',
    # 2WikiMultihop 原始类型
    'compositional': 'Compositional',
    'inference': 'Inference',
    'bridge_comparison': 'Bridge\nComparison',
}


def load_json_file(filepath: str) -> Dict[str, Any]:
    """加载单个 JSON 文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_framework_name(filepath: str, data: Dict) -> str:
    """从文件路径或数据中提取框架名称"""
    if 'framework' in data:
        return data['framework']

    # 从文件路径推断
    path_parts = Path(filepath).parts
    for part in path_parts:
        part_lower = part.lower()
        if 'clearrag' in part_lower:
            return 'clearrag'
        elif 'linearrag' in part_lower:
            return 'linearrag'
        elif 'lightrag' in part_lower:
            return 'lightrag'
        elif 'graphrag' in part_lower:
            return 'graphrag'

    return 'unknown'


def extract_dataset_name(filepath: str) -> str:
    """从文件路径提取数据集名称"""
    filename = Path(filepath).stem
    # 提取 medical, hotpotqa 等数据集名称
    for dataset in ['medical', 'hotpotqa', '2wikimultihop', 'musique']:
        if dataset in filename.lower():
            return dataset.capitalize()
    return 'Unknown'


def detect_format(data: Dict[str, Any]) -> str:
    """检测评估数据的格式类型

    Returns:
        'unified': unified_eval 格式（包含 grouped_scores）
        'retrieval_generation': retrieval_eval/generation_eval 格式（问题类型为顶级键）
    """
    # unified_eval 特有字段
    if 'grouped_scores' in data or 'average_scores' in data:
        return 'unified'

    # retrieval_eval / generation_eval 格式
    # 顶级键应为问题类型，值为指标字典
    for key, value in data.items():
        if isinstance(value, dict):
            # 检查是否包含已知指标
            known_metrics = {
                'context_relevancy', 'evidence_recall',  # retrieval
                'rouge_score', 'answer_correctness', 'coverage_score', 'faithfulness',  # generation
                'answer_em', 'answer_f1', 'rouge_score', 'evidence_coverage',  # unified
            }
            if any(k in known_metrics for k in value.keys()):
                return 'retrieval_generation'

    return 'unknown'


def normalize_metrics_data(data: Dict[str, Any], format_type: str) -> Dict[str, Dict[str, float]]:
    """将不同格式的评估数据标准化为统一结构

    Args:
        data: 原始 JSON 数据
        format_type: 'unified' 或 'retrieval_generation'

    Returns:
        Dict[问题类型, Dict[指标名, 指标值]]
    """
    if format_type == 'unified':
        # 从 grouped_scores 提取
        grouped = data.get('grouped_scores', {})
        result = {}
        for q_type, scores in grouped.items():
            # 过滤非指标字段（如 count, *_n）
            metrics = {
                k: v for k, v in scores.items()
                if not k.endswith('_n') and k != 'count' and isinstance(v, (int, float))
            }
            if metrics:
                result[q_type] = metrics
        return result

    elif format_type == 'retrieval_generation':
        # 已经是标准格式，直接返回
        return {k: v for k, v in data.items() if isinstance(v, dict)}

    return {}


def parse_evaluation_data(filepaths: List[str]) -> List[Dict[str, Any]]:
    """解析多个评估文件，支持多种格式"""
    results = []
    for filepath in filepaths:
        data = load_json_file(filepath)
        framework = extract_framework_name(filepath, data)
        dataset = extract_dataset_name(filepath)

        # 检测格式并标准化
        format_type = detect_format(data)
        metrics_data = normalize_metrics_data(data, format_type)

        if not metrics_data:
            print(f"Warning: No metrics found in {filepath}")
            continue

        results.append({
            'filepath': filepath,
            'framework': framework,
            'dataset': dataset,
            'metrics': metrics_data,
            'format': format_type
        })
    return results


def filter_metrics(available_metrics: List[str], selected_metrics: List[str] = None) -> List[str]:
    """筛选要展示的指标

    Args:
        available_metrics: 数据中可用的指标列表
        selected_metrics: 用户指定的指标列表（可选）

    Returns:
        要展示的指标列表
    """
    if selected_metrics:
        # 用户指定指标，只保留存在的
        return [m for m in selected_metrics if m in available_metrics]

    # 自动选择核心指标
    for metric_type, core in CORE_METRICS.items():
        if any(m in available_metrics for m in core):
            selected = [m for m in core if m in available_metrics]
            if selected:
                return selected

    return available_metrics


def plot_grouped_bar(results: List[Dict], output_path: str, title: str = None,
                      metrics_filter: List[str] = None):
    """绘制分组柱状图 - 按问题类型分组，比较不同框架

    优化布局：指标数 > 3 时使用多行布局
    """
    if not results:
        print("No data to plot")
        return

    # 获取所有问题类型和指标
    question_types = list(results[0]['metrics'].keys())
    all_metrics = list(results[0]['metrics'][question_types[0]].keys())
    metrics = filter_metrics(all_metrics, metrics_filter)

    if not metrics:
        print("No metrics to plot after filtering")
        return

    frameworks = [r['framework'] for r in results]
    n_frameworks = len(frameworks)
    n_types = len(question_types)
    n_metrics = len(metrics)

    # 智能布局：超过3个指标时使用2行
    if n_metrics > 3:
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
    else:
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]

    x = np.arange(n_types)
    width = 0.8 / max(n_frameworks, 1)

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for i, result in enumerate(results):
            framework = result['framework']
            values = [result['metrics'][qt].get(metric, 0) for qt in question_types]
            color = COLORS.get(framework.lower(), COLORS['default'])

            bars = ax.bar(x + i * width, values, width,
                         label=framework.upper() if metric_idx == 0 else "",
                         color=color, alpha=0.85, edgecolor='white', linewidth=0.5)

            # 在柱子上添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height > 0.01:  # 只显示非零值
                    ax.annotate(f'{val:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xlabel('Question Type', fontsize=10)
        ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=10)
        ax.set_xticks(x + width * (n_frameworks - 1) / 2)
        ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 添加全局图例
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(f.lower(), COLORS['default']))
               for f in frameworks]
    labels = [f.upper() for f in frameworks]
    fig.legend(handles, labels, loc='upper center', ncol=len(frameworks),
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(results: List[Dict], output_path: str, title: str = None,
                      metrics_filter: List[str] = None):
    """绘制雷达图 - 展示各框架在不同问题类型上的整体表现"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    all_metrics = list(results[0]['metrics'][question_types[0]].keys())
    metrics = filter_metrics(all_metrics, metrics_filter)

    if not metrics:
        return

    # 为每个指标创建雷达图
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(question_types), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for result in results:
            framework = result['framework']
            values = [result['metrics'][qt].get(metric, 0) for qt in question_types]
            values += values[:1]  # 闭合

            color = COLORS.get(framework.lower(), COLORS['default'])
            ax.plot(angles, values, 'o-', linewidth=2, label=framework.upper(), color=color, markersize=6)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)

        chart_title = f"{METRIC_NAMES.get(metric, metric)} Comparison"
        if title:
            chart_title = f"{title} - {METRIC_NAMES.get(metric, metric)}"
        plt.title(chart_title, fontsize=12, fontweight='bold', pad=15)

        # 保存 - 使用规范命名
        metric_filename = metric.replace('_', '-')
        save_path = output_path.replace('.png', f'_{metric_filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_heatmap(results: List[Dict], output_path: str, title: str = None,
                  metrics_filter: List[str] = None):
    """绘制热力图 - 展示各框架在各问题类型的综合得分"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    all_metrics = list(results[0]['metrics'][question_types[0]].keys())
    metrics = filter_metrics(all_metrics, metrics_filter)

    if not metrics:
        return

    frameworks = [r['framework'] for r in results]

    # 计算平均得分
    data_matrix = []
    for result in results:
        row = []
        for qt in question_types:
            avg_score = np.mean([result['metrics'][qt].get(m, 0) for m in metrics])
            row.append(avg_score)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    fig, ax = plt.subplots(figsize=(8, max(4, len(frameworks) * 1.5)))
    im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(question_types)))
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=10)
    ax.set_yticklabels([f.upper() for f in frameworks], fontsize=11)

    # 在格子中显示数值
    for i in range(len(frameworks)):
        for j in range(len(question_types)):
            val = data_matrix[i, j]
            text_color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.3f}',
                    ha="center", va="center", color=text_color, fontsize=10, fontweight='bold')

    ax.set_xlabel('Question Type', fontsize=11)
    ax.set_ylabel('Framework', fontsize=11)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Score', fontsize=10)

    if title:
        plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_line(results: List[Dict], output_path: str, title: str = None,
                          metrics_filter: List[str] = None):
    """绘制折线对比图 - 展示不同框架在各问题类型上的性能变化"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    all_metrics = list(results[0]['metrics'][question_types[0]].keys())
    metrics = filter_metrics(all_metrics, metrics_filter)

    if not metrics:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
    if n_metrics == 1:
        axes = [axes]

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for result in results:
            framework = result['framework']
            values = [result['metrics'][qt].get(metric, 0) for qt in question_types]
            color = COLORS.get(framework.lower(), COLORS['default'])
            marker = 'o' if framework.lower() in COLORS else 's'

            ax.plot(range(len(question_types)), values, marker=marker, linewidth=2,
                   markersize=8, label=framework.upper(), color=color)

        ax.set_xlabel('Question Type', fontsize=10)
        ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=10)
        ax.set_xticks(range(len(question_types)))
        ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: List[Dict], output_path: str,
                            metrics_filter: List[str] = None):
    """生成汇总表格（Markdown 格式）"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    all_metrics = list(results[0]['metrics'][question_types[0]].keys())
    metrics = filter_metrics(all_metrics, metrics_filter)

    if not metrics:
        return

    # 表头
    headers = ['Framework'] + question_types + ['Average']
    lines = []
    lines.append('# Evaluation Results Summary\n')
    lines.append(f'**Metrics**: {", ".join([METRIC_NAMES.get(m, m) for m in metrics])}\n')
    lines.append('')
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

    # 数据行
    for result in results:
        framework = result['framework'].upper()
        row_values = [framework]
        all_scores = []

        for qt in question_types:
            avg = np.mean([result['metrics'][qt].get(m, 0) for m in metrics])
            row_values.append(f'{avg:.3f}')
            all_scores.append(avg)

        row_values.append(f'{np.mean(all_scores):.3f}')
        lines.append('| ' + ' | '.join(row_values) + ' |')

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        f.write('\n')

    print(f"Saved: {output_path}")


def generate_output_filename(results: List[Dict], chart_type: str, prefix: str = None) -> str:
    """生成规范的输出文件名

    格式: {framework}_{dataset}_{chart_type}.png
    多框架时: multi_{datasets}_{chart_type}.png
    """
    if not results:
        return f"unknown_{chart_type}.png"

    # 提取框架名称
    frameworks = sorted(set(r['framework'].lower() for r in results))
    if len(frameworks) == 1:
        framework_part = frameworks[0]
    else:
        framework_part = 'multi'

    # 提取数据集名称
    datasets = sorted(set(r['dataset'].lower() for r in results if r['dataset'] != 'Unknown'))
    if len(datasets) == 0:
        dataset_part = 'unknown'
    elif len(datasets) == 1:
        dataset_part = datasets[0]
    else:
        dataset_part = '-'.join(datasets[:2])  # 最多显示2个数据集

    # 用户指定的前缀
    if prefix:
        return f"{prefix}_{framework_part}_{dataset_part}_{chart_type}.png"

    return f"{framework_part}_{dataset_part}_{chart_type}.png"


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results from JSON files')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input JSON file(s) path')
    parser.add_argument('--output', '-o', default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--type', '-t', choices=['bar', 'radar', 'heatmap', 'line', 'all'],
                       default='bar', help='Type of plot to generate')
    parser.add_argument('--metrics', '-m', nargs='+', default=None,
                       help='Metrics to display (e.g., answer_f1 rouge_score). Auto-selects core metrics if not specified.')
    parser.add_argument('--title', help='Title for the plot')
    parser.add_argument('--prefix', '-p', default=None,
                       help='Output filename prefix (optional). Auto-generates from framework and dataset if not specified.')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 解析数据
    results = parse_evaluation_data(args.input)
    print(f"Loaded {len(results)} evaluation result(s)")

    for r in results:
        print(f"  - {r['framework']}: {r['dataset']}")

    # 解析指标筛选
    metrics_filter = args.metrics

    # 生成图表
    if args.type in ['bar', 'all']:
        filename = generate_output_filename(results, 'bar', args.prefix)
        plot_grouped_bar(results, os.path.join(args.output, filename), args.title, metrics_filter)

    if args.type in ['radar', 'all']:
        base_filename = generate_output_filename(results, 'radar', args.prefix)
        plot_radar_chart(results, os.path.join(args.output, base_filename), args.title, metrics_filter)

    if args.type in ['heatmap', 'all']:
        filename = generate_output_filename(results, 'heatmap', args.prefix)
        plot_heatmap(results, os.path.join(args.output, filename), args.title, metrics_filter)

    if args.type in ['line', 'all']:
        filename = generate_output_filename(results, 'line', args.prefix)
        plot_comparison_line(results, os.path.join(args.output, filename), args.title, metrics_filter)

    # 生成汇总表格
    summary_filename = generate_output_filename(results, 'summary', args.prefix).replace('.png', '.md')
    generate_summary_table(results, os.path.join(args.output, summary_filename), metrics_filter)

    print("\nDone!")


if __name__ == '__main__':
    main()
