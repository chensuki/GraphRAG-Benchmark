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
    'clearrag': '#2196F3',      # 蓝色
    'linearrag': '#4CAF50',     # 绿色
    'lightrag': '#FF9800',      # 橙色
    'graphrag': '#9C27B0',      # 紫色
    'default': '#607D8B',       # 灰色
}

# 指标显示名称
METRIC_NAMES = {
    'context_relevancy': 'Context Relevancy',
    'evidence_recall': 'Evidence Recall',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1': 'F1 Score',
    'accuracy': 'Accuracy',
}

# 问题类型显示名称
QUESTION_TYPE_NAMES = {
    'Complex Reasoning': 'Complex\nReasoning',
    'Contextual Summarize': 'Contextual\nSummarize',
    'Creative Generation': 'Creative\nGeneration',
    'Fact Retrieval': 'Fact\nRetrieval',
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


def parse_evaluation_data(filepaths: List[str]) -> List[Dict[str, Any]]:
    """解析多个评估文件"""
    results = []
    for filepath in filepaths:
        data = load_json_file(filepath)
        framework = extract_framework_name(filepath, data)
        dataset = extract_dataset_name(filepath)

        # 移除 framework 键，只保留指标数据
        metrics_data = {k: v for k, v in data.items() if k != 'framework'}

        results.append({
            'filepath': filepath,
            'framework': framework,
            'dataset': dataset,
            'metrics': metrics_data
        })
    return results


def plot_grouped_bar(results: List[Dict], output_path: str, title: str = None):
    """绘制分组柱状图 - 按问题类型分组，比较不同框架"""
    if not results:
        print("No data to plot")
        return

    # 获取所有问题类型和指标
    question_types = list(results[0]['metrics'].keys())
    metrics = list(results[0]['metrics'][question_types[0]].keys())

    frameworks = [r['framework'] for r in results]
    n_frameworks = len(frameworks)
    n_types = len(question_types)
    n_metrics = len(metrics)

    # 为每个指标创建一个子图
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_types)
    width = 0.8 / n_frameworks

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx]

        for i, result in enumerate(results):
            framework = result['framework']
            values = [result['metrics'][qt].get(metric, 0) for qt in question_types]
            color = COLORS.get(framework.lower(), COLORS['default'])

            bars = ax.bar(x + i * width, values, width,
                         label=framework.upper(), color=color, alpha=0.85)

            # 在柱子上添加数值标签
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Question Type', fontsize=11)
        ax.set_ylabel(METRIC_NAMES.get(metric, metric), fontsize=11)
        ax.set_xticks(x + width * (n_frameworks - 1) / 2)
        ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_radar_chart(results: List[Dict], output_path: str, title: str = None):
    """绘制雷达图 - 展示各框架在不同问题类型上的整体表现"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    metrics = list(results[0]['metrics'][question_types[0]].keys())

    # 为每个指标创建雷达图
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(question_types), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for result in results:
            framework = result['framework']
            values = [result['metrics'][qt].get(metric, 0) for qt in question_types]
            values += values[:1]  # 闭合

            color = COLORS.get(framework.lower(), COLORS['default'])
            ax.plot(angles, values, 'o-', linewidth=2, label=framework.upper(), color=color)
            ax.fill(angles, values, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)

        chart_title = f"{METRIC_NAMES.get(metric, metric)} Comparison"
        if title:
            chart_title = f"{title} - {METRIC_NAMES.get(metric, metric)}"
        plt.title(chart_title, fontsize=14, fontweight='bold', pad=20)

        # 保存
        metric_filename = metric.replace('_', '-')
        save_path = output_path.replace('.png', f'_{metric_filename}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved: {save_path}")


def plot_heatmap(results: List[Dict], output_path: str, title: str = None):
    """绘制热力图 - 展示各框架在各问题类型的综合得分"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    metrics = list(results[0]['metrics'][question_types[0]].keys())
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

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(data_matrix, cmap='YlGn', aspect='auto', vmin=0, vmax=1)

    # 设置刻度
    ax.set_xticks(np.arange(len(question_types)))
    ax.set_yticks(np.arange(len(frameworks)))
    ax.set_xticklabels([QUESTION_TYPE_NAMES.get(qt, qt) for qt in question_types], fontsize=10)
    ax.set_yticklabels([f.upper() for f in frameworks], fontsize=11)

    # 在格子中显示数值
    for i in range(len(frameworks)):
        for j in range(len(question_types)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11)

    ax.set_xlabel('Question Type', fontsize=12)
    ax.set_ylabel('Framework', fontsize=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Score', fontsize=11)

    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def plot_comparison_line(results: List[Dict], output_path: str, title: str = None):
    """绘制折线对比图 - 展示不同框架在各问题类型上的性能变化"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    metrics = list(results[0]['metrics'][question_types[0]].keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
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
        ax.set_ylim(0, 1.1)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_summary_table(results: List[Dict], output_path: str):
    """生成汇总表格（Markdown 格式）"""
    if not results:
        return

    question_types = list(results[0]['metrics'].keys())
    metrics = list(results[0]['metrics'][question_types[0]].keys())

    # 表头
    headers = ['Framework'] + question_types + ['Average']
    lines = []
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
        f.write('# Evaluation Results Summary\n\n')
        f.write('\n'.join(lines))
        f.write('\n')

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot evaluation results from JSON files')
    parser.add_argument('--input', '-i', nargs='+', required=True,
                       help='Input JSON file(s) path')
    parser.add_argument('--output', '-o', default='figures/',
                       help='Output directory for figures')
    parser.add_argument('--type', '-t', choices=['bar', 'radar', 'heatmap', 'line', 'all'],
                       default='bar', help='Type of plot to generate')
    parser.add_argument('--title', help='Title for the plot')
    parser.add_argument('--prefix', default='evaluation', help='Output filename prefix')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 解析数据
    results = parse_evaluation_data(args.input)
    print(f"Loaded {len(results)} evaluation result(s)")

    for r in results:
        print(f"  - {r['framework']}: {r['dataset']}")

    # 生成图表
    base_path = os.path.join(args.output, args.prefix)

    if args.type in ['bar', 'all']:
        plot_grouped_bar(results, f'{base_path}_bar.png', args.title)

    if args.type in ['radar', 'all']:
        plot_radar_chart(results, f'{base_path}_radar.png', args.title)

    if args.type in ['heatmap', 'all']:
        plot_heatmap(results, f'{base_path}_heatmap.png', args.title)

    if args.type in ['line', 'all']:
        plot_comparison_line(results, f'{base_path}_line.png', args.title)

    # 生成汇总表格
    generate_summary_table(results, f'{base_path}_summary.md')

    print("\nDone!")


if __name__ == '__main__':
    main()
