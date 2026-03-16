"""
数据集截取脚本

功能：
1. 从 MuSiQue、HotpotQA、2WikiMultihop 数据集中截取指定规模子集
2. 确保问题的依据（evidence）在语料库中存在
3. 过滤语料库，只保留问题涉及的文章
4. 按原始数据集的问题类型比例进行均衡抽取

使用方法：
    python Datasets/subset_datasets.py --dataset musique --num_questions 100
    python Datasets/subset_datasets.py --dataset hotpotqa --num_questions 500
    python Datasets/subset_datasets.py --dataset 2wikimultihop --num_questions 200
    python Datasets/subset_datasets.py --dataset all --num_questions 100

输出文件命名：
    - Corpus/{dataset}_subset_{num_questions}.json
    - Questions/{dataset}_questions_subset_{num_questions}.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 数据集配置
DATASET_CONFIGS = {
    "musique": {
        "corpus_file": "musique.json",
        "questions_file": "musique_questions.json",
        "title_format": "list",  # supporting_paragraph_titles 是列表
    },
    "hotpotqa": {
        "corpus_file": "hotpotqa_distractor.json",
        "questions_file": "hotpotqa_distractor_questions.json",
        "title_format": "supporting_facts",  # supporting_facts.title
    },
    "2wikimultihop": {
        "corpus_file": "2wikimultihop.json",
        "questions_file": "2wikimultihop_questions.json",
        "title_format": "supporting_facts",  # supporting_facts.title
    },
}


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str) -> None:
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"已保存: {path}")


def extract_supporting_titles(question: Dict, dataset_type: str) -> List[str]:
    """
    从问题中提取支持段落标题
    
    Args:
        question: 问题字典
        dataset_type: 数据集类型
    
    Returns:
        去重后的支持段落标题列表
    """
    config = DATASET_CONFIGS.get(dataset_type, {})
    title_format = config.get("title_format", "list")
    
    if title_format == "list":
        # MuSiQue 格式: supporting_paragraph_titles 是列表
        return question.get("supporting_paragraph_titles", [])
    
    elif title_format == "supporting_facts":
        # HotpotQA / 2WikiMultihop 格式: supporting_facts.title
        supporting_facts = question.get("supporting_facts", {})
        titles = supporting_facts.get("title", [])
        # 去重并保持顺序
        seen = set()
        result = []
        for t in titles:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result
    
    else:
        return []


def extract_question_type(question: Dict, dataset_type: str) -> str:
    """
    提取问题的类型标识
    
    Args:
        question: 问题字典
        dataset_type: 数据集类型
    
    Returns:
        问题类型字符串
    """
    # MuSiQue: question_type = "2hop", "3hop", "4hop"
    # HotpotQA: question_type = "bridge", "comparison"
    # 2WikiMultihop: question_type = "compositional", "comparison"
    return question.get("question_type", "unknown")


def analyze_type_distribution(questions: List[Dict], dataset_type: str) -> Dict[str, int]:
    """
    统计原始问题集中的类型分布
    
    Args:
        questions: 问题列表
        dataset_type: 数据集类型
    
    Returns:
        类型 -> 数量 的映射
    """
    distribution = defaultdict(int)
    for q in questions:
        qtype = extract_question_type(q, dataset_type)
        distribution[qtype] += 1
    return dict(distribution)


def validate_question(
    question: Dict,
    corpus_map: Dict[str, Dict],
    dataset_type: str
) -> Tuple[bool, List[str]]:
    """
    验证问题的依据是否在语料库中存在
    
    返回:
        (是否有效, 缺失的标题列表)
    """
    supporting_titles = extract_supporting_titles(question, dataset_type)
    missing_titles = [t for t in supporting_titles if t not in corpus_map]
    return len(missing_titles) == 0, missing_titles


def create_subset(
    questions: List[Dict],
    corpus: List[Dict],
    dataset_type: str,
    num_questions: int,
) -> Tuple[List[Dict], List[Dict], Dict[str, Any]]:
    """
    创建数据集子集，按原始数据集的问题类型比例进行均衡抽取
    
    Args:
        questions: 原始问题列表
        corpus: 原始语料库列表
        dataset_type: 数据集类型
        num_questions: 目标问题数量
    
    Returns:
        (子集问题列表, 子集语料库列表, 统计信息)
    """
    # 构建语料库映射: title -> corpus_item
    corpus_map = {item["title"]: item for item in corpus}
    logger.info(f"原始语料库大小: {len(corpus_map)} 个段落")
    logger.info(f"原始问题数量: {len(questions)}")
    
    # 1. 分析原始类型分布
    type_distribution = analyze_type_distribution(questions, dataset_type)
    total_questions = len(questions)
    
    logger.info(f"原始类型分布: {type_distribution}")
    
    # 2. 按类型分组问题
    questions_by_type: Dict[str, List[Dict]] = defaultdict(list)
    for q in questions:
        qtype = extract_question_type(q, dataset_type)
        questions_by_type[qtype].append(q)
    
    # 3. 计算每个类型的目标数量（按原始比例）
    type_targets: Dict[str, int] = {}
    allocated = 0
    
    # 先按比例分配整数部分
    for qtype, count in sorted(type_distribution.items()):
        ratio = count / total_questions
        target = int(ratio * num_questions)
        type_targets[qtype] = target
        allocated += target
    
    # 剩余名额按比例大小分配给类型
    remaining = num_questions - allocated
    if remaining > 0:
        # 按原始数量降序排列类型，依次分配剩余名额
        sorted_types = sorted(type_distribution.items(), key=lambda x: x[1], reverse=True)
        for i in range(remaining):
            qtype = sorted_types[i % len(sorted_types)][0]
            type_targets[qtype] += 1
    
    logger.info(f"各类型目标数量: {type_targets}")
    
    # 4. 从各类型中选取有效问题
    required_titles: Set[str] = set()
    selected_questions: List[Dict] = []
    type_selected: Dict[str, int] = defaultdict(int)
    invalid_count = 0
    
    for qtype, target_count in type_targets.items():
        type_questions = questions_by_type[qtype]
        selected_from_type = 0
        
        for question in type_questions:
            if selected_from_type >= target_count:
                break
            
            is_valid, missing_titles = validate_question(question, corpus_map, dataset_type)
            
            if is_valid:
                selected_questions.append(question)
                supporting_titles = extract_supporting_titles(question, dataset_type)
                required_titles.update(supporting_titles)
                selected_from_type += 1
            else:
                invalid_count += 1
        
        type_selected[qtype] = selected_from_type
    
    # 5. 如果某些类型选不够，从其他类型补充
    actual_count = len(selected_questions)
    if actual_count < num_questions:
        shortfall = num_questions - actual_count
        logger.info(f"部分类型问题不足，尝试从其他类型补充 {shortfall} 个")
        
        # 找出还有剩余有效问题的类型
        for qtype, target_count in type_targets.items():
            if shortfall <= 0:
                break
            
            type_questions = questions_by_type[qtype]
            already_selected_ids = {q["id"] for q in selected_questions}
            
            for question in type_questions:
                if shortfall <= 0:
                    break
                if question["id"] in already_selected_ids:
                    continue
                
                is_valid, missing_titles = validate_question(question, corpus_map, dataset_type)
                
                if is_valid:
                    selected_questions.append(question)
                    supporting_titles = extract_supporting_titles(question, dataset_type)
                    required_titles.update(supporting_titles)
                    shortfall -= 1
                    type_selected[qtype] += 1
    
    actual_count = len(selected_questions)
    if actual_count < num_questions:
        logger.warning(f"有效问题不足: 目标 {num_questions}, 实际 {actual_count}")
    
    # 6. 过滤语料库
    filtered_corpus = [
        corpus_map[title] for title in required_titles if title in corpus_map
    ]
    
    # 7. 统计选中问题的类型分布
    selected_type_distribution = analyze_type_distribution(selected_questions, dataset_type)
    
    # 8. 统计信息
    stats = {
        "dataset_type": dataset_type,
        "target_questions": num_questions,
        "actual_questions": actual_count,
        "original_corpus_size": len(corpus),
        "filtered_corpus_size": len(filtered_corpus),
        "invalid_questions": invalid_count,
        "required_titles": len(required_titles),
        "original_type_distribution": type_distribution,
        "target_type_distribution": type_targets,
        "actual_type_distribution": selected_type_distribution,
    }
    
    logger.info(f"选中问题: {actual_count}/{num_questions}")
    logger.info(f"选中类型分布: {selected_type_distribution}")
    logger.info(f"过滤后语料库: {len(filtered_corpus)} 个段落")
    logger.info(f"无效问题（依据缺失）: {invalid_count}")
    
    return selected_questions, filtered_corpus, stats


def process_dataset(
    dataset_type: str,
    num_questions: int,
    base_dir: str,
) -> Dict[str, Any]:
    """
    处理单个数据集
    
    Args:
        dataset_type: 数据集类型
        num_questions: 目标问题数量
        base_dir: 基础目录
    
    Returns:
        统计信息
    """
    config = DATASET_CONFIGS.get(dataset_type)
    if not config:
        logger.error(f"不支持的数据集类型: {dataset_type}")
        return {"error": f"Unsupported dataset: {dataset_type}"}
    
    # 确定文件路径
    corpus_dir = os.path.join(base_dir, "Corpus")
    questions_dir = os.path.join(base_dir, "Questions")
    
    corpus_path = os.path.join(corpus_dir, config["corpus_file"])
    questions_path = os.path.join(questions_dir, config["questions_file"])
    
    # 检查文件是否存在
    if not os.path.exists(corpus_path):
        logger.error(f"语料库文件不存在: {corpus_path}")
        return {"error": f"Corpus file not found: {corpus_path}"}
    
    if not os.path.exists(questions_path):
        logger.error(f"问题文件不存在: {questions_path}")
        return {"error": f"Questions file not found: {questions_path}"}
    
    # 加载数据
    logger.info(f"加载数据集: {dataset_type}")
    corpus = load_json(corpus_path)
    questions = load_json(questions_path)
    
    # 创建子集
    selected_questions, filtered_corpus, stats = create_subset(
        questions=questions,
        corpus=corpus,
        dataset_type=dataset_type,
        num_questions=num_questions,
    )
    
    # 添加创建时间
    from datetime import datetime
    stats["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 统一命名格式: {dataset}_subset_{num_questions}.json
    output_corpus_name = f"{dataset_type}_subset_{num_questions}.json"
    output_questions_name = f"{dataset_type}_questions_subset_{num_questions}.json"
    
    output_corpus_path = os.path.join(corpus_dir, output_corpus_name)
    output_questions_path = os.path.join(questions_dir, output_questions_name)
    
    # 更新 corpus_name 以避免工作目录冲突
    # 例如: hotpotqa_distractor -> hotpotqa_500
    subset_corpus_name = f"{dataset_type}_{num_questions}"
    for item in filtered_corpus:
        if "corpus_name" in item:
            item["corpus_name"] = subset_corpus_name
    
    # 更新问题中的 source 字段
    for q in selected_questions:
        if "source" in q:
            q["source"] = subset_corpus_name
    
    # 保存结果
    save_json(filtered_corpus, output_corpus_path)
    save_json(selected_questions, output_questions_path)
    
    stats["output_corpus"] = output_corpus_path
    stats["output_questions"] = output_questions_path
    
    logger.info(f"子集已生成: {dataset_type}_{num_questions}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="截取数据集子集，确保问题依据完整且语料库精简"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["musique", "hotpotqa", "2wikimultihop", "all"],
        default="all",
        help="要处理的数据集 (all=处理全部)"
    )
    parser.add_argument(
        "--num_questions",
        type=int,
        default=100,
        help="目标问题数量"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="基础目录（默认为脚本所在目录）"
    )
    
    args = parser.parse_args()
    
    # 确定基础目录
    base_dir = args.base_dir or str(Path(__file__).parent)
    
    # 确定要处理的数据集
    if args.dataset == "all":
        datasets = ["musique", "hotpotqa", "2wikimultihop"]
    else:
        datasets = [args.dataset]
    
    # 处理数据集
    all_stats = {}
    
    for dataset_type in datasets:
        logger.info("=" * 50)
        logger.info(f"处理 {dataset_type} 数据集")
        logger.info("=" * 50)
        
        stats = process_dataset(
            dataset_type=dataset_type,
            num_questions=args.num_questions,
            base_dir=base_dir,
        )
        all_stats[dataset_type] = stats
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("处理完成汇总")
    print("=" * 60)
    
    for dataset, stats in all_stats.items():
        if "error" in stats:
            print(f"\n{dataset}: {stats['error']}")
        else:
            print(f"\n{dataset}:")
            print(f"  目标问题数: {stats['target_questions']}")
            print(f"  实际问题数: {stats['actual_questions']}")
            print(f"  原始语料库: {stats['original_corpus_size']} 段落")
            print(f"  过滤后语料库: {stats['filtered_corpus_size']} 段落")
            if stats['original_corpus_size'] > 0:
                ratio = stats['filtered_corpus_size'] / stats['original_corpus_size'] * 100
                print(f"  压缩比: {ratio:.1f}%")
            # 显示类型分布
            if 'original_type_distribution' in stats:
                print(f"  原始类型分布: {stats['original_type_distribution']}")
            if 'actual_type_distribution' in stats:
                print(f"  选中类型分布: {stats['actual_type_distribution']}")
            print(f"  输出文件:")
            print(f"    - {stats.get('output_corpus', 'N/A')}")
            print(f"    - {stats.get('output_questions', 'N/A')}")


if __name__ == "__main__":
    main()
