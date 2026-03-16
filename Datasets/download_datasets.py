"""
通用数据集下载与转换工具（支持评估格式兼容）

支持的数据集：
- hotpotqa/hotpot_qa (HotpotQA)
- TommyChien/UltraDomain
- framolfese/2WikiMultihopQA
- bdsaglam/musique

特点：
- 输出评估格式（兼容 retrieval_eval.py 和 generation_eval.py）
- 同时生成 parquet 和 json 格式
- 默认从本地目录加载（跳过网络下载）

目录结构：
    GraphRAG-Benchmark/
    ├── Datasets/
    │   ├── raw/                  # 原始数据集（手动下载）
    │   │   ├── hotpot_qa/
    │   │   ├── musique/
    │   │   └── ...
    │   ├── Corpus/               # 解析后语料库
    │   └── Questions/            # 解析后问题集
    └── hf_cache/                 # 网络下载缓存（备用）

使用方法：
    # 从本地目录加载（默认）
    python Datasets/download_datasets.py --datasets hotpotqa

    # 从网络下载（需要 --download 参数）
    python Datasets/download_datasets.py --datasets hotpotqa --download

    # 查看支持的数据集
    python Datasets/download_datasets.py --list
"""
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

import pandas as pd

# HuggingFace 镜像配置
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str                    # 数据集名称
    hf_path: str                 # HuggingFace 数据集路径
    subset: Optional[str]        # 子集名称
    split: str                   # 数据集分割
    description: str             # 描述

    # 标准字段映射
    id_field: str = "id"
    question_field: str = "question"
    answer_field: str = "answer"
    context_field: str = "context"
    evidence_field: str = "supporting_facts"

    # 问题类型默认值
    question_type_default: str = "Complex Reasoning"

    # 上下文处理
    has_nested_context: bool = False
    per_question_corpus: bool = False

    # 额外保留字段（评估格式中使用）
    extra_standard_fields: List[str] = field(default_factory=list)

    # 原始字段保存（原始格式已禁用，保留配置以备扩展）
    preserve_all_original_fields: bool = False


# 支持的数据集配置
DATASET_CONFIGS = {
    "hotpotqa_distractor": DatasetConfig(
        name="hotpotqa_distractor",
        hf_path="hotpotqa/hotpot_qa",
        subset="distractor",
        split="validation",
        description="HotpotQA Distractor: 预填充干扰文档版本",
        has_nested_context=True,
        evidence_field="supporting_facts",
        preserve_all_original_fields=True,
        extra_standard_fields=["type", "level", "supporting_facts"],
    ),
    "hotpotqa_fullwiki": DatasetConfig(
        name="hotpotqa_fullwiki",
        hf_path="hotpotqa/hotpot_qa",
        subset="fullwiki",
        split="validation",
        description="HotpotQA Fullwiki: 完整 Wikipedia 检索版本",
        has_nested_context=True,
        evidence_field="supporting_facts",
        extra_standard_fields=["type", "level", "supporting_facts"],
    ),
    "ultradomain": DatasetConfig(
        name="ultradomain",
        hf_path="TommyChien/UltraDomain",
        subset=None,
        split="train",
        description="UltraDomain/LongBench: 长文本问答数据集",
        id_field="_id",
        question_field="input",
        answer_field="answers",
        context_field="context",
        evidence_field=None,
        has_nested_context=False,
        preserve_all_original_fields=True,
    ),
    "2wikimultihop": DatasetConfig(
        name="2wikimultihop",
        hf_path="framolfese/2WikiMultihopQA",
        subset=None,
        split="validation",
        description="2WikiMultihopQA: 两跳维基百科问答",
        has_nested_context=True,
        evidence_field="supporting_facts",
        preserve_all_original_fields=True,
        extra_standard_fields=["type", "level", "evidences", "supporting_facts"],
    ),
    "musique": DatasetConfig(
        name="musique",
        hf_path="bdsaglam/musique",
        subset="answerable",
        split="validation",
        description="MuSiQue: 多跳问答数据集（仅包含可回答样本）",
        context_field="paragraphs",
        evidence_field="question_decomposition",
        has_nested_context=False,
        preserve_all_original_fields=True,
    ),
}


# 本地数据集目录名称映射（用于处理有 subset 的数据集）
LOCAL_DIR_MAPPING = {
    "hotpotqa_distractor": "hotpot_qa",
    "hotpotqa_fullwiki": "hotpot_qa",
}


def setup_mirror() -> str:
    """设置 HuggingFace 镜像"""
    current = os.environ.get("HF_ENDPOINT", "")
    if current:
        return current

    for mirror in HF_MIRRORS:
        try:
            import urllib.request
            urllib.request.urlopen(mirror, timeout=5)
            os.environ["HF_ENDPOINT"] = mirror
            print(f"[INFO] 使用镜像: {mirror}")
            return mirror
        except:
            continue

    print("[WARN]  使用官方源: https://huggingface.co")
    return "https://huggingface.co"


def check_dataset_exists(output_dir: Path, dataset_name: str) -> bool:
    """检查数据集是否已存在"""
    corpus_parquet = output_dir / "Corpus" / f"{dataset_name}.parquet"
    corpus_json = output_dir / "Corpus" / f"{dataset_name}.json"
    questions_parquet = output_dir / "Questions" / f"{dataset_name}_questions.parquet"
    questions_json = output_dir / "Questions" / f"{dataset_name}_questions.json"

    corpus_exists = corpus_parquet.exists() or corpus_json.exists()
    questions_exists = questions_parquet.exists() or questions_json.exists()

    if corpus_exists and questions_exists:
        print(f"   [FILE] 语料库: {corpus_parquet if corpus_parquet.exists() else corpus_json}")
        print(f"   [FILE] 问题集: {questions_parquet if questions_parquet.exists() else questions_json}")

    return corpus_exists and questions_exists


def load_dataset_with_retry(
    config: DatasetConfig,
    cache_dir: Path,
    max_retries: int = 3
) -> Optional[Any]:
    """带重试的数据集加载"""
    from datasets import load_dataset

    dataset_cache_dir = cache_dir / config.name
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        for mirror in HF_MIRRORS:
            try:
                os.environ["HF_ENDPOINT"] = mirror
                print(f"   尝试从 {mirror} 下载...")

                kwargs = {
                    "path": config.hf_path,
                    "split": config.split,
                    "cache_dir": str(dataset_cache_dir)
                }
                if config.subset:
                    kwargs["name"] = config.subset

                dataset = load_dataset(**kwargs)
                print(f"   [OK] 下载成功: {len(dataset)} 条数据")
                return dataset

            except Exception as e:
                print(f"   [FAIL] 失败: {str(e)[:100]}")
                continue

        if attempt < max_retries - 1:
            print(f"   重试 {attempt + 2}/{max_retries}...")

    return None


def load_dataset_from_local(local_path: Path, config: DatasetConfig) -> Optional[Any]:
    """
    从本地目录加载数据集

    支持 subset 子目录结构，例如：
    - local_path: hotpot_qa/
    - subset: distractor → 从 hotpot_qa/distractor/ 加载
    - subset: fullwiki → 从 hotpot_qa/fullwiki/ 加载

    支持 subset 文件名模式，例如：
    - local_path: musique/
    - subset: answerable → 加载 musique_ans_v1.0_dev.jsonl
    - subset: default → 加载 musique_full_v1.0_dev.jsonl
    """
    from datasets import load_dataset, Dataset

    local_path = Path(local_path)

    # 如果指定了 subset，尝试从 local_path/subset 加载
    if config.subset:
        subset_path = local_path / config.subset
        if subset_path.exists() and subset_path.is_dir():
            local_path = subset_path

    if not local_path.exists():
        print(f"   [FAIL] 本地路径不存在: {local_path}")
        return None

    print(f"   [DIR] 从本地加载: {local_path}")

    if local_path.is_dir():
        split = config.split

        # 统一切分名称映射（validation -> dev）
        split_mapping = {"train": "train", "validation": "dev"}
        actual_split = split_mapping.get(config.split, config.split)

        # MuSiQue 特殊处理：根据 subset 选择正确的文件
        if config.name == "musique" and config.subset:
            # subset="answerable" 对应 musique_ans_v1.0_*.jsonl
            # subset="default" 对应 musique_full_v1.0_*.jsonl
            subset_prefix = "full" if config.subset == "default" else "ans"
            split_patterns = [
                f"**/musique_{subset_prefix}_v1.0_{actual_split}.jsonl",
                f"**/{actual_split}*.parquet",
                f"**/{actual_split}/*.parquet",
                f"**/{actual_split}*.arrow",
                f"**/{actual_split}/*.arrow",
            ]
        else:
            split_patterns = [
                f"**/{split}*.parquet",
                f"**/{split}/*.parquet",
                f"**/{split}*.arrow",
                f"**/{split}/*.arrow",
            ]

        # 搜索匹配的文件
        split_files = []
        for pattern in split_patterns:
            found = list(local_path.glob(pattern))
            if found:
                split_files.extend(found)
                break

        # 分类文件类型
        if split_files:
            parquet_files = [f for f in split_files if f.suffix == '.parquet']
            arrow_files = [f for f in split_files if f.suffix == '.arrow']
            jsonl_files = [f for f in split_files if f.suffix == '.jsonl']
        else:
            # 后备：全局搜索
            parquet_files = list(local_path.rglob("*.parquet"))
            arrow_files = list(local_path.rglob("*.arrow"))
            jsonl_files = list(local_path.rglob("*.jsonl"))

        # 优先加载 jsonl 文件（用于 MuSiQue）
        if jsonl_files:
            print(f"   找到 {len(jsonl_files)} 个 jsonl 文件")
            try:
                # 构建目标文件名（使用已定义的 actual_split 和 subset_prefix）
                target_filename = f"musique_{subset_prefix}_v1.0_{actual_split}.jsonl"
                filtered_files = [f for f in jsonl_files if target_filename in str(f)]

                if filtered_files:
                    print(f"   加载: {[f.name for f in filtered_files]}")
                    # 指定 split="train" 直接获取 Dataset 而非 DatasetDict
                    dataset = load_dataset("json", data_files=[str(f) for f in filtered_files], split="train")
                    print(f"   [OK] 加载成功: {len(dataset)} 条数据")
                    return dataset
                else:
                    print(f"   [WARN] 未找到匹配的文件")
                    print(f"   期望文件: {target_filename}")
                    print(f"   可用文件: {[f.name for f in jsonl_files]}")
            except Exception as e:
                print(f"   [WARN] 加载 jsonl 失败: {e}")

        if parquet_files:
            print(f"   找到 {len(parquet_files)} 个 parquet 文件")
            try:
                dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
                print(f"   [OK] 加载成功: {len(dataset)} 条数据")
                return dataset
            except Exception as e:
                print(f"   [WARN] 加载 parquet 失败: {e}")

        if arrow_files:
            print(f"   找到 {len(arrow_files)} 个 arrow 文件")
            try:
                dataset = Dataset.from_file(str(arrow_files[0]))
                print(f"   [OK] 加载成功: {len(dataset)} 条数据")
                return dataset
            except Exception as e:
                print(f"   [WARN] 加载 arrow 失败: {e}")

        # 最后尝试：使用 datasets 库直接加载
        try:
            dataset = load_dataset(str(local_path), split=config.split)
            print(f"   [OK] 加载成功: {len(dataset)} 条数据")
            return dataset
        except Exception as e:
            print(f"   [WARN] 直接加载失败: {str(e)[:100]}")

    if local_path.is_file():
        suffix = local_path.suffix.lower()

        if suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(local_path), split="train")
            print(f"   [OK] 加载成功: {len(dataset)} 条数据")
            return dataset
        elif suffix == ".arrow":
            dataset = Dataset.from_file(str(local_path))
            print(f"   [OK] 加载成功: {len(dataset)} 条数据")
            return dataset
        elif suffix == ".json":
            dataset = load_dataset("json", data_files=str(local_path), split="train")
            print(f"   [OK] 加载成功: {len(dataset)} 条数据")
            return dataset

    print(f"   [FAIL] 无法识别的数据格式")
    return None


def normalize_to_list(data: Any) -> List:
    """将数据转换为列表"""
    if data is None:
        return []
    if hasattr(data, 'tolist'):
        return data.tolist()
    if isinstance(data, list):
        return data
    if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
        return list(data)
    return [data]


def normalize_nested_data(data: Any) -> Any:
    """递归将数据转换为原生 Python 类型"""
    if hasattr(data, "tolist"):
        data = data.tolist()
    if isinstance(data, dict):
        return {str(k): normalize_nested_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_nested_data(item) for item in data]
    if isinstance(data, tuple):
        return [normalize_nested_data(item) for item in data]
    return data


def extract_context_documents(item: Dict, config: DatasetConfig) -> Tuple[str, List[Dict]]:
    """
    从数据项中提取上下文文档

    返回: (合并后的语料文本, 文档列表)
    """
    context_data = item.get(config.context_field, {})
    documents = []
    all_text = []

    if config.has_nested_context and isinstance(context_data, dict):
        titles = normalize_to_list(context_data.get('title', []))
        sentences_list = normalize_to_list(context_data.get('sentences', []))

        for title, sentences in zip(titles, sentences_list):
            sentences = normalize_to_list(sentences)
            text = " ".join(str(s) for s in sentences if s)
            if text.strip() and title:
                doc = {
                    "title": str(title),
                    "text": text,
                    "sentences": sentences
                }
                documents.append(doc)
                all_text.append(f"[{title}]\n{text}")

    elif isinstance(context_data, list):
        for item_data in context_data:
            if isinstance(item_data, (list, tuple)) and len(item_data) >= 2:
                title = str(item_data[0])
                sentences = normalize_to_list(item_data[1])
                text = " ".join(str(s) for s in sentences if s)
                if text.strip():
                    documents.append({
                        "title": title,
                        "text": text,
                        "sentences": sentences
                    })
                    all_text.append(f"[{title}]\n{text}")
            elif isinstance(item_data, dict):
                title = item_data.get("title", item_data.get("paragraph_title", ""))
                text = item_data.get("text", item_data.get("paragraph_text", ""))
                if text.strip():
                    documents.append({
                        "title": str(title),
                        "text": text,
                        "sentences": [text.strip()]  # 整个段落作为单个单元
                    })
                    all_text.append(f"[{title}]\n{text}")

    elif isinstance(context_data, str):
        if context_data.strip():
            documents.append({
                "title": item.get("context_id", ""),
                "text": context_data
            })
            all_text.append(context_data)

    return "\n\n".join(all_text), documents



def extract_evidence(item: Dict, config: DatasetConfig, documents: List[Dict]) -> str:
    """
    提取证据文本
    
    对于不同数据集格式：
    - HotpotQA: 从 supporting_facts 提取支持句子
    - MuSiQue: 从 paragraphs 中提取 is_supporting=True 的段落
    - 其他: 从 evidence_field 提取
    """
    evidence_parts = []
    
    # 优先处理 paragraphs 格式（如 MuSiQue）
    paragraphs = item.get("paragraphs", [])
    if isinstance(paragraphs, list):
        supporting = [p for p in paragraphs if p.get("is_supporting")]
        if supporting:
            for p in supporting:
                text = p.get("paragraph_text", "").strip()
                if text:
                    evidence_parts.append(text)
            if evidence_parts:
                return " ".join(evidence_parts)
    
    # 处理 supporting_facts 格式（如 HotpotQA）
    if config.evidence_field is None:
        return ""
    evidence_data = item.get(config.evidence_field, [])

    if isinstance(evidence_data, dict):
        titles = normalize_to_list(evidence_data.get('title', []))
        sent_ids = normalize_to_list(evidence_data.get('sent_id', []))

        doc_map = {doc['title']: doc for doc in documents if 'title' in doc}

        for title, sent_id in zip(titles, sent_ids):
            if title in doc_map:
                doc = doc_map[title]
                if 'sentences' in doc:
                    sentences = doc['sentences']
                    if 0 <= sent_id < len(sentences):
                        evidence_parts.append(str(sentences[sent_id]).strip())
                elif 'text' in doc:
                    sentences = doc['text'].split('. ')
                    if 0 <= sent_id < len(sentences):
                        evidence_parts.append(sentences[sent_id].strip())

    elif isinstance(evidence_data, list):
        # 跳过 question_decomposition 格式（由 extract_reasoning_chain 处理）
        if evidence_data and isinstance(evidence_data[0], dict) and "paragraph_support_idx" in evidence_data[0]:
            return ""  # MuSiQue 格式，evidence 由 paragraphs 处理
        for ev in evidence_data:
            if isinstance(ev, dict):
                q = ev.get("question", "")
                a = ev.get("answer", "")
                if q and a:
                    evidence_parts.append(f"Q: {q} A: {a}")
            elif isinstance(ev, str):
                evidence_parts.append(ev)

    elif isinstance(evidence_data, str):
        evidence_parts.append(evidence_data)

    return " ".join(evidence_parts)


def extract_question_type_from_id(item: Dict, config: DatasetConfig) -> str:
    """从 ID 中提取问题类型（用于 MuSiQue 等数据集）"""
    import re
    item_id = str(item.get(config.id_field, ""))

    # 匹配模式：数字 + hop
    match = re.match(r'^(\d+)hop', item_id, re.IGNORECASE)
    if match:
        return f"{match.group(1)}hop"

    return config.question_type_default


def extract_reasoning_steps(item: Dict) -> List[Dict]:
    """
    提取结构化推理步骤（用于多跳问答数据集如 MuSiQue）

    返回:
        [{"step": 1, "question": "...", "answer": "...", "entity": "...", "relation": "..."}, ...]

    注意: MuSiQue 有两种格式：
    - 标准格式: "entity >> relation" (36%)
    - 自然语言格式: 完整问句 (64%)

    对于自然语言格式，entity 和 relation 为空，使用 question 字段。
    """
    decomposition = item.get("question_decomposition", [])
    if not isinstance(decomposition, list):
        return []

    steps = []
    for i, step in enumerate(decomposition):
        if isinstance(step, dict):
            q = step.get("question", "")
            a = step.get("answer", "")
            if q and a:
                step_data = {
                    "step": i + 1,
                    "question": q,  # 保留原始问题
                    "answer": a,
                }

                # 尝试解析标准格式 "entity >> relation"
                if " >> " in q:
                    parts = q.split(" >> ", 1)
                    step_data["entity"] = parts[0].strip()
                    step_data["relation"] = parts[1].strip() if len(parts) > 1 else ""
                else:
                    # 自然语言格式，entity/relation 为空
                    step_data["entity"] = ""
                    step_data["relation"] = ""

                steps.append(step_data)

    return steps


def extract_supporting_paragraph_titles(item: Dict) -> List[str]:
    """
    提取支持段落标题（用于 MuSiQue 等数据集）

    返回:
        支持段落标题列表

    注意: 删除了 supporting_paragraph_indices 字段，因为语料库中无 idx 字段无法关联。
    """
    paragraphs = item.get("paragraphs", [])
    if not isinstance(paragraphs, list):
        return []

    titles = []
    for p in paragraphs:
        if isinstance(p, dict) and p.get("is_supporting"):
            title = p.get("title", "")
            if title:
                titles.append(title)

    return titles


def process_item(
    item: Dict,
    config: DatasetConfig,
    idx: int
) -> Tuple[List[Dict], Dict, Dict]:
    """
    处理单个数据项

    返回: (语料库项列表, 问题项-评估格式, 问题项-原始格式)

    对于多文档数据集（如 HotpotQA），一个问题对应多个文档，
    每个文档独立输出，便于后续按 title 去重。
    """
    # 提取上下文和文档（用于语料库）
    context_text, documents = extract_context_documents(item, config)

    # 提取证据（用于评估）
    evidence_text = extract_evidence(item, config, documents)

    # 提取推理步骤（用于多跳问答数据集）
    reasoning_steps = extract_reasoning_steps(item)

    # 提取支持段落标题（用于 MuSiQue 等数据集）
    supporting_titles = extract_supporting_paragraph_titles(item)

    # 提取标准字段
    question_id = str(item.get(config.id_field, f"{config.name}-{idx}"))
    question_text = str(item.get(config.question_field, ""))

    # 处理答案（标准答案）
    answer_data = item.get(config.answer_field, "")
    if isinstance(answer_data, list):
        answer_text = answer_data[0] if answer_data else ""
    else:
        answer_text = str(answer_data)

    # 提取答案别名（用于 MuSiQue 等数据集）
    answer_aliases = item.get("answer_aliases", [])
    if isinstance(answer_aliases, list):
        # 过滤空值并去重
        answer_aliases = [str(a) for a in answer_aliases if a]
        if not answer_aliases:
            answer_aliases = None

    corpus_name = config.name

    # 确定 question_type
    question_type = config.question_type_default
    has_raw_type = "type" in item
    if has_raw_type:
        question_type = str(item.get("type", config.question_type_default))
    elif config.name == "musique":
        # MuSiQue 数据集：从 ID 中解析 hop 类型
        question_type = extract_question_type_from_id(item, config)

    # 构建语料库项：每个文档独立输出，保留原始句子用于后续合并
    corpus_items: List[Dict[str, Any]] = []
    for doc in documents:
        title = str(doc.get("title", "")).strip()
        text = str(doc.get("text", "")).strip()
        sentences = doc.get("sentences", [])  # 原始句子列表
        if not text:
            continue
        context = f"[{title}]\n{text}" if title else text
        corpus_items.append({
            "corpus_name": corpus_name,
            "title": title,
            "context": context,
            "sentences": sentences,  # 用于合并同标题内容
        })

    # 如果没有文档但有 context_text（如 ultradomain），输出单个语料库项
    if not corpus_items and context_text.strip():
        corpus_items.append({
            "corpus_name": corpus_name,
            "context": context_text,
        })

    # 构建评估格式问题项
    question_eval = {
        "id": question_id,
        "source": config.name,
        "question": question_text,
        "answer": answer_text,
        "evidence": evidence_text,
        "question_type": question_type,
    }

    # 添加推理步骤（如果有）- 替代原来的 reasoning_chain
    if reasoning_steps:
        question_eval["reasoning_steps"] = reasoning_steps

    # 添加支持段落标题（如果有）- 替代原来的 indices
    if supporting_titles:
        question_eval["supporting_paragraph_titles"] = supporting_titles

    # 添加答案别名（如果有）
    if answer_aliases:
        question_eval["answer_aliases"] = answer_aliases

    # 保留额外字段（不覆盖已存在的评估字段）
    for field in config.extra_standard_fields:
        # type 字段已用作 question_type，不再重复保存
        if field == "type" and has_raw_type:
            continue
        if field in item and field not in question_eval:
            question_eval[field] = normalize_nested_data(item[field])

    # 构建原始格式问题项（保留配置以备扩展）
    if config.preserve_all_original_fields:
        question_original = normalize_nested_data(item)
    else:
        question_original = {}

    return corpus_items, question_eval, question_original


def convert_dataset(
    dataset: Any,
    config: DatasetConfig,
    max_questions: Optional[int] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    转换数据集

    Returns:
        (语料库列表, 评估格式问题列表, 原始格式问题列表)
    """
    print(f"\n[PROC] 转换数据集: {config.name}")

    total = len(dataset)
    if max_questions:
        process_count = min(total, max_questions)
        print(f"   限制处理: {process_count}/{total} 条")
    else:
        process_count = total
        print(f"   处理全部: {total} 条")

    is_dataframe = isinstance(dataset, pd.DataFrame)

    corpus_list = []
    questions_eval = []
    questions_original = []

    # 用于合并同标题内容：(corpus_name, title) -> set of sentences
    title_to_sentences: Dict[Tuple[str, str], Set[str]] = {}

    for i in range(process_count):
        if (i + 1) % 1000 == 0:
            print(f"   进度: {i + 1}/{process_count}")

        if is_dataframe:
            item = dataset.iloc[i].to_dict()
        else:
            item = dataset[i]

        if not isinstance(item, dict):
            continue

        try:
            corpus_items, q_eval, q_orig = process_item(item, config, i)

            for corpus_item in corpus_items:
                if config.per_question_corpus:
                    corpus_list.append(corpus_item)
                    continue

                # 合并同标题内容：收集所有原始句子
                title = corpus_item.get("title", "")
                corpus_name = corpus_item.get("corpus_name", "")
                dedupe_key = (corpus_name, title)

                # 使用原始句子列表（如果有），避免重新解析
                sentences = corpus_item.get("sentences", [])
                if dedupe_key not in title_to_sentences:
                    title_to_sentences[dedupe_key] = set()
                # 将句子转为字符串并存入集合（自动去重）
                for sent in sentences:
                    sent_str = str(sent).strip()
                    if sent_str:
                        title_to_sentences[dedupe_key].add(sent_str)

            questions_eval.append(q_eval)
            questions_original.append(q_orig)

        except Exception as e:
            print(f"   [WARN] 跳过第 {i} 条: {str(e)[:50]}")
            continue

    # 从合并后的句子生成语料库
    for (corpus_name, title), sentences in title_to_sentences.items():
        if sentences:
            # 将句子按字典序排序后用空格连接（保持一致性）
            text = " ".join(sorted(sentences))
            context = f"[{title}]\n{text}"
            corpus_list.append({
                "corpus_name": corpus_name,
                "title": title,
                "context": context,
            })

    print(f"   [OK] 转换完成: {len(corpus_list)} 个文档, {len(questions_eval)} 个问题")

    return corpus_list, questions_eval, questions_original


def save_dataset(
    corpus_list: List[Dict],
    questions_eval: List[Dict],
    questions_original: List[Dict],
    output_dir: Path,
    dataset_name: str
) -> Tuple[Path, Path]:
    """
    保存数据集

    输出文件：
    - Corpus/{dataset_name}.parquet / .json
    - Questions/{dataset_name}_questions.parquet / .json (评估格式)
    """
    corpus_dir = output_dir / "Corpus"
    questions_dir = output_dir / "Questions"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)

    # 保存语料库
    corpus_parquet = corpus_dir / f"{dataset_name}.parquet"
    corpus_json = corpus_dir / f"{dataset_name}.json"

    corpus_df = pd.DataFrame(corpus_list)
    corpus_df.to_parquet(corpus_parquet, index=False)
    with open(corpus_json, 'w', encoding='utf-8') as f:
        json.dump(corpus_list, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 语料库已保存:")
    print(f"   Parquet: {corpus_parquet}")
    print(f"   JSON: {corpus_json}")
    print(f"   文档数: {len(corpus_list)}")

    # 保存问题集（评估格式）
    questions_parquet = questions_dir / f"{dataset_name}_questions.parquet"
    questions_json = questions_dir / f"{dataset_name}_questions.json"

    questions_df = pd.DataFrame(questions_eval)
    questions_df.to_parquet(questions_parquet, index=False)
    with open(questions_json, 'w', encoding='utf-8') as f:
        json.dump(questions_eval, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] 问题集已保存:")
    print(f"   Parquet: {questions_parquet}")
    print(f"   JSON: {questions_json}")
    print(f"   问题数: {len(questions_eval)}")
    print(f"   字段: {list(questions_eval[0].keys()) if questions_eval else 'N/A'}")

    # 保存原始格式问题集（如果配置了 preserve_all_original_fields）
    if questions_original and any(questions_original):
        original_json = questions_dir / f"{dataset_name}_questions_original.json"
        with open(original_json, 'w', encoding='utf-8') as f:
            json.dump(questions_original, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] 问题集（完整格式）已保存:")
        print(f"   JSON: {original_json}")

    return corpus_parquet, questions_parquet


def process_dataset(
    config: DatasetConfig,
    output_dir: Path,
    cache_dir: Path,
    local_path: Optional[Path] = None,
    max_questions: Optional[int] = None,
    force: bool = False
) -> bool:
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"[DIR] 数据集: {config.name}")
    print(f"   描述: {config.description}")
    print(f"{'='*60}")

    if not force and check_dataset_exists(output_dir, config.name):
        print(f"[WARN]  数据集已存在，跳过下载。使用 --force 强制重新下载。")
        return True

    print(f"   解析输出目录: {output_dir}")

    if local_path:
        print(f"[DIR] 从本地加载...")
        dataset = load_dataset_from_local(local_path, config)
    else:
        print(f"[DOWN] 下载数据集...")
        print(f"   缓存目录: {cache_dir / config.name}")
        dataset = load_dataset_with_retry(config, cache_dir)

    if dataset is None:
        print(f"[FAIL] 加载失败: {config.name}")
        return False

    corpus_list, q_eval, q_orig = convert_dataset(dataset, config, max_questions)

    if not corpus_list or not q_eval:
        print(f"[FAIL] 转换失败: {config.name}")
        return False

    save_dataset(corpus_list, q_eval, q_orig, output_dir, config.name)

    return True


def list_datasets():
    """列出所有支持的数据集"""
    print("\n[LIST] 支持的数据集:")
    print("-" * 80)
    for name, config in DATASET_CONFIGS.items():
        print(f"  {name:15} - {config.description}")
        print(f"                  HuggingFace: {config.hf_path}")
    print("-" * 80)
    print(f"\n使用方法:")
    print(f"  python Datasets/download_datasets.py --datasets hotpotqa musique")
    print(f"  python Datasets/download_datasets.py --datasets all --max-questions 500")


def main():
    parser = argparse.ArgumentParser(
        description="通用数据集下载与转换工具（支持评估格式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["all"],
        help="要处理的数据集名称"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="最大问题数量"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="解析后数据输出目录（默认: Datasets/）"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="原始数据缓存目录（默认: Datasets/data/）"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有支持的数据集"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="从网络下载"
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="指定本地数据目录"
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    base_dir = Path(__file__).parent
    project_root = base_dir.parent
    raw_dir = base_dir / "raw"
    cache_dir = project_root / "hf_cache"
    output_dir = base_dir

    download_mode = args.download

    if not download_mode:
        print("[INFO] 默认模式: 从本地目录加载（使用 --download 从网络下载）")

    if download_mode:
        setup_mirror()

    if "all" in args.datasets:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_process = args.datasets

    invalid = [d for d in datasets_to_process if d not in DATASET_CONFIGS]
    if invalid:
        print(f"[FAIL] 未知的数据集: {invalid}")
        print(f"   支持的数据集: {list(DATASET_CONFIGS.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"[START] 开始处理 {len(datasets_to_process)} 个数据集")
    print(f"   模式: {'网络下载' if download_mode else '本地加载'}")
    if download_mode:
        print(f"   缓存目录: {cache_dir}")
    else:
        print(f"   原始数据目录: {raw_dir}")
    print(f"   解析输出目录: {output_dir}")
    print(f"   问题限制: {args.max_questions or '无限制'}")
    print(f"{'='*60}")

    results = {}
    for dataset_name in datasets_to_process:
        config = DATASET_CONFIGS[dataset_name]

        if args.local:
            local_path = Path(args.local)
        elif not download_mode:
            # 使用映射获取本地目录名称
            local_dir_name = LOCAL_DIR_MAPPING.get(dataset_name, dataset_name)
            local_path = raw_dir / local_dir_name
        else:
            local_path = None

        success = process_dataset(
            config, output_dir, cache_dir,
            local_path=local_path,
            max_questions=args.max_questions,
            force=args.force
        )
        results[dataset_name] = "[OK] 成功" if success else "[FAIL] 失败"

    print(f"\n{'='*60}")
    print(f"[RESULT] 处理结果汇总:")
    for name, status in results.items():
        print(f"   {name:15} {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
