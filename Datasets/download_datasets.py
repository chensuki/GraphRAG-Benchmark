"""
通用数据集下载与转换工具

支持的数据集：
- hotpotqa/hotpot_qa (HotpotQA)
- TommyChien/UltraDomain  
- framolfese/2WikiMultihopQA
- bdsaglam/musique

特点：
- 保留原始数据集的所有字段
- 输出标准格式（用于评估）+ 原始格式（用于扩展）
- 同时生成 parquet 和 json 格式
- 默认从本地目录加载（跳过网络下载）

目录结构：
    GraphRAG-Benchmark/
    ├── Datasets/
    │   ├── raw/                  # 原始数据集（手动下载）
    │   │   ├── hotpotqa/
    │   │   ├── ultradomain/
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
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# HuggingFace 镜像配置
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str                    # 数据集名称（用于命令行参数）
    hf_path: str                 # HuggingFace 数据集路径
    subset: Optional[str]        # 子集名称（如果有）
    split: str                   # 数据集分割
    description: str             # 描述
    
    # 原始字段映射（用于保留原始数据）
    original_fields: List[str] = field(default_factory=list)
    
    # 标准字段映射（用于框架评估）
    id_field: str = "id"
    question_field: str = "question"
    answer_field: str = "answer"
    context_field: str = "context"
    evidence_field: str = "supporting_facts"
    
    # 特殊处理标志
    has_nested_context: bool = False  # context 是否为嵌套结构
    question_type_field: Optional[str] = "type"  # 问题类型字段
    question_type_default: str = "Complex Reasoning"


# 支持的数据集配置（保留原始字段）
DATASET_CONFIGS = {
    "hotpotqa": DatasetConfig(
        name="hotpotqa",
        hf_path="hotpotqa/hotpot_qa",
        subset="fullwiki",
        split="validation",
        description="HotpotQA: 多跳问答数据集",
        original_fields=["id", "question", "answer", "type", "level", "supporting_facts", "context"],
        has_nested_context=True,
        question_type_field="type",
    ),
    "ultradomain": DatasetConfig(
        name="ultradomain",
        hf_path="TommyChien/UltraDomain",
        subset=None,
        split="train",
        description="UltraDomain/LongBench: 长文本问答数据集",
        original_fields=["_id", "input", "answers", "context", "length", "dataset", "label"],
        id_field="_id",
        question_field="input",
        answer_field="answers",
        context_field="context",
        evidence_field=None,  # 该数据集无显式证据字段
        has_nested_context=False,
        question_type_field="dataset",  # 使用 dataset 字段标识来源
    ),
    "2wikimultihop": DatasetConfig(
        name="2wikimultihop",
        hf_path="framolfese/2WikiMultihopQA",
        subset=None,
        split="validation",
        description="2WikiMultihopQA: 两跳维基百科问答",
        original_fields=["id", "question", "answer", "type", "evidences", "supporting_facts", "context"],
        id_field="id",
        question_field="question",
        answer_field="answer",
        context_field="context",
        evidence_field="supporting_facts",  # 使用 supporting_facts 提取证据句子
        has_nested_context=True,
        question_type_field="type",
    ),
    "musique": DatasetConfig(
        name="musique",
        hf_path="bdsaglam/musique",
        subset=None,
        split="validation",
        description="MuSiQue: 多跳问答数据集",
        original_fields=["id", "question", "answer", "question_decomposition", "paragraphs", "answerable"],
        id_field="id",
        question_field="question",
        answer_field="answer",
        context_field="paragraphs",
        evidence_field="question_decomposition",
        has_nested_context=False,
        question_type_field="answerable",
    ),
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
            print(f"📌 使用镜像: {mirror}")
            return mirror
        except:
            continue
    
    print("⚠️  使用官方源: https://huggingface.co")
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
        print(f"   📁 语料库: {corpus_parquet if corpus_parquet.exists() else corpus_json}")
        print(f"   📁 问题集: {questions_parquet if questions_parquet.exists() else questions_json}")
    
    return corpus_exists and questions_exists


def load_dataset_with_retry(
    config: DatasetConfig, 
    cache_dir: Path,
    max_retries: int = 3
) -> Optional[Any]:
    """带重试的数据集加载，保存原始数据到 cache_dir"""
    from datasets import load_dataset
    
    # 设置缓存目录
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
                    "cache_dir": str(dataset_cache_dir)  # 原始数据缓存位置
                }
                if config.subset:
                    kwargs["name"] = config.subset
                
                dataset = load_dataset(**kwargs)
                
                print(f"   ✅ 下载成功: {len(dataset)} 条数据")
                print(f"   📦 原始数据缓存: {dataset_cache_dir}")
                return dataset
                
            except Exception as e:
                print(f"   ❌ 失败: {str(e)[:100]}")
                continue
        
        if attempt < max_retries - 1:
            print(f"   重试 {attempt + 2}/{max_retries}...")
    
    return None


def load_dataset_from_local(local_path: Path, config: DatasetConfig) -> Optional[Any]:
    """从本地目录加载数据集
    
    支持:
    - HuggingFace 缓存目录 (包含 dataset_info.json)
    - Parquet 文件 (*.parquet)
    - Arrow 文件 (*.arrow)
    - JSON 文件 (*.json)
    """
    from datasets import load_dataset, Dataset
    
    local_path = Path(local_path)
    
    if not local_path.exists():
        print(f"   ❌ 本地路径不存在: {local_path}")
        return None
    
    print(f"   📂 从本地加载: {local_path}")
    
    # 1. 尝试加载 HuggingFace 缓存目录
    if local_path.is_dir():
        # 查找指定 split 的 parquet 文件（优先）
        split = config.split
        split_patterns = [
            f"**/{split}*.parquet",      # validation-00000-of-00001.parquet
            f"**/{split}/*.parquet",      # validation/xxx.parquet
            f"**/{split}*.arrow",
            f"**/{split}/*.arrow",
        ]
        
        split_files = []
        for pattern in split_patterns:
            found = list(local_path.glob(pattern))
            if found:
                split_files.extend(found)
                break
        
        # 如果找到了指定 split 的文件，只加载这些
        if split_files:
            parquet_files = [f for f in split_files if f.suffix == '.parquet']
            arrow_files = [f for f in split_files if f.suffix == '.arrow']
        else:
            # 否则加载所有文件
            parquet_files = list(local_path.rglob("*.parquet"))
            arrow_files = list(local_path.rglob("*.arrow"))
        
        if parquet_files:
            print(f"   找到 {len(parquet_files)} 个 parquet 文件 (split: {split})")
            try:
                dataset = load_dataset("parquet", data_files=[str(f) for f in parquet_files], split="train")
                print(f"   ✅ 加载成功: {len(dataset)} 条数据")
                return dataset
            except Exception as e:
                print(f"   ⚠️ 加载 parquet 失败: {e}")
        
        if arrow_files:
            print(f"   找到 {len(arrow_files)} 个 arrow 文件")
            try:
                dataset = Dataset.from_file(str(arrow_files[0]))
                print(f"   ✅ 加载成功: {len(dataset)} 条数据")
                return dataset
            except Exception as e:
                print(f"   ⚠️ 加载 arrow 失败: {e}")
        
        # 尝试作为 HuggingFace 数据集目录加载
        try:
            dataset = load_dataset(str(local_path), split=config.split)
            print(f"   ✅ 加载成功: {len(dataset)} 条数据")
            return dataset
        except:
            pass
    
    # 2. 尝试加载单个文件
    if local_path.is_file():
        suffix = local_path.suffix.lower()
        
        if suffix == ".parquet":
            dataset = load_dataset("parquet", data_files=str(local_path), split="train")
            print(f"   ✅ 加载成功: {len(dataset)} 条数据")
            return dataset
        
        elif suffix == ".arrow":
            dataset = Dataset.from_file(str(local_path))
            print(f"   ✅ 加载成功: {len(dataset)} 条数据")
            return dataset
        
        elif suffix == ".json":
            dataset = load_dataset("json", data_files=str(local_path), split="train")
            print(f"   ✅ 加载成功: {len(dataset)} 条数据")
            return dataset
    
    print(f"   ❌ 无法识别的数据格式")
    return None


def normalize_to_list(data: Any) -> List:
    """将数据转换为列表（处理 numpy 数组等）"""
    if data is None:
        return []
    if hasattr(data, 'tolist'):
        return data.tolist()
    if isinstance(data, list):
        return data
    if hasattr(data, '__iter__') and not isinstance(data, (str, dict)):
        return list(data)
    return [data]


def extract_context_documents(item: Dict, config: DatasetConfig) -> Tuple[str, List[Dict], List[Dict]]:
    """
    从数据项中提取上下文文档
    
    返回: (合并后的语料文本, 文档列表用于语料库, 证据文档列表)
    """
    context_data = item.get(config.context_field, {})
    documents = []
    all_text = []
    
    if config.has_nested_context and isinstance(context_data, dict):
        # HotpotQA / 2WikiMultihopQA 风格: {'title': [...], 'sentences': [[...], ...]}
        titles = normalize_to_list(context_data.get('title', []))
        sentences_list = normalize_to_list(context_data.get('sentences', []))
        
        for title, sentences in zip(titles, sentences_list):
            sentences = normalize_to_list(sentences)
            text = " ".join(str(s) for s in sentences if s)
            if text.strip() and title:
                doc = {
                    "title": str(title),
                    "text": text,
                    "sentences": sentences  # 保留原始句子列表
                }
                documents.append(doc)
                all_text.append(f"[{title}]\n{text}")
    
    elif isinstance(context_data, list):
        # 列表格式: [[title, [sentences]], ...] 或 [{...}, ...]
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
                # 字典列表格式（musique 风格）
                title = item_data.get("title", item_data.get("paragraph_title", ""))
                text = item_data.get("text", item_data.get("paragraph_text", ""))
                if text.strip():
                    documents.append({
                        "title": str(title),
                        "text": text
                    })
                    all_text.append(f"[{title}]\n{text}")
    
    elif isinstance(context_data, str):
        # 纯文本格式（UltraDomain 风格）
        if context_data.strip():
            documents.append({
                "title": item.get("context_id", ""),
                "text": context_data
            })
            all_text.append(context_data)
    
    return "\n\n".join(all_text), documents


def extract_evidence(item: Dict, config: DatasetConfig, documents: List[Dict]) -> str:
    """提取证据文本"""
    if config.evidence_field is None:
        return ""  # 无证据字段
    evidence_data = item.get(config.evidence_field, [])
    evidence_parts = []
    
    if isinstance(evidence_data, dict):
        # HotpotQA 风格: {'title': [...], 'sent_id': [...]}
        titles = normalize_to_list(evidence_data.get('title', []))
        sent_ids = normalize_to_list(evidence_data.get('sent_id', evidence_data.get('sent_idx', [])))
        
        doc_map = {doc['title']: doc for doc in documents if 'title' in doc}
        
        for title, sent_id in zip(titles, sent_ids):
            if title in doc_map:
                doc = doc_map[title]
                if 'sentences' in doc:
                    sentences = doc['sentences']
                    if 0 <= sent_id < len(sentences):
                        evidence_parts.append(str(sentences[sent_id]).strip())
                elif 'text' in doc:
                    # 按句号分割取对应句子
                    sentences = doc['text'].split('. ')
                    if 0 <= sent_id < len(sentences):
                        evidence_parts.append(sentences[sent_id].strip())
    
    elif isinstance(evidence_data, list):
        # 列表格式证据
        for ev in evidence_data:
            if isinstance(ev, dict):
                # musique 风格
                q = ev.get("question", "")
                a = ev.get("answer", "")
                if q and a:
                    evidence_parts.append(f"Q: {q} A: {a}")
            elif isinstance(ev, str):
                evidence_parts.append(ev)
    
    elif isinstance(evidence_data, str):
        evidence_parts.append(evidence_data)
    
    return " ".join(evidence_parts)


def process_item(
    item: Dict, 
    config: DatasetConfig, 
    idx: int
) -> Tuple[Dict, Dict, Dict]:
    """
    处理单个数据项
    
    返回: (语料库项, 问题项-标准格式, 问题项-原始格式)
    """
    # 提取上下文
    context_text, documents = extract_context_documents(item, config)
    
    # 提取证据
    evidence_text = extract_evidence(item, config, documents)
    
    # 提取标准字段
    question_id = str(item.get(config.id_field, f"{config.name}-{idx}"))
    question_text = str(item.get(config.question_field, ""))
    
    # 处理答案（可能是列表）
    answer_data = item.get(config.answer_field, "")
    if isinstance(answer_data, list):
        answer_text = answer_data[0] if answer_data else ""
    else:
        answer_text = str(answer_data)
    
    # 问题类型
    if config.question_type_field:
        q_type = str(item.get(config.question_type_field, config.question_type_default))
    else:
        q_type = config.question_type_default
    
    # 构建语料库项
    corpus_item = {
        "corpus_name": config.name,
        "context": context_text,
        # 额外保留文档列表
        "documents": documents
    }
    
    # 构建标准格式问题项
    question_standard = {
        "id": question_id,
        "source": config.name,
        "question": question_text,
        "answer": answer_text,
        "question_type": q_type,
        "evidence": evidence_text,
    }
    
    # 构建原始格式问题项（保留所有原始字段）
    question_original = {"_original": {}}
    for field in config.original_fields:
        if field in item:
            value = item[field]
            # 处理 numpy 数组
            if hasattr(value, 'tolist'):
                value = value.tolist()
            question_original["_original"][field] = value
    
    # 添加标准字段到原始格式
    question_original.update(question_standard)
    
    return corpus_item, question_standard, question_original


def convert_dataset(
    dataset: Any,
    config: DatasetConfig,
    max_questions: Optional[int] = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    转换数据集
    
    Returns:
        (语料库列表, 标准格式问题列表, 原始格式问题列表)
    """
    print(f"\n🔄 转换数据集: {config.name}")
    print(f"   原始字段: {config.original_fields}")
    
    total = len(dataset)
    if max_questions:
        process_count = min(total, max_questions)
        print(f"   限制处理: {process_count}/{total} 条")
    else:
        process_count = total
        print(f"   处理全部: {total} 条")
    
    is_dataframe = isinstance(dataset, pd.DataFrame)
    
    corpus_list = []
    questions_standard = []
    questions_original = []
    seen_corpus = set()
    
    for i in range(process_count):
        if (i + 1) % 1000 == 0:
            print(f"   进度: {i + 1}/{process_count}")
        
        # 获取数据项
        if is_dataframe:
            item = dataset.iloc[i].to_dict()
        else:
            item = dataset[i]
        
        if not isinstance(item, dict):
            continue
        
        try:
            corpus_item, q_std, q_orig = process_item(item, config, i)
            
            # 去重语料库
            corpus_hash = hash(corpus_item["context"][:500])
            if corpus_hash not in seen_corpus:
                seen_corpus.add(corpus_hash)
                corpus_list.append(corpus_item)
            
            questions_standard.append(q_std)
            questions_original.append(q_orig)
            
        except Exception as e:
            print(f"   ⚠️ 跳过第 {i} 条: {str(e)[:50]}")
            continue
    
    print(f"   ✅ 转换完成: {len(corpus_list)} 个文档, {len(questions_standard)} 个问题")
    
    return corpus_list, questions_standard, questions_original


def save_dataset(
    corpus_list: List[Dict],
    questions_standard: List[Dict],
    questions_original: List[Dict],
    output_dir: Path,
    dataset_name: str
) -> Tuple[Path, Path]:
    """保存数据集（标准格式 + 原始格式）"""
    corpus_dir = output_dir / "Corpus"
    questions_dir = output_dir / "Questions"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 保存语料库 ==========
    corpus_parquet = corpus_dir / f"{dataset_name}.parquet"
    corpus_json = corpus_dir / f"{dataset_name}.json"
    
    # 简化语料库（移除 documents 列表，仅保留文本）
    corpus_simple = [
        {"corpus_name": item["corpus_name"], "context": item["context"]}
        for item in corpus_list
    ]
    
    corpus_df = pd.DataFrame(corpus_simple)
    corpus_df.to_parquet(corpus_parquet, index=False)
    with open(corpus_json, 'w', encoding='utf-8') as f:
        json.dump(corpus_simple, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 语料库已保存:")
    print(f"   Parquet: {corpus_parquet}")
    print(f"   JSON: {corpus_json}")
    print(f"   文档数: {len(corpus_list)}")
    
    # ========== 保存问题集（标准格式） ==========
    questions_parquet = questions_dir / f"{dataset_name}_questions.parquet"
    questions_json = questions_dir / f"{dataset_name}_questions.json"
    
    questions_df = pd.DataFrame(questions_standard)
    questions_df.to_parquet(questions_parquet, index=False)
    with open(questions_json, 'w', encoding='utf-8') as f:
        json.dump(questions_standard, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 问题集（标准格式）已保存:")
    print(f"   Parquet: {questions_parquet}")
    print(f"   JSON: {questions_json}")
    print(f"   问题数: {len(questions_standard)}")
    
    # ========== 保存问题集（原始格式，保留所有字段） ==========
    if questions_original and "_original" in questions_original[0]:
        original_json = questions_dir / f"{dataset_name}_questions_full.json"
        with open(original_json, 'w', encoding='utf-8') as f:
            json.dump(questions_original, f, indent=2, ensure_ascii=False)
        print(f"\n✅ 问题集（完整格式）已保存:")
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
    """处理单个数据集
    
    Args:
        config: 数据集配置
        output_dir: 解析后数据输出目录 (Corpus/Questions)
        cache_dir: 原始下载缓存目录 (data/{dataset_name})
        local_path: 本地数据集路径（如果提供，不从网络下载）
        max_questions: 最大问题数量
        force: 是否强制重新下载
    """
    print(f"\n{'='*60}")
    print(f"📂 数据集: {config.name}")
    print(f"   描述: {config.description}")
    print(f"{'='*60}")
    
    if not force and check_dataset_exists(output_dir, config.name):
        print(f"⚠️  数据集已存在，跳过下载。使用 --force 强制重新下载。")
        return True
    
    print(f"   解析输出目录: {output_dir}")
    
    # 从本地加载或从网络下载
    if local_path:
        print(f"📂 从本地加载...")
        dataset = load_dataset_from_local(local_path, config)
    else:
        print(f"📥 下载数据集...")
        print(f"   缓存目录: {cache_dir / config.name}")
        dataset = load_dataset_with_retry(config, cache_dir)
    
    if dataset is None:
        print(f"❌ 加载失败: {config.name}")
        return False
    
    corpus_list, q_std, q_orig = convert_dataset(dataset, config, max_questions)
    
    if not corpus_list or not q_std:
        print(f"❌ 转换失败: {config.name}")
        return False
    
    save_dataset(corpus_list, q_std, q_orig, output_dir, config.name)
    
    return True


def list_datasets():
    """列出所有支持的数据集"""
    print("\n📚 支持的数据集:")
    print("-" * 80)
    for name, config in DATASET_CONFIGS.items():
        print(f"  {name:15} - {config.description}")
        print(f"                  HuggingFace: {config.hf_path}")
        print(f"                  原始字段: {config.original_fields}")
    print("-" * 80)
    print(f"\n使用方法:")
    print(f"  python Datasets/download_datasets.py --datasets hotpotqa musique")
    print(f"  python Datasets/download_datasets.py --datasets all --max-questions 500")


def main():
    parser = argparse.ArgumentParser(
        description="通用数据集下载与转换工具",
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
        help="最大问题数量（不限制语料规模）"
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
        help="从网络下载（默认从本地 raw/ 目录加载）"
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="指定本地数据目录（默认: Datasets/raw/{dataset_name}/）"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    # 默认路径配置
    base_dir = Path(__file__).parent
    project_root = base_dir.parent  # 项目根目录
    
    # 默认原始数据目录
    raw_dir = base_dir / "raw"
    cache_dir = project_root / "hf_cache"
    output_dir = base_dir
    
    # 是否从网络下载
    download_mode = args.download
    
    if not download_mode:
        print("📌 默认模式: 从本地目录加载（使用 --download 从网络下载）")
    
    if download_mode:
        setup_mirror()
    
    if "all" in args.datasets:
        datasets_to_process = list(DATASET_CONFIGS.keys())
    else:
        datasets_to_process = args.datasets
    
    invalid = [d for d in datasets_to_process if d not in DATASET_CONFIGS]
    if invalid:
        print(f"❌ 未知的数据集: {invalid}")
        print(f"   支持的数据集: {list(DATASET_CONFIGS.keys())}")
        return
    
    print(f"\n{'='*60}")
    print(f"🚀 开始处理 {len(datasets_to_process)} 个数据集")
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
        
        # 确定本地路径
        if args.local:
            local_path = Path(args.local)
        elif not download_mode:
            local_path = raw_dir / dataset_name
        else:
            local_path = None
        
        success = process_dataset(
            config, output_dir, cache_dir,
            local_path=local_path,
            max_questions=args.max_questions,
            force=args.force
        )
        results[dataset_name] = "✅ 成功" if success else "❌ 失败"
    
    print(f"\n{'='*60}")
    print(f"📊 处理结果汇总:")
    for name, status in results.items():
        print(f"   {name:15} {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
