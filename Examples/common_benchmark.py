import json
import os
from typing import Any, Dict, List

from datasets import load_dataset


def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by source field."""
    grouped_questions: Dict[str, List[dict]] = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


def ensure_context_list(context: Any) -> List[str]:
    """Normalize context to benchmark-required list[str]."""
    if context is None:
        return []
    if isinstance(context, str):
        return [context] if context else []
    if isinstance(context, list):
        return [str(item) for item in context]
    return [str(context)]


def build_error_result(
    question: dict,
    source: str,
    error: Exception,
    *,
    question_type_default: str = "",
    message_prefix: str = "Query failed",
) -> dict:
    """Return a benchmark-compatible error record, preserving all original fields."""
    result = dict(question)
    result.update({
        "source": source,
        "context": [],
        "generated_answer": f"{message_prefix}: {error}",
        "ground_truth": question.get("answer", ""),
    })
    return result


def load_corpus_records(corpus_path: str) -> List[dict]:
    """Load corpus records from parquet/json into benchmark schema."""
    if corpus_path.endswith(".json"):
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise ValueError(f"Unsupported corpus JSON type: {type(data)}")
        return [dict(item) for item in data]

    corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
    return [dict(item) for item in corpus_dataset]


def merge_corpus_by_name(corpus_records: List[dict]) -> List[dict]:
    """
    Merge corpus records by corpus_name.

    All documents sharing the same corpus_name are grouped together.
    Context is simply concatenated with double newlines.

    Args:
        corpus_records: List of {"corpus_name": str, "context": str, "title": str} records

    Returns:
        List of merged records, one per unique corpus_name:
        [{"corpus_name": str, "context": str}]
    """
    grouped: Dict[str, List[str]] = {}

    for item in corpus_records:
        name = item.get("corpus_name", "Unknown")
        context = item.get("context", "")

        if name not in grouped:
            grouped[name] = []

        if context:
            grouped[name].append(context)

    return [
        {
            "corpus_name": name,
            "context": "\n\n".join(contexts),
        }
        for name, contexts in grouped.items()
        if contexts
    ]


def load_question_records(questions_path: str) -> List[dict]:
    """Load question records from parquet/json, preserving all original fields."""
    if questions_path.endswith(".json"):
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        if not isinstance(question_data, list):
            raise ValueError(f"Question data must be a list, got: {type(question_data)}")
        return question_data

    questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
    return [dict(item) for item in questions_dataset]


def build_output_path(output_dir: str, filename: str) -> str:
    """Create output directory and return full output file path."""
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, filename)


def save_results_json(output_path: str, results: List[dict]) -> None:
    """Save benchmark results to json with unified formatting."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)