import asyncio
import argparse
import json
import numpy as np
import os
from typing import Dict, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to deprecated import for backward compatibility
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings as HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from Evaluation.metrics import compute_context_relevance, compute_evidence_recall
from langchain_ollama import OllamaEmbeddings
from Evaluation.llm import OllamaClient,OllamaWrapper

SEED = 42

async def evaluate_dataset(
    dataset: Dataset,
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 1,
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate context relevance and recall for a dataset"""
    results = {
        "context_relevancy": [],
        "evidence_recall": []
    }
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    contexts_list = dataset["contexts"]
    evidences = dataset["evidences"]

    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                contexts=contexts_list[i],
                evidences=evidences[i],
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "contexts": contexts_list[i],
                    "evidences": evidences[i],
                    "metrics": sample_metrics
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    sample_results = []
    completed = 0
    
    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output and detailed_results is not None:
                detailed_results.append(result)
                # metrics aggregation (guard types for linters)
                if isinstance(result, dict):
                    metrics_dict = result.get("metrics")
                    if isinstance(metrics_dict, dict):
                        for metric, score in metrics_dict.items():
                            if isinstance(score, (int, float)) and not np.isnan(score):
                                results[metric].append(score)
            else:
                sample_results.append(result)
                if isinstance(result, dict):
                    for metric, score in result.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            completed += 1
            print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            completed += 1
    
    avg_results = {
        "context_relevancy": np.nanmean(results["context_relevancy"]),
        "evidence_recall": np.nanmean(results["evidence_recall"])
    }
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results


async def evaluate_sample(
    question: str,
    contexts: List[str],
    evidences: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate retrieval metrics for a single sample"""
    # Evaluate both metrics in parallel
    relevance_task = compute_context_relevance(question, contexts, llm)
    recall_task = compute_evidence_recall(question, contexts, evidences, llm)
    
    # Wait for both tasks to complete
    relevance_score, recall_score = await asyncio.gather(relevance_task, recall_task)

    print(f"Relevance Score: {relevance_score}, Recall Score: {recall_score}")

    return {
        "context_relevancy": relevance_score,
        "evidence_recall": recall_score
    }

async def main(args: argparse.Namespace):
    """Main retrieval evaluation function"""
    if args.mode == "API":
        # Check API key
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
        
        # Initialize models
        # Wrap API key in SecretStr to satisfy type hints
        from pydantic import SecretStr
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.0,
            max_retries=3,
            timeout=120,
            top_p=1,
            seed=SEED,
            presence_penalty=0,
            frequency_penalty=0
        )
        
        # Initialize the embedding model
        embedding = HuggingFaceEmbeddings(model_name=args.embedding_model)

    elif args.mode == "ollama":
        ollama_client = OllamaClient(base_url=args.base_url)
        llm = OllamaWrapper(
            ollama_client,
            args.model,
            default_options={
                "temperature": 0.0,
                "top_p": 1,
                "num_ctx": 32768,
                "seed": SEED
            }
        )
        ollama_embeddings = OllamaEmbeddings(
            model=args.embedding_model,
            base_url=args.base_url
        )
        embedding = LangchainEmbeddingsWrapper(embeddings=ollama_embeddings)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Load evaluation data with memory-efficient streaming for large files
    print(f"Loading evaluation data from {args.data_file}...")
    
    # Check file size to decide loading strategy
    file_size = os.path.getsize(args.data_file)
    file_size_mb = file_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Group data by question type while loading (memory efficient)
    grouped_data = {}
    
    # Use streaming JSON parser for large files (>100MB)
    if file_size > 100 * 1024 * 1024:  # 100MB threshold
        print("Large file detected, using streaming parser...")
        try:
            import ijson
            item_count = 0
            with open(args.data_file, 'rb') as f:
                # Parse JSON array items one by one
                parser = ijson.items(f, 'item')
                for item in parser:
                    q_type = item.get("question_type", "Uncategorized")
                    if q_type not in grouped_data:
                        grouped_data[q_type] = []
                    
                    # Only keep items if we haven't reached the limit for this type
                    if args.num_samples is None or len(grouped_data[q_type]) < args.num_samples:
                        grouped_data[q_type].append(item)
                    
                    item_count += 1
                    if item_count % 1000 == 0:
                        print(f"Processed {item_count} items...")
            
            print(f"Loaded {item_count} items from file (kept {sum(len(v) for v in grouped_data.values())} for evaluation)")
        except ImportError:
            print("ERROR: ijson library is required for large files (>100MB)")
            print("Please install it with: pip install ijson")
            raise ImportError("ijson library is required for large files. Install with: pip install ijson")
    else:
        # Standard loading for smaller files
        with open(args.data_file, 'r', encoding='utf-8') as f:
            file_data = json.load(f)
        
        print(f"Loaded {len(file_data)} items from file")
        
        # Group data by question type
        for item in file_data:
            q_type = item.get("question_type", "Uncategorized")
            if q_type not in grouped_data:
                grouped_data[q_type] = []
            grouped_data[q_type].append(item)
        
        # Apply num_samples limit if specified
        if args.num_samples:
            for q_type in grouped_data:
                grouped_data[q_type] = grouped_data[q_type][:args.num_samples]
    
    all_results = {}
    
    # Evaluate each question type
    for question_type in list(grouped_data.keys()):
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]

        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        
        # Convert evidence to list format (handle both string and list)
        evidences = []
        for item in group_items:
            evidence = item.get('evidence', '')
            if isinstance(evidence, str):
                # Split by semicolon if it's a string with multiple statements
                if ';' in evidence:
                    evidence_list = [e.strip() for e in evidence.split(';') if e.strip()]
                else:
                    evidence_list = [evidence] if evidence else []
            elif isinstance(evidence, list):
                evidence_list = evidence
            else:
                evidence_list = []
            evidences.append(evidence_list)
        
        # Ensure contexts is a list of strings (extract from unified format if needed)
        contexts = []
        for item in group_items:
            context = item.get('context', [])
            if isinstance(context, str):
                context_list = [context] if context else []
            elif isinstance(context, list):
                # 统一格式：[{"type": "chunk", "content": "..."}, ...]
                context_list = []
                for ctx_item in context:
                    if isinstance(ctx_item, dict):
                        # 统一取 content 字段
                        context_list.append(ctx_item.get("content", ""))
                    elif isinstance(ctx_item, str):
                        context_list.append(ctx_item)
            else:
                context_list = []
            contexts.append(context_list)
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "contexts": contexts,
            "evidences": evidences
        }
        dataset = Dataset.from_dict(data)
        
        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(args.num_samples))])
        
        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            llm=llm, 
            embeddings=embedding,
            detailed_output=args.detailed_output
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if args.detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
            
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    await llm.close() if args.mode == "ollama" else None
    print('\nEvaluation complete.')

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval performance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add command-line arguments
    parser.add_argument(
        "--mode", 
        required=True,
        choices=["API", "ollama"],
        type=str,
        default="API",
        help="Use API or ollama for LLM"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API"
    )
    
    parser.add_argument(
        "--embedding_model", 
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="HuggingFace model for embeddings"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="retrieval_results.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int,
        default=None,
        help="Number of samples per question type to evaluate (optional)"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the main function
    asyncio.run(main(args))
