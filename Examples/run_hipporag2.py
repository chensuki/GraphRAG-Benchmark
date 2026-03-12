import os
import asyncio
import argparse
import logging
from typing import Dict, List
from dotenv import load_dotenv
from transformers import AutoTokenizer
from tqdm import tqdm
from common_benchmark import (
    build_output_path,
    build_error_result,
    ensure_context_list,
    group_questions_by_source,
    load_corpus_records,
    load_question_records,
    merge_corpus_by_name,
    save_results_json,
)
from subset_registry import get_subset_paths, get_supported_subsets

# Load environment variables
load_dotenv()

# Import HippoRAG components after setting environment
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hipporag_processing.log")
    ]
)

def split_text(
    text: str, 
    tokenizer: AutoTokenizer, 
    chunk_token_size: int = 256, 
    chunk_overlap_token_size: int = 32
) -> List[str]:
    """Split text into chunks based on token length with overlap"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    top_k: int,
    questions: Dict[str, List[dict]],
    sample: int,
    output_dir_root: str = "./results/hipporag2",
    skip_build: bool = False,
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = os.path.join(output_dir_root, corpus_name)
    output_path = build_output_path(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize tokenizer for text splitting
    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        logging.info(f"✅ Loaded tokenizer: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return
    
    # Split text into chunks
    chunks = split_text(context, tokenizer)
    logging.info(f"✂️ Split corpus into {len(chunks)} chunks")
    
    # Format chunks as documents
    docs = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare queries and gold answers
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q['answer']] for q in corpus_questions]
    
    # Configure HippoRAG
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=llm_base_url,
        llm_name=model_name,
        embedding_model_name=embed_model_path,
        force_index_from_scratch=not skip_build,
        force_openie_from_scratch=not skip_build,
        rerank_dspy_file_path="hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=top_k,
        linking_top_k=top_k,
        max_qa_steps=3,
        qa_top_k=top_k,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )
    
    # Override LLM configuration for Ollama mode
    if mode == "ollama":
        config.llm_mode = "ollama"
        logging.info(f"✅ Using Ollama mode: {model_name} at {llm_base_url}")
    else:
        config.llm_mode = "openai"
        logging.info(f"✅ Using OpenAI mode: {model_name} at {llm_base_url}")
    
    # Initialize HippoRAG
    hipporag = HippoRAG(global_config=config)
    
    # Index the corpus content
    if not skip_build:
        hipporag.index(docs)
        logging.info(f"✅ Indexed corpus: {corpus_name}")
    else:
        logging.info(f"⏭️  Skipping indexing (assuming corpus {corpus_name} is already indexed)")
    
    # Process questions
    results = []

    try:
        queries_solutions, _, _, _, _ = hipporag.rag_qa(
            queries=all_queries,
            gold_docs=None,
            gold_answers=gold_answers,
        )
        solutions = [query.to_dict() for query in queries_solutions]
    except Exception as e:
        logging.error(f"❌ Batch QA failed for corpus {corpus_name}: {e}")
        solutions = []
    
    for question in corpus_questions:
        solution = next((sol for sol in solutions if sol['question'] == question['question']), None)
        if solution:
            results.append({
                "id": question["id"],
                "question": question["question"],
                "source": corpus_name,
                "context": ensure_context_list(solution.get("docs", [])),
                "evidence": question.get("evidence", ""),
                "question_type": question.get("question_type", ""),
                "generated_answer": solution.get("answer", ""),
                "ground_truth": question.get("answer", "")
            })
        else:
            results.append(
                build_error_result(
                    question,
                    corpus_name,
                    LookupError("No matching HippoRAG solution for question"),
                )
            )
    
    # Save results
    save_results_json(output_path, results)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    supported_subsets = get_supported_subsets("hipporag2")

    parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument(
        "--subset",
        required=True,
        choices=supported_subsets,
        help=f"Subset to process ({', '.join(supported_subsets)})",
    )
    parser.add_argument("--base_dir", default="./hipporag2_workspace", 
                        help="Base working directory for HippoRAG")
    parser.add_argument("--output_dir", default="./results/hipporag2",
                        help="Output directory for predictions")
    
    # Model configuration
    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="gpt-4o-mini", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/contriever", 
                        help="Path to embedding model directory")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    parser.add_argument("--corpus_sample", type=int, default=None,
                        help="Number of corpora to process")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Unified top-k for retrieval/linking/qa")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip indexing phase and reuse existing index")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use OPENAI_API_KEY environment variable)")

    args = parser.parse_args()
    
    logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset}")
    
    # Get file paths for this subset
    try:
        corpus_path, questions_path = get_subset_paths(args.subset)
    except ValueError as e:
        logging.error(f"❌ {e}")
        return
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_data = load_corpus_records(corpus_path)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    # Merge corpus by name (critical for multi-hop QA datasets)
    original_count = len(corpus_data)
    corpus_data = merge_corpus_by_name(corpus_data)
    logging.info(f"📖 Merged {original_count} documents into {len(corpus_data)} corpora")

    total_corpora = len(corpus_data)
    if args.corpus_sample and args.corpus_sample < len(corpus_data):
        corpus_data = corpus_data[:args.corpus_sample]
        logging.info(f"Sampled {args.corpus_sample} corpora from {total_corpora} total")
    
    # Load question data
    try:
        question_data = load_question_records(questions_path)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process each corpus concurrently using asyncio + threads
    async def _run_all():
        tasks = []
        for item in corpus_data:
            tasks.append(asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.mode,
                args.model_name,
                args.embed_model_path,
                args.llm_base_url,
                api_key,
                args.top_k,
                grouped_questions,
                args.sample,
                args.output_dir,
                args.skip_build,
            ))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"❌ Task failed: {r}")

    asyncio.run(_run_all())

if __name__ == "__main__":
    main()
