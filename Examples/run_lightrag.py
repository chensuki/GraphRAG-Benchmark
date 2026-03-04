# lightrag_example.py
import asyncio
import os
import logging
import nest_asyncio
import argparse
from typing import Dict, List

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.llm.hf import hf_embed
from lightrag.llm.zhipu import zhipu_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from common_benchmark import (
    build_output_path,
    group_questions_by_source,
    load_corpus_records,
    load_question_records,
    save_results_json,
)
from subset_registry import get_subset_paths, get_supported_subsets

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Knowledge Base---
{context_data}
"""

def create_llm_model_func(model_name: str, base_url: str, api_key: str):
    """Create an LLM model function with bound API configuration"""
    async def llm_model_func(
        prompt: str,
        system_prompt: str = None,
        history_messages: list | None = None,
        keyword_extraction: bool = False,
        **kwargs
    ) -> str:
        """LLM interface function using OpenAI-compatible API"""
        # Use bound API configuration, but allow kwargs to override
        final_model_name = kwargs.pop("model_name", model_name)
        final_base_url = kwargs.pop("base_url", base_url)
        final_api_key = kwargs.pop("api_key", api_key)
        
        # Fallback to environment variable if api_key is empty
        if not final_api_key:
            final_api_key = os.getenv("LLM_API_KEY", "")
        
        if not final_api_key:
            raise ValueError(
                "LLM API Key is required but not provided. "
                "Please set --llm_api_key or LLM_API_KEY environment variable."
            )
        
        if final_base_url and "deepseek" in final_base_url.lower():
            if not (final_api_key.startswith("sk-or-v1-") or final_api_key.startswith("sk-")):
                logging.warning(
                    f"DeepSeek API Key should start with 'sk-or-v1-' or 'sk-', but got '{final_api_key[:12]}...'. "
                    f"Please verify your API key is correct."
                )
        
        final_history_messages = history_messages or []

        return await openai_complete_if_cache(
            final_model_name,
            prompt,
            system_prompt=system_prompt,
            history_messages=final_history_messages,
            base_url=final_base_url,
            api_key=final_api_key,
            keyword_extraction=keyword_extraction,
            **kwargs
        )
    return llm_model_func

async def initialize_rag(
    base_dir: str,
    source: str,
    mode:str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    embed_provider: str,
    embed_api_key: str | None = None
) -> LightRAG:
    """Initialize LightRAG instance for a specific corpus"""
    working_dir = os.path.join(base_dir, source)
    
    # Create directory for this corpus
    os.makedirs(working_dir, exist_ok=True)
    
    if mode == "API":
        if embed_provider == "zhipu":
            embedding_func = EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: zhipu_embedding(
                    texts, model=embed_model_name, api_key=embed_api_key
                ),
            )
        elif embed_provider == "openai":
            embedding_func = EmbeddingFunc(
                embedding_dim=1536,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embed_model_name,
                    base_url=llm_base_url,
                    api_key=embed_api_key,
                ),
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
            embed_model = AutoModel.from_pretrained(embed_model_name)
            # Initialize local HF embedding
            embedding_func = EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: hf_embed(texts, tokenizer, embed_model),
            )
        
        # Create LLM configuration
        # Validate API key format
        if llm_api_key and not llm_api_key.startswith("sk-"):
            logging.warning(
                f"API Key does not start with 'sk-'. "
                f"DeepSeek API keys typically start with 'sk-'. "
                f"Key preview: {llm_api_key[:8]}...{llm_api_key[-4:] if len(llm_api_key) > 12 else '***'}"
            )
        
        llm_kwargs = {
            "model_name": model_name,
            "base_url": llm_base_url,
            "api_key": llm_api_key
        }

        # Create a bound LLM model function with API configuration
        llm_model_func = create_llm_model_func(model_name, llm_base_url, llm_api_key)
    elif mode == "ollama":
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=llm_base_url
            ),
        )

        llm_kwargs = {
            "host": llm_base_url,
            "options": {"num_ctx": 32768},
        }

        llm_model_func = ollama_model_complete
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API' or 'ollama'.")
    
    # Create RAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func,
        llm_model_name=model_name,
        llm_model_max_async=4,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    embed_provider: str,
    embed_api_key: str | None,
    questions: List[dict],
    sample: int,
    retrieve_topk: int,
    skip_build: bool = False,
    output_dir_root: str = "./results/lightrag",
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Initialize RAG for this corpus
    rag = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        mode=mode,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            embed_provider=embed_provider,
            embed_api_key=embed_api_key,
    )
    
    # Index the corpus content (skip if skip_build is True)
    if not skip_build:
        rag.insert(context)
        logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
    else:
        logging.info(f"⏭️  Skipping indexing (assuming corpus {corpus_name} is already indexed)")
    
    corpus_questions = questions.get(corpus_name, [])
    
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare output path
    output_dir = os.path.join(output_dir_root, corpus_name)
    output_path = build_output_path(output_dir, f"predictions_{corpus_name}.json")
    
    # Process questions
    results = []
    query_type = 'hybrid'
    
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        # Prepare query parameters
        query_param = QueryParam(
            mode=query_type,
            top_k=retrieve_topk,
            max_entity_tokens=4000,
            max_relation_tokens=4000,
            max_total_tokens=4000
        )
        
        # Execute query
        try:
            # Get response text
            query_result = rag.query(
                q["question"],
                param=query_param,
                system_prompt=SYSTEM_PROMPT
            )
            
            # Handle both async and sync responses
            if asyncio.iscoroutine(query_result):
                query_result = await query_result
            
            # Get response text
            if query_result is None:
                logging.warning(f"Query returned None for question {q['id']}")
                response = "Query failed: No response from LLM"
                predicted_answer = response
                context = []
            else:
                response = str(query_result) if query_result else "Query failed: No response"
                predicted_answer = response
                
                # Get context chunks from aquery_data
                try:
                    query_data = await rag.aquery_data(
                        q["question"],
                        param=query_param
                    )
                    
                    # Extract context chunks from the data structure
                    context = []
                    if query_data and isinstance(query_data, dict):
                        status = query_data.get("status", "unknown")
                        message = query_data.get("message", "")
                        
                        if status == "failure":
                            logging.warning(f"aquery_data failed for question {q['id']}: {message}")
                            context = []
                        else:
                            data = query_data.get("data", {})
                            if data:
                                # Extract chunks content
                                chunks = data.get("chunks", [])
                                if chunks:
                                    for chunk in chunks:
                                        if isinstance(chunk, dict) and "content" in chunk:
                                            context.append(chunk["content"])
                                        elif isinstance(chunk, str):
                                            context.append(chunk)
                                
                                # If no chunks, try to extract from entities and relationships
                                if not context:
                                    entities = data.get("entities", [])
                                    relationships = data.get("relationships", [])
                                    
                                    # Extract entity descriptions
                                    for entity in entities:
                                        if isinstance(entity, dict):
                                            desc = entity.get("description", "")
                                            if desc:
                                                context.append(desc)
                                    
                                    # Extract relationship descriptions
                                    for rel in relationships:
                                        if isinstance(rel, dict):
                                            desc = rel.get("description", "")
                                            if desc:
                                                context.append(desc)
                            else:
                                logging.debug(f"No data field in query_data for question {q['id']}")
                    else:
                        logging.warning(f"Invalid query_data format for question {q['id']}: {type(query_data)}")
                except Exception as e:
                    logging.warning(f"Failed to extract context for question {q['id']}: {str(e)}")
                    import traceback
                    logging.debug(traceback.format_exc())
                    context = []
            
            # Ensure context is a list
            if not isinstance(context, list):
                if isinstance(context, str):
                    context = [context] if context else []
                else:
                    context = []
                
        except Exception as e:
            logging.error(f"Query failed for question {q['id']}: {str(e)}")
            response = f"Query failed: {str(e)}"
            context = []
            predicted_answer = response

        # Collect results
        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": context,
            "evidence": q["evidence"],
            "question_type": q["question_type"],
            "generated_answer": predicted_answer,
            "ground_truth": q.get("answer"),

        })
    
    # Save results
    save_results_json(output_path, results)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    supported_subsets = get_supported_subsets("lightrag")

    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument(
        "--subset",
        required=True,
        choices=supported_subsets,
        help=f"Subset to process ({', '.join(supported_subsets)})",
    )
    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")
    parser.add_argument("--output_dir", default="./results/lightrag", help="Output directory for predictions")
    
    # Model configuration
    parser.add_argument("--mode", required=True, choices=["API", "ollama"], help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", help="LLM model identifier")
    parser.add_argument("--embed_model", default="bge-base-en", help="Embedding model name")
    parser.add_argument("--embed_provider", default="hf", choices=["hf", "zhipu", "openai"], help="Embedding provider")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    parser.add_argument("--corpus_sample", type=int, default=None, help="Number of corpora to process")
    parser.add_argument("--skip-build", action="store_true", help="Skip indexing phase, only run queries (assumes corpus is already indexed)")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")
    parser.add_argument("--embed_api_key", default="", 
                        help="API key for embedding service (can also use ZHIPUAI_API_KEY or OPENAI_API_KEY environment variables)")

    args = parser.parse_args()
    
    # Validate mode
    if args.mode not in ["API", "ollama"]:
        logging.error(f'Invalid mode: {args.mode}. Valid options: {["API", "ollama"]}')
        return
    
    # Get file paths for this subset
    try:
        corpus_path, questions_path = get_subset_paths(args.subset)
    except ValueError as e:
        logging.error(str(e))
        return
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    
    embed_api_key = (
        args.embed_api_key
        or os.getenv("ZHIPUAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or api_key
    )
    if args.embed_provider in ["zhipu", "openai"] and not embed_api_key:
        logging.warning(f"{args.embed_provider} embedding selected but no embedding API key provided.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_data = load_corpus_records(corpus_path)
        logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    total_corpora = len(corpus_data)
    if args.corpus_sample and args.corpus_sample < len(corpus_data):
        corpus_data = corpus_data[:args.corpus_sample]
        logging.info(f"Sampled {args.corpus_sample} corpora from {total_corpora} total")
    
    # Load question data
    try:
        question_data = load_question_records(questions_path)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return
    
    # Process each corpus concurrently in a single event loop
    async def _run_all():
        tasks = []
        for item in corpus_data:
            tasks.append(
                process_corpus(
                    corpus_name=item["corpus_name"],
                    context=item["context"],
                    base_dir=args.base_dir,
                    mode=args.mode,
                    model_name=args.model_name,
                    embed_model_name=args.embed_model,
                    llm_base_url=args.llm_base_url,
                    llm_api_key=api_key,
                    embed_provider=args.embed_provider,
                    embed_api_key=embed_api_key,
                    questions=grouped_questions,
                    sample=args.sample,
                    retrieve_topk=args.retrieve_topk,
                    skip_build=args.skip_build,
                    output_dir_root=args.output_dir,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"Task failed: {r}")

    asyncio.run(_run_all())

if __name__ == "__main__":
    main()
