import asyncio
import sys
import os
import logging
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from dotenv import load_dotenv
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService
from fast_graphrag._llm._base import BaseEmbeddingService
from tqdm import tqdm
import httpx
from common_benchmark import (
    build_output_path,
    build_error_result,
    group_questions_by_source,
    load_corpus_records,
    load_question_records,
    save_results_json,
)
from subset_registry import get_subset_paths, get_supported_subsets

# Add project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Configuration constants
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

@dataclass
class OpenAICompatibleEmbeddingService(BaseEmbeddingService):
    """Embedding service using OpenAI-compatible API (e.g., Zhipu, DeepSeek)"""
    
    model: str = "embedding-3"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key: str = ""
    embedding_dim: int = 2048  # 智谱 embedding-3 维度
    
    async def encode(self, texts: List[str], model: Optional[str] = None) -> np.ndarray:
        """Encode texts using OpenAI-compatible embedding API"""
        model_to_use = model or self.model
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                # Build request with dimensions parameter (required for Zhipu embedding-3)
                request_body = {
                    "model": model_to_use,
                    "input": texts
                }
                
                # Add dimensions parameter for Zhipu embedding-3 model
                if "embedding-3" in model_to_use:
                    request_body["dimensions"] = self.embedding_dim
                
                response = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_body
                )
                
                response.raise_for_status()
                data = response.json()
                
                # Extract embeddings in order
                embeddings = [item["embedding"] for item in data["data"]]
                return np.array(embeddings)
            except httpx.HTTPStatusError as e:
                logging.error(f"Embedding API error: {e.response.status_code} - {e.response.text}")
                raise

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_provider: str,
    embed_model_path: str,
    embed_base_url: str,
    embed_api_key: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int,
    output_dir_root: str = "./results/fast-graphrag",
    skip_build: bool = False,
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = os.path.join(output_dir_root, corpus_name)
    output_path = build_output_path(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize embedding service based on provider
    if embed_provider == "api":
        # Use OpenAI-compatible API (Zhipu, DeepSeek, etc.)
        embedding_service = OpenAICompatibleEmbeddingService(
            model=embed_model_path,  # e.g., "embedding-3"
            base_url=embed_base_url,
            api_key=embed_api_key,
            embedding_dim=2048 if "embedding-3" in embed_model_path else 1024
        )
        logging.info(f"✅ Using API embedding service: {embed_model_path} at {embed_base_url}")
    else:
        # Use local HuggingFace model - requires transformers package
        try:
            from transformers import AutoTokenizer, AutoModel
            from fast_graphrag._llm import HuggingFaceEmbeddingService as HFEmbedding
            
            embedding_tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
            embedding_model = AutoModel.from_pretrained(embed_model_path)
            embedding_service = HFEmbedding(
                model=embedding_model,
                tokenizer=embedding_tokenizer,
                embedding_dim=1024,
                max_token_size=8192
            )
            logging.info(f"✅ Loaded local embedding model: {embed_model_path}")
        except ImportError as e:
            logging.error(f"❌ Local embedding requires 'transformers' package: pip install transformers")
            return
        except Exception as e:
            logging.error(f"❌ Failed to load embedding model: {e}")
            return
    
    # Initialize LLM service based on mode
    if mode == "ollama":
        # Create Ollama client (lazy import)
        from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper
        ollama_client = OllamaClient(base_url=llm_base_url)
        llm_service = OllamaWrapper(ollama_client, model_name)
        logging.info(f"✅ Using Ollama LLM service: {model_name} at {llm_base_url}")
    else:
        # Use OpenAI-compatible service
        llm_service = OpenAILLMService(
            model=model_name,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        logging.info(f"✅ Using OpenAI-compatible LLM service: {model_name} at {llm_base_url}")

    # Initialize GraphRAG
    grag = GraphRAG(
        working_dir=os.path.join(base_dir, corpus_name),
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=embedding_service,
        ),
    )
    
    # Index the corpus content
    if not skip_build:
        logging.info(f"🔧 Starting indexing for corpus: {corpus_name}")
        try:
            grag.insert(context)
            logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
        except Exception as e:
            logging.error(f"❌ Indexing failed for corpus {corpus_name}: {e}")
            raise
    else:
        logging.info(f"⏭️  Skipping indexing (assuming corpus {corpus_name} is already indexed)")
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            # Execute query
            response = grag.query(q["question"])
            response_dict = response.to_dict()
            logging.info(f"Query response structure: {list(response_dict.keys())}")
            
            # Check context structure
            if 'context' in response_dict:
                logging.info(f"Context keys: {list(response_dict['context'].keys())}")
                if 'chunks' in response_dict['context']:
                    context_chunks = response_dict['context']['chunks']
                    logging.info(f"Found {len(context_chunks)} context chunks")
                    contexts = [item[0]["content"] for item in context_chunks] if context_chunks else []
                else:
                    logging.warning("No 'chunks' key in context")
                    contexts = []
            else:
                logging.warning("No 'context' key in response")
                contexts = []
                
            predicted_answer = response.response
            logging.info(f"Generated answer: {predicted_answer[:100]}...")

            # Collect results
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append(build_error_result(q, corpus_name, e))
    
    # Save results
    save_results_json(output_path, results)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    supported_subsets = get_supported_subsets("fast-graphrag")

    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument(
        "--subset",
        required=True,
        choices=supported_subsets,
        help=f"Subset to process ({', '.join(supported_subsets)})",
    )
    parser.add_argument("--base_dir", default="./fast-graphrag_workspace", 
                        help="Base working directory for GraphRAG")
    parser.add_argument("--output_dir", default="./results/fast-graphrag",
                        help="Output directory for predictions")
    
    # LLM configuration
    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="deepseek-chat", 
                        help="LLM model identifier")
    parser.add_argument("--llm_base_url", default="https://api.deepseek.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")
    
    # Embedding configuration
    parser.add_argument("--embed_provider", choices=["api", "local"], default="api",
                        help="Embedding provider: 'api' for OpenAI-compatible API, 'local' for HuggingFace model")
    parser.add_argument("--embed_model", default="embedding-3", 
                        help="Embedding model name (for API) or path (for local)")
    parser.add_argument("--embed_base_url", default="https://open.bigmodel.cn/api/paas/v4", 
                        help="Base URL for embedding API (Zhipu default)")
    parser.add_argument("--embed_api_key", default="", 
                        help="API key for embedding service (default: use ZHIPUAI_API_KEY env var)")
    
    # Other options
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip indexing phase and reuse existing index")

    args = parser.parse_args()
    
    # Configure logging for Windows (fix emoji encoding issues)
    import sys
    if sys.platform == "win32":
        # Set UTF-8 encoding for Windows to handle emoji characters
        # Also configure StreamHandler to handle UTF-8 properly
        class UTF8StreamHandler(logging.StreamHandler):
            def __init__(self, stream=None):
                super().__init__(stream)
                self.stream = sys.stdout if stream is None else stream
                self.stream.reconfigure(encoding='utf-8')
    
        # Use custom handler to properly handle UTF-8
        file_handler = logging.FileHandler(f"graphrag_{args.subset}.log", encoding='utf-8')
        console_handler = UTF8StreamHandler()
        
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[console_handler, file_handler]
        )
    else:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging.INFO,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"graphrag_{args.subset}.log", encoding='utf-8')
            ]
        )
    
    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")
    
    # Get file paths for this subset
    try:
        corpus_path, questions_path = get_subset_paths(args.subset)
    except ValueError as e:
        logging.error(f"❌ {e}")
        return
    
    # Handle API keys
    llm_api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not llm_api_key:
        logging.warning("⚠️ No LLM API key provided! Requests may fail.")
    
    embed_api_key = args.embed_api_key or os.getenv("ZHIPUAI_API_KEY", "")
    if args.embed_provider == "api" and not embed_api_key:
        logging.warning("⚠️ No embedding API key provided! Embedding requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_data = load_corpus_records(corpus_path)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]
        # Also limit context length to avoid memory issues
        if corpus_data:
            max_words = 5000  # Limit to first 5000 words
            context = corpus_data[0]["context"]
            words = context.split()
            if len(words) > max_words:
                limited_context = " ".join(words[:max_words])
                corpus_data[0]["context"] = limited_context
                logging.info(f"Limited context to {max_words} words for testing")
    
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
                args.embed_provider,
                args.embed_model,
                args.embed_base_url,
                embed_api_key,
                args.llm_base_url,
                llm_api_key,
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
