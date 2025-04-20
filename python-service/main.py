"""
Main interface module for the RAG system.
This module provides a simple interface for asking questions and getting answers.
"""

import logging
import time
import argparse
import json
from typing import Dict, Any, Optional
from rag_agent import RAGAgent
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a singleton RAG agent
_rag_agent = None

def get_rag_agent() -> RAGAgent:
    """Get the singleton RAG agent instance."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent

"""
Main interface module for the RAG system.
This module provides a simple interface for asking questions and getting answers.
"""

import logging
import time
import argparse
import json
from typing import Dict, Any, Optional
from rag_agent import RAGAgent
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a singleton RAG agent
_rag_agent = None

def get_rag_agent() -> RAGAgent:
    """Get the singleton RAG agent instance."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGAgent()
    return _rag_agent

def ask_question(
    question: str, 
    namespace: str = "default", 
    search_mode: str = "direct",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Ask a question and get an answer from the RAG system.
    
    Args:
        question: The question to ask
        namespace: The namespace to use for this query (for memory/context)
        search_mode: The search mode to use ('direct' or 'deepsearch')
        verbose: Whether to print detailed information
    
    Returns:
        A dictionary containing the answer and metadata
    """
    agent = get_rag_agent()
    start_time = time.time()
    
    result = agent.answer_question(question, namespace, search_mode)
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"QUERY: {question}")
        print(f"NAMESPACE: {namespace}")
        print(f"SEARCH MODE: {search_mode}")
        print(f"LLM USED: {result['metrics']['llm_used']}")
        print(f"PROCESSING TIME: {result['processing_time']['total']:.2f} seconds")
        
        if "sub_questions" in result and result["sub_questions"]:
            print("\nSUB-QUESTIONS:")
            for i, subq in enumerate(result["sub_questions"]):
                print(f"  {i+1}. {subq}")
        
        print("\nANSWER:")
        print("-" * 80)
        print(result["answer"])
        print("=" * 80)
    
    return result

def clean_answer(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the raw result from ask_question into a cleaner format.
    
    Args:
        result: The raw result dictionary from ask_question
        
    Returns:
        A clean dictionary with separated answer, sources, and logs
    """
    # Start with the raw answer
    raw_answer = result.get("answer", "")
    
    # Default empty values
    clean_answer = raw_answer
    sources = ""
    
    # Extract sources if they exist in the answer
    if "Sources:" in raw_answer:
        parts = raw_answer.split("Sources:", 1)
        clean_answer = parts[0].strip()
        sources = "Sources:" + parts[1]
    
    # Prepare logs
    logs = []
    logs.append(f"NAMESPACE: {result.get('namespace', 'default')}")
    logs.append(f"SEARCH MODE: {result.get('metrics', {}).get('search_mode', 'unknown')}")
    logs.append(f"LLM USED: {result.get('metrics', {}).get('llm_used', 'unknown')}")
    logs.append(f"PROCESSING TIME: {result.get('processing_time', {}).get('total', 0):.2f} seconds")
    
    if "sub_questions" in result and result["sub_questions"]:
        sub_q_log = ["SUB-QUESTIONS:"]
        for i, subq in enumerate(result["sub_questions"]):
            sub_q_log.append(f"  {i+1}. {subq}")
        logs.append("\n".join(sub_q_log))
    
    # Add all processing times
    time_logs = ["TIMING DETAILS:"]
    for step, duration in result.get("processing_time", {}).items():
        if isinstance(duration, (int, float)):
            time_logs.append(f"  {step}: {duration:.2f} seconds")
    logs.append("\n".join(time_logs))
    
    # Add document counts
    logs.append(f"DOCUMENTS RETRIEVED: {result.get('metrics', {}).get('documents_retrieved', 0)}")
    logs.append(f"SOURCES FOUND: {result.get('metrics', {}).get('sources_found', 0)}")
    
    # Combine logs
    logs_text = "\n".join(logs)
    
    return {
        "question": result.get("question", ""),
        "answer": clean_answer,
        "sources": sources,
        "logs": logs_text,
        "raw_result": result  # Include the original result for reference if needed
    }

def process_batch(
    input_file: str, 
    output_file: str, 
    namespace: str = "default", 
    search_mode: str = "direct"
) -> None:
    """
    Process a batch of questions from a file and save results to another file.
    
    Args:
        input_file: Path to input file with questions (one per line)
        output_file: Path to output file for results
        namespace: The namespace to use
        search_mode: The search mode to use
    """
    results = []
    
    try:
        with open(input_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(questions)} questions from {input_file}")
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = ask_question(question, namespace, search_mode)
            results.append(result)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")

def add_namespace_config(
    namespace: str,
    direct_top_n: int = 10,
    direct_llm: str = "gemini",
    deepsearch_top_n: int = 6,
    deepsearch_sub_query_top_n: int = 3,
    deepsearch_llm: str = "openai"
) -> None:
    """
    Add or update a namespace configuration.
    
    Args:
        namespace: The namespace to configure
        direct_top_n: Number of documents to retrieve for direct search
        direct_llm: LLM to use for direct search
        deepsearch_top_n: Number of documents to retrieve for simple queries in deepsearch
        deepsearch_sub_query_top_n: Number of documents per sub-question for complex queries
        deepsearch_llm: LLM to use for deepsearch
    """
    config.SEARCH_CONFIG[namespace] = {
        "direct": {
            "top_n": direct_top_n,
            "llm": direct_llm,
            "rerank": True
        },
        "deepsearch": {
            "top_n": deepsearch_top_n,
            "sub_query_top_n": deepsearch_sub_query_top_n,
            "llm": deepsearch_llm,
            "rerank": True
        }
    }
    
    logger.info(f"Added/updated configuration for namespace: {namespace}")

def main():
    """Command-line interface for the RAG system."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    
    # Command type
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single question command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--namespace", "-n", default="default", help="Namespace to use")
    ask_parser.add_argument("--mode", "-m", default="direct", choices=["direct", "deepsearch"], 
                           help="Search mode to use")
    ask_parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of questions")
    batch_parser.add_argument("input", help="Input file with questions")
    batch_parser.add_argument("output", help="Output file for results")
    batch_parser.add_argument("--namespace", "-n", default="default", help="Namespace to use")
    batch_parser.add_argument("--mode", "-m", default="direct", choices=["direct", "deepsearch"], 
                             help="Search mode to use")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Add or update namespace configuration")
    config_parser.add_argument("namespace", help="Namespace to configure")
    config_parser.add_argument("--direct-top-n", type=int, default=10, 
                              help="Number of documents for direct search")
    config_parser.add_argument("--direct-llm", default="gemini", choices=["gemini", "openai"],
                              help="LLM to use for direct search")
    config_parser.add_argument("--deepsearch-top-n", type=int, default=6,
                              help="Number of documents for simple queries in deepsearch")
    config_parser.add_argument("--deepsearch-sub-query-top-n", type=int, default=3,
                              help="Number of documents per sub-question for complex queries")
    config_parser.add_argument("--deepsearch-llm", default="openai", choices=["gemini", "openai"],
                              help="LLM to use for deepsearch")
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    if args.command == "ask":
        result = ask_question(args.question, args.namespace, args.mode, args.verbose)
        if not args.verbose:
            print(result["answer"])
    
    elif args.command == "batch":
        process_batch(args.input, args.output, args.namespace, args.mode)
    
    elif args.command == "config":
        add_namespace_config(
            args.namespace,
            args.direct_top_n,
            args.direct_llm,
            args.deepsearch_top_n,
            args.deepsearch_sub_query_top_n,
            args.deepsearch_llm
        )
    
    else:
        parser.print_help()

# Simple demo code
if __name__ == "__main__":
    # If run directly, use the command-line interface
    main()


def process_batch(
    input_file: str, 
    output_file: str, 
    namespace: str = "default", 
    search_mode: str = "direct"
) -> None:
    """
    Process a batch of questions from a file and save results to another file.
    
    Args:
        input_file: Path to input file with questions (one per line)
        output_file: Path to output file for results
        namespace: The namespace to use
        search_mode: The search mode to use
    """
    results = []
    
    try:
        with open(input_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Processing {len(questions)} questions from {input_file}")
        
        for i, question in enumerate(questions):
            logger.info(f"Processing question {i+1}/{len(questions)}")
            result = ask_question(question, namespace, search_mode)
            results.append(result)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
    
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")

def add_namespace_config(
    namespace: str,
    direct_top_n: int = 10,
    direct_llm: str = "gemini",
    deepsearch_top_n: int = 6,
    deepsearch_sub_query_top_n: int = 3,
    deepsearch_llm: str = "openai"
) -> None:
    """
    Add or update a namespace configuration.
    
    Args:
        namespace: The namespace to configure
        direct_top_n: Number of documents to retrieve for direct search
        direct_llm: LLM to use for direct search
        deepsearch_top_n: Number of documents to retrieve for simple queries in deepsearch
        deepsearch_sub_query_top_n: Number of documents per sub-question for complex queries
        deepsearch_llm: LLM to use for deepsearch
    """
    config.SEARCH_CONFIG[namespace] = {
        "direct": {
            "top_n": direct_top_n,
            "llm": direct_llm,
            "rerank": True
        },
        "deepsearch": {
            "top_n": deepsearch_top_n,
            "sub_query_top_n": deepsearch_sub_query_top_n,
            "llm": deepsearch_llm,
            "rerank": True
        }
    }
    
    logger.info(f"Added/updated configuration for namespace: {namespace}")

def main():
    """Command-line interface for the RAG system."""
    parser = argparse.ArgumentParser(description="RAG System CLI")
    
    # Command type
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single question command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--namespace", "-n", default="default", help="Namespace to use")
    ask_parser.add_argument("--mode", "-m", default="direct", choices=["direct", "deepsearch"], 
                           help="Search mode to use")
    ask_parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of questions")
    batch_parser.add_argument("input", help="Input file with questions")
    batch_parser.add_argument("output", help="Output file for results")
    batch_parser.add_argument("--namespace", "-n", default="default", help="Namespace to use")
    batch_parser.add_argument("--mode", "-m", default="direct", choices=["direct", "deepsearch"], 
                             help="Search mode to use")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Add or update namespace configuration")
    config_parser.add_argument("namespace", help="Namespace to configure")
    config_parser.add_argument("--direct-top-n", type=int, default=10, 
                              help="Number of documents for direct search")
    config_parser.add_argument("--direct-llm", default="gemini", choices=["gemini", "openai"],
                              help="LLM to use for direct search")
    config_parser.add_argument("--deepsearch-top-n", type=int, default=6,
                              help="Number of documents for simple queries in deepsearch")
    config_parser.add_argument("--deepsearch-sub-query-top-n", type=int, default=3,
                              help="Number of documents per sub-question for complex queries")
    config_parser.add_argument("--deepsearch-llm", default="openai", choices=["gemini", "openai"],
                              help="LLM to use for deepsearch")
    
    # Parse arguments and execute command
    args = parser.parse_args()
    
    if args.command == "ask":
        result = ask_question(args.question, args.namespace, args.mode, args.verbose)
        if not args.verbose:
            print(result["answer"])
    
    elif args.command == "batch":
        process_batch(args.input, args.output, args.namespace, args.mode)
    
    elif args.command == "config":
        add_namespace_config(
            args.namespace,
            args.direct_top_n,
            args.direct_llm,
            args.deepsearch_top_n,
            args.deepsearch_sub_query_top_n,
            args.deepsearch_llm
        )
    
    else:
        parser.print_help()

# Simple demo code
if __name__ == "__main__":
    # If run directly, use the command-line interface
    main()