"""
Configuration module for the RAG system.
Contains all configurable parameters in one centralized location.
"""

import logging
from typing import Dict, Any
from dotenv import load_dotenv
import os 
from dotenv import load_dotenv
load_dotenv()
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Replace these with your actual API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY =os.getenv("PINECONE_API_KEY")

# Pinecone index name
PINECONE_INDEX_NAME = "askneu"

# Models configuration
MODEL_CONFIG = {
    "openai": {
        "model_name": "gpt-4.1",
        "temperature": 0.1,
        "max_retries": 2,
        "api_key": OPENAI_API_KEY
    },
    "gemini": {
        "model_name": "gemini-2.0-flash",  
        "temperature": 0.1,
        "max_retries": 2,
        "api_key": GOOGLE_API_KEY
    },
    "embeddings": {
        "model_name": "text-embedding-3-small",
        "api_key": OPENAI_API_KEY
    }
}


SEARCH_CONFIG = {
    "default": {  # Default namespace
        "direct": {
            "top_n": 7,
            "llm": "gemini",  # Using Gemini for direct search
            "rerank": True
        },
        "deepsearch": {
            "top_n": 6,  # For simple queries
            "sub_query_top_n": 4,  # For each sub-question in complex queries
            "llm": "openai",  # Use OpenAI for deep search
            "rerank": True
        }
    },
    "classroom": {
        "direct": {
            "top_n": 6,
            "llm": "gemini",
            "rerank": True
        },
        "deepsearch": {
            "top_n": 6,
            "sub_query_top_n": 4,
            "llm": "openai",
            "rerank": False
        }
    },
    "course": {
        "direct": {
            "top_n": 6,
            "llm": "gemini",
            "rerank": True
        },
        "deepsearch": {
            "top_n": 6,
            "sub_query_top_n": 4,
            "llm": "openai",
            "rerank": True
        }
    }
}



# Synthesis prompt template
SYNTHESIS_PROMPT_TEMPLATE = """
CONTEXT:
{contexts}

QUESTION:
{question}

EXTRACTED SOURCES:
{sources}

You are a knowledgeable assistant specializing in Northeastern University (NEU) information. Your purpose is to provide friendly, straightforward responses that sound natural and conversational.

Guidelines:
- Present information about NEU as factual knowledge without mentioning "context," "provided information," or any references to your information sources in the main body of your answer
- DO NOT mention or refer to sources within your main answer
- Answer the question comprehensively using the provided context
- After your complete answer, ALWAYS include a "Sources" section that lists all the URLs provided in the EXTRACTED SOURCES section above
- Format the sources section exactly like this:
  
  Sources:
  - URL1
  - URL2
  ...and so on
  
- If you lack sufficient information to answer fully, simply state: "I don't have enough information about this topic. Please contact Northeastern University directly for more details."
- Never fabricate information
- Maintain professional language regardless of user input

Your responses should sound natural and helpful, as if you're simply sharing knowledge about Northeastern University without revealing how you obtained that information.

IMPORTANT: YOU MUST INCLUDE THE SOURCES SECTION AT THE END OF YOUR RESPONSE WITH ALL THE URLS LISTED.
"""

# Query analyzer prompt template
QUERY_ANALYZER_TEMPLATE = """
Analyze this complex question and break it down into 2-3 sub-questions that will help answer the main question comprehensively:
{question}

Each sub-question should:
1. Address a specific part of the main question
2. Be answerable independently
3. Be clear and focused

Return a JSON object that follows this structure:
{{
    "sub_questions": ["sub-question 1", "sub-question 2", ...]
}}
"""

# Query router prompt template
ROUTER_PROMPT_TEMPLATE = """
Analyze this question's complexity:
{question}

Respond with 'simple' if the question:
- Asks for a single piece of information
- Has a straightforward answer
- Does not require connecting multiple concepts

Respond with 'complex' if the question:
- Contains multiple questions
- Requires comparing or connecting different pieces of information
- Asks for analysis or insights across topics

Respond ONLY with 'simple' or 'complex'
"""

def get_namespace_config(namespace: str, search_mode: str) -> Dict[str, Any]:
    """Get configuration for a specific namespace and search mode, with fallback to defaults."""
    if namespace in SEARCH_CONFIG:
        if search_mode in SEARCH_CONFIG[namespace]:
            return SEARCH_CONFIG[namespace][search_mode]
    
    # Fallback to default namespace
    return SEARCH_CONFIG["default"][search_mode]

