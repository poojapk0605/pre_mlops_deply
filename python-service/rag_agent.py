"""
RAG Agent module that defines the LangGraph workflow and all node functions.
This module handles the core RAG processing logic.
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
import cohere
import config
from pinecone import Pinecone

# Set up logging
logger = logging.getLogger(__name__)

# Try different import paths for the MemorySaver
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    try:
        from langgraph.persist import MemorySaver
    except ImportError:
        # If both fail, create a dummy class that doesn't break the code
        logger.warning("Could not import MemorySaver, state will not persist between calls")
        class MemorySaver:
            pass

class SubQuery(BaseModel):
    sub_questions: List[str] = Field(..., description="List of decomposed sub-questions")

# State definition for LangGraph
class RAGState(TypedDict):
    query: str                      # Original user query
    query_type: Optional[str]       # 'simple' or 'complex'
    sub_questions: List[str]        # Decomposed sub-questions if complex
    docs: List[Any]                 # Retrieved documents
    answer: Optional[str]           # Final answer
    sources: List[str]              # Extracted sources
    timing: Dict[str, float]        # Timing information
    error: Optional[str]            # Any error information
    namespace: str                  # The namespace for this query
    search_mode: str                # 'direct' or 'deepsearch'
    config: Dict[str, Any]          # Configuration for this query

class RAGAgent:
    """RAG Agent implementation with LangGraph workflow."""
    
    def __init__(self):
        """Initialize the RAG Agent with necessary components."""
        # Initialize Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=config.MODEL_CONFIG["embeddings"]["model_name"],
            api_key=config.MODEL_CONFIG["embeddings"]["api_key"]
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index=pc.Index(config.PINECONE_INDEX_NAME),
            embedding=self.embeddings,
            text_key="text"
        )
        
        # Initialize Cohere client
        self.cohere_client = cohere.Client(api_key=config.COHERE_API_KEY)
        
        # Create memory saver for persisting state
        self.memory_saver = MemorySaver()
        
        # Create and compile the workflow
        self.rag_graph = self._create_workflow().compile()
        
    def get_llm(self, config_name: str) -> BaseChatModel:
        """Get the appropriate LLM based on configuration name."""
        if config_name == "gemini":
            model_config = config.MODEL_CONFIG["gemini"]
            return ChatGoogleGenerativeAI(
                model=model_config["model_name"],
                temperature=model_config["temperature"],
                max_retries=model_config["max_retries"],
                google_api_key=model_config["api_key"]
            )
        else:  # Default to OpenAI
            model_config = config.MODEL_CONFIG["openai"]
            return ChatOpenAI(
                model=model_config["model_name"],
                temperature=model_config["temperature"],
                max_retries=model_config["max_retries"],
                api_key=model_config["api_key"]
            )
    
    def create_query_analyzer(self, llm):
        """Create a query analyzer with the specified LLM."""
        query_analyzer_prompt = ChatPromptTemplate.from_template(config.QUERY_ANALYZER_TEMPLATE)
        return query_analyzer_prompt | llm | JsonOutputParser(pydantic_object=SubQuery)
    
    def create_query_router(self, llm):
        """Create a query router with the specified LLM."""
        router_prompt = ChatPromptTemplate.from_template(config.ROUTER_PROMPT_TEMPLATE)
        return router_prompt | llm | StrOutputParser()
    
    def extract_sources_from_metadata(self, docs: List[Any]) -> List[str]:
        """Extract source URLs from document metadata."""
        sources = []
        
        for doc in docs:
            if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                source = doc.metadata.get('source')
                if source and isinstance(source, str):
                    sources.append(source)
        
        return list(set(sources))
    
    def rerank_documents(self, docs, query):
        """Rerank documents using Cohere's reranking API."""
        if not docs:
            logger.warning("No documents to rerank")
            return docs
        
        try:
            logger.info(f"Attempting to rerank {len(docs)} documents using Cohere")
            rerank_start = time.time()
            
            docs_for_reranking = [doc.page_content for doc in docs]
            
            rerank_response = self.cohere_client.rerank(
                model='rerank-v3.5',
                query=query,
                documents=docs_for_reranking,
                top_n=min(len(docs_for_reranking), len(docs_for_reranking) - 1)  # Ensure top_n is valid
            )
            
            reordered_docs = []
            
            for result in rerank_response.results:
                reordered_docs.append(docs[result.index])
            
            logger.info(f"Reranking completed in {time.time() - rerank_start:.2f} seconds")
            return reordered_docs
        
        except Exception as e:
            logger.warning(f"Cohere reranking failed, using original document order: {str(e)}")
            return docs
    
    # LangGraph node functions
    def route_query(self, state: RAGState) -> Dict[str, Any]:
        """Determine if the query is simple or complex."""
        query = state["query"]
        start_time = time.time()
        node_config = state["config"]
        
        llm_name = node_config.get("llm", "openai")
        llm = self.get_llm(llm_name)
        query_router = self.create_query_router(llm)
        
        logger.info(f"Routing query using {llm_name}: {query}")
        query_type = query_router.invoke({"question": query}).strip().lower()
        
        timing = state.get("timing", {})
        timing["routing"] = time.time() - start_time
        
        logger.info(f"Query classified as: {query_type} in {timing['routing']:.2f} seconds")
        
        return {
            **state,
            "query_type": query_type,
            "timing": timing
        }
    
    def decompose_query(self, state: RAGState) -> Dict[str, Any]:
        """Break down complex queries into sub-questions."""
        query = state["query"]
        node_config = state["config"]
        start_time = time.time()
        
        llm_name = node_config.get("llm", "openai")
        llm = self.get_llm(llm_name)
        query_analyzer = self.create_query_analyzer(llm)
        
        logger.info(f"Decomposing complex query using {llm_name}: {query}")
        
        try:
            decomposition_result = query_analyzer.invoke({"question": query})
            try:
                sub_questions = decomposition_result.sub_questions
            except AttributeError:
                logger.warning("Failed to parse SubQuery model, falling back to dict access")
                sub_questions = decomposition_result.get("sub_questions", [])
        except Exception as e:
            logger.error(f"Error during query decomposition: {str(e)}")
            sub_questions = []
            
        timing = state.get("timing", {})
        timing["decomposition"] = time.time() - start_time
        
        logger.info(f"Decomposed into {len(sub_questions)} sub-questions in {timing['decomposition']:.2f} seconds")
        
        return {
            **state,
            "sub_questions": sub_questions,
            "timing": timing
        }
    
    def retrieve_documents_simple(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents for simple queries."""
        query = state["query"]
        node_config = state["config"]
        namespace = state["namespace"]  # Get the namespace from state
        start_time = time.time()
        
        top_n = node_config.get("top_n", 6)
        should_rerank = node_config.get("rerank", True)
        
        logger.info(f"Retrieving documents for simple query with top_n={top_n} in namespace '{namespace}': {query}")
        
        if namespace == "default" or not namespace: # Check if it's default or empty/None
             logger.info("Querying default (unnamed) namespace.")
             docs = self.vectorstore.similarity_search(
                 query,
                 k=top_n,
             )
        else:
             logger.info(f"Querying specific namespace: {namespace}")
             docs = self.vectorstore.similarity_search(
                 query,
                 k=top_n,
                 namespace=namespace # Add namespace parameter here for specific namespaces
             )    

        timing = state.get("timing", {})
        timing["search"] = time.time() - start_time
        
        logger.info(f"Retrieved {len(docs)} documents in {timing['search']:.2f} seconds")
        
        if should_rerank:
            rerank_start = time.time()
            try:
                docs = self.rerank_documents(docs, query)
                timing["reranking"] = time.time() - rerank_start
                logger.info(f"Reranking completed in {timing['reranking']:.2f} seconds")
            except Exception as e:
                timing["reranking"] = time.time() - rerank_start
                logger.warning(f"Reranking failed: {str(e)}")
        
        sources = self.extract_sources_from_metadata(docs)
        
        return {
            **state,
            "docs": docs,
            "sources": sources,
            "timing": timing
        }
    
   

    def retrieve_documents_simple(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents for simple queries."""
        query = state["query"]
        node_config = state["config"]
        namespace = state["namespace"] 
        start_time = time.time()
        
        top_n = node_config.get("top_n", 6)
        should_rerank = node_config.get("rerank", True)
        
        logger.info(f"Retrieving documents for simple query with top_n={top_n} in namespace '{namespace}': {query}")
        
        if namespace == "default":
            docs = self.vectorstore.similarity_search(query, k=top_n)
        else:
            docs = self.vectorstore.similarity_search(
                query, 
                k=top_n,
                namespace=namespace
            )
        
        timing = state.get("timing", {})
        timing["search"] = time.time() - start_time
        
        logger.info(f"Retrieved {len(docs)} documents in {timing['search']:.2f} seconds")
        
        if should_rerank:
            rerank_start = time.time()
            try:
                docs = self.rerank_documents(docs, query)
                timing["reranking"] = time.time() - rerank_start
                logger.info(f"Reranking completed in {timing['reranking']:.2f} seconds")
            except Exception as e:
                timing["reranking"] = time.time() - rerank_start
                logger.warning(f"Reranking failed: {str(e)}")
        
        sources = self.extract_sources_from_metadata(docs)
        
        return {
            **state,
            "docs": docs,
            "sources": sources,
            "timing": timing
        }
    
    def retrieve_documents_complex(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve documents for complex queries using sub-questions."""
        sub_questions = state["sub_questions"]
        node_config = state["config"]
        namespace = state["namespace"]  # Get the namespace from state
        start_time = time.time()
        
        top_n = node_config.get("sub_query_top_n", 3)
        should_rerank = node_config.get("rerank", True)
        
        logger.info(f"Retrieving documents for {len(sub_questions)} sub-questions with top_n={top_n} in namespace '{namespace}'")
        
        all_docs = []
        
        for idx, sub_q in enumerate(sub_questions):
            logger.info(f"Processing sub-question {idx+1}/{len(sub_questions)}: {sub_q}")
            
            if namespace == "default" :
                logger.info("Querying default  namespace.")
                sub_docs = self.vectorstore.similarity_search(
                    sub_q,
                    k=top_n
                )
            else:
                logger.info(f"Querying specific namespace: {namespace}")
                sub_docs = self.vectorstore.similarity_search(
                    sub_q,
                    k=top_n,
                    namespace=namespace 
                )
                
            logger.info(f"Retrieved {len(sub_docs)} documents for sub-question {idx+1}")
            
            if should_rerank:
                try:
                    reranked_docs = self.rerank_documents(sub_docs, sub_q)
                    all_docs.extend(reranked_docs)
                except Exception as e:
                    logger.warning(f"Reranking failed for sub-question {idx+1}: {str(e)}")
                    all_docs.extend(sub_docs)
            else:
                all_docs.extend(sub_docs)
        
        # Deduplicate documents by content
        unique_docs = {}
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in unique_docs:
                unique_docs[content_hash] = doc
        
        docs = list(unique_docs.values())
        
        sources = self.extract_sources_from_metadata(docs)
        
        timing = state.get("timing", {})
        timing["search"] = time.time() - start_time
        
        logger.info(f"Retrieved {len(docs)} unique documents in {timing['search']:.2f} seconds")
        
        return {
            **state,
            "docs": docs,
            "sources": sources,
            "timing": timing
        }
    
    def direct_search(self, state: RAGState) -> Dict[str, Any]:
        """Perform a direct vector search."""
        query = state["query"]
        node_config = state["config"]
        namespace = state["namespace"]  # Get the namespace from state
        start_time = time.time()
        
        top_n = node_config.get("top_n", 10)
        should_rerank = node_config.get("rerank", True)
        
        logger.info(f"Performing direct search for query with top_n={top_n} in namespace '{namespace}': {query}")
        

        if namespace == "default": 
             logger.info("Querying default (unnamed) namespace.")
             docs = self.vectorstore.similarity_search(
                 query,
                 k=top_n,
             )
        else:
             logger.info(f"Querying specific namespace: {namespace}")
             docs = self.vectorstore.similarity_search(
                 query,
                 k=top_n,
                 namespace=namespace 
             )         
        
        timing = state.get("timing", {})
        timing["search"] = time.time() - start_time
        
        logger.info(f"Retrieved {len(docs)} documents in {timing['search']:.2f} seconds")
        
        if should_rerank:
            rerank_start = time.time()
            
            try:
                docs = self.rerank_documents(docs, query)
                timing["reranking"] = time.time() - rerank_start
                logger.info(f"Reranking completed in {timing['reranking']:.2f} seconds")
            except Exception as e:
                timing["reranking"] = time.time() - rerank_start
                logger.warning(f"Reranking failed: {str(e)}")
        
        sources = self.extract_sources_from_metadata(docs)
        
        return {
            **state,
            "docs": docs,
            "sources": sources,
            "timing": timing,
            "query_type": "direct"
        }
    
    def synthesize_answer(self, state: RAGState) -> Dict[str, Any]:
        """Generate a final answer from the retrieved documents."""
        query = state["query"]
        docs = state["docs"]
        sources = state["sources"]
        node_config = state["config"]
        start_time = time.time()
        
        llm_name = node_config.get("llm", "openai")
        llm = self.get_llm(llm_name)
        
        logger.info(f"Synthesizing answer using {llm_name} from {len(docs)} documents")
        
        if not docs:
            answer = "I don't have enough information to answer this question about Northeastern University."
        else:
            sources_list = "\n".join(sources)
            
            synthesis_prompt = ChatPromptTemplate.from_template(config.SYNTHESIS_PROMPT_TEMPLATE)
            
            try:
                answer = (synthesis_prompt | llm | StrOutputParser()).invoke({
                    "question": query,
                    "contexts": "\n\n".join([doc.page_content for doc in docs]),
                    "sources": sources_list
                })
                
                if "Sources:" not in answer and answer != "I don't have enough information to answer this question.":
                    logger.info("Adding missing Sources section to response")
                    answer = answer.strip() + "\n\nSources:\n" + (sources_list if sources else "No relevant sources found.")
            except Exception as e:
                logger.error(f"Error during answer synthesis: {str(e)}")
                answer = f"I encountered an error while synthesizing an answer: {str(e)[:100]}..."
        
        timing = state.get("timing", {})
        timing["synthesis"] = time.time() - start_time
        
        logger.info(f"Synthesis completed in {timing['synthesis']:.2f} seconds")
        
        return {
            **state,
            "answer": answer,
            "timing": timing
        }
    
    def determine_search_path(self, state):
        """Determine which search path to take based on search mode and query type."""
        search_mode = state.get("search_mode", "direct")
        
        if search_mode == "direct":
            return "direct_search"
        elif state["query_type"] == "simple":
            return "retrieve_documents_simple"
        else:
            return "decompose_query"
    
    def _create_workflow(self):
        """Create the LangGraph workflow for RAG processing."""
        workflow = StateGraph(RAGState)
        
        # Add nodes for each step in the workflow
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("decompose_query", self.decompose_query)
        workflow.add_node("retrieve_documents_simple", self.retrieve_documents_simple)
        workflow.add_node("retrieve_documents_complex", self.retrieve_documents_complex)
        workflow.add_node("direct_search", self.direct_search)
        workflow.add_node("synthesize_answer", self.synthesize_answer)
        
        # Define routing based on search mode
        try:
            workflow.add_conditional_edges(
                "route_query",
                self.determine_search_path
            )
        except (TypeError, AttributeError):
            # Fallback to direct edge definition
            logger.info("Using direct edge definition for search mode routing")
            workflow.add_edge("route_query", "direct_search", 
                             lambda x: x.get("search_mode") == "direct")
            workflow.add_edge("route_query", "retrieve_documents_simple", 
                             lambda x: x.get("search_mode") == "deepsearch" and x.get("query_type") == "simple")
            workflow.add_edge("route_query", "decompose_query", 
                             lambda x: x.get("search_mode") == "deepsearch" and x.get("query_type") == "complex")
        
        # After decomposition, retrieve documents for complex queries
        workflow.add_edge("decompose_query", "retrieve_documents_complex")
        
        # All document retrieval paths lead to answer synthesis
        workflow.add_edge("retrieve_documents_simple", "synthesize_answer")
        workflow.add_edge("retrieve_documents_complex", "synthesize_answer")
        workflow.add_edge("direct_search", "synthesize_answer")
        
        # Final node leads to END
        workflow.add_edge("synthesize_answer", END)
        
        # Set the entry point
        workflow.set_entry_point("route_query")
        
        return workflow

    def initialize_graph(self):
        """Initialize the graph by compiling it with memory checkpointing if available."""
        try:
            # Try to compile with checkpoint
            self.rag_graph = self._create_workflow().compile(
                checkpointer=self.memory_saver
            )
        except TypeError:
            # Fallback to simple compile without checkpoint
            logger.warning("Compiling without checkpoint functionality")
            self.rag_graph = self._create_workflow().compile()
        
        return self.rag_graph
    
    def answer_question(self, question: str, namespace: str = "default", search_mode: str = "direct") -> Dict[str, Any]:
        """Process a user question and return a comprehensive answer."""
        logger.info("=" * 50)
        logger.info(f"Processing question in namespace '{namespace}' with search mode '{search_mode}': {question}")
        
        start_time = time.time()
        
        # Validate search_mode
        if search_mode not in ["direct", "deepsearch"]:
            logger.warning(f"Invalid search_mode '{search_mode}', defaulting to 'direct'")
            search_mode = "direct"
        
        # Get configuration for this namespace and search mode
        node_config = config.get_namespace_config(namespace, search_mode)
        logger.info(f"Using configuration: {node_config}")
        
        try:
            # Initialize the state
            initial_state = {
                "query": question,
                "query_type": None,
                "sub_questions": [],
                "docs": [],
                "answer": None,
                "sources": [],
                "timing": {},
                "error": None,
                "namespace": namespace,
                "search_mode": search_mode,
                "config": node_config
            }
            
            # Run the graph with the initial state
            try:
                # Try with config for thread_id support
                result = self.rag_graph.invoke(
                    initial_state,
                    config={"configurable": {"thread_id": namespace}},
                )
            except (TypeError, ValueError):
                # Fallback if config is not supported
                logger.info("Using alternate invocation method without config")
                result = self.rag_graph.invoke(initial_state)
            
            total_time = time.time() - start_time
            logger.info(f"Question answered in {total_time:.2f} seconds using {search_mode} mode with {node_config.get('llm')} LLM")
            
            # Create a simplified report
            report = {
                "question": question,
                "answer": result["answer"],
                "processing_time": {
                    "total": total_time,
                    **result.get("timing", {})
                },
                "sub_questions": result.get("sub_questions", []),
                "metrics": {
                    "sources_found": len(result.get("sources", [])),
                    "documents_retrieved": len(result.get("docs", [])),
                    "query_type": result.get("query_type", "unknown"),
                    "search_mode": search_mode,
                    "llm_used": node_config.get("llm", "unknown")
                },
                "namespace": namespace
            }
            
            return report
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"Error processing question: {str(e)}", exc_info=True)
            
            return {
                "question": question, 
                "answer": f"I encountered an unexpected error while searching for information about Northeastern University. Technical details: {str(e)[:100]}...",
                "processing_time": {
                    "total": error_time
                },
                "error": str(e),
                "namespace": namespace,
                "search_mode": search_mode
            }


