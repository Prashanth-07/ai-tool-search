import os
import time
import re
import json
import logging
import urllib3
from uuid import uuid4
from typing import List, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import requests
from langchain_groq import ChatGroq
from langchain.embeddings.base import Embeddings
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import asyncio
import sys
# Check Python version for asyncio support
if sys.version_info < (3, 7):
    raise RuntimeError("Python 3.7 or higher is required for async functionality")

# Force NumPy implementation to avoid SimSIMD issues
os.environ["USE_NUMPY"] = "1"

# Disable SSL warning messages when verification is disabled
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vectorstore.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_VERIFY_SSL = os.getenv("OLLAMA_VERIFY_SSL", "true").lower() == "true"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "ai-tool-search")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DIMENSION = 768  # for nomic-embed-text

# Verify required environment variables
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="AI Tool Search API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Global state tracking
app_state = {
    "index_ready": False,
    "initialization_started": False,
    "vectors_loaded": 0,
    "total_vectors": 0,
    "bm25_ready": False
}

async def load_vectors_for_bm25():
    """Load vectors in batches for BM25 indexing"""
    fetch_size = 20  # Smaller batch size
    total_vectors = app_state["total_vectors"]
    
    for i in range(0, total_vectors, fetch_size):
        batch_size = min(fetch_size, total_vectors - i)
        logger.info(f"Loading vector batch {i//fetch_size + 1}/{(total_vectors-1)//fetch_size + 1} ({batch_size} vectors)")
        
        try:
            # Load batch of vectors
            vector_store = get_vector_store()
            results = vector_store.similarity_search(
                "",
                k=batch_size
            )
            
            # Add each document to the BM25 index
            for doc in results:
                metadata = doc.metadata
                if "tool_id" in metadata and "rid" in metadata:
                    tool = Tool(
                        name=metadata.get("name", "Unknown"),
                        tool_id=metadata.get("tool_id", ""),
                        description=metadata.get("description", ""),
                        pros=metadata.get("pros", []),
                        cons=metadata.get("cons", []),
                        categories=metadata.get("categories", ""),
                        usage=metadata.get("usage", ""),
                        unique_features=metadata.get("unique_features", ""),
                        pricing=metadata.get("pricing", "")
                    )
                    bm25_index.add_tool(tool, metadata.get("rid"))
            
            # Update progress
            app_state["vectors_loaded"] = min(i + batch_size, total_vectors)
            logger.info(f"Progress: {app_state['vectors_loaded']}/{total_vectors} vectors loaded")
            
            # Give other tasks a chance to run
            await asyncio.sleep(0.01)
            
        except Exception as e:
            logger.error(f"Error loading vector batch: {str(e)}")
    
    # Build the BM25 index after adding all documents
    logger.info("Building BM25 index")
    bm25_index.rebuild_index()
    app_state["bm25_ready"] = True
    logger.info("BM25 index built successfully")

async def initialize_indexes():
    """Load indexes in background with progress tracking"""
    try:
        logger.info("Starting background initialization of vector store and BM25 index")
        
        # Get total vector count first
        try:
            index = pc.Index(INDEX_NAME)
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            app_state["total_vectors"] = total_vectors
            logger.info(f"Found {total_vectors} vectors to load")
        except Exception as e:
            logger.error(f"Error getting vector count: {str(e)}")
            app_state["total_vectors"] = 0
        
        # Initialize Pinecone client and start loading vectors in batches
        _ = get_or_create_index()
        
        # Initialize BM25 index if needed
        if app_state["total_vectors"] > 0:
            logger.info("Initializing BM25 index from existing vectors")
            
            # Load vectors in batches for BM25 indexing
            await load_vectors_for_bm25()
            
            # Mark initialization as complete
            app_state["index_ready"] = True
            logger.info("Background initialization complete - all indexes ready")
        else:
            # No vectors to load
            app_state["index_ready"] = True
            logger.info("No vectors to load, initialization complete")
    except Exception as e:
        logger.error(f"Error during background initialization: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize basic services first, then start background tasks"""
    try:
        logger.info("Application starting up - initializing basic services")
        
        # Verify required environment variables
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        # Start background tasks for heavier initialization
        app_state["initialization_started"] = True
        asyncio.create_task(initialize_indexes())
        
        logger.info("Basic startup complete - API ready for requests")
    except Exception as e:
        logger.error(f"Error during startup initialization: {str(e)}")

# @app.on_event("startup")
# async def startup_event():
#     """Initialize BM25 index on startup"""
#     try:
#         # Force initialization of the vector store and BM25 index
#         logger.info("Application starting up - initializing vector store and BM25 index")
#         _ = get_or_create_index()
        
#         # Check if BM25 index is initialized
#         if not bm25_index.is_initialized:
#             logger.warning("BM25 index not initialized during startup - forcing rebuild")
#             bm25_index.rebuild_index()
            
#         logger.info("Startup initialization complete")
#     except Exception as e:
#         logger.error(f"Error during startup initialization: {str(e)}")

# Helper function to check if URL is HTTPS
def is_https_url(url):
    """Check if URL is HTTPS"""
    return url and url.lower().startswith("https://")

# Get Ollama URL from headers or default
def get_ollama_url(headers=None):
    """
    Get Ollama URL from headers or environment variable
    Headers take precedence over environment variables
    """
    if headers and "OLLAMA_URL" in headers:
        url = headers.get("OLLAMA_URL")
        if url and url.strip():
            logger.info(f"Using custom Ollama URL from headers: {url}")
            return url
    
    logger.info(f"Using default Ollama URL from environment: {OLLAMA_BASE_URL}")
    return OLLAMA_BASE_URL

# Determine if SSL verification should be enabled
def should_verify_ssl(headers=None):
    """
    Determine if SSL verification should be enabled based on headers or env vars
    Headers take precedence over environment variables
    """
    # First check headers
    if headers and "OLLAMA_VERIFY_SSL" in headers:
        verify_ssl = headers.get("OLLAMA_VERIFY_SSL", "true").lower() == "true"
        logger.info(f"SSL verification setting from headers: {verify_ssl}")
        return verify_ssl
    
    # Default to environment variable
    logger.info(f"SSL verification setting from environment: {OLLAMA_VERIFY_SSL}")
    return OLLAMA_VERIFY_SSL

# Get Ollama API key from headers or environment
def get_ollama_api_key(headers=None):
    """Get Ollama API key from headers or environment"""
    if headers and "OLLAMA_API_KEY" in headers:
        api_key = headers.get("OLLAMA_API_KEY")
        if api_key and api_key.strip():
            logger.info("Using API key from headers")
            return api_key
    
    api_key = os.getenv("OLLAMA_API_KEY", "")
    if api_key:
        logger.info("Using API key from environment")
    return api_key

# Get current model based on headers or environment
def get_current_model(headers=None):
    """
    Get the current model based on headers or environment variables
    """
    # Default to environment variable if no headers provided
    if not headers:
        environment = os.getenv("ENVIRONMENT", "DEV")
        if environment == "DEV":
            return os.getenv("DEV_MODEL", "mistral-7b-instruct")
        else:
            return os.getenv("PROD_MODEL", "llama3-8b-8192")
    
    # Use header if provided
    model_choice = headers.get("MODEL_CHOICE", "DEV_MODEL")
    if model_choice == "PROD_MODEL":
        return os.getenv("PROD_MODEL", "llama3-8b-8192")
    else:
        return os.getenv("DEV_MODEL", "mistral-7b-instruct")

# Tool search cache with TTL
class ToolSearchCache:
    def __init__(self, max_size=1000, expiry_minutes=60):
        self.cache = {}
        self.max_size = max_size
        self.expiry = timedelta(minutes=expiry_minutes)
        
    def get(self, query: str):
        if query in self.cache:
            result, timestamp = self.cache[query]
            if datetime.now() - timestamp < self.expiry:
                return result
            else:
                del self.cache[query]
        return None
        
    def set(self, query: str, result: dict):
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
            
        self.cache[query] = (result, datetime.now())

# Initialize cache
tool_search_cache = ToolSearchCache(max_size=1000, expiry_minutes=60)

# Initialize BM25 index
class BM25IndexManager:
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []
        self.tool_data = {}
        self.is_initialized = False
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text):
        """Tokenize and remove stopwords from text"""
        if not text:
            return []
        tokens = word_tokenize(str(text).lower())
        return [token for token in tokens if token.isalnum() and token not in self.stop_words]
    
    def create_document_text(self, tool):
        """Create a searchable text representation of a tool"""
        # Combine relevant fields with appropriate weighting (repeating important fields)
        doc_text = (
            f"{tool.name} {tool.name} {tool.name} "  # Name is most important (repeated)
            f"{tool.description} {tool.description} "
            f"{tool.categories} {tool.categories} {tool.categories}"
            f"{tool.usage} "
            f"{tool.unique_features} "
            f"{' '.join(tool.pros)} "
            f"{' '.join(tool.cons)} "
            f"{tool.pricing}"
        )
        return doc_text
    
    def add_tool(self, tool, rid):
        """Add a single tool to the BM25 index"""
        doc_text = self.create_document_text(tool)
        tokenized_doc = self.preprocess_text(doc_text)
        
        # Store the tool and its tokenized representation
        self.tool_data[rid] = {
            "tool": tool,
            "tokenized_doc": tokenized_doc
        }
        
        # Flag that we need to rebuild the index
        self.is_initialized = False
    
    def rebuild_index(self, batch_size=20):
        """Rebuild the BM25 index with all current documents in batches"""
        self.doc_ids = list(self.tool_data.keys())
        self.tokenized_corpus = []
        
        if not self.doc_ids:
            logger.warning("No documents to index for BM25")
            self.is_initialized = False
            return
        
        # Process in batches
        total_batches = (len(self.doc_ids) - 1) // batch_size + 1
        logger.info(f"Rebuilding BM25 index with {len(self.doc_ids)} documents in {total_batches} batches")
        
        for i in range(0, len(self.doc_ids), batch_size):
            batch_end = min(i + batch_size, len(self.doc_ids))
            batch_ids = self.doc_ids[i:batch_end]
            
            # Process this batch
            batch_docs = [self.tool_data[doc_id]["tokenized_doc"] for doc_id in batch_ids]
            self.tokenized_corpus.extend(batch_docs)
            
            # Log progress
            logger.info(f"Processed batch {i//batch_size + 1}/{total_batches} for BM25 index")
        
        # Create BM25 index after all batches are processed
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_initialized = True
        logger.info(f"BM25 index successfully built with {len(self.doc_ids)} documents")
    
    def search(self, query, top_k=20):
        """Search the BM25 index for the query"""
        if not self.is_initialized:
            self.rebuild_index()
            
        if not self.is_initialized or not self.bm25:
            logger.warning("BM25 index not initialized, returning empty results")
            return []
            
        # Tokenize and preprocess the query
        tokenized_query = self.preprocess_text(query)
        logger.info(f"BM25 search query: '{query}' tokenized as {tokenized_query}")
        
        if not tokenized_query:
            logger.warning("Empty query after preprocessing, returning empty results")
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Log scores for specific tools if needed for debugging
        for idx, doc_id in enumerate(self.doc_ids):
            tool = self.tool_data[doc_id]["tool"]
            if hasattr(tool, 'tool_id') and tool.tool_id in ['iconme-023', 'contentgoblinai-002']:
                logger.info(f"Tool ID: {tool.tool_id}, Score: {scores[idx]}")
        
        # Get top-k results
        top_k = min(top_k, len(self.doc_ids))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                doc_id = self.doc_ids[idx]
                tool = self.tool_data[doc_id]["tool"]
                results.append({
                    "rid": doc_id,
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "description": tool.description,
                    "categories": tool.categories,
                    "usage": tool.usage,
                    "unique_features": tool.unique_features,
                    "pros": tool.pros,
                    "cons": tool.cons,
                    "pricing": tool.pricing,
                    "score": float(scores[idx])
                })
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results

bm25_index = BM25IndexManager()

# Pydantic models
class Tool(BaseModel):
    name: str
    tool_id: str
    description: str
    pros: List[str]
    cons: List[str]
    categories: str
    usage: str
    unique_features: str
    pricing: str

class ToolResponse(BaseModel):
    id: str
    tool: Tool
    status: str = "added"

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

class DeleteResponse(BaseModel):
    success: bool
    deleted_tool: str

class BulkToolRequest(BaseModel):
    tools: List[Tool]

class BulkToolResponse(BaseModel):
    results: List[ToolResponse]

class BulkUpdateRequest(BaseModel):
    tools: List[Tool]

class BulkUpdateResponse(BaseModel):
    results: List[ToolResponse]

class ClearIndexRequest(BaseModel):
    api_key: str = Field(..., description="API key for authorization to clear the index")

# Get total vectors in the index
def get_total_vectors() -> int:
    """Get total number of vectors in the index"""
    try:
        index = pc.Index(INDEX_NAME)
        stats = index.describe_index_stats()
        total_vectors = stats.total_vector_count
        logger.info(f"Total vectors in index: {total_vectors}")
        return total_vectors
    except Exception as e:
        logger.error(f"Error getting vector count: {str(e)}")
        return 0

# Vector store initialization
def get_or_create_index():
    """Create Pinecone index if it doesn't exist"""
    try:
        # Check if index exists
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        
        if INDEX_NAME not in index_names:
            print(f"Creating new index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine"
            )
            print(f"Index {INDEX_NAME} created successfully")
        
        # Initialize BM25 index if needed - we're changing this condition
        # from: if get_total_vectors() > 0 and not bm25_index.is_initialized:
        # to:   if get_total_vectors() > 0: (Always populate if vectors exist)
        if get_total_vectors() > 0:
            logger.info("Initializing BM25 index from existing vectors")
            index = pc.Index(INDEX_NAME)
            
            # Get all vectors (in batches if there are many)
            fetch_size = 1000
            total_vectors = get_total_vectors()
            
            for i in range(0, total_vectors, fetch_size):
                # Use vector_store to get batches of documents
                vector_store = get_vector_store()
                results = vector_store.similarity_search(
                    "",
                    k=min(fetch_size, total_vectors - i)
                )
                
                # Add each document to the BM25 index
                for doc in results:
                    metadata = doc.metadata
                    if "tool_id" in metadata and "rid" in metadata:
                        # Reconstruct the Tool object from metadata
                        tool = Tool(
                            name=metadata.get("name", "Unknown"),
                            tool_id=metadata.get("tool_id", ""),
                            description=metadata.get("description", ""),
                            pros=metadata.get("pros", []),
                            cons=metadata.get("cons", []),
                            categories=metadata.get("categories", ""),
                            usage=metadata.get("usage", ""),
                            unique_features=metadata.get("unique_features", ""),
                            pricing=metadata.get("pricing", "")
                        )
                        bm25_index.add_tool(tool, metadata.get("rid"))
                
                logger.info(f"Added batch of {len(results)} documents to BM25 index")
            
            # Build the index after adding all documents
            bm25_index.rebuild_index()
            print(f"===== BM25 INDEX CONTENTS =====")
            print(f"Total tools indexed in BM25: {len(bm25_index.tool_data)}")
            print(f"Is 'iconme-023' in BM25 index: {'iconme-023' in [tool.tool_id for rid, data in bm25_index.tool_data.items() for tool in [data['tool']]]}")
            print(f"Is 'contentgoblinai-002' in BM25 index: {'contentgoblinai-002' in [tool.tool_id for rid, data in bm25_index.tool_data.items() for tool in [data['tool']]]}")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error in get_or_create_index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Pinecone index: {str(e)}"
        )
    
class NomicAtlasEmbeddings(Embeddings):
    """Wrapper for Nomic Atlas embedding API."""
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api-atlas.nomic.ai/v1/embedding/text",
        task_type: str = "search_document",
        max_tokens_per_text: int = 8192,
        dimensionality: int = 768,
    ):
        """Initialize the Nomic Atlas embeddings wrapper."""
        self.api_key = api_key
        self.api_url = api_url
        self.task_type = task_type
        self.max_tokens_per_text = max_tokens_per_text
        self.dimensionality = dimensionality
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using the Nomic Atlas API."""
        max_retires=3
        retry_delay=1
        try:
            payload = {
                "texts": texts,
                "task_type": self.task_type,
                "max_tokens_per_text": self.max_tokens_per_text,
                "dimensionality": self.dimensionality
            }
            
            response = requests.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            return result["embeddings"]
        except Exception as e:
            logger.warning(f"Error calling Nomic Atlas API (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise ValueError(f"Error calling Nomic Atlas API: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using the Nomic Atlas API."""
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
# Global cache for vector store instance
_vector_store_cache = None
_vector_store_timestamp = None
_cache_lifetime = 3600 
def get_vector_store(headers=None):
    """Initialize or return existing vector store using Nomic Atlas API."""
    global _vector_store_cache, _vector_store_timestamp
    # Check if cache is valid
    current_time = time.time()
    if (_vector_store_cache is not None and 
        _vector_store_timestamp is not None and 
        current_time - _vector_store_timestamp < _cache_lifetime):
        logger.info("Using cached vector store instance")
        return _vector_store_cache

    try:
        # Get the API key from environment variables
        nomic_api_key = os.getenv("NOMIC_API_KEY")
        if not nomic_api_key:
            raise ValueError("NOMIC_API_KEY is not set in the environment.")
        
        # Create an embeddings instance using our custom wrapper
        embeddings = NomicAtlasEmbeddings(
            api_key=nomic_api_key,
            api_url="https://api-atlas.nomic.ai/v1/embedding/text",
            task_type="search_document",
            max_tokens_per_text=8192,
            dimensionality=DIMENSION  # This uses the global DIMENSION variable (768)
        )
        
        # Initialize the vector store using Langchain's Pinecone integration
        vector_store = LangchainPinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        _vector_store_cache = vector_store
        _vector_store_timestamp = current_time
        return vector_store
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize vector store: {str(e)}"
        )


# Get vector store - keep using Ollama for embeddings

# def get_vector_store(headers=None):
#     """Initialize or return existing vector store"""
#     try:
#         # Get URL and SSL verification setting
#         ollama_url = get_ollama_url(headers)
#         verify_ssl = should_verify_ssl(headers)
#         api_key = get_ollama_api_key(headers)
        
#         if is_https_url(ollama_url):
#             logger.info(f"Using HTTPS for Ollama embeddings with SSL verification: {verify_ssl}")

#         # Create client_kwargs with authentication if available
#         client_kwargs = {"verify": verify_ssl}
#         if api_key:
#             client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}
#             logger.info("Added authorization header for embeddings")
            
#         embeddings = OllamaEmbeddings(
#             model='nomic-embed-text',
#             base_url=ollama_url,
#             client_kwargs=client_kwargs
#         )
        
#         # Initialize the vector store using the new syntax
#         vector_store = LangchainPinecone.from_existing_index(
#             index_name=INDEX_NAME,
#             embedding=embeddings,
#             text_key="text"
#         )
        
#         return vector_store
#     except Exception as e:
#         print(f"Error in get_vector_store: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to initialize vector store: {str(e)}"
#         )
    
def format_tool_for_indexing(tool: Tool, rid: str) -> str:
    """Format tool data for embedding"""
    return (
        # f"document: " 
        f"RID: {rid}\n"
        f"Name: {tool.name}\n"
        f"Description: {tool.description}\n"
        f"Pros: {', '.join(tool.pros)}\n"
        f"Cons: {', '.join(tool.cons)}\n"
        f"Categories: {tool.categories}\n"
        f"Usage: {tool.usage}\n"
        f"Unique Features: {tool.unique_features}\n"
        f"Pricing: {tool.pricing}"
    )

def post_process_llm_response(response_text):
    """
    Post-processing to:
    1. Remove <think> tags
    2. Remove markdown code block delimiters (```json ... ```)
    3. Return clean JSON or text
    """
    try:
        # Remove thinking section if present
        if "<think>" in response_text:
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()
        
        # Check if the response is wrapped in markdown code block
        code_block_match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```\s*$', response_text, re.DOTALL)
        if code_block_match:
            # Extract just the content between the code block delimiters
            response_text = code_block_match.group(1).strip()
        
        # Return the cleaned response
        return response_text
    
    except Exception as e:
        # If anything fails, just return the original response
        logger.warning(f"Error in post-processing: {str(e)}")
        return response_text

async def check_duplicate_tool(vector_store, tool: Tool) -> bool:
    """
    Check if a tool with the same tool_id already exists.
    Returns True if duplicate found, False otherwise.
    """
    try:
        # Search specifically for the tool_id
        results = vector_store.similarity_search(
            "",  # Empty query string
            k=1,  # We only need to find one match
            filter={"tool_id": tool.tool_id}  # Filter by the unique tool_id
        )
        
        # If we got any results, a duplicate exists
        return len(results) > 0
    
    except Exception as e:
        logger.error(f"Error checking for duplicate: {str(e)}")
        return False  # Assume no duplicate in case of error, safer to check manually

def fuse_search_results(vector_results, bm25_results, alpha=0.7):
    """
    Fuse vector search and BM25 search results using a weighted approach.
    
    Args:
        vector_results: List of results from vector search with scores
        bm25_results: List of results from BM25 search with scores
        alpha: Weight for vector search (1-alpha will be the weight for BM25)
        
    Returns:
        List of fused results ordered by combined score
    """
    # Collect all unique document IDs
    all_ids = set()
    for result in vector_results:
        all_ids.add(result.get("rid", ""))
    for result in bm25_results:
        all_ids.add(result.get("rid", ""))
    
    # Remove empty IDs
    if "" in all_ids:
        all_ids.remove("")
    
    # Create score maps
    vector_scores = {result.get("rid", ""): result.get("score", 0) for result in vector_results}
    bm25_scores = {result.get("rid", ""): result.get("score", 0) for result in bm25_results}
    
    # Normalize scores within each method
    if vector_scores:
        max_vector_score = max(vector_scores.values()) if vector_scores.values() else 1
        vector_scores = {k: v/max_vector_score for k, v in vector_scores.items()}
    
    if bm25_scores:
        max_bm25_score = max(bm25_scores.values()) if bm25_scores.values() else 1
        bm25_scores = {k: v/max_bm25_score for k, v in bm25_scores.items()}
    
    # Combine scores
    combined_results = []
    for doc_id in all_ids:
        v_score = vector_scores.get(doc_id, 0)
        b_score = bm25_scores.get(doc_id, 0)
        combined_score = alpha * v_score + (1 - alpha) * b_score
        
        # Find the full result object
        result_obj = None
        for result in vector_results:
            if result.get("rid", "") == doc_id:
                result_obj = result
                break
        
        if not result_obj:
            for result in bm25_results:
                if result.get("rid", "") == doc_id:
                    result_obj = result
                    break
        
        if result_obj:
            # Create a new result with combined score
            combined_result = dict(result_obj)
            combined_result["score"] = combined_score
            combined_result["vector_score"] = v_score
            combined_result["bm25_score"] = b_score
            combined_results.append(combined_result)
    
    # Sort by combined score
    combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return combined_results

# API endpoints
@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "active",
        "message": "AI Tool Search API is running (with Groq LLM integration)",
        "version": "1.0"
    }

@app.get("/health")
async def health_check():
    """Check if API and Pinecone connection are healthy"""
    try:
        logger.info("Health check started")
        
        # Check if all required environment variables are set
        logger.info("Checking environment variables")
        if not PINECONE_API_KEY:
            logger.error("PINECONE_API_KEY is not set")
            raise ValueError("PINECONE_API_KEY environment variable is not set")
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not set")
            raise ValueError("GROQ_API_KEY environment variable is not set")
        
        logger.info("Initializing Pinecone client")
        # Initialize Pinecone client
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        logger.info("Verifying Pinecone connection")
        # Verify Pinecone connection
        _ = get_or_create_index()
        
        logger.info("Health check completed successfully")
        return {
            "status": "healthy",
            "api_version": "1.0",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )
    
@app.get("/basic-health")
async def basic_health_check():
    """Simple health check that doesn't test Pinecone connection"""
    try:
        return {
            "status": "api_running",
            "api_version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "note": "This endpoint only checks if the API is running, not the Pinecone connection"
        }
    except Exception as e:
        logger.error(f"Basic health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/initialization-status")
async def get_initialization_status():
    """Get the current initialization status"""
    return {
        "initialized": app_state["index_ready"],
        "initialization_started": app_state["initialization_started"],
        "vectors_loaded": app_state["vectors_loaded"],
        "total_vectors": app_state["total_vectors"],
        "loading_percentage": (app_state["vectors_loaded"] / max(app_state["total_vectors"], 1)) * 100,
        "bm25_index_ready": app_state["bm25_ready"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/query", response_model=QueryResponse)
async def query_tools(request: QueryRequest, request_headers: Request):
    """Query tools based on user input using hybrid search (vector + keyword)."""
    start_time = time.time()
    try:
        logger.info(f"Processing query: {request.query}")
        headers = request_headers.headers

        # Try to get cached response
        cached_response = tool_search_cache.get(request.query)
        if cached_response:
            logger.info("Returning cached response")
            return QueryResponse(response=cached_response)
        
        # Check if indexes are ready
        if not app_state["initialization_started"]:
            logger.warning("Query received before initialization started")
            error_response = json.dumps({
                "error": "system_not_initialized",
                "message": "The system has not started initialization yet. Please try again later.",
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            return QueryResponse(response=error_response)
        
        if not app_state["index_ready"]:
            # System is still initializing - provide status and limited functionality
            progress = (app_state["vectors_loaded"] / max(app_state["total_vectors"], 1)) * 100
            logger.info(f"Query received during initialization. Progress: {progress:.1f}%")
            initializing_response = json.dumps({
                "status": "initializing",
                "message": f"The system is still initializing. Currently loaded {app_state['vectors_loaded']} of {app_state['total_vectors']} tools ({progress:.1f}%).",
                "progress_percentage": progress,
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            return QueryResponse(response=initializing_response)
        
        # Get the vector store
        vector_store = get_vector_store(headers)

        # Log total vectors for reference
        total_vectors = get_total_vectors()
        logger.info(f"Total vectors in store: {total_vectors}")
        
        # Hybrid search implementation
        try:
            # Step 1: Perform vector search with scores
            vector_k = 15  # Increase to get more candidates
            vector_results_with_scores = vector_store.similarity_search_with_score(
                request.query,
                k=vector_k
            )
            
            # Convert to a list of result dictionaries
            processed_vector_results = []
            for doc, score in vector_results_with_scores:
                metadata = doc.metadata
                processed_vector_results.append({
                    "rid": metadata.get("rid", ""),
                    "tool_id": metadata.get("tool_id", ""),
                    "name": metadata.get("name", ""),
                    "description": metadata.get("description", ""),
                    "categories": metadata.get("categories", ""),
                    "usage": metadata.get("usage", ""),
                    "unique_features": metadata.get("unique_features", ""),
                    "pros": metadata.get("pros", []),
                    "cons": metadata.get("cons", []),
                    "pricing": metadata.get("pricing", ""),
                    "score": float(1.0 - score)  # Convert distance to similarity score
                })
            
            logger.info(f"Vector search returned {len(processed_vector_results)} results")
            
            # Step 2: Perform BM25 search
            bm25_k = 30  # Get similar number of results
            bm25_results = bm25_index.search(request.query, top_k=bm25_k)
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            
            # Step 3: Fuse the results (using 0.5 weight for vector search, 0.5 for BM25)
            hybrid_results = fuse_search_results(
                processed_vector_results, 
                bm25_results,
                alpha=0.5  # Equal weight for vector and keyword search
            )
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            
            # Log details of specific tools if needed
            tool_ids = [result.get('tool_id', 'N/A') for result in hybrid_results]
            logger.debug(f"Tools in hybrid results: {tool_ids}")
            
            # Step 4: Take top results (up to 10) for LLM processing
            top_results = hybrid_results[:10]
            
            # Format the documents for LLM
            formatted_docs = []
            for result in top_results:
                formatted_doc = (
                    f"Tool ID: {result.get('tool_id', 'N/A')}\n"
                    f"Name: {result.get('name', 'N/A')}\n"
                    f"Description: {result.get('description', 'N/A')}\n"
                    f"Categories: {result.get('categories', 'N/A')}\n"
                    f"Usage: {result.get('usage', 'N/A')}\n"
                    f"Unique Features: {result.get('unique_features', 'N/A')}\n"
                    f"Pros: {', '.join(result.get('pros', []))}\n"
                    f"Cons: {', '.join(result.get('cons', []))}\n"
                    f"Pricing: {result.get('pricing', 'N/A')}"
                )
                formatted_docs.append(formatted_doc)
                # Log details for debugging if needed
                logger.debug(f"Tool: {result.get('tool_id', 'N/A')}, Score: V={result.get('vector_score', 0):.4f}, BM25={result.get('bm25_score', 0):.4f}, Combined={result.get('score', 0):.4f}")
            
            # Join the formatted documents with separators
            context = "\n\n---\n\n".join(formatted_docs)
            
            # Get the model based on headers
            current_model = get_current_model(headers)
            logger.info(f"Using model: {current_model} with Groq API")

            system_text= f"""
You are a tool retrieval assistant tasked with finding, picking and ranking all relevant tools from a provided Tool Data based on a User Query.
Instructions:
- Analyze the User Query to understand their needs.
- Find all relevant tools that could help with this specific task.
- Carefully evaluate each tool's relevance to the query.
- Rank them from most to least relevant.
- For each tool, generate:
  - A short summary (1–2 lines) explaining how it helps with the query.
  - 2–3 bullet points showing key features that make it useful for this task.
- Format your response as a JSON object with the following schema:
{{
  "tool_id": ["most_relevant_id", "next_most_relevant_id", ...],
  "tools": [
    {{
      "id": "tool_id",
      "name": "Tool Name",
      "description": "How this tool helps the user based on their query",
      "bullets": [
        "Feature or benefit 1",
        "Feature or benefit 2",
        "Optional feature or benefit 3"
      ]
    }},
    ...
  ]
}}

Guidelines:
- Strict ordering: The most relevant tool MUST be listed first, followed by decreasing relevance.
- Include all the relevant tools that matches the User Query.
- Only include tools that are relevant to the User Query.
- Match the tool description to the user's specific query.
- Output ONLY the JSON. No preamble No extra note..
"""
            
            # Create prompt for LangChain
            prompt_text = f"""
User Query: {request.query}

Tool Data: {context}
"""

            # Initialize LangChain's ChatGroq
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=current_model,
                temperature=0.1
            )
            # print(f"\n===== TOOLS SENT TO LLM (Tool Data) =====")
            # print(context)
            # print("============================\n")
            # print(f"\n===== TOOLS SENT TO LLM =====")
            # for idx, doc in enumerate(formatted_docs):
            #     print(f"Tool {idx+1}:\n{doc}\n")
            # print("============================\n")
            
            # Get response from Groq LLM with timeout handling
            try:
                response = llm.invoke([
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt_text}
                ])

                
                # Extract content from response
                llm_response = response.content
                # print(f"\n===== LLM RESPONSE =====")
                # print(llm_response)
                # print("============================\n")
                
                # Post-process the response
                processed_response = post_process_llm_response(llm_response)
                
                # Try to parse as JSON but don't fail if it's not valid
                try:
                    # Parse and validate response
                    response_data = json.loads(processed_response)
                    
                    # Clean response
                    clean_response = json.dumps(response_data, indent=2)
                    logger.info("Processed valid JSON response")
                except json.JSONDecodeError:
                    # If not valid JSON, just return the processed response as-is
                    clean_response = processed_response
                    logger.warning("Response is not valid JSON, returning as-is")
                    
                    # Try to construct a minimal valid JSON if parsing failed
                    if not processed_response or processed_response.strip() == "":
                        clean_response = json.dumps({
                            "tool_id": [],
                            "tools": [],
                            "error": "No valid response generated"
                        })
                
                # Cache the processed response
                tool_search_cache.set(request.query, clean_response)
                
                # Log total processing time
                elapsed_time = time.time() - start_time
                logger.info(f"Total processing time: {elapsed_time:.2f}s")
                
                return QueryResponse(response=clean_response)
            except Exception as e:
                # Handle LLM-specific errors
                logger.error(f"Error in LLM call: {str(e)}")
                error_response = json.dumps({
                    "error": "llm_error",
                    "message": f"Error processing query with language model: {str(e)}",
                    "tools": [],
                    "timestamp": datetime.now().isoformat()
                })
                return QueryResponse(response=error_response)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in hybrid search: {error_msg}")
            
            # Create a detailed error response
            fallback_response = json.dumps({
                "error": "search_error",
                "message": f"Failed to process search: {error_msg}",
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            
            return QueryResponse(response=fallback_response)
            
    except Exception as e:
        logger.error(f"Error in query_tools: {str(e)}")
        
        # Log total processing time even on error
        elapsed_time = time.time() - start_time
        logger.info(f"Query processing completed with error in {elapsed_time:.2f}s")
        
        # Return a detailed error response
        error_response = json.dumps({
            "error": "api_error",
            "message": f"An unexpected error occurred: {str(e)}",
            "tools": [],
            "timestamp": datetime.now().isoformat()
        })
        return QueryResponse(response=error_response)

@app.post("/add-tools", response_model=BulkToolResponse)
async def add_tools(bulk_request: BulkToolRequest):
    """Add multiple tools to the vector store with duplicate checking."""
    try:
        vector_store = get_vector_store()
        results = []
        skipped_tools = []
        added_tools = []
        
        # First check for duplicates for all tools
        for tool in bulk_request.tools:
            is_duplicate = await check_duplicate_tool(vector_store, tool)
            if is_duplicate:
                skipped_tools.append(tool.tool_id)
                logger.info(f"Skipping duplicate tool: {tool.name} (Tool ID: {tool.tool_id})")
                results.append(ToolResponse(
                    id=f"duplicate-{tool.tool_id}",  # Use a placeholder ID
                    tool=tool,
                    status="skipped_duplicate"
                ))
            else:
                added_tools.append(tool)
        
        # Now add only the non-duplicate tools
        for tool in added_tools:
            # Generate unique RID for each tool
            rid = str(uuid4())
            # Format tool data
            tool_text = format_tool_for_indexing(tool, rid)
            # Prepare metadata
            metadata = {
                "rid": rid,
                "tool_id": tool.tool_id,
                "name": tool.name,
                **tool.model_dump()
            }
            # Add document to vector store
            vector_store.add_texts(
                texts=[tool_text],
                metadatas=[metadata],
                ids=[rid]
            )
            
            # Also add to BM25 index
            bm25_index.add_tool(tool, rid)
            
            logger.info(f"Added tool: {tool.name} (Tool ID: {tool.tool_id})")
            results.append(ToolResponse(id=rid, tool=tool, status="added"))
        
        # Rebuild BM25 index if any tools were added
        if added_tools:
            bm25_index.rebuild_index()
        
        new_count = get_total_vectors()
        logger.info(f"Vector count after addition: {new_count}")
        
        if skipped_tools:
            logger.info(f"Skipped {len(skipped_tools)} duplicate tools: {', '.join(skipped_tools)}")
        
        return BulkToolResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error in add_tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add tools: {str(e)}"
        )
    
@app.delete("/delete-tool/{tool_id}", response_model=DeleteResponse)
async def delete_tool(tool_id: str):
    """Delete a tool by its provided tool_id."""
    try:
        # Log initial count
        initial_count = get_total_vectors()
        logger.info(f"Current vector count before deletion: {initial_count}")
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Retrieve the tool using a filter on 'tool_id'
        search_results = vector_store.similarity_search(
            "",
            k=1,
            filter={"tool_id": tool_id}
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        tool_name = search_results[0].metadata.get("name", "Unknown tool")
        # Retrieve the unique generated RID from the metadata
        rid = search_results[0].metadata.get("rid")
        if not rid:
            raise HTTPException(status_code=500, detail="RID not found for the tool")
        
        # Delete the vector using the retrieved RID
        index = get_or_create_index()
        index.delete(ids=[rid])
        
        # Also remove from BM25 index if it exists
        if rid in bm25_index.tool_data:
            del bm25_index.tool_data[rid]
            # Mark that the index needs rebuilding
            bm25_index.is_initialized = False
            logger.info(f"Removed tool from BM25 index: {tool_name} (RID: {rid})")
        
        # Log new vector count
        new_count = get_total_vectors()
        logger.info(f"Vector count after deletion: {new_count}")
        logger.info(f"Deleted tool: {tool_name} (Tool ID: {tool_id}, RID: {rid})")
        
        return DeleteResponse(
            success=True,
            deleted_tool=tool_name
        )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in delete_tool: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete tool: {str(e)}"
        )

@app.put("/update-tools", response_model=BulkUpdateResponse)
async def update_tools(request: BulkUpdateRequest):
    """Update multiple tools in bulk using their tool_ids."""
    try:
        vector_store = get_vector_store()
        results = []
        updated_rids = []
        
        for tool in request.tools:
            try:
                # Search for the existing record using tool_id
                search_results = vector_store.similarity_search(
                    "",
                    k=1,
                    filter={"tool_id": tool.tool_id}
                )
                
                if not search_results:
                    logger.warning(f"Tool not found: {tool.tool_id}")
                    continue
                
                # Get the existing RID
                existing_rid = search_results[0].metadata.get("rid")
                if not existing_rid:
                    logger.warning(f"RID not found for tool: {tool.tool_id}")
                    continue
                
                # Format tool data for update
                tool_text = format_tool_for_indexing(tool, existing_rid)
                
                # Prepare metadata
                metadata = {
                    "rid": existing_rid,
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    **tool.model_dump()
                }
                
                # Update the vector
                vector_store.add_texts(
                    texts=[tool_text],
                    metadatas=[metadata],
                    ids=[existing_rid]
                )
                
                # Update BM25 index
                bm25_index.add_tool(tool, existing_rid)
                updated_rids.append(existing_rid)
                
                logger.info(f"Updated tool: {tool.name} (Tool ID: {tool.tool_id}, RID: {existing_rid})")
                results.append(ToolResponse(id=existing_rid, tool=tool, status="updated"))
                
            except Exception as e:
                logger.error(f"Error updating tool {tool.tool_id}: {str(e)}")
                continue
        
        # Rebuild BM25 index if any tools were updated
        if updated_rids:
            bm25_index.rebuild_index()
        
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No tools were updated successfully"
            )
        
        return BulkUpdateResponse(results=results)
        
    except Exception as e:
        logger.error(f"Error in bulk update: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process bulk update: {str(e)}"
        )
    
@app.get("/test-connection")
async def test_connection(request: Request):
    """Test connection to Nomic Atlas and Groq API with current settings"""
    try:
        # Save the original headers
        original_headers = request.headers
        model_choice = original_headers.get("MODEL_CHOICE", "DEV_MODEL")
        
        # Add debugging
        logger.info(f"Test connection with model_choice: {model_choice}")
        
        test_results = {
            "nomic_atlas": "enabled",
            "groq_api": "enabled",
            "endpoints_tested": []
        }
        
        # Test Nomic Atlas embeddings endpoint
        try:
            nomic_api_key = os.getenv("NOMIC_API_KEY")
            if not nomic_api_key:
                raise ValueError("NOMIC_API_KEY environment variable is not set")
                
            nomic_headers = {  # Use a different variable name
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {nomic_api_key}"
            }
            
            embed_response = requests.post(
                "https://api-atlas.nomic.ai/v1/embedding/text",
                headers=nomic_headers,  # Use the renamed variable
                json={"texts": ["Test"], "task_type": "search_document", "max_tokens_per_text": 8192, "dimensionality": 768},
                timeout=3  # Reduced timeout
            )
            
            embed_status = {
                "endpoint": "nomic_atlas_embeddings",
                "status_code": embed_response.status_code,
                "success": embed_response.status_code == 200
            }
            
            if embed_response.status_code == 200:
                data = embed_response.json()
                embed_status["embedding_size"] = len(data.get("embeddings", [])[0])
            else:
                embed_status["error"] = embed_response.text
                
            test_results["endpoints_tested"].append(embed_status)
        except Exception as e:
            test_results["endpoints_tested"].append({
                "endpoint": "nomic_atlas_embeddings",
                "success": False,
                "error": str(e)
            })
        
        # Test Groq chat endpoint with LangChain
        try:
            model = get_current_model(original_headers)  # Use original headers
            logger.info(f"Testing Groq with model: {model}")
            
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=model,
                max_tokens=20  # Very limited response
            )
            
            response = llm.invoke("Hi")
            
            chat_status = {
                "endpoint": "groq_chat_completions",
                "model": model,
                "status_code": 200,
                "success": True,
                "response": "Groq chat endpoint working"
            }
            
            test_results["endpoints_tested"].append(chat_status)
        except Exception as e:
            test_results["endpoints_tested"].append({
                "endpoint": "groq_chat_completions",
                "success": False,
                "error": str(e)
            })
        
        # Determine overall status
        all_success = all(endpoint["success"] for endpoint in test_results["endpoints_tested"])
        
        if all_success:
            test_results["status"] = "success"
            test_results["message"] = "Successfully connected to all endpoints"
            return test_results
        else:
            test_results["status"] = "partial" if any(endpoint["success"] for endpoint in test_results["endpoints_tested"]) else "error"
            test_results["message"] = "Some endpoints failed connection test"
            return test_results
            
    except Exception as e:
        return {
            "status": "error",
            "message": f"Exception: {str(e)}",
            "details": {
                "error_type": type(e).__name__
            }
        }
    
@app.get("/model-info")
async def get_model_info(request: Request):
    """Get current model information"""
    try:
        headers = request.headers
        current_model = get_current_model(headers)
        return {
            "current_model": current_model,
            "provider": "Groq (LLM) / Nomic Atlas (Embeddings)",
            "environment": os.getenv("ENVIRONMENT", "DEV")
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.get("/stats")
async def get_stats(show_all: bool = True):
    """Get vector store statistics and vector metadata"""
    try:
        # Get basic stats
        total_vectors = get_total_vectors()
        index = get_or_create_index()
        stats = index.describe_index_stats()
        
        # Get vector store to fetch metadata
        vector_store = get_vector_store()
        
        # Set fetch limit based on show_all parameter
        max_vectors_to_fetch = total_vectors if show_all else min(total_vectors, 50)
        
        # Query vectors with limit
        results = vector_store.similarity_search(
            "",
            k=max_vectors_to_fetch
        )
        
        # Extract vector details
        vectors_info = []
        for doc in results:
            vectors_info.append({
                "name": doc.metadata.get("name", "N/A"),
                "tool_id": doc.metadata.get("tool_id", "N/A"),
                "rid": doc.metadata.get("rid", "N/A"),
                # "description": doc.metadata.get("description", "N/A")[:100] + "...",
                "description": doc.metadata.get("description", "N/A"),  # Truncate long descriptions
                "categories": doc.metadata.get("categories", "N/A"),
                "pricing": doc.metadata.get("pricing", "N/A")
            })
        
        # Create complete stats dictionary
        stats_dict = {
            "total_vectors": total_vectors,
            "vectors_shown": len(vectors_info),
            "dimension": DIMENSION,
            "index_fullness": float(stats.index_fullness) if hasattr(stats, 'index_fullness') else 0.0,
            "namespaces": {},
            "vectors": vectors_info
        }
        
        # Add namespace information if available
        if hasattr(stats, 'namespaces'):
            for namespace, ns_stats in stats.namespaces.items():
                stats_dict["namespaces"][namespace] = {
                    "vector_count": getattr(ns_stats, 'vector_count', 0),
                    "metadata": {}
                }
        
        return stats_dict
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )

# @app.get("/stats")
# async def get_stats():
#     """Get vector store statistics and vector metadata"""
#     try:
#         # Get basic stats
#         total_vectors = get_total_vectors()
#         index = get_or_create_index()
#         stats = index.describe_index_stats()
        
#         # Get vector store to fetch metadata
#         vector_store = get_vector_store()
        
#         # Limit results for large indices to avoid slow response
#         max_vectors_to_fetch = min(total_vectors, 50)  # Only show up to 50 vectors
        
#         # Query vectors with limit
#         results = vector_store.similarity_search(
#             "",
#             k=max_vectors_to_fetch
#         )
        
#         # Extract vector details
#         vectors_info = []
#         for doc in results:
#             vectors_info.append({
#                 "name": doc.metadata.get("name", "N/A"),
#                 "tool_id": doc.metadata.get("tool_id", "N/A"),
#                 "rid": doc.metadata.get("rid", "N/A"),
#                 "description": doc.metadata.get("description", "N/A")[:100] + "...",  # Truncate long descriptions
#                 "categories": doc.metadata.get("categories", "N/A"),
#                 "pricing": doc.metadata.get("pricing", "N/A")
#             })
        
#         # Create complete stats dictionary
#         stats_dict = {
#             "total_vectors": total_vectors,
#             "vectors_shown": len(vectors_info),
#             "dimension": DIMENSION,
#             "index_fullness": float(stats.index_fullness) if hasattr(stats, 'index_fullness') else 0.0,
#             "namespaces": {},
#             "vectors": vectors_info
#         }
        
#         # Add namespace information if available
#         if hasattr(stats, 'namespaces'):
#             for namespace, ns_stats in stats.namespaces.items():
#                 stats_dict["namespaces"][namespace] = {
#                     "vector_count": getattr(ns_stats, 'vector_count', 0),
#                     "metadata": {}
#                 }
        
#         return stats_dict
        
#     except Exception as e:
#         logger.error(f"Error getting stats: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to get stats: {str(e)}"
#         )

@app.delete("/clear-index", response_model=Dict[str, Any])
async def clear_index(request: ClearIndexRequest):
    """Delete all vectors in the Pinecone index. Requires Pinecone API key authorization."""
    try:
        # Verify API key using the existing PINECONE_API_KEY
        if request.api_key != PINECONE_API_KEY:
            logger.warning(f"Unauthorized clear-index attempt with incorrect API key")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid API key"
            )
        
        # Log initial count
        initial_count = get_total_vectors()
        logger.info(f"Current vector count before clearing: {initial_count}")
        
        if initial_count == 0:
            return {
                "success": True,
                "message": "Index already empty",
                "deleted_count": 0
            }
        
        # Get the Pinecone index
        index = get_or_create_index()
        
        # Delete all vectors
        index.delete(delete_all=True)
        
        # Clear BM25 index as well
        bm25_index.tool_data = {}
        bm25_index.is_initialized = False
        logger.info("Cleared BM25 index")
        
        # Verify deletion
        new_count = get_total_vectors()
        
        if new_count > 0:
            logger.warning(f"Not all vectors were deleted. Remaining: {new_count}")
        
        logger.info(f"Cleared index. Deleted {initial_count} vectors.")
        
        return {
            "success": True,
            "message": "Successfully cleared the vector index",
            "deleted_count": initial_count - new_count,
            "remaining_count": new_count
        }
    
    except HTTPException as he:
        # Re-raise HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear index: {str(e)}"
        )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)