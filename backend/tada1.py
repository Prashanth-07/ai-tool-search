import os
import time
import re
import json
import logging
import urllib3
import tiktoken
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
from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Union
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

def clean_text(text: str) -> str:
    # strip emojis (broad unicode ranges)…
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)
    no_emoji = emoji_pattern.sub("", text)
    # collapse multiple blank lines
    no_blank = re.sub(r'\n\s*\n+', '\n', no_emoji)
    # collapse tabs/spaces
    single_space = re.sub(r'[ \t]+', ' ', no_blank)
    return single_space.strip()

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
                        tool_id=metadata.get("tool_id", ""),
                        name=metadata.get("name", "Unknown"),
                        category_subcat=metadata.get("category_subcat", ""),
                        url=metadata.get("url", "https://example.com"),  # Default URL required
                        description=metadata.get("description", ""),
                        image_url=metadata.get("image_url", None),
                        owner=metadata.get("owner", None),
                        status=metadata.get("status", None)
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

class ToolDetails(BaseModel):
    introduction: Optional[str] = None
    usage: Optional[str] = None
    speciality: Optional[str] = None

class ToolFeaturesPros(BaseModel):
    pros: List[str] = []
    cons: List[str] = []

class ToolMetrics(BaseModel):
    functionality: float = 0
    innovation: float = 0
    performance: float = 0
    overall: float = 0
    easeOfUse: float = 0
    valueForMoney: float = 0

class Category(BaseModel):
    Category: str

class PricingPlan(BaseModel):
    planName: str
    price: str
    features: List[str] = []
    isPopular: bool = False
    _id: Optional[str] = None

class QAItem(BaseModel):
    question: str
    answer: str

class Tool(BaseModel):
    tool_id: str
    name: str
    url: Union[HttpUrl, str]
    description: Optional[str] = None
    category_subcat: Optional[str] = None
    image_url: Optional[Union[HttpUrl, str]] = None
    owner: Optional[str] = None
    status: Optional[str] = None
    
    # New fields for the updated schema
    pricingType: Optional[str] = None
    details: Optional[ToolDetails] = None
    features: Optional[ToolFeaturesPros] = None
    metrics: Optional[ToolMetrics] = None
    categories: List[Category] = []
    pricing: List[PricingPlan] = []
    qaSection: List[QAItem] = []
    
    class Config:
        # Allow additional fields
        extra = "allow"
        
    def generate_category_subcat(self) -> str:
        """Generate category_subcat field from categories for backward compatibility"""
        if self.categories:
            return ", ".join([cat.Category for cat in self.categories])
        return self.category_subcat or ""
    
    def generate_description(self) -> str:
        """Generate description from details for backward compatibility"""
        if not self.description and self.details:
            parts = []
            if self.details.introduction:
                parts.append(self.details.introduction)
            if self.details.usage:
                parts.append(f"Usage: {self.details.usage}")
            if self.details.speciality:
                parts.append(f"Specialty: {self.details.speciality}")
            if parts:
                return "\n\n".join(parts)
        return self.description or ""

class ToolResponse(BaseModel):
    id: str
    tool: Tool
    status: str = "added"

# Initialize BM25 index
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

    def create_document_text(self, tool: Tool):
        parts = [
            tool.name, tool.name,
            tool.category_subcat,
            tool.description,
            tool.owner or "",
            tool.status or ""
            ]
        
        # Add categories if available
        if hasattr(tool, 'categories') and tool.categories:
            for cat in tool.categories:
                parts.append(cat.Category)
        
        # Add details if available
        if hasattr(tool, 'details') and tool.details:
            if tool.details.introduction:
                parts.append(tool.details.introduction)
            if tool.details.usage:
                parts.append(tool.details.usage)
            if tool.details.speciality:
                parts.append(tool.details.speciality)
        
        # Add features if available
        if hasattr(tool, 'features') and tool.features:
            if tool.features.pros:
                parts.extend(tool.features.pros)
            if tool.features.cons:
                parts.extend(tool.features.cons)
        
        # Add pricing type
        if hasattr(tool, 'pricingType') and tool.pricingType:
            parts.append(tool.pricingType)
        
        # Add Q&A content for better search
        if hasattr(tool, 'qaSection') and tool.qaSection:
            for qa in tool.qaSection:
                parts.append(qa.question)
                parts.append(qa.answer)
                
        return " ".join(parts)
    
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
                    "category_subcat": tool.category_subcat,
                    "url": str(tool.url),
                    "image_url": tool.image_url or "",
                    "owner": tool.owner or "",
                    "status": tool.status or "",
                    "score": float(scores[idx])
                })
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results

bm25_index = BM25IndexManager()

# Pydantic models

class QueryRequest(BaseModel):
    query: str
    searchFrom: Optional[List[str]] = None

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

def clean_tool_data(tool: Tool) -> Tool:
    """
    Clean tool data before adding to vector database.
    Removes emojis, normalizes whitespace, and cleans fields.
    
    Args:
        tool: The tool object to clean
        
    Returns:
        Cleaned tool object
    """
    import re
    from urllib.parse import urlparse
    
    # Create a copy of the tool
    cleaned_tool = tool.model_copy() if hasattr(tool, 'model_copy') else Tool(**tool.dict())
    
    # Ensure category_subcat is present
    if not cleaned_tool.category_subcat:
        cleaned_tool.category_subcat = cleaned_tool.generate_category_subcat()
    
    # Ensure description is present
    if not cleaned_tool.description:
        cleaned_tool.description = cleaned_tool.generate_description()
    
    # 1. Clean name field - if it's a URL, extract domain name
    if cleaned_tool.name and cleaned_tool.name.startswith('http'):
        try:
            parsed_url = urlparse(cleaned_tool.name)
            # Extract domain name without protocol and www
            cleaned_tool.name = parsed_url.netloc.replace('www.', '')
        except Exception as e:
            logger.warning(f"Failed to parse URL in name: {cleaned_tool.name}")
    
    # 2. Remove emojis and special characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # geometric shapes
        "\U0001F800-\U0001F8FF"  # supplemental arrows
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"  # chess symbols
        "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-A
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "]+", 
        flags=re.UNICODE
    )
    
    # Special characters pattern (checkmarks, etc.)
    special_chars_pattern = re.compile(r'[✅❌⚠️⚙️🔧🤔📝🌱🏗️📅]')
    
    if cleaned_tool.description:
        # Remove emojis and special characters
        cleaned_tool.description = emoji_pattern.sub('', cleaned_tool.description)
        cleaned_tool.description = special_chars_pattern.sub('', cleaned_tool.description)
        
        # Normalize whitespace
        cleaned_tool.description = re.sub(r'[ \t]+', ' ', cleaned_tool.description)  # collapse spaces
        cleaned_tool.description = re.sub(r'\n{3,}', '\n\n', cleaned_tool.description)  # limit newlines
        cleaned_tool.description = cleaned_tool.description.strip()  # trim edges
    
    # 3. Clean category_subcat field
    if cleaned_tool.category_subcat:
        cleaned_tool.category_subcat = re.sub(r'[ \t]+', ' ', cleaned_tool.category_subcat)
        cleaned_tool.category_subcat = re.sub(r'\n{3,}', '\n\n', cleaned_tool.category_subcat)
        cleaned_tool.category_subcat = cleaned_tool.category_subcat.strip()
    
    # 4. Clean status and image_url fields
    if cleaned_tool.status == "Not Added":
        cleaned_tool.status = ""
    
    if cleaned_tool.image_url == "Added":
        cleaned_tool.image_url = ""
    
    return cleaned_tool

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
                                tool_id=metadata.get("tool_id", ""),
                                name=metadata.get("name", "Unknown"),
                                category_subcat=metadata.get("category_subcat", ""),
                                url=metadata.get("url", "https://example.com"),  # Default URL required
                                description=metadata.get("description", ""),
                                image_url=metadata.get("image_url", None),
                                owner=metadata.get("owner", None),
                                status=metadata.get("status", None)
                        )
                        bm25_index.add_tool(tool, metadata.get("rid"))
                
                logger.info(f"Added batch of {len(results)} documents to BM25 index")
            
            # Build the index after adding all documents
            bm25_index.rebuild_index()
            print(f"===== BM25 INDEX CONTENTS =====")
            print(f"Total tools indexed in BM25: {len(bm25_index.tool_data)}")
            # print(f"Is 'iconme-023' in BM25 index: {'iconme-023' in [tool.tool_id for rid, data in bm25_index.tool_data.items() for tool in [data['tool']]]}")
            # print(f"Is 'contentgoblinai-002' in BM25 index: {'contentgoblinai-002' in [tool.tool_id for rid, data in bm25_index.tool_data.items() for tool in [data['tool']]]}")
        return pc.Index(INDEX_NAME)
    except Exception as e:
        print(f"Error in get_or_create_index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize Pinecone index: {str(e)}"
        )
    
class NomicAtlasEmbeddings(Embeddings):
    """Wrapper for Nomic Atlas embedding API with task-specific embedding support."""
    
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api-atlas.nomic.ai/v1/embedding/text",
        document_task_type: str = "search_document",
        query_task_type: str = "search_query",
        max_tokens_per_text: int = 8192,
        dimensionality: int = 768,
    ):
        """Initialize the Nomic Atlas embeddings wrapper with task-specific types."""
        self.api_key = api_key
        self.api_url = api_url
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.max_tokens_per_text = max_tokens_per_text
        self.dimensionality = dimensionality
        
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for the API request."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _embed_with_task_type(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Internal method to embed texts with a specific task type."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "texts": texts,
                    "task_type": task_type,
                    "max_tokens_per_text": self.max_tokens_per_text,
                    "dimensionality": self.dimensionality
                }
                
                logger.info(f"Embedding {len(texts)} texts with task_type: {task_type}")
                
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
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using the Nomic Atlas API with document task type."""
        return self._embed_with_task_type(texts, self.document_task_type)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text using the Nomic Atlas API with query task type."""
        embeddings = self._embed_with_task_type([text], self.query_task_type)
        return embeddings[0]
    
# Global cache for vector store instance
# Global cache for document‑optimized vector store
_vector_store_cache = None
_vector_store_timestamp = None

# Global cache for query‑optimized vector store
_query_vector_store_cache = None
_query_vector_store_timestamp = None

# How long (in seconds) to keep any cache entry
_cache_lifetime = 3600

def get_vector_store(headers=None, for_query=False):
    """
    Initialize or return existing vector store using Nomic Atlas API.
    
    Args:
        headers: Optional HTTP headers
        for_query: If True, creates an embeddings instance optimized for queries
                   If False, creates an embeddings instance optimized for documents
    
    Returns:
        LangchainPinecone vector store instance
    """
    global _vector_store_cache, _vector_store_timestamp, _query_vector_store_cache, _query_vector_store_timestamp
    
    # Use different caches for document vs query vector stores
    if for_query:
        # Check if query cache is valid
        current_time = time.time()
        if (_query_vector_store_cache is not None and 
            _query_vector_store_timestamp is not None and 
            current_time - _query_vector_store_timestamp < _cache_lifetime):
            logger.info("Using cached query vector store instance")
            return _query_vector_store_cache
    else:
        # Check if document cache is valid
        current_time = time.time()
        if (_vector_store_cache is not None and 
            _vector_store_timestamp is not None and 
            current_time - _vector_store_timestamp < _cache_lifetime):
            logger.info("Using cached document vector store instance")
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
            document_task_type="search_document",
            query_task_type="search_query",
            max_tokens_per_text=8192,
            dimensionality=DIMENSION  # This uses the global DIMENSION variable (768)
        )
        
        # Initialize the vector store using Langchain's Pinecone integration
        vector_store = LangchainPinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        
        # Cache the appropriate vector store
        if for_query:
            _query_vector_store_cache = vector_store
            _query_vector_store_timestamp = current_time
            logger.info("Created and cached new query vector store instance")
        else:
            _vector_store_cache = vector_store
            _vector_store_timestamp = current_time
            logger.info("Created and cached new document vector store instance")
            
        return vector_store
    except Exception as e:
        logger.error(f"Error in get_vector_store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize vector store: {str(e)}"
        )
    
def format_tool_for_indexing(tool: Tool, rid: str) -> str:
    """Format tool data for embedding"""
    # Start with basic info
    formatted_text = (
        f"RID: {rid}\n"
        f"Tool ID: {tool.tool_id}\n"
        f"Name: {tool.name}\n\n"
    )
    
    # Add categories
    if tool.categories:
        categories_text = ", ".join([cat.Category for cat in tool.categories])
        formatted_text += f"Categories: {categories_text}\n\n"
    elif tool.category_subcat:
        formatted_text += f"Category/Sub‑cat: {tool.category_subcat}\n\n"
    
    # Add URL
    formatted_text += f"URL: {tool.url}\n\n"
    
    # Add pricing type
    if tool.pricingType:
        formatted_text += f"Pricing Type: {tool.pricingType}\n\n"
    
    # Add details section
    if tool.details:
        formatted_text += "Details:\n"
        if tool.details.introduction:
            formatted_text += f"Introduction: {tool.details.introduction}\n"
        if tool.details.usage:
            formatted_text += f"Usage: {tool.details.usage}\n"
        if tool.details.speciality:
            formatted_text += f"Specialty: {tool.details.speciality}\n"
        formatted_text += "\n"
    
    # Add features
    if tool.features and (tool.features.pros or tool.features.cons):
        formatted_text += "Features:\n"
        if tool.features.pros:
            formatted_text += "Pros:\n" + "\n".join([f"- {pro}" for pro in tool.features.pros]) + "\n"
        if tool.features.cons:
            formatted_text += "Cons:\n" + "\n".join([f"- {con}" for con in tool.features.cons]) + "\n"
        formatted_text += "\n"
    
    # Add description (for backward compatibility)
    if tool.description:
        formatted_text += f"Description:\n{tool.description}\n\n"
    
    # Add metrics if available
    if tool.metrics:
        formatted_text += "Metrics:\n"
        metrics_dict = tool.metrics.dict()
        for key, value in metrics_dict.items():
            formatted_text += f"{key}: {value}\n"
        formatted_text += "\n"
    
    # Add pricing plans
    if tool.pricing:
        formatted_text += "Pricing Plans:\n"
        for plan in tool.pricing:
            formatted_text += f"- {plan.planName}: {plan.price}\n"
            if plan.features:
                formatted_text += "  Features: " + ", ".join(plan.features) + "\n"
        formatted_text += "\n"
    
    # Add Q&A section
    if tool.qaSection:
        formatted_text += "FAQ:\n"
        for qa in tool.qaSection:
            formatted_text += f"Q: {qa.question}\nA: {qa.answer}\n\n"
    
    # Add remaining metadata fields for backward compatibility
    formatted_text += (
        f"Image URL: {tool.image_url or 'N/A'}\n"
        f"Owner: {tool.owner or 'Unassigned'}\n"
        f"Status: {tool.status or 'Unknown'}"
    )
    
    return formatted_text

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

@app.post("/reindex-tools")
async def reindex_tools():
    """
    Re-index all existing tools with task-specific embeddings.
    This doesn't delete any tools but updates their vector representations.
    """
    try:
        # Initialize document-optimized vector store
        document_vector_store = get_vector_store(for_query=False)
        
        # Get total vectors for reference
        total_vectors = get_total_vectors()
        logger.info(f"Starting re-indexing of {total_vectors} tools")
        
        if total_vectors == 0:
            return {
                "success": True,
                "message": "No tools to re-index",
                "count": 0
            }
        
        # Get the Pinecone index
        index = get_or_create_index()
        
        # Track progress
        updated_count = 0
        failed_count = 0
        
        # Process in batches for large collections
        batch_size = 20
        total_batches = (total_vectors - 1) // batch_size + 1
        
        # Get the raw Pinecone index for faster updates
        pinecone_index = pc.Index(INDEX_NAME)
        
        for batch_num in range(total_batches):
            # Calculate current batch range
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_vectors)
            current_batch_size = end_idx - start_idx
            
            logger.info(f"Processing batch {batch_num+1}/{total_batches} (tools {start_idx+1}-{end_idx})")
            
            # Retrieve current batch of tools using old embeddings
            results = document_vector_store.similarity_search(
                "",  # Empty query to get all vectors
                k=current_batch_size,
                filter={}  # No filtering
            )
            
            # Re-index each tool with task-specific embeddings
            for doc in results:
                try:
                    metadata = doc.metadata
                    rid = metadata.get("rid")
                    
                    if not rid:
                        logger.warning(f"Skipping tool with missing RID")
                        failed_count += 1
                        continue
                    
                    # Reconstruct the tool object
                    if "tool_id" in metadata:
                        tool = Tool(
                            tool_id=metadata.get("tool_id", ""),
                            name=metadata.get("name", "Unknown"),
                            category_subcat=metadata.get("category_subcat", ""),
                            url=metadata.get("url", "https://example.com"),  # Default URL required
                            description=metadata.get("description", ""),
                            image_url=metadata.get("image_url", None),
                            owner=metadata.get("owner", None),
                            status=metadata.get("status", None)
                        )
                        
                        # Format tool data for embedding using the document task type
                        tool_text = format_tool_for_indexing(tool, rid)
                        
                        # Update the vector in Pinecone
                        document_vector_store.add_texts(
                            texts=[tool_text],
                            metadatas=[metadata],
                            ids=[rid]
                        )
                        
                        updated_count += 1
                    else:
                        logger.warning(f"Skipping document with missing tool_id")
                        failed_count += 1
                
                except Exception as e:
                    logger.error(f"Error re-indexing tool: {str(e)}")
                    failed_count += 1
            
            # Log progress
            logger.info(f"Re-indexed batch {batch_num+1}/{total_batches} - {updated_count} tools updated so far")
        
        # Refresh BM25 index - no need to rebuild as metadata hasn't changed
        if updated_count > 0 and not bm25_index.is_initialized:
            bm25_index.rebuild_index()
            logger.info("Rebuilt BM25 index")
        
        return {
            "success": True,
            "message": f"Successfully re-indexed {updated_count} tools with task-specific embeddings",
            "total_vectors": total_vectors,
            "updated_count": updated_count,
            "failed_count": failed_count
        }
        
    except Exception as e:
        logger.error(f"Error in reindex_tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to re-index tools: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_tools(request: QueryRequest, request_headers: Request):
    """Query tools based on user input using hybrid search (vector + keyword)."""
    start_time = time.time()
    try:
        logger.info(f"Processing query: {request.query}")
        headers = request_headers.headers

        # Log if searching from specific tools
        if request.searchFrom:
            logger.info(f"Searching only within {len(request.searchFrom)} specified tools: {request.searchFrom[:3]}...")

        # Try to get cached response - only use cache if not doing filtered search
        cached_response = None
        if not request.searchFrom:
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
        vector_store = get_vector_store(headers, for_query=True)

        # Log total vectors for reference
        total_vectors = get_total_vectors()
        logger.info(f"Total vectors in store: {total_vectors}")
        
        # Hybrid search implementation
        try:
            # Create a filter if searchFrom is provided
            search_filter = None
            if request.searchFrom and len(request.searchFrom) > 0:
                # Filter to only include specified tool_ids
                search_filter = {"tool_id": {"$in": request.searchFrom}}
                logger.info(f"Applied filter to search only within specified tools: {search_filter}")
            
            # Step 1: Perform vector search with scores, applying filter if provided
            vector_k = 15  # Increase to get more candidates
            
            try:
                vector_results_with_scores = vector_store.similarity_search_with_score(
                    request.query,
                    k=vector_k,
                    filter=search_filter  # Apply the filter here
                )
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                vector_results_with_scores = []
            
            # Convert to a list of result dictionaries
            processed_vector_results = []
            for doc, score in vector_results_with_scores:
                metadata = doc.metadata
                processed_vector_results.append({
                    "rid": metadata["rid"],
                    "tool_id": metadata["tool_id"],
                    "name": metadata["name"],
                    "category_subcat": metadata.get("category_subcat",""),
                    "url": metadata.get("url",""),
                    "description": metadata.get("description",""),
                    "image_url": metadata.get("image_url",""),
                    "owner": metadata.get("owner",""),
                    "status": metadata.get("status",""),
                    "score": float(1.0 - score)
                })

            logger.info(f"Vector search returned {len(processed_vector_results)} results")
            
            # Step 2: Perform BM25 search - we need to filter these results manually
            bm25_k = 30  # Get similar number of results
            
            try:
                bm25_results = bm25_index.search(request.query, top_k=bm25_k)
                logger.info(f"BM25 search returned {len(bm25_results)} results")
                
                # Filter BM25 results if searchFrom is provided
                if request.searchFrom and len(request.searchFrom) > 0:
                    filtered_bm25_results = [
                        result for result in bm25_results 
                        if result.get("tool_id") in request.searchFrom
                    ]
                    logger.info(f"Filtered BM25 results from {len(bm25_results)} to {len(filtered_bm25_results)}")
                    bm25_results = filtered_bm25_results
            except Exception as e:
                logger.error(f"Error in BM25 search: {str(e)}")
                bm25_results = []
            
            # Step 3: Fuse the results (using 0.5 weight for vector search, 0.5 for BM25)
            hybrid_results = fuse_search_results(
                processed_vector_results, 
                bm25_results,
                alpha=0.5  # Equal weight for vector and keyword search
            )
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            
            # If there are no results and we have a filter, it might be too restrictive
            if not hybrid_results and request.searchFrom:
                logger.warning(f"No results found with the provided filter. Check if tool IDs are valid.")
                # Return a helpful message
                filter_response = json.dumps({
                    "tool_id": [],
                    "tools": [],
                    "message": "No tools found matching your search within the specified tools.",
                    "search_filter_applied": True,
                    "searched_within_tool_ids": request.searchFrom,
                    "timestamp": datetime.now().isoformat()
                })
                return QueryResponse(response=filter_response)
            
            # Step 4: Take top results (up to 5) for LLM processing
            top_results = hybrid_results[:5]

            print("\n===== TOOL NAMES BEING SENT TO LLM =====")
            for result in top_results:
                print(f"- {result.get('name', 'N/A')}")
            print("============================\n")
            
            # Format the documents for LLM
            formatted_docs = []
            for result in top_results:
                formatted_doc = (
                    f"Tool ID: {result.get('tool_id', 'N/A')}\n"
                    f"Name: {result.get('name', 'N/A')}\n"
                    f"Category/Sub‑cat: {result.get('category_subcat', 'N/A')}\n"
                    f"URL: {result.get('url', 'N/A')}\n"
                    f"Description: {result.get('description', 'N/A')}\n"
                    f"Image URL: {result.get('image_url', 'N/A')}\n"
                    f"Owner: {result.get('owner', 'N/A')}\n"
                    f"Status: {result.get('status', 'N/A')}"
                )
                formatted_docs.append(formatted_doc)
            
            # If we have no tools to process, return early
            if not formatted_docs:
                empty_response = json.dumps({
                    "tool_id": [],
                    "tools": [],
                    "message": "No tools found matching your search criteria.",
                    "search_filter_applied": request.searchFrom is not None,
                    "searched_within_tool_ids": request.searchFrom,
                    "timestamp": datetime.now().isoformat()
                })
                return QueryResponse(response=empty_response)
            
            # 2) Determine model early
            current_model = get_current_model(headers)
            logger.info(f"Using model: {current_model} with Groq API")
            try:
                encoder = tiktoken.encoding_for_model(current_model)
                logger.info(f"Using tiktoken encoding_for_model for {current_model}")
            except Exception:
                encoder = tiktoken.get_encoding("cl100k_base")
                logger.info("Falling back to cl100k_base encoding")
            context = "\n\n---\n\n".join(formatted_docs)

            # 4) Count tokens before cleaning
            raw_tokens = len(encoder.encode(context))
            print(f"DEBUG: tokens before cleaning: {raw_tokens}")

            # 5) Clean and count again
            cleaned_context = clean_text(context)
            clean_tokens = len(encoder.encode(cleaned_context))
            print(f"DEBUG: tokens after cleaning:  {clean_tokens}")

            system_text= f"""
You are a tool retrieval assistant tasked with finding, picking and ranking relevant tools from a provided Tool Data that closely relate to the User Query.
Instructions:
- Analyze the User Query and understand their needs, including any quantity specifications (e.g., "top 1", "best 3", etc.).
- If the User Query specifies a number of tools (e.g., "top 1", "best 3", "show me 2"), return EXACTLY that number of tools.
- If no quantity is specified, analyze each tool in the given tools in the Tool Data and include the relevant tools based on the analysis of User Query and needs.
- Rank them from most to least relevant.
- For each tool, generate:
  - A short summary (1–2 lines) explaining how that particular tool helps with the User Query.
  - 2–3 bullet points showing key features that make it useful for the User Query.
- Format your response as a JSON object with the following schema:
{{
  "tool_id": ["most_relevant_id", "next_most_relevant_id", ...],
  "tools": [
    {{
      "id": "tool_id",
      "name": "Tool Name",
      "description": "How this tool helps User Query",
      "bullets": [
        "Feature or benefit 1 which helps User Query",
        "Feature or benefit 2",
        "Optional feature or benefit 3"
      ]
    }},
    ...
  ]
}}

Guidelines:
- Pay special attention to any quantity specifications in the query (e.g., "top 1", "best 3") and strictly adhere to them.
- Carefully evaluate each included tool's semantic relevance to the User Query and include relevant tools excluding irrelevant tools in the output JSON.
- Strict ordering: The most relevant tool MUST be listed first, followed by decreasing relevance.
- Strict Warning: Do not include tools that are irrelevant to the User Query in the output JSON.
- Match the tool description to the User Query.
- Output ONLY the JSON. No preamble No extra note.
"""
            
            # Create prompt for LangChain
            prompt_text = f"""
User Query: {request.query}

Tool Data: {context}
"""
            full_prompt = system_text + "\n\n" + prompt_text
            token_count = len(encoder.encode(full_prompt))
            print(f"DEBUG: total tokens in prompt = {token_count}")
            # Initialize LangChain's ChatGroq
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=current_model,
                temperature=0.1
            )
            
            # Get response from Groq LLM with timeout handling
            try:
                response = llm.invoke([
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt_text}
                ])

                
                # Extract content from response
                llm_response = response.content
                print(f"\n===== LLM RESPONSE =====")
                print(llm_response)
                print("============================\n")
                
                # Post-process the response
                processed_response = post_process_llm_response(llm_response)
                
                # Try to parse as JSON but don't fail if it's not valid
                try:
                    # Parse and validate response
                    response_data = json.loads(processed_response)
                    
                    # Add search filter info if applied
                    if request.searchFrom:
                        response_data["search_filter_applied"] = True
                        response_data["searched_within_tool_ids"] = request.searchFrom

                    # Ensure proper tool data is included - create a fallback if needed
                    if "tools" not in response_data or not response_data["tools"]:
                        # Include the original tools if the LLM didn't format them correctly
                        response_data["tools"] = []
                        response_data["tool_id"] = []
                        
                        for result in top_results:
                            tool_data = {
                                "id": result.get("tool_id", ""),
                                "name": result.get("name", ""),
                                "description": result.get("description", ""),
                                "bullets": [
                                    "This tool was found in your search",
                                    "Check the description for more details"
                                ]
                            }
                            response_data["tools"].append(tool_data)
                            response_data["tool_id"].append(result.get("tool_id", ""))
                            
                        response_data["message"] = "Structured response incomplete, showing raw search results instead."
                    
                    # Clean response
                    clean_response = json.dumps(response_data, indent=2)
                    logger.info("Processed valid JSON response")
                except json.JSONDecodeError:
                    # If not valid JSON, construct a valid JSON with the tools
                    fallback_data = {
                        "tool_id": [result.get("tool_id", "") for result in top_results],
                        "tools": [],
                        "message": "Failed to parse LLM response into JSON, showing raw search results",
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Include basic info about the tools
                    for result in top_results:
                        tool_data = {
                            "id": result.get("tool_id", ""),
                            "name": result.get("name", ""),
                            "description": result.get("description", ""),
                            "bullets": [
                                "This tool was found in your search",
                                "Check the description for more details"
                            ]
                        }
                        fallback_data["tools"].append(tool_data)
                    
                    # Add search filter info if applied
                    if request.searchFrom:
                        fallback_data["search_filter_applied"] = True
                        fallback_data["searched_within_tool_ids"] = request.searchFrom
                    
                    clean_response = json.dumps(fallback_data, indent=2)
                    logger.warning("Created fallback JSON response")
                
                # Only cache the result if this wasn't a filtered search
                if not request.searchFrom:
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
        vector_store = get_vector_store(for_query=False)
        results = []
        skipped_tools = []
        added_tools = []

        # First clean all tools and ensure backward compatibility fields
        cleaned_tools = []
        for tool in bulk_request.tools:
            # Map tool_id from _id if needed
            if hasattr(tool, '_id') and tool._id and not tool.tool_id:
                tool.tool_id = tool._id
                
            # Ensure backward compatibility fields
            if not tool.category_subcat:
                tool.category_subcat = tool.generate_category_subcat()
                
            if not tool.description:
                tool.description = tool.generate_description()
                
            # Now clean the tool data
            cleaned_tool = clean_tool_data(tool)
            cleaned_tools.append(cleaned_tool)

        # Check for duplicates
        for tool in cleaned_tools:
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
        
        # Add non-duplicate tools
        for tool in added_tools:
            # Generate unique RID for each tool
            rid = str(uuid4())
            
            # Format tool data for indexing
            tool_text = format_tool_for_indexing(tool, rid)
            
            # Prepare metadata - convert complex objects to simple types for Pinecone
            metadata = {
                "rid": rid,
                "tool_id": tool.tool_id,
                "name": tool.name,
                "category_subcat": tool.category_subcat,
                "url": str(tool.url),
                "description": tool.description,
                "image_url": tool.image_url or "",
                "owner": tool.owner or "",
                "status": tool.status or ""
            }
            
            # Add new fields with properly serialized data
            if tool.pricingType:
                metadata["pricingType"] = tool.pricingType
            
            # Convert complex nested objects to strings for metadata
            if tool.details:
                metadata["details_introduction"] = tool.details.introduction or ""
                metadata["details_usage"] = tool.details.usage or ""
                metadata["details_speciality"] = tool.details.speciality or ""
            
            if tool.features:
                if tool.features.pros:
                    metadata["features_pros"] = ",".join(tool.features.pros)
                if tool.features.cons:
                    metadata["features_cons"] = ",".join(tool.features.cons)
                
            if tool.metrics:
                for key, value in tool.metrics.dict().items():
                    metadata[f"metrics_{key}"] = value
                
            if tool.categories:
                # For categories, extract just the category values as a list of strings
                metadata["categories_list"] = [cat.Category for cat in tool.categories]
                
            if tool.pricing:
                # For pricing, extract key information as simple strings
                metadata["pricing_plans"] = ",".join([plan.planName for plan in tool.pricing])
                metadata["pricing_prices"] = ",".join([plan.price for plan in tool.pricing])
                
            if tool.qaSection:
                # For QA, store as concatenated strings
                metadata["qa_questions"] = ",".join([qa.question for qa in tool.qaSection])
                metadata["qa_answers"] = ",".join([qa.answer for qa in tool.qaSection])
            
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
        vector_store = get_vector_store(for_query=False)
        results = []
        updated_rids = []

        cleaned_tools = [clean_tool_data(tool) for tool in request.tools]
        
        for tool in cleaned_tools:
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
                
                # Prepare metadata - convert complex objects to simple types for Pinecone (same as add_tools)
                metadata = {
                    "rid": existing_rid,
                    "tool_id": tool.tool_id,
                    "name": tool.name,
                    "category_subcat": tool.category_subcat,
                    "url": str(tool.url),
                    "description": tool.description,
                    "image_url": tool.image_url or "",
                    "owner": tool.owner or "",
                    "status": tool.status or ""
                }
                
                # Add new fields with properly serialized data (same as add_tools)
                if tool.pricingType:
                    metadata["pricingType"] = tool.pricingType
                
                # Convert complex nested objects to strings for metadata
                if tool.details:
                    metadata["details_introduction"] = tool.details.introduction or ""
                    metadata["details_usage"] = tool.details.usage or ""
                    metadata["details_speciality"] = tool.details.speciality or ""
                
                if tool.features:
                    if tool.features.pros:
                        metadata["features_pros"] = ",".join(tool.features.pros)
                    if tool.features.cons:
                        metadata["features_cons"] = ",".join(tool.features.cons)
                    
                if tool.metrics:
                    for key, value in tool.metrics.dict().items():
                        metadata[f"metrics_{key}"] = value
                    
                if tool.categories:
                    # For categories, extract just the category values as a list of strings
                    metadata["categories_list"] = [cat.Category for cat in tool.categories]
                    
                if tool.pricing:
                    # For pricing, extract key information as simple strings
                    metadata["pricing_plans"] = ",".join([plan.planName for plan in tool.pricing])
                    metadata["pricing_prices"] = ",".join([plan.price for plan in tool.pricing])
                    
                if tool.qaSection:
                    # For QA, store as concatenated strings
                    metadata["qa_questions"] = ",".join([qa.question for qa in tool.qaSection])
                    metadata["qa_answers"] = ",".join([qa.answer for qa in tool.qaSection])
                
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