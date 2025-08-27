"""
AI Tool Search API 

This module provides a comprehensive FastAPI-based search API for AI tools
using vector similarity search (Nomic Atlas) and keyword-based search (BM25)
with LLM-powered query processing through Groq.

Key Features:
- Hybrid search combining vector and keyword matching
- Task-specific embeddings for documents vs queries
- BM25 keyword search integration
- Tool management (CRUD operations)
- Query suggestions and keyword extraction
- Comprehensive caching and error handling
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import nltk
import numpy as np
import requests
import tiktoken
import urllib3
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings.base import Embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pinecone import Pinecone
from pydantic import BaseModel, Field, HttpUrl
from rank_bm25 import BM25Okapi
from openai import OpenAI
# MongoDB imports
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Application configuration management."""
    
    # Environment validation
    MIN_PYTHON_VERSION = (3, 7)
    DIMENSION = 768  # Nomic Atlas embedding dimension
    
    # Cache settings
    TOOL_SEARCH_CACHE_SIZE = 1000
    TOOL_SEARCH_CACHE_TTL_MINUTES = 60
    VECTOR_STORE_CACHE_LIFETIME = 3600  # seconds
    
    # Search settings
    VECTOR_SEARCH_K = 40
    BM25_SEARCH_K = 50
    HYBRID_ALPHA = 0.7  # Weight for vector vs BM25 search
    POPULAR_TOOLS_LIMIT = 4
    
    # Score-based selection settings
    HYBRID_MIN_SCORE = 0.5   # Minimum relevance threshold
    HYBRID_MAX_TOOLS = 50       # Maximum tools to send to LLM
    HYBRID_FALLBACK_COUNT = 1  # Minimum tools if none meet threshold
    
    # Processing settings
    DEFAULT_BATCH_SIZE = 100
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1
    REQUEST_TIMEOUT = 10
    
    # NEW: Vector loading setting3
    LOADING_FIRST_BATCH_MIN = 50        # Minimum first batch size
    LOADING_FIRST_BATCH_MAX = 100       # Maximum first batch size  
    LOADING_FIRST_BATCH_RATIO = 5       # total_vectors // this ratio
    LOADING_SINGLE_QUERY_THRESHOLD = 10000  # Use single query if vectors <= this
    LOADING_COVERAGE_TARGET = 0.95      # Stop multi-term search at 95% coverage
    LOADING_MULTI_TERM_BATCH_SIZE = 100 # Batch size for multi-term strategy
    LOADING_TERM_DELAY_SECONDS = 0.1    # Delay between search terms
# MongoDB settings
    MONGO_DB_NAME = "aitoolbook"
    MONGO_COLLECTION_TOOLS = "aitools"
    MONGO_COLLECTION_CATEGORIES = "categories"
    MONGO_COLLECTION_USECASES = "usecases"

# System Validation
if sys.version_info < Config.MIN_PYTHON_VERSION:
    raise RuntimeError(f"Python {'.'.join(map(str, Config.MIN_PYTHON_VERSION))} or higher is required")

# Environment setup
os.environ["USE_NUMPY"] = "1"
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# LOGGING SETUP
# ============================================================================

class LoggerSetup:
    """Centralized logging configuration."""
    
    @staticmethod
    def setup():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('vectorstore.log')
            ]
        )
        return logging.getLogger(__name__)

logger = LoggerSetup.setup()

# ============================================================================
# ENVIRONMENT VARIABLES MANAGEMENT
# ============================================================================

class Environment:
    """Environment variables management with validation."""
    
    def __init__(self):
        load_dotenv()
        self._validate_required_vars()
    
    def _validate_required_vars(self):
        """Validate that all required environment variables are set."""
        required_vars = ["PINECONE_API_KEY", "GROQ_API_KEY", "NOMIC_API_KEY"]
        # Check for OpenAI key if we want to use it for query tools
        if os.getenv("OPENAI_API_KEY"):
            logger.info("OpenAI API key found - will be used for query tools endpoint")
        else:
            logger.warning("OpenAI API key not found - query tools will use Groq fallback")
    
        missing_vars = [var for var in required_vars if not os.getenv(var)]
    
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    @property
    def pinecone_api_key(self) -> str:
        return os.getenv("PINECONE_API_KEY")
    
    @property
    def groq_api_key(self) -> str:
        return os.getenv("GROQ_API_KEY")
    
    @property
    def nomic_api_key(self) -> str:
        return os.getenv("NOMIC_API_KEY")
    
    @property
    def index_name(self) -> str:
        return os.getenv("INDEX_NAME", "ai-tool-search")
    
    @property
    def ollama_base_url(self) -> str:
        return os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    
    @property
    def ollama_verify_ssl(self) -> bool:
        return os.getenv("OLLAMA_VERIFY_SSL", "true").lower() == "true"
    
    @property
    def dev_model(self) -> str:
        return os.getenv("DEV_MODEL", "mistral-7b-instruct")
    
    @property
    def prod_model(self) -> str:
        return os.getenv("PROD_MODEL", "llama-3.1-8b-instant")
    
    @property
    def environment(self) -> str:
        return os.getenv("ENVIRONMENT", "DEV")
    
    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY")

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    @property
    def mongo_uri(self) -> str:
        return os.getenv("MONGO_URI", "mongodb+srv://techatb01:v0cshOyjd3axvxMa@cluster0.wuyqyki.mongodb.net/aitoolbook?retryWrites=true&w=majority&appName=Cluster0")

env = Environment()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ToolDetails(BaseModel):
    """Tool details section."""
    introduction: Optional[str] = None
    usage: Optional[str] = None
    speciality: Optional[str] = None

class ToolFeaturesPros(BaseModel):
    """Tool features and pros/cons."""
    pros: List[str] = []
    cons: List[str] = []

class ToolMetrics(BaseModel):
    """Tool performance metrics."""
    functionality: float = 0
    innovation: float = 0
    performance: float = 0
    overall: float = 0
    easeOfUse: float = 0
    valueForMoney: float = 0

class Category(BaseModel):
    """Tool category."""
    Category: str

class PricingPlan(BaseModel):
    """Pricing plan information."""
    planName: str
    price: str
    features: List[str] = []
    isPopular: bool = False
    _id: Optional[str] = None

class QAItem(BaseModel):
    """Q&A item."""
    question: str
    answer: str

class Tool(BaseModel):
    """Complete tool model with all features."""
    tool_id: str
    name: str
    url: Union[HttpUrl, str]
    description: Optional[str] = None
    category_subcat: Optional[str] = None
    image_url: Optional[Union[HttpUrl, str]] = None
    owner: Optional[str] = None
    status: Optional[str] = None
    pricingType: Optional[str] = Field(None, alias="pricing_type")  # Accept both
    details: Optional[ToolDetails] = None
    features: Optional[ToolFeaturesPros] = None
    metrics: Optional[ToolMetrics] = None
    categories: List[Category] = []
    pricing: List[PricingPlan] = []
    qaSection: List[QAItem] = Field([], alias="qa_section")  # Accept both

    class Config:
        extra = "allow"
        populate_by_name = True  # This allows both field names and aliases
    
    def generate_category_subcat(self) -> str:
        """Generate category_subcat field from categories for backward compatibility."""
        if self.categories:
            return ", ".join([cat.Category for cat in self.categories])
        return self.category_subcat or ""
    
    def generate_description(self) -> str:
        """Generate description from details for backward compatibility."""
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

# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    searchFrom: Optional[List[str]] = None
    limit: Optional[int] = Field(default=5, gt=0, le=50)

class QueryResponse(BaseModel):
    """Query response model."""
    response: str

class StructuredToolResult(BaseModel):
    """Individual tool result for structured outputs."""
    id: str
    name: str
    description: str
    bullets: List[str] = []
    
    class Config:
        extra = "forbid"  # This generates "additionalProperties": false

class StructuredQueryResponse(BaseModel):
    """Structured query response for OpenAI structured outputs."""
    tool_id: List[str]
    tools: List[StructuredToolResult]
    message: str = ""
    
    class Config:
        extra = "forbid"  # This generates "additionalProperties": false

class QuerySuggestionsRequest(BaseModel):
    """Query suggestions request."""
    query: str


class QuerySuggestionsResponse(BaseModel):
    """Query suggestions response."""
    original_query: str
    suggestions: List[str]

class PopularByUseCaseRequest(BaseModel):
    """Popular tools by use case request."""
    use_case: str

class PopularByUseCaseResponse(BaseModel):
    """Popular tools by use case response."""
    use_case: str
    tool_id: List[str]
    tools: List[Dict[str, Any]]
    ranking_criteria: str

class ExtractKeywordsRequest(BaseModel):
    """Extract keywords request."""
    reviews: List[str]

class KeywordItem(BaseModel):
    """Keyword item with sentiment."""
    keyword: str
    sentiment: str

class ExtractKeywordsResponse(BaseModel):
    """Extract keywords response."""
    keywords: List[KeywordItem]

# Tool Categorization Models
class ToolCategorizationRequest(BaseModel):
    """Tool categorization request - tool_id will come from path parameter."""
    pass

class CategoryMatch(BaseModel):
    """Individual category match."""
    id: str
    name: str


class ToolCategorizationResponse(BaseModel):
    """Tool categorization response."""
    tool_id: str
    tool_name: str
    matched_categories: List[CategoryMatch]
    total_available_categories: int

# Tool Use Case Categorization Models

class UseCaseMatch(BaseModel):
    """Individual use case match."""
    id: str
    name: str

class ToolUseCaseCategorizationResponse(BaseModel):
    """Tool use case categorization response."""
    tool_id: str
    tool_name: str
    matched_usecases: List[UseCaseMatch]
    total_available_usecases: int

class ToolResponse(BaseModel):
    """Tool operation response."""
    id: str
    tool: Tool
    status: str = "added"

class DeleteResponse(BaseModel):
    """Delete operation response."""
    success: bool
    deleted_tool: str

class BulkToolRequest(BaseModel):
    """Bulk tool operation request."""
    tools: List[Tool]

class BulkToolResponse(BaseModel):
    """Bulk tool operation response."""
    results: List[ToolResponse]

class BulkUpdateRequest(BaseModel):
    """Bulk update request."""
    tools: List[Tool]

class BulkUpdateResponse(BaseModel):
    """Bulk update response."""
    results: List[ToolResponse]

class ClearIndexRequest(BaseModel):
    """Clear index request."""
    api_key: str = Field(..., description="API key for authorization to clear the index")


# ============================================================================
# UTILITY CLASSES
# ============================================================================

class TextCleaner:
    """Text cleaning utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing emojis and normalizing whitespace."""
        if not text:
            return ""
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
        
        no_emoji = emoji_pattern.sub("", text)
        no_blank = re.sub(r'\n\s*\n+', '\n', no_emoji)
        single_space = re.sub(r'[ \t]+', ' ', no_blank)
        return single_space.strip()

class ModelUtils:
   """Model and header utilities."""
   
   @staticmethod
   def get_ollama_url(headers=None) -> str:
       """Get Ollama URL from headers or environment."""
       if headers and "OLLAMA_URL" in headers:
           url = headers.get("OLLAMA_URL")
           if url and url.strip():
               logger.info(f"Using custom Ollama URL from headers: {url}")
               return url
       
       logger.info(f"Using default Ollama URL from environment: {env.ollama_base_url}")
       return env.ollama_base_url
   
   @staticmethod
   def should_verify_ssl(headers=None) -> bool:
       """Determine if SSL verification should be enabled."""
       if headers and "OLLAMA_VERIFY_SSL" in headers:
           verify_ssl = headers.get("OLLAMA_VERIFY_SSL", "true").lower() == "true"
           logger.info(f"SSL verification setting from headers: {verify_ssl}")
           return verify_ssl
       
       logger.info(f"SSL verification setting from environment: {env.ollama_verify_ssl}")
       return env.ollama_verify_ssl
   
   @staticmethod
   def get_current_model(headers=None) -> str:
       """Get current model based on headers or environment."""
       current_env = env.environment
       dev_model = env.dev_model
       prod_model = env.prod_model
       
       logger.info(f"DEBUG: get_current_model called - env={current_env}, dev={dev_model}, prod={prod_model}")
       logger.info(f"DEBUG: headers type: {type(headers)}, headers present: {headers is not None}")
       
       if not headers:
           result = dev_model if current_env == "DEV" else prod_model
           logger.info(f"DEBUG: No headers - env={current_env}, returning {result}")
           return result
       
       # Check if MODEL_CHOICE is explicitly set in headers
       model_choice = headers.get("MODEL_CHOICE")  # No default!
       logger.info(f"DEBUG: MODEL_CHOICE from headers: {model_choice}")
       
       if model_choice == "PROD_MODEL":
           logger.info(f"DEBUG: Explicit PROD_MODEL requested, returning {prod_model}")
           return prod_model
       elif model_choice == "DEV_MODEL":
           logger.info(f"DEBUG: Explicit DEV_MODEL requested, returning {dev_model}")
           return dev_model
       else:
           # No MODEL_CHOICE specified, use environment setting
           result = dev_model if current_env == "DEV" else prod_model
           logger.info(f"DEBUG: No MODEL_CHOICE specified, using environment {current_env}, returning {result}")
           return result
   
   @staticmethod
   def call_openai_for_query_tools(system_prompt: str, user_prompt: str, tier1_system: str=None, tier1_user: str=None) -> dict:
       """Call OpenAI with Structured Outputs - AUTOMATIC CACHING ENABLED."""
       import os
       
       try:
           openai_api_key = os.getenv("OPENAI_API_KEY")
           openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
           
           if not openai_api_key:
               raise ValueError("OPENAI_API_KEY not found in environment variables")
           
           logger.info(f"🚀 Using OpenAI model: {openai_model} with AUTOMATIC PROMPT CACHING")
           
           # ✅ TRY 1: Structured Outputs with Automatic Caching
           try:
               from openai import OpenAI
               
               client = OpenAI(
                   api_key=openai_api_key,
                   timeout=60.0,
                   max_retries=3
               )
               final_system = tier1_system if tier1_system else system_prompt
               final_user = tier1_user if tier1_user else user_prompt

               logger.info(f"📏 System prompt: {len(final_system)} chars, User prompt: {len(final_user)} chars")
               logger.info("🎯 TIER 1: Using Structured Outputs with optimized prompts")
               
               # Log prompt lengths for caching analysis
               logger.info(f"📏 System prompt: {len(system_prompt)} chars, User prompt: {len(user_prompt)} chars")
               
               # ✅ STRUCTURED OUTPUTS (Caching happens automatically)
               response = client.responses.create(
                   model=openai_model,
                   input=[
                       {"role": "system", "content": final_system},
                       {"role": "user", "content": final_user}
                   ],
                   text={
                       "format": {
                           "type": "json_schema",
                           "name": "structured_query_response", 
                           "strict": True,
                           "schema": {
                               "type": "object",
                               "properties": {
                                   "tool_id": {
                                       "type": "array",
                                       "items": {"type": "string"}
                                   },
                                   "tools": {
                                       "type": "array",
                                       "items": {
                                           "type": "object",
                                           "properties": {
                                               "id": {"type": "string"},
                                               "name": {"type": "string"}, 
                                               "description": {"type": "string"},
                                               "bullets": {"type": "array", "items": {"type": "string"}}
                                           },
                                           "required": ["id", "name", "description", "bullets"],
                                           "additionalProperties": False
                                       }
                                   }
                               },
                               "required": ["tool_id", "tools"],
                               "additionalProperties": False
                           }
                       }
                   }
               )
               
               # ✅ ANALYZE TOKENS & CACHING
               cached_tokens = 0
               cache_hit = False
               cache_percentage = 0.0
               
               try:
                   import tiktoken
                   encoder = tiktoken.encoding_for_model(openai_model)
                   input_text = system_prompt + user_prompt
                   output_text = response.output_text or ""
                   input_tokens = len(encoder.encode(input_text))
                   output_tokens = len(encoder.encode(output_text))
                   total_tokens = input_tokens + output_tokens
                   
                   # Extract cache information from API response
                   usage = response.usage
                   if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
                       cached_tokens = getattr(usage.prompt_tokens_details, 'cached_tokens', 0)
                       cache_hit = cached_tokens > 0
                       cache_percentage = (cached_tokens / max(input_tokens, 1)) * 100
                   
                   # ✅ COMPREHENSIVE TOKEN LOGGING
                   logger.info(f"🔢 TOKENS → Input: {input_tokens} | Output: {output_tokens} | Cached: {cached_tokens} | Total: {total_tokens}")
                   
                   if cache_hit:
                       logger.info(f"⚡ CACHE HIT! {cached_tokens} tokens ({cache_percentage:.1f}%) from cache")
                       logger.info(f"💰 Savings: ~{cache_percentage:.1f}% cost reduction + significant speed boost")
                   else:
                       logger.info(f"💾 CACHE MISS - Building cache for next request")
                       if input_tokens >= 1024:
                           logger.info(f"✅ Qualifies for caching ({input_tokens} ≥ 1024 tokens)")
                       else:
                           logger.info(f"❌ Too short for caching ({input_tokens} < 1024 tokens)")
                   
               except Exception as e:
                   logger.warning(f"Token analysis failed: {str(e)}")
               
               logger.info("✅ OpenAI Structured Outputs successful")
               
               return {
                   "content": response.output_text,
                   "model": openai_model,
                   "success": True,
                   "structured": True,
                   "cached_tokens": cached_tokens,
                   "cache_hit": cache_hit,
                   "usage": response.usage.model_dump() if hasattr(response.usage, 'model_dump') else str(response.usage)
               }
               
           except Exception as structured_error:
               logger.warning(f"Structured Outputs failed: {str(structured_error)}")
               logger.info("Falling back to Chat Completions...")
               
               # ✅ TRY 2: Chat Completions Fallback (also has caching)
               import requests
               
               headers = {
                   "Authorization": f"Bearer {openai_api_key}",
                   "Content-Type": "application/json"
               }
               
               data = {
                   "model": openai_model,
                   "messages": [
                       {"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_prompt}
                   ],
                   "temperature": 0.05,
                   "response_format": {"type": "json_object"},
                   "max_tokens": 4000
               }
               
               response = requests.post(
                   "https://api.openai.com/v1/chat/completions",
                   headers=headers,
                   json=data,
                   timeout=60
               )
               
               if response.status_code == 200:
                   response_data = response.json()
                   content = response_data["choices"][0]["message"]["content"]
                   usage = response_data.get("usage", {})
                   
                   # ✅ ANALYZE TOKENS & CACHING FOR FALLBACK TOO
                   cached_tokens = 0
                   cache_hit = False
                   cache_percentage = 0.0
                   
                   try:
                       import tiktoken
                       encoder = tiktoken.encoding_for_model(openai_model)
                       input_text = system_prompt + user_prompt
                       input_tokens = len(encoder.encode(input_text))
                       output_tokens = len(encoder.encode(content))
                       total_tokens = input_tokens + output_tokens
                       
                       # Extract cache information from Chat Completions response
                       if 'prompt_tokens_details' in usage and usage['prompt_tokens_details']:
                           cached_tokens = usage['prompt_tokens_details'].get('cached_tokens', 0)
                           cache_hit = cached_tokens > 0
                           cache_percentage = (cached_tokens / max(input_tokens, 1)) * 100
                       
                       # ✅ COMPREHENSIVE TOKEN LOGGING
                       logger.info(f"🔢 TOKENS → Input: {input_tokens} | Output: {output_tokens} | Cached: {cached_tokens} | Total: {total_tokens}")
                       
                       if cache_hit:
                           logger.info(f"⚡ CACHE HIT! {cached_tokens} tokens ({cache_percentage:.1f}%) from cache")
                       else:
                           logger.info(f"💾 CACHE MISS - Building cache for next request")
                           
                   except Exception as e:
                       logger.warning(f"Token analysis failed: {str(e)}")
                   
                   logger.info("✅ Chat Completions fallback successful")
                   
                   return {
                       "content": content,
                       "model": openai_model,
                       "success": True,
                       "structured": False,
                       "cached_tokens": cached_tokens,
                       "cache_hit": cache_hit,
                       "usage": usage
                   }
               else:
                   raise Exception(f"Chat Completions failed: {response.status_code}: {response.text}")
                   
       except Exception as e:
           logger.error(f"❌ OpenAI API call failed: {str(e)}")
           return {
               "content": None,
               "error": str(e),
               "success": False,
               "cached_tokens": 0,
               "cache_hit": False
           }
# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

class ToolSearchCache:
    """Tool search cache with TTL support."""
    
    def __init__(self, max_size: int = Config.TOOL_SEARCH_CACHE_SIZE, 
                 expiry_minutes: int = Config.TOOL_SEARCH_CACHE_TTL_MINUTES):
        self.cache = {}
        self.max_size = max_size
        self.expiry = timedelta(minutes=expiry_minutes)
    
    def get(self, query: str) -> Optional[dict]:
        """Get cached result if not expired."""
        if query in self.cache:
            result, timestamp = self.cache[query]
            if datetime.now() - timestamp < self.expiry:
                return result
            else:
                del self.cache[query]
        return None
    
    def set(self, query: str, result: dict) -> None:
        """Set cache entry, removing oldest if at capacity."""
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[query] = (result, datetime.now())

# ============================================================================
# CUSTOM EMBEDDINGS IMPLEMENTATION
# ============================================================================

class NomicAtlasEmbeddings(Embeddings):
    """Nomic Atlas embedding API wrapper with task-specific support."""
    
    def __init__(self, api_key: str, api_url: str = "https://api-atlas.nomic.ai/v1/embedding/text",
                 document_task_type: str = "search_document", query_task_type: str = "search_query",
                 max_tokens_per_text: int = 8192, dimensionality: int = Config.DIMENSION):
        self.api_key = api_key
        self.api_url = api_url
        self.document_task_type = document_task_type
        self.query_task_type = query_task_type
        self.max_tokens_per_text = max_tokens_per_text
        self.dimensionality = dimensionality
    
    def _get_headers(self) -> Dict[str, str]:
        """Get API request headers."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def _embed_with_task_type(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Embed texts with specific task type."""
        for attempt in range(Config.MAX_RETRIES):
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
                    timeout=Config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                result = response.json()
                return result["embeddings"]
                
            except Exception as e:
                logger.warning(f"Error calling Nomic Atlas API (attempt {attempt+1}/{Config.MAX_RETRIES}): {str(e)}")
                if attempt < Config.MAX_RETRIES - 1:
                    time.sleep(Config.BASE_RETRY_DELAY * (2 ** attempt))
                else:
                    raise ValueError(f"Error calling Nomic Atlas API: {str(e)}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using document task type."""
        return self._embed_with_task_type(texts, self.document_task_type)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using query task type."""
        embeddings = self._embed_with_task_type([text], self.query_task_type)
        return embeddings[0]


# ============================================================================
# BM25 INDEX MANAGEMENT
# ============================================================================

class BM25IndexManager:
    """BM25 keyword search index manager."""
    
    def __init__(self):
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []
        self.tool_data = {}
        self.is_initialized = False
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK resources."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> List[str]:
        """Enhanced tokenization with compound word handling."""
        if not text:
            return []
        
        tokens = word_tokenize(str(text).lower())
        enhanced_tokens = []
        
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                enhanced_tokens.append(token)
                
                # Handle compound words like "texttovideo" → ["text", "to", "video"]
                if len(token) > 6:  # Only process longer words to avoid false splits
                    compound_parts = self._split_compound_word(token)
                    enhanced_tokens.extend(compound_parts)
        
        return enhanced_tokens

    def _split_compound_word(self, word: str) -> List[str]:
        """Split compound words into component parts using dynamic patterns."""
        parts = []
        
        # Dynamic pattern-based splitting for common connectors
        connectors = ['to', 'and', 'or', 'with', 'from', 'into', 'onto']
        
        for connector in connectors:
            if connector in word and len(word) > len(connector) + 4:  # Ensure meaningful parts
                idx = word.find(connector)
                if idx > 2 and idx + len(connector) < len(word) - 2:  # Ensure both parts are meaningful
                    before = word[:idx]
                    after = word[idx + len(connector):]
                    
                    # Only split if both parts are valid words (alphanumeric, reasonable length)
                    if (len(before) >= 2 and len(after) >= 2 and 
                        before.isalnum() and after.isalnum()):
                        parts.extend([before, connector, after])
                        break
        
        # Additional pattern: split AI-related compounds (ai + word)
        if word.startswith('ai') and len(word) > 4:
            remainder = word[2:]
            if remainder.isalnum() and len(remainder) >= 3:
                parts.extend(['ai', remainder])
        
        # Pattern: split common tech suffixes (word + tech/tool/app)
        tech_suffixes = ['tech', 'tool', 'app', 'bot', 'gen']
        for suffix in tech_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                prefix = word[:-len(suffix)]
                if prefix.isalnum() and len(prefix) >= 3:
                    parts.extend([prefix, suffix])
                    break
        
        # Only return valid, meaningful parts
        return [part for part in parts if len(part) >= 2 and part.isalnum()]
    
    def create_document_text(self, tool: Tool) -> str:
        """Create searchable text from tool data."""
        parts = [
            tool.name, tool.name,  # Double weight for name
            tool.category_subcat,
            tool.description,
            tool.owner or "",
            tool.status or ""
        ]
        
        # Add enhanced data if available
        if hasattr(tool, 'categories') and tool.categories:
            parts.extend([cat.Category for cat in tool.categories])
        
        if hasattr(tool, 'details') and tool.details:
            for attr in ['introduction', 'usage', 'speciality']:
                value = getattr(tool.details, attr, None)
                if value:
                    parts.append(value)
        
        if hasattr(tool, 'features') and tool.features:
            parts.extend(tool.features.pros or [])
            parts.extend(tool.features.cons or [])
        
        if hasattr(tool, 'pricingType') and tool.pricingType:
            parts.append(tool.pricingType)
        
        if hasattr(tool, 'qaSection') and tool.qaSection:
            for qa in tool.qaSection:
                parts.extend([qa.question, qa.answer])
        
        return " ".join(filter(None, parts))
    
    def add_tool(self, tool: Tool, rid: str) -> None:
        """Add tool to BM25 index with complete metadata preservation."""
        doc_text = self.create_document_text(tool)
        tokenized_doc = self.preprocess_text(doc_text)
        
        # Create comprehensive metadata for this tool
        complete_metadata = {
            "rid": rid,
            "tool_id": tool.tool_id,
            "name": tool.name,
            "category_subcat": tool.category_subcat or "",
            "url": str(tool.url),
            "description": tool.description or "",
            "image_url": tool.image_url or "",
            "owner": tool.owner or "",
            "status": tool.status or ""
        }
        
        # Add pricingType if available
        if hasattr(tool, 'pricingType') and tool.pricingType:
            complete_metadata["pricingType"] = tool.pricingType
        
        # Add detailed information if available
        if hasattr(tool, 'details') and tool.details:
            if tool.details.introduction:
                complete_metadata["details_introduction"] = tool.details.introduction
            if tool.details.usage:
                complete_metadata["details_usage"] = tool.details.usage
            if tool.details.speciality:
                complete_metadata["details_speciality"] = tool.details.speciality
        
        # Add features if available
        if hasattr(tool, 'features') and tool.features:
            if tool.features.pros:
                complete_metadata["features_pros"] = ",".join(tool.features.pros)
            if tool.features.cons:
                complete_metadata["features_cons"] = ",".join(tool.features.cons)
        
        # Add metrics if available
        if hasattr(tool, 'metrics') and tool.metrics:
            for key, value in tool.metrics.dict().items():
                complete_metadata[f"metrics_{key}"] = value
        
        # Add categories list if available
        if hasattr(tool, 'categories') and tool.categories:
            complete_metadata["categories_list"] = [cat.Category for cat in tool.categories]
        
        # Add pricing information if available
        if hasattr(tool, 'pricing') and tool.pricing:
            complete_metadata["pricing_plans"] = ",".join([plan.planName for plan in tool.pricing])
            complete_metadata["pricing_prices"] = ",".join([plan.price for plan in tool.pricing])
        
        # Add QA section if available
        if hasattr(tool, 'qaSection') and tool.qaSection:
            complete_metadata["qa_questions"] = ",".join([qa.question for qa in tool.qaSection])
            complete_metadata["qa_answers"] = ",".join([qa.answer for qa in tool.qaSection])
        
        # Store both tool object and complete metadata
        self.tool_data[rid] = {
            "tool": tool,
            "tokenized_doc": tokenized_doc,
            "complete_metadata": complete_metadata  # Store complete metadata
        }
        self.is_initialized = False
    
    def rebuild_index(self, batch_size: int = Config.DEFAULT_BATCH_SIZE) -> None:
        """Rebuild BM25 index with all documents."""
        self.doc_ids = list(self.tool_data.keys())
        self.tokenized_corpus = []
        
        if not self.doc_ids:
            logger.warning("No documents to index for BM25")
            return
        
        total_batches = (len(self.doc_ids) - 1) // batch_size + 1
        logger.info(f"Rebuilding BM25 index with {len(self.doc_ids)} documents in {total_batches} batches")
        
        for i in range(0, len(self.doc_ids), batch_size):
            batch_end = min(i + batch_size, len(self.doc_ids))
            batch_ids = self.doc_ids[i:batch_end]
            
            batch_docs = [self.tool_data[doc_id]["tokenized_doc"] for doc_id in batch_ids]
            self.tokenized_corpus.extend(batch_docs)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{total_batches} for BM25 index")
        
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_initialized = True
        logger.info(f"BM25 index successfully built with {len(self.doc_ids)} documents")
    
    def search(self, query: str, top_k: int = Config.BM25_SEARCH_K) -> List[Dict[str, Any]]:
        """Search BM25 index and return complete metadata."""
        if not self.is_initialized:
            self.rebuild_index()
        
        if not self.is_initialized or not self.bm25:
            logger.warning("BM25 index not initialized")
            return []
        
        tokenized_query = self.preprocess_text(query)
        if not tokenized_query:
            return []
        
        scores = self.bm25.get_scores(tokenized_query)
        top_k = min(top_k, len(self.doc_ids))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id = self.doc_ids[idx]
                
                # Get the complete metadata that was stored with this tool
                complete_metadata = dict(self.tool_data[doc_id]["complete_metadata"])
                complete_metadata["score"] = float(scores[idx])
                
                results.append(complete_metadata)
        
        return results

# ============================================================================
# APPLICATION STATE MANAGEMENT
# ============================================================================

class ApplicationState:
    """Global application state management."""
    
    def __init__(self):
        self.index_ready = False
        self.initialization_started = False
        self.vectors_loaded = 0
        self.total_vectors = 0
        self.bm25_ready = False
        self.vectors_loading = False
        self.bm25_building = False
        self.first_search_processed = False
# ============================================================================
# TOOL MANAGEMENT UTILITIES
# ============================================================================

class ToolManager:
    """Tool management utilities."""
    
    @staticmethod
    def clean_tool_data(tool: Tool) -> Tool:
        """Clean and validate tool data."""
        cleaned_tool = tool.model_copy() if hasattr(tool, 'model_copy') else Tool(**tool.dict())
        
        # Ensure backward compatibility
        if not cleaned_tool.category_subcat:
            cleaned_tool.category_subcat = cleaned_tool.generate_category_subcat()
        
        if not cleaned_tool.description:
            cleaned_tool.description = cleaned_tool.generate_description()
        
        # Clean name if it's a URL
        if cleaned_tool.name and cleaned_tool.name.startswith('http'):
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(cleaned_tool.name)
                cleaned_tool.name = parsed_url.netloc.replace('www.', '')
            except Exception:
                logger.warning(f"Failed to parse URL in name: {cleaned_tool.name}")
        
        # Clean text fields
        if cleaned_tool.description:
            cleaned_tool.description = TextCleaner.clean_text(cleaned_tool.description)
        
        if cleaned_tool.category_subcat:
            cleaned_tool.category_subcat = TextCleaner.clean_text(cleaned_tool.category_subcat)
        
        # Clean status and image_url
        if cleaned_tool.status == "Not Added":
            cleaned_tool.status = ""
        
        if cleaned_tool.image_url == "Added":
            cleaned_tool.image_url = ""
        
        return cleaned_tool
    
    @staticmethod
    def format_tool_for_indexing(tool: Tool, rid: str) -> str:
        """Format tool data for embedding."""
        sections = [
            f"RID: {rid}",
            f"Tool ID: {tool.tool_id}",
            f"Name: {tool.name}",
        ]
        
        if tool.categories:
            categories_text = ", ".join([cat.Category for cat in tool.categories])
            sections.append(f"Categories: {categories_text}")
        elif tool.category_subcat:
            sections.append(f"Category/Sub-cat: {tool.category_subcat}")
        
        sections.append(f"URL: {tool.url}")
        
        if tool.pricingType:
            sections.append(f"Pricing Type: {tool.pricingType}")
        
        if tool.details:
            details_section = ["Details:"]
            for attr in ['introduction', 'usage', 'speciality']:
                value = getattr(tool.details, attr, None)
                if value:
                    details_section.append(f"{attr.title()}: {value}")
            if len(details_section) > 1:
                sections.append("\n".join(details_section))
        
        if tool.features and (tool.features.pros or tool.features.cons):
            features_section = ["Features:"]
            if tool.features.pros:
                features_section.append("Pros:")
                features_section.extend([f"- {pro}" for pro in tool.features.pros])
            if tool.features.cons:
                features_section.append("Cons:")
                features_section.extend([f"- {con}" for con in tool.features.cons])
            sections.append("\n".join(features_section))
        
        if tool.description:
            sections.append(f"Description:\n{tool.description}")
        
        if tool.metrics:
            metrics_section = ["Metrics:"]
            for key, value in tool.metrics.dict().items():
                metrics_section.append(f"{key}: {value}")
            sections.append("\n".join(metrics_section))
        
        if tool.pricing:
            pricing_section = ["Pricing Plans:"]
            for plan in tool.pricing:
                pricing_section.append(f"- {plan.planName}: {plan.price}")
                if plan.features:
                    pricing_section.append("  Features: " + ", ".join(plan.features))
            sections.append("\n".join(pricing_section))
        
        if tool.qaSection:
            qa_section = ["FAQ:"]
            for qa in tool.qaSection:
                qa_section.extend([f"Q: {qa.question}", f"A: {qa.answer}"])
            sections.append("\n".join(qa_section))
        
        # Add metadata
        sections.extend([
            f"Image URL: {tool.image_url or 'N/A'}",
            f"Owner: {tool.owner or 'Unassigned'}",
            f"Status: {tool.status or 'Unknown'}"
        ])
        
        return "\n\n".join(sections)

# ============================================================================
# SEARCH AND RANKING UTILITIES
# ============================================================================

class SearchUtils:
    """Search and ranking utilities."""
    
    @staticmethod
    def fuse_search_results(vector_results: List[Dict], bm25_results: List[Dict], 
                          alpha: float = Config.HYBRID_ALPHA) -> List[Dict]:
        """Fuse vector and BM25 search results."""
        all_ids = set()
        for result in vector_results + bm25_results:
            rid = result.get("rid", "")
            if rid:
                all_ids.add(rid)
        
        vector_scores = {result.get("rid", ""): result.get("score", 0) for result in vector_results}
        bm25_scores = {result.get("rid", ""): result.get("score", 0) for result in bm25_results}
        
        # Normalize scores
        if vector_scores:
            max_vector = max(vector_scores.values()) or 1
            vector_scores = {k: v/max_vector for k, v in vector_scores.items()}
        
        if bm25_scores:
            max_bm25 = max(bm25_scores.values()) or 1
            bm25_scores = {k: v/max_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores
        combined_results = []
        for doc_id in all_ids:
            v_score = vector_scores.get(doc_id, 0)
            b_score = bm25_scores.get(doc_id, 0)
            combined_score = alpha * v_score + (1 - alpha) * b_score
            
            # Find result object
            result_obj = None
            for result in vector_results + bm25_results:
                if result.get("rid", "") == doc_id:
                    result_obj = result
                    break
            
            if result_obj:
                combined_result = dict(result_obj)
                combined_result.update({
                    "score": combined_score,
                    "vector_score": v_score,
                    "bm25_score": b_score
                })
                combined_results.append(combined_result)
        
        return sorted(combined_results, key=lambda x: x.get("score", 0), reverse=True)
    
    @staticmethod
    def calculate_metadata_completeness_score(tool: Tool) -> float:
        """Calculate metadata completeness score (0-80 points)."""
        score = 0
        
        # Basic metadata (10 points each)
        if tool.description and tool.description.strip():
            score += 10
        if tool.category_subcat or (tool.categories and len(tool.categories) > 0):
            score += 10
        
        # Advanced metadata (15 points each)
        if tool.pricing and len(tool.pricing) > 0:
            score += 15
        if tool.qaSection and len(tool.qaSection) > 0:
            score += 15
        
        # Medium metadata (10 points)
        if tool.features and (tool.features.pros or tool.features.cons):
            score += 10
        
        # Basic info (5 points)
        if tool.owner and tool.owner.strip():
            score += 5
        
        # Complete profile bonus (15 points)
        if all([
            tool.description and tool.description.strip(),
            tool.category_subcat or (tool.categories and len(tool.categories) > 0),
            tool.pricing and len(tool.pricing) > 0
        ]):
            score += 15
        
        return min(score, 80)
    
    @staticmethod
    def calculate_use_case_relevance_score(tool: Tool, use_case: str) -> float:
        """Calculate use case relevance score (0-50 points)."""
        use_case_lower = use_case.lower().strip()
        score = 0
        
        # Check categories for exact match
        if tool.categories:
            for category in tool.categories:
                if use_case_lower in category.Category.lower():
                    score += 50
                    break
        elif tool.category_subcat and use_case_lower in tool.category_subcat.lower():
            score += 50
        
        # Partial matches if no exact match
        if score == 0:
            if tool.categories:
                for category in tool.categories:
                    category_words = category.Category.lower().split()
                    use_case_words = use_case_lower.split()
                    if any(word in category_words for word in use_case_words):
                        score += 30
                        break
            elif tool.category_subcat:
                category_words = tool.category_subcat.lower().split()
                use_case_words = use_case_lower.split()
                if any(word in category_words for word in use_case_words):
                    score += 30
        
        # Additional scoring
        if tool.description and use_case_lower in tool.description.lower():
            score += 20
        
        if tool.details and tool.details.speciality and use_case_lower in tool.details.speciality.lower():
            score += 25
        
        if tool.name and use_case_lower in tool.name.lower():
            score += 20
        
        return min(score, 50)
    
    @staticmethod
    def calculate_popularity_score(tool: Tool, use_case: str) -> float:
        """Calculate composite popularity score."""
        metadata_score = SearchUtils.calculate_metadata_completeness_score(tool)
        relevance_score = SearchUtils.calculate_use_case_relevance_score(tool, use_case)
        
        composite_score = (metadata_score * 0.6) + (relevance_score * 0.4)
        
        # Apply use case specific boosts
        if tool.name and use_case.lower() in tool.name.lower():
            composite_score *= 1.2
        
        return composite_score
    
    @staticmethod
    def ensure_tool_diversity(tools: List[Dict], max_similar: int = 1) -> List[Dict]:
        """Ensure diversity by limiting similar tools."""
        if len(tools) <= 4:
            return tools
        
        diverse_tools = []
        name_words_seen = set()
        
        for tool in tools:
            tool_name_words = set(tool.get('name', '').lower().split())
            
            is_similar = False
            for seen_words in name_words_seen:
                overlap = len(tool_name_words.intersection(seen_words))
                if overlap >= 2:
                    is_similar = True
                    break
            
            if not is_similar or len(diverse_tools) < 2:
                diverse_tools.append(tool)
                name_words_seen.add(frozenset(tool_name_words))
            
            if len(diverse_tools) >= 4:
                break
        
        return diverse_tools
    
    @staticmethod
    def generate_popularity_indicators(tool: Tool, use_case: str, score: float) -> List[str]:
        """Generate popularity indicators based on scoring factors."""
        indicators = []
        
        metadata_score = SearchUtils.calculate_metadata_completeness_score(tool)
        if metadata_score >= 60:
            indicators.append("Complete tool profile with comprehensive metadata")
        elif metadata_score >= 40:
            indicators.append("Well-documented tool with good metadata coverage")
        
        relevance_score = SearchUtils.calculate_use_case_relevance_score(tool, use_case)
        if relevance_score >= 40:
            indicators.append(f"Exact category match for {use_case}")
        elif relevance_score >= 20:
            indicators.append(f"Strong relevance to {use_case} use case")
        
        if tool.pricing and len(tool.pricing) > 0:
            indicators.append("Detailed pricing information available")
        
        if tool.features and (tool.features.pros or tool.features.cons):
            indicators.append("Comprehensive feature analysis provided")
        
        if tool.qaSection and len(tool.qaSection) > 0:
            indicators.append("Extensive FAQ and user guidance available")
        
        if len(indicators) < 2:
            indicators.append("Identified as relevant tool for the use case")
        
        return indicators[:3]
    
    @staticmethod
    def select_tools_by_score(hybrid_results: List[Dict], 
                            min_score: float = Config.HYBRID_MIN_SCORE, 
                            max_tools: int = Config.HYBRID_MAX_TOOLS,
                            fallback_count: int = Config.HYBRID_FALLBACK_COUNT,
                            high_individual_threshold: float = 0.9) -> List[Dict]:
       """Select tools based on score thresholds with special handling for high individual scores."""
       qualified_tools = []
       
       logger.info(f"Score-based selection: min_score={min_score}, max_tools={max_tools}, high_individual_threshold={high_individual_threshold}")
       
       for i, result in enumerate(hybrid_results):
           score = result.get("score", 0)
           tool_name = result.get("name", "Unknown")
           tool_id = result.get("tool_id", "Unknown")
           vector_score = result.get("vector_score", 0)
           bm25_score = result.get("bm25_score", 0)

           # Check inclusion conditions
           meets_min_threshold = score >= min_score
           high_vector_score = vector_score >= high_individual_threshold
           high_bm25_score = bm25_score >= high_individual_threshold
           
           # Include if ANY condition is met
           if meets_min_threshold or high_vector_score or high_bm25_score:
               qualified_tools.append(result)
               
               # Log with appropriate reason
               reasons = []
               if meets_min_threshold:
                   reasons.append("min_threshold")
               if high_vector_score:
                   reasons.append("high_vector")
               if high_bm25_score:
                   reasons.append("high_bm25")
               
               reason_str = "+".join(reasons)
               logger.info("Rank %2d: '%s' (ID: %s) qualified (%s) - Score: %.3f (V:%.3f + B:%.3f)", 
                          i+1, str(tool_name), str(tool_id), reason_str, score, vector_score, bm25_score)
           else:
               logger.info("Rank %2d: '%s' (ID: %s) rejected - Score: %.3f (V:%.3f + B:%.3f)", 
                          i+1, str(tool_name), str(tool_id), score, vector_score, bm25_score)

           # Don't exceed maximum to avoid token limits
           if len(qualified_tools) >= max_tools:
               logger.info(f"Reached maximum {max_tools} tools, stopping selection")
               break
       
       # Ensure we send at least fallback_count tools even if scores are low
       if len(qualified_tools) == 0 and len(hybrid_results) > 0:
           qualified_tools = hybrid_results[:fallback_count]
           logger.warning(f"No tools met any threshold, using fallback: top {fallback_count} tools")
       
       logger.info(f"Selected {len(qualified_tools)} tools for LLM processing")
       return qualified_tools

# ============================================================================
# KEYWORD EXTRACTION UTILITIES
# ============================================================================

class KeywordExtractor:
    """Keyword extraction utilities."""
    
    @staticmethod
    def get_statistical_candidates(reviews: List[str]) -> List[Dict[str, Any]]:
        """Extract keyword candidates using TF-IDF and n-grams."""
        try:
            cleaned_reviews = [TextCleaner.clean_text(review) for review in reviews]
            cleaned_reviews = [review for review in cleaned_reviews if review.strip()]
            
            if not cleaned_reviews:
                return []
            
            all_text = " ".join(cleaned_reviews)
            
            try:
                tokens = word_tokenize(all_text.lower())
            except Exception:
                tokens = all_text.lower().split()
            
            stop_words = set(stopwords.words('english')) if 'english' in stopwords.fileids() else set()
            domain_stopwords = {'tool', 'app', 'software', 'product', 'service', 'platform', 'application'}
            all_stopwords = stop_words.union(domain_stopwords)
            
            filtered_tokens = [
                token for token in tokens 
                if token.isalnum() and len(token) > 1 and token not in all_stopwords
            ]
            
            candidates = []
            
            # 1-grams
            unigram_counts = Counter(filtered_tokens)
            for word, freq in unigram_counts.items():
                candidates.append({
                    "term": word,
                    "frequency": freq,
                    "tfidf_score": freq / len(filtered_tokens),
                    "type": "unigram"
                })
            
            # 2-grams
            if len(filtered_tokens) >= 2:
                bigrams = [f"{filtered_tokens[i]} {filtered_tokens[i+1]}" 
                          for i in range(len(filtered_tokens)-1)]
                bigram_counts = Counter(bigrams)
                
                for bigram, freq in bigram_counts.items():
                    candidates.append({
                        "term": bigram,
                        "frequency": freq,
                        "tfidf_score": freq / len(bigrams) if bigrams else 0,
                        "type": "bigram"
                    })
            
            # 3-grams
            if len(filtered_tokens) >= 3:
                trigrams = [f"{filtered_tokens[i]} {filtered_tokens[i+1]} {filtered_tokens[i+2]}" 
                           for i in range(len(filtered_tokens)-2)]
                trigram_counts = Counter(trigrams)
                
                for trigram, freq in trigram_counts.items():
                    if freq > 1:
                        candidates.append({
                            "term": trigram,
                            "frequency": freq,
                            "tfidf_score": freq / len(trigrams) if trigrams else 0,
                            "type": "trigram"
                        })
            
            candidates.sort(key=lambda x: (x["frequency"], x["tfidf_score"]), reverse=True)
            
            # Filter candidates
            min_freq = max(1, len(cleaned_reviews) * 0.02)
            max_freq = len(cleaned_reviews) * 0.8
            
            filtered_candidates = [
                c for c in candidates 
                if min_freq <= c["frequency"] <= max_freq or c["frequency"] >= 2
            ]
            
            return filtered_candidates[:30]
            
        except Exception as e:
            logger.error(f"Error in statistical candidate extraction: {str(e)}")
            return []
    
    @staticmethod
    def process_keywords_with_llm(candidates: List[Dict], reviews: List[str], 
                                target_count: int) -> List[Dict]:
        """Process candidates with LLM to get refined keywords."""
        try:
            sample_reviews = reviews[:5]
            
            system_prompt = f"""You are a keyword analysis expert. I'll provide candidate keywords from statistical analysis and sample reviews.

Your task: Select and refine exactly {target_count} keywords that best represent the review themes.

Rules:
1. Group similar terms (e.g., "expensive" + "overpriced" → "expensive")
2. Create meaningful phrases (e.g., "features" → "great features" if reviews mention it)
3. Add sentiment for each keyword (positive/negative/neutral)
4. Remove noise/irrelevant terms
5. Focus on the most impactful keywords that users care about

Output format: JSON array only
[
  {{"keyword": "user-friendly", "sentiment": "positive"}},
  {{"keyword": "expensive", "sentiment": "negative"}}
]"""

            user_prompt = f"""Candidate Keywords (from statistical analysis):
{json.dumps(candidates, indent=2)}

Sample Reviews for context:
{json.dumps(sample_reviews, indent=2)}

Select and refine the best {target_count} keywords from the candidates above."""

            llm = ChatGroq(
                groq_api_key=env.groq_api_key,
                model_name=env.dev_model,
                temperature=0.1,
                max_tokens=500
            )
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            llm_response = response.content.strip()
            
            if llm_response.startswith("```"):
                llm_response = re.sub(r'^```(?:json)?\s*', '', llm_response)
                llm_response = re.sub(r'\s*```$', '', llm_response)
            
            keywords = json.loads(llm_response)
            
            if isinstance(keywords, list):
                valid_keywords = []
                for kw in keywords:
                    if isinstance(kw, dict) and "keyword" in kw and "sentiment" in kw:
                        valid_keywords.append({
                            "keyword": str(kw["keyword"]).strip(),
                            "sentiment": str(kw["sentiment"]).lower()
                        })
                return valid_keywords[:target_count]
            
            return []
            
        except Exception as e:
            logger.error(f"Error in LLM processing: {str(e)}")
            return []
    
    @staticmethod
    def apply_basic_sentiment(candidates: List[Dict]) -> List[KeywordItem]:
        """Apply basic rule-based sentiment analysis as fallback."""
        positive_words = {
            'great', 'good', 'excellent', 'amazing', 'awesome', 'fantastic', 'wonderful',
            'easy', 'simple', 'user-friendly', 'intuitive', 'fast', 'quick', 'reliable',
            'helpful', 'useful', 'recommend', 'love', 'like', 'perfect', 'best'
        }
        
        negative_words = {
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'slow', 'difficult', 'hard',
            'expensive', 'overpriced', 'buggy', 'broken', 'confusing', 'complicated',
            'hate', 'dislike', 'worst', 'useless', 'frustrating', 'annoying'
        }
        
        result = []
        for candidate in candidates:
            term = candidate["term"].lower()
            
            sentiment = "neutral"
            if any(pos_word in term for pos_word in positive_words):
                sentiment = "positive"
            elif any(neg_word in term for neg_word in negative_words):
                sentiment = "negative"
            
            result.append(KeywordItem(
                keyword=candidate["term"],
                sentiment=sentiment
            ))
        
        return result



class RelatedToolsUtils:
    """Utilities for finding related tools using hybrid search."""
    
    @staticmethod
    def create_search_query_from_tool(tool_metadata: Dict[str, Any]) -> str:
        """Create a search query from tool metadata."""
        query_parts = []
        
        # Add tool name (most important)
        name = tool_metadata.get("name", "")
        if name:
            query_parts.append(name)
        
        # Add categories
        category_subcat = tool_metadata.get("category_subcat", "")
        if category_subcat:
            query_parts.append(category_subcat)
        
        # Add description (truncated to avoid too long queries)
        description = tool_metadata.get("description", "")
        if description:
            # Take first 100 characters to keep query manageable
            desc_short = description[:100].strip()
            query_parts.append(desc_short)
        
        # Add pricing type if available
        pricing_type = tool_metadata.get("pricingType", "")
        if pricing_type:
            query_parts.append(pricing_type)
        
        # Join with spaces and clean
        search_query = " ".join(query_parts)
        return TextCleaner.clean_text(search_query)
    
    @staticmethod
    def filter_and_rank_results(hybrid_results: List[Dict], 
                               original_tool_id: str, 
                               min_score: float = 0.4,
                               max_results: int = 12) -> List[str]:
        """Filter hybrid search results and return tool IDs."""
        
        # Filter out original tool and low-quality results
        filtered_results = [
            result for result in hybrid_results
            if (result.get("tool_id") != original_tool_id and 
                result.get("score", 0) >= min_score)
        ]
        
        # Sort by score (highest first)
        filtered_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top results and extract tool IDs
        top_results = filtered_results[:max_results]
        tool_ids = [result.get("tool_id") for result in top_results if result.get("tool_id")]
        
        return tool_ids

# ============================================================================
# QUERY PROCESSING UTILITIES
# ============================================================================

class QueryProcessor:
    """Query processing and suggestion utilities."""
    
    @staticmethod
    def post_process_llm_response(response_text: str) -> str:
        """Post-process LLM response by removing think tags and code blocks."""
        try:
            if "<think>" in response_text:
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
                response_text = response_text.strip()
            
            code_block_match = re.match(r'^```(?:json)?\s*([\s\S]*?)\s*```\s*$', response_text, re.DOTALL)
            if code_block_match:
                response_text = code_block_match.group(1).strip()
            
            return response_text
        
        except Exception as e:
            logger.warning(f"Error in post-processing: {str(e)}")
            return response_text
    
    @staticmethod
    def generate_fallback_suggestions(query: str) -> List[str]:
        """Generate fallback suggestions using templates when LLM fails."""
        query_lower = query.lower()
        
        templates = [
            f"free {query}",
            f"{query} for beginners",
            f"best {query}",
            f"{query} online",
            f"AI-powered {query}"
        ]
        
        # Specialized templates based on keywords
        if any(word in query_lower for word in ['tool', 'software', 'app', 'platform']):
            templates = [
                f"free {query}",
                f"{query} for small business",
                f"online {query}",
                f"{query} with collaboration features",
                f"enterprise {query}"
            ]
        elif any(word in query_lower for word in ['editing', 'editor', 'create', 'design']):
            templates = [
                f"{query} for beginners",
                f"professional {query}",
                f"{query} with AI features",
                f"online {query}",
                f"{query} for teams"
            ]
        elif any(word in query_lower for word in ['management', 'organize', 'track']):
            templates = [
                f"{query} software",
                f"{query} for teams",
                f"simple {query} tool",
                f"{query} with reporting",
                f"cloud-based {query}"
            ]
        
        return templates[:5]


# ============================================================================
# MONGODB MANAGER
# ============================================================================

class MongoDBManager:
    """MongoDB connection and operations manager."""
    
    def __init__(self):
        self.client: AsyncIOMotorClient = None
        self.database = None
    
    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(env.mongo_uri)
            self.database = self.client[Config.MONGO_DB_NAME]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ MongoDB connected successfully")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {str(e)}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def _convert_objectids_to_strings(self, data):
        """Recursively convert ObjectIds to strings for JSON serialization."""
        if isinstance(data, ObjectId):
            return str(data)
        elif isinstance(data, dict):
            return {key: self._convert_objectids_to_strings(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_objectids_to_strings(item) for item in data]
        else:
            return data
    
    async def get_tool_by_id(self, tool_id: str) -> Optional[Dict]:
        """Get tool by _id from MongoDB."""
        try:
            collection = self.database[Config.MONGO_COLLECTION_TOOLS]
            
            # Convert string to ObjectId
            object_id = ObjectId(tool_id)
            tool = await collection.find_one({"_id": object_id})
            
            if tool:
                # Convert all ObjectIds to strings recursively
                tool = self._convert_objectids_to_strings(tool)
                logger.info(f"Found tool: {tool.get('name', 'Unknown')}")
                return tool
            else:
                logger.warning(f"Tool not found with ID: {tool_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching tool {tool_id}: {str(e)}")
            return None
    
    async def get_all_categories(self) -> List[Dict]:
        """Get all categories from MongoDB."""
        try:
            collection = self.database[Config.MONGO_COLLECTION_CATEGORIES]
            categories = []
            
            async for category in collection.find({}):
                # Convert ObjectIds to strings
                category = self._convert_objectids_to_strings(category)
                categories.append(category)
            
            logger.info(f"Fetched {len(categories)} categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error fetching categories: {str(e)}")
            return []
    
    async def get_all_usecases(self) -> List[Dict]:
        """Get all use cases from MongoDB."""
        try:
            collection = self.database[Config.MONGO_COLLECTION_USECASES]
            usecases = []
            
            async for usecase in collection.find({}):
                # Convert ObjectIds to strings
                usecase = self._convert_objectids_to_strings(usecase)
                usecases.append(usecase)
            
            logger.info(f"Fetched {len(usecases)} use cases")
            return usecases
            
        except Exception as e:
            logger.error(f"Error fetching use cases: {str(e)}")
            return []

def deduplicate_results(results: List[Dict], id_field: str) -> List[Dict]:
    """Remove duplicate results based on ID field."""
    seen_ids = set()
    deduplicated = []
    
    for result in results:
        result_id = result.get(id_field, "")
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            deduplicated.append(result)
    
    return deduplicated

# ============================================================================
# VECTOR STORE MANAGEMENT
# ============================================================================

class VectorStoreManager:
    """Vector store management with caching."""
    
    def __init__(self):
        self._vector_store_cache = None
        self._vector_store_timestamp = None
        self._query_vector_store_cache = None
        self._query_vector_store_timestamp = None
        self.cache_lifetime = Config.VECTOR_STORE_CACHE_LIFETIME
        self.pc = Pinecone(api_key=env.pinecone_api_key)
    
    def get_vector_store(self, headers=None, for_query=False):
        """Get vector store instance with caching."""
        current_time = time.time()
        
        if for_query:
            if (self._query_vector_store_cache is not None and 
                self._query_vector_store_timestamp is not None and 
                current_time - self._query_vector_store_timestamp < self.cache_lifetime):
                logger.info("Using cached query vector store instance")
                return self._query_vector_store_cache
        else:
            if (self._vector_store_cache is not None and 
                self._vector_store_timestamp is not None and 
                current_time - self._vector_store_timestamp < self.cache_lifetime):
                logger.info("Using cached document vector store instance")
                return self._vector_store_cache
        
        try:
            embeddings = NomicAtlasEmbeddings(
                api_key=env.nomic_api_key,
                dimensionality=Config.DIMENSION
            )
            pinecone_index = self.pc.Index(env.index_name)
            vector_store = PineconeVectorStore(
                index=pinecone_index,
                embedding=embeddings,
                text_key="text"
            )
            
            if for_query:
                self._query_vector_store_cache = vector_store
                self._query_vector_store_timestamp = current_time
                logger.info("Created and cached new query vector store instance")
            else:
                self._vector_store_cache = vector_store
                self._vector_store_timestamp = current_time
                logger.info("Created and cached new document vector store instance")
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error in get_vector_store: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize vector store: {str(e)}"
            )
    
    def get_or_create_index(self):
        """Create Pinecone index if it doesn't exist."""
        try:
            indexes = self.pc.list_indexes()
            index_names = [index.name for index in indexes]
            
            if env.index_name not in index_names:
                logger.info(f"Creating new index: {env.index_name}")
                self.pc.create_index(
                    name=env.index_name,
                    dimension=Config.DIMENSION,
                    metric="cosine"
                )
                logger.info(f"Index {env.index_name} created successfully")
            
            return self.pc.Index(env.index_name)
            
        except Exception as e:
            logger.error(f"Error in get_or_create_index: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize Pinecone index: {str(e)}"
            )
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the index."""
        try:
            index = self.pc.Index(env.index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            logger.info(f"Total vectors in index: {total_vectors}")
            return total_vectors
        except Exception as e:
            logger.error(f"Error getting vector count: {str(e)}")
            return 0

# ============================================================================
# INITIALIZE GLOBAL COMPONENTS
# ============================================================================

app_state = ApplicationState()
tool_search_cache = ToolSearchCache()
bm25_index = BM25IndexManager()
vector_store_manager = VectorStoreManager()
mongodb_manager = MongoDBManager()

# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="AI Tool Search API",
    description="Production-ready AI tool search API with hybrid vector and keyword search",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# APPLICATION EVENT HANDLERS
# ============================================================================

def build_bm25_lazy():
    """Build BM25 index lazily (called on first search)."""
    try:
        if app_state.bm25_ready or app_state.bm25_building:
            return
            
        app_state.bm25_building = True
        logger.info("🔧 Building BM25 index lazily (first search detected)")
        
        start_time = time.time()
        bm25_index.rebuild_index()
        build_time = time.time() - start_time
        
        app_state.bm25_ready = True
        app_state.bm25_building = False
        app_state.first_search_processed = True
        
        logger.info(f"✅ BM25 index built successfully in {build_time:.2f}s ({len(bm25_index.tool_data)} documents)")
        
    except Exception as e:
        logger.error(f"Error building BM25 index lazily: {str(e)}")
        app_state.bm25_building = False

async def load_vectors_optimized_serverless(total_vectors: int, first_batch_size: int) -> int:
    """OPTIONAL: Load vectors using proper Pinecone serverless methods (only if new SDK available)."""
    try:
        # Only try this if the new SDK is available
        index = vector_store_manager.pc.Index(env.index_name)
        
        # Check if serverless list methods are available
        if not hasattr(index, 'list_paginated'):
            logger.info("list_paginated not available, skipping optimized serverless loading")
            raise Exception("Serverless methods not available")
        
        total_loaded = 0
        processed_rids = set()
        
        logger.info(f"Using Pinecone serverless list_paginated() method for batch loading")
        
        batch_count = 0
        pagination_token = None
        
        # Load vectors in batches using list_paginated
        while total_loaded < total_vectors:
            try:
                # Determine batch size - smaller for first batch, larger for subsequent
                if batch_count == 0:
                    limit = first_batch_size
                else:
                    limit = min(150, total_vectors - total_loaded)
                
                # Use list_paginated for controlled batch loading
                if pagination_token:
                    results = index.list_paginated(
                        limit=limit,
                        pagination_token=pagination_token
                    )
                else:
                    results = index.list_paginated(limit=limit)
                
                # Get vector IDs from this batch
                vector_ids = [vector.id for vector in results.vectors]
                
                if not vector_ids:
                    logger.info("No more vectors to fetch")
                    break
                
                # Fetch the actual vector data using the IDs
                fetched_vectors = index.fetch(ids=vector_ids)
                
                batch_loaded = 0
                for vector_id, vector_data in fetched_vectors.vectors.items():
                    try:
                        if vector_id in processed_rids:
                            continue
                            
                        processed_rids.add(vector_id)
                        metadata = vector_data.metadata
                        
                        if not metadata or "tool_id" not in metadata:
                            continue
                        
                        tool = Tool(
                            tool_id=metadata.get("tool_id", ""),
                            name=metadata.get("name", "Unknown"),
                            category_subcat=metadata.get("category_subcat", ""),
                            url=metadata.get("url", "https://example.com"),
                            description=metadata.get("description", ""),
                            image_url=metadata.get("image_url", None),
                            owner=metadata.get("owner", None),
                            status=metadata.get("status", None)
                        )
                        
                        bm25_index.add_tool(tool, vector_id)
                        batch_loaded += 1
                        total_loaded += 1
                        
                    except Exception as e:
                        logger.warning(f"Error processing vector {vector_id}: {str(e)}")
                        continue
                
                batch_count += 1
                logger.info(f"Optimized batch {batch_count}: loaded {batch_loaded} vectors (total: {total_loaded})")
                
                # Update app state after first batch
                if batch_count == 1:
                    app_state.vectors_loaded = total_loaded
                    app_state.index_ready = True
                    logger.info(f"🚀 System ready after first optimized batch! Loaded {total_loaded} vectors")
                
                # Check for more data
                if hasattr(results, 'pagination') and hasattr(results.pagination, 'next'):
                    pagination_token = results.pagination.next
                else:
                    logger.info("No more pages available")
                    break
                
                # Small delay between batches
                if batch_count > 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as batch_error:
                logger.error(f"Error in optimized batch {batch_count}: {str(batch_error)}")
                break
        
        return total_loaded
        
    except Exception as e:
        logger.warning(f"Optimized serverless loading not available: {str(e)}")
        return 0

async def load_vectors_batch_fallback(total_vectors: int, first_batch_size: int) -> int:
    """RELIABLE: Batch loading using existing similarity_search method (preserves current working approach)."""
    try:
        vector_store = vector_store_manager.get_vector_store()
        total_loaded = 0
        
        # Load first batch quickly
        logger.info(f"Loading first batch of {first_batch_size} vectors (similarity_search method)")
        first_results = vector_store.similarity_search("", k=first_batch_size)
        
        for doc in first_results:
            try:
                metadata = doc.metadata
                rid = metadata.get("rid")
                
                if not rid or "tool_id" not in metadata:
                    continue
                
                tool = Tool(
                    tool_id=metadata.get("tool_id", ""),
                    name=metadata.get("name", "Unknown"),
                    category_subcat=metadata.get("category_subcat", ""),
                    url=metadata.get("url", "https://example.com"),
                    description=metadata.get("description", ""),
                    image_url=metadata.get("image_url", None),
                    owner=metadata.get("owner", None),
                    status=metadata.get("status", None)
                )
                
                bm25_index.add_tool(tool, rid)
                total_loaded += 1
                
            except Exception as e:
                continue
        
        # READY AFTER FIRST BATCH
        app_state.vectors_loaded = total_loaded
        app_state.index_ready = True
        logger.info(f"🚀 System ready after first batch! Loaded {total_loaded} vectors")
        
        # Continue loading remaining vectors in background
        if total_loaded < total_vectors:
            logger.info(f"Loading remaining {total_vectors - total_loaded} vectors in background...")
            remaining_results = vector_store.similarity_search("", k=total_vectors)
            processed_rids = {metadata.get("rid") for doc in first_results for metadata in [doc.metadata] if metadata.get("rid")}
            
            for doc in remaining_results:
                try:
                    metadata = doc.metadata
                    rid = metadata.get("rid")
                    
                    if not rid or "tool_id" not in metadata or rid in processed_rids:
                        continue
                    
                    processed_rids.add(rid)
                    
                    tool = Tool(
                        tool_id=metadata.get("tool_id", ""),
                        name=metadata.get("name", "Unknown"),
                        category_subcat=metadata.get("category_subcat", ""),
                        url=metadata.get("url", "https://example.com"),
                        description=metadata.get("description", ""),
                        image_url=metadata.get("image_url", None),
                        owner=metadata.get("owner", None),
                        status=metadata.get("status", None)
                    )
                    
                    bm25_index.add_tool(tool, rid)
                    total_loaded += 1
                    
                except Exception as e:
                    continue
            
            logger.info(f"Background loading complete: {total_loaded} total vectors")
        
        return total_loaded
        
    except Exception as e:
        logger.error(f"Batch fallback loading failed: {str(e)}")
        return 0

async def load_vectors_for_bm25():
    """Clean batch loading with minimal logging."""
    try:
        total_vectors = app_state.total_vectors
        logger.info(f"Loading {total_vectors} vectors")
        
        if total_vectors == 0:
            app_state.index_ready = True
            app_state.bm25_ready = False
            return

        first_batch_size = min(
            Config.LOADING_FIRST_BATCH_MAX, 
            max(Config.LOADING_FIRST_BATCH_MIN, total_vectors // Config.LOADING_FIRST_BATCH_RATIO)
        )
        
        app_state.vectors_loading = True
        total_loaded = 0
        
        # Strategy 1: Batch loading
        try:
            total_loaded = await load_vectors_batch_fallback(total_vectors, first_batch_size)
            
            if total_loaded > 0:
                logger.info(f"✅ Loaded {total_loaded}/{total_vectors} vectors ({(total_loaded/total_vectors)*100:.1f}%)")
            else:
                raise Exception("Batch loading failed")
                
        except Exception as e:
            logger.warning(f"Batch loading failed, using single query fallback")
            
            # Strategy 2: Single query fallback
            try:
                vector_store = vector_store_manager.get_vector_store()
                
                if total_vectors <= Config.LOADING_SINGLE_QUERY_THRESHOLD:
                    results = vector_store.similarity_search("", k=total_vectors)
                    
                    for doc in results:
                        try:
                            metadata = doc.metadata
                            rid = metadata.get("rid")
                            
                            if not rid or "tool_id" not in metadata:
                                continue
                            
                            tool = Tool(
                                tool_id=metadata.get("tool_id", ""),
                                name=metadata.get("name", "Unknown"),
                                category_subcat=metadata.get("category_subcat", ""),
                                url=metadata.get("url", "https://example.com"),
                                description=metadata.get("description", ""),
                                image_url=metadata.get("image_url", None),
                                owner=metadata.get("owner", None),
                                status=metadata.get("status", None)
                            )
                            
                            bm25_index.add_tool(tool, rid)
                            total_loaded += 1
                            
                        except Exception:
                            continue
                    
                else:
                    # Multi-term strategy for large datasets
                    processed_ids = set()
                    search_terms = ["", "tool", "ai", "software", "app", "data", "design", "business", "platform"]
                    coverage_target = total_vectors * Config.LOADING_COVERAGE_TARGET
                    
                    for term in search_terms:
                        if total_loaded >= coverage_target:
                            break
                            
                        results = vector_store.similarity_search(term, k=Config.LOADING_MULTI_TERM_BATCH_SIZE)
                        
                        for doc in results:
                            metadata = doc.metadata
                            rid = metadata.get("rid")
                            
                            if not rid or rid in processed_ids or "tool_id" not in metadata:
                                continue
                            
                            processed_ids.add(rid)
                            
                            tool = Tool(
                                tool_id=metadata.get("tool_id", ""),
                                name=metadata.get("name", "Unknown"),
                                category_subcat=metadata.get("category_subcat", ""),
                                url=metadata.get("url", "https://example.com"),
                                description=metadata.get("description", ""),
                                image_url=metadata.get("image_url", None),
                                owner=metadata.get("owner", None),
                                status=metadata.get("status", None)
                            )
                            
                            bm25_index.add_tool(tool, rid)
                            total_loaded += 1
                        
                        await asyncio.sleep(Config.LOADING_TERM_DELAY_SECONDS)
                
                app_state.index_ready = True
                logger.info(f"✅ Fallback loaded {total_loaded}/{total_vectors} vectors ({(total_loaded/total_vectors)*100:.1f}%)")
                
            except Exception:
                app_state.index_ready = True
                app_state.bm25_ready = False
        
        app_state.vectors_loaded = total_loaded
        app_state.vectors_loading = False
        app_state.bm25_ready = False  # Always lazy
        
    except Exception as e:
        logger.error(f"Loading failed: {str(e)}")
        app_state.index_ready = True
        app_state.bm25_ready = False

async def initialize_indexes():
    """Initialize indexes in background with progress tracking."""
    try:
        logger.info("Starting background initialization of vector store and BM25 index")
        
        try:
            index = vector_store_manager.pc.Index(env.index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.total_vector_count
            app_state.total_vectors = total_vectors
            logger.info(f"Found {total_vectors} vectors to load")
        except Exception as e:
            logger.error(f"Error getting vector count: {str(e)}")
            app_state.total_vectors = 0
        
        _ = vector_store_manager.get_or_create_index()
        
        if app_state.total_vectors > 0:
            logger.info("Initializing BM25 index from existing vectors")
            await load_vectors_for_bm25()
            app_state.index_ready = True
            logger.info("Background initialization complete - all indexes ready")
        else:
            app_state.index_ready = True
            logger.info("No vectors to load, initialization complete")
            
    except Exception as e:
        logger.error(f"Error during background initialization: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize basic services and start background tasks."""
    try:
        logger.info("Application starting up - initializing basic services")
        
        # Validate environment
        env._validate_required_vars()
        
        # Connect to MongoDB
        await mongodb_manager.connect()
        
        app_state.initialization_started = True
        asyncio.create_task(initialize_indexes())
        
        logger.info("Basic startup complete - API ready for requests")
    except Exception as e:
        logger.error(f"Error during startup initialization: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    await mongodb_manager.disconnect()

# ============================================================================
# UTILITY FUNCTIONS FOR ENDPOINTS
# ============================================================================

async def check_duplicate_tool(vector_store, tool: Tool) -> bool:
    """Check if a tool with the same tool_id already exists."""
    try:
        results = vector_store.similarity_search(
            "",
            k=1,
            filter={"tool_id": tool.tool_id}
        )
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking for duplicate: {str(e)}")
        return False

def create_tool_metadata(tool: Tool, rid: str) -> Dict[str, Any]:
    """Create metadata dictionary for tool storage."""
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
    
    if tool.pricingType:
        metadata["pricingType"] = tool.pricingType
    
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
        metadata["categories_list"] = [cat.Category for cat in tool.categories]
    
    if tool.pricing:
        metadata["pricing_plans"] = ",".join([plan.planName for plan in tool.pricing])
        metadata["pricing_prices"] = ",".join([plan.price for plan in tool.pricing])
    
    if tool.qaSection:
        metadata["qa_questions"] = ",".join([qa.question for qa in tool.qaSection])
        metadata["qa_answers"] = ",".join([qa.answer for qa in tool.qaSection])
    
    return metadata

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint to verify API is running."""
    return {
        "status": "active",
        "message": "AI Tool Search API is running (Production v2.0)",
        "version": "2.0.0",
        "features": ["Hybrid Search", "Vector Embeddings", "BM25 Keyword Search", "LLM Processing"]
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for API and connections."""
    try:
        logger.info("Health check started")
        
        # Validate environment variables
        env._validate_required_vars()
        
        # Test Pinecone connection
        _ = vector_store_manager.get_or_create_index()
        
        logger.info("Health check completed successfully")
        return {
            "status": "healthy",
            "version": "2.0.0",  # Add this line
            "timestamp": datetime.now().isoformat(),
            "dependencies": {      # Add this section
                "pinecone": "connected",
                "groq": "available", 
                "nomic": "available",
                "bm25": "ready" if app_state.bm25_ready else "initializing"
            },
            "components": {
                "pinecone": "connected",
                "groq": "available",
                "nomic": "available",
                "bm25": "ready" if app_state.bm25_ready else "initializing"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/basic-health")
async def basic_health_check():
    """Simple health check without external dependencies."""
    return {
        "status": "api_running",
        "version": "2.0.0",  # Changed from "api_version" to "version"
        "timestamp": datetime.now().isoformat(),
        "note": "Basic health check - API is running"
    }

@app.get("/initialization-status")
async def get_initialization_status():
    """Get current initialization status."""
    return {
        "initialized": app_state.index_ready,
        "initialization_started": app_state.initialization_started,
        "vectors_loaded": app_state.vectors_loaded,
        "total_vectors": app_state.total_vectors,
        "loading_percentage": (app_state.vectors_loaded / max(app_state.total_vectors, 1)) * 100,
        "bm25_index_ready": app_state.bm25_ready,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model-info")
async def get_model_info(request: Request):
    """Get current model information."""
    try:
        headers = request.headers
        current_model = ModelUtils.get_current_model(headers)
        
        # Check which LLM will be used for query tools
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        query_tools_llm = "OpenAI" if openai_available else "Groq"
        
        return {
            "current_groq_model": current_model,
            "query_tools_llm": query_tools_llm,
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if openai_available else "Not configured",
            "provider": "OpenAI (Query Tools) / Groq (Other Endpoints) / Nomic Atlas (Embeddings)",
            "environment": env.environment,
            "api_version": "2.0.0"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.get("/stats")
async def get_stats(show_all: bool = True):
    """Get vector store statistics and vector metadata."""
    try:
        total_vectors = vector_store_manager.get_total_vectors()
        index = vector_store_manager.get_or_create_index()
        stats = index.describe_index_stats()
        
        vector_store = vector_store_manager.get_vector_store()
        
        max_vectors_to_fetch = total_vectors if show_all else min(total_vectors, 50)
        
        results = vector_store.similarity_search("", k=max_vectors_to_fetch)
        
        vectors_info = []
        for doc in results:
            # Return raw metadata exactly as stored in the database
            # This gives complete visibility into what's actually there
            metadata = doc.metadata
            
            # Create a copy of all metadata without any filtering or formatting
            vector_info = dict(metadata)
            
            vectors_info.append(vector_info)
        
        stats_dict = {
            "total_vectors": total_vectors,
            "vectors_shown": len(vectors_info),
            "dimension": Config.DIMENSION,
            "index_fullness": float(stats.index_fullness) if hasattr(stats, 'index_fullness') else 0.0,
            "namespaces": {},
            "vectors": vectors_info,
            "bm25_index_status": {
                "initialized": bm25_index.is_initialized,
                "document_count": len(bm25_index.tool_data)
            }
        }
        
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

@app.get("/test-connection")
async def test_connection(request: Request):
    """Test connection to Nomic Atlas, Groq API, and OpenAI with current settings."""
    try:
        original_headers = request.headers
        model_choice = original_headers.get("MODEL_CHOICE", "DEV_MODEL")
        
        logger.info(f"Test connection with model_choice: {model_choice}")
        
        test_results = {
            "nomic_atlas": "enabled",
            "groq_api": "enabled",
            "openai_api": "enabled" if os.getenv("OPENAI_API_KEY") else "disabled",
            "endpoints_tested": []
        }
        
        # Test Nomic Atlas embeddings
        try:
            nomic_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {env.nomic_api_key}"
            }
            
            embed_response = requests.post(
                "https://api-atlas.nomic.ai/v1/embedding/text",
                headers=nomic_headers,
                json={
                    "texts": ["Test"], 
                    "task_type": "search_document", 
                    "max_tokens_per_text": 8192, 
                    "dimensionality": Config.DIMENSION
                },
                timeout=3
            )
            
            embed_status = {
                "endpoint": "nomic_atlas_embeddings",
                "status_code": embed_response.status_code,
                "success": embed_response.status_code == 200
            }
            
            if embed_response.status_code == 200:
                data = embed_response.json()
                embed_status["embedding_size"] = len(data.get("embeddings", [])[0]) if data.get("embeddings") else 0
            else:
                embed_status["error"] = embed_response.text
                
            test_results["endpoints_tested"].append(embed_status)
            
        except Exception as e:
            test_results["endpoints_tested"].append({
                "endpoint": "nomic_atlas_embeddings",
                "success": False,
                "error": str(e)
            })
        
        # Test Groq chat endpoint
        try:
            model = ModelUtils.get_current_model(original_headers)
            logger.info(f"DEBUG: /test-connection using Groq model: {model}")
            
            llm = ChatGroq(
                groq_api_key=env.groq_api_key,
                model_name=model,
                max_tokens=20
            )
            
            response = llm.invoke("Hi, respond with 'Groq test successful'")
            
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
        
        # Test OpenAI API if available
        if os.getenv("OPENAI_API_KEY"):
            try:
                logger.info("Testing OpenAI API connection")
                
                openai_result = ModelUtils.call_openai_for_query_tools(
                    system_prompt="You are a test assistant. Respond only with valid JSON.",
                    user_prompt='Respond with: {"status": "test_successful", "message": "OpenAI connection working"}'
                )
                
                openai_status = {
                    "endpoint": "openai_chat_completions",
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "success": openai_result["success"]
                }
                
                if openai_result["success"]:
                    openai_status["response"] = "OpenAI endpoint working"
                    # Try to parse the JSON response for additional validation
                    try:
                        parsed_response = json.loads(openai_result["content"])
                        openai_status["test_response"] = parsed_response
                    except json.JSONDecodeError:
                        openai_status["note"] = "Response received but not valid JSON"
                else:
                    openai_status["error"] = openai_result["error"]
                
                test_results["endpoints_tested"].append(openai_status)
                
            except Exception as e:
                test_results["endpoints_tested"].append({
                    "endpoint": "openai_chat_completions",
                    "success": False,
                    "error": str(e)
                })
        else:
            # OpenAI not configured
            test_results["endpoints_tested"].append({
                "endpoint": "openai_chat_completions",
                "success": False,
                "error": "OpenAI API key not configured",
                "note": "Query tools will use Groq fallback"
            })
        
        # Determine overall status
        successful_endpoints = [ep for ep in test_results["endpoints_tested"] if ep["success"]]
        failed_endpoints = [ep for ep in test_results["endpoints_tested"] if not ep["success"]]
        
        # Generate summary
        if len(failed_endpoints) == 0:
            test_results["status"] = "success"
            test_results["message"] = "Successfully connected to all available endpoints"
        elif len(successful_endpoints) > 0:
            test_results["status"] = "partial"
            test_results["message"] = f"{len(successful_endpoints)} endpoints working, {len(failed_endpoints)} failed"
            test_results["failed_endpoints"] = [ep["endpoint"] for ep in failed_endpoints]
        else:
            test_results["status"] = "error"
            test_results["message"] = "All endpoints failed connection test"
        
        # Add configuration summary
        test_results["configuration"] = {
            "groq_model": ModelUtils.get_current_model(original_headers),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
            "query_tools_llm": "OpenAI" if os.getenv("OPENAI_API_KEY") else "Groq",
            "environment": env.environment
        }
        
        return test_results
            
    except Exception as e:
        logger.error(f"Exception in test_connection: {str(e)}")
        return {
            "status": "error",
            "message": f"Exception during connection test: {str(e)}",
            "endpoints_tested": [],
            "details": {
                "error_type": type(e).__name__
            }
        }

# def extract_tool_bullets(result):
#     """Extract meaningful bullets from tool metadata instead of static text."""
#     bullets = []
    
#     # Try to get bullets from features.pros
#     if result.get("features_pros"):
#         pros = result.get("features_pros", "").split(",")
#         bullets.extend([pro.strip() for pro in pros[:3] if pro.strip()])
    
#     # Try to get bullets from features.cons (mark as limitations)
#     if result.get("features_cons") and len(bullets) < 2:
#         cons = result.get("features_cons", "").split(",")
#         for con in cons[:1]:
#             if con.strip():
#                 bullets.append(f"Limitation: {con.strip()}")
    
#     # Try to get bullets from QA section
#     if result.get("qa_questions") and len(bullets) < 2:
#         questions = result.get("qa_questions", "").split(",")
#         for q in questions[:2]:
#             if q.strip():
#                 bullets.append(f"Supports: {q.strip()}")
    
#     # Try to get bullets from details
#     if result.get("details_speciality") and len(bullets) < 2:
#         speciality = result.get("details_speciality", "")
#         if speciality:
#             bullets.append(f"Specializes in: {speciality[:50]}...")
    
#     # Try to get bullets from pricing
#     if result.get("pricing_plans") and len(bullets) < 2:
#         plans = result.get("pricing_plans", "").split(",")
#         if plans and plans[0].strip():
#             bullets.append(f"Pricing: {plans[0].strip()} available")
    
#     # Try to get bullets from categories
#     if result.get("category_subcat") and len(bullets) < 2:
#         categories = result.get("category_subcat", "").split(",")
#         if categories and categories[0].strip():
#             bullets.append(f"Category: {categories[0].strip()}")
    
#     # Fallback bullets if no metadata found
#     if not bullets:
#         bullets = [
#             "Found via search relevance",
#             "Check description for details"
#         ]
    
#     # Ensure we have exactly 2 bullets
#     while len(bullets) < 2:
#         bullets.append("Additional features available")
    
#     return bullets[:2]

def extract_tool_bullets(result):
    """Return empty bullets array - no fabricated bullet points."""
    return []

def validate_and_fix_tool_ids(response_data: dict, llm_tools: List[Dict]) -> dict:
    """Validate and fix tool IDs in LLM response - ensure actual tool_id values are used."""
    
    # Create mapping from tool names to tool IDs
    name_to_id = {}
    id_to_tool = {}
    
    for tool in llm_tools:
        tool_name = tool.get("name", "").strip()
        tool_id = tool.get("tool_id", "").strip()
        
        if tool_name and tool_id:
            name_to_id[tool_name] = tool_id
            id_to_tool[tool_id] = tool
    
    # Fix tool_id array if it contains names instead of IDs
    if "tool_id" in response_data and isinstance(response_data["tool_id"], list):
        fixed_tool_ids = []
        
        for item in response_data["tool_id"]:
            item_str = str(item).strip()
            
            # Check if it's already a valid tool_id (not a name)
            if item_str in id_to_tool:
                fixed_tool_ids.append(item_str)
            # If it's a tool name, convert to tool_id
            elif item_str in name_to_id:
                fixed_tool_ids.append(name_to_id[item_str])
                logger.info(f"Fixed tool_id: '{item_str}' → '{name_to_id[item_str]}'")
            else:
                logger.warning(f"Could not resolve tool identifier: '{item_str}'")
        
        response_data["tool_id"] = fixed_tool_ids
    
    # Fix tools array - ensure id field uses tool_id not name
    if "tools" in response_data and isinstance(response_data["tools"], list):
        for tool_entry in response_data["tools"]:
            if isinstance(tool_entry, dict) and "id" in tool_entry:
                current_id = str(tool_entry["id"]).strip()
                
                # If id field contains a name instead of ID, fix it
                if current_id in name_to_id:
                    tool_entry["id"] = name_to_id[current_id]
                    logger.info(f"Fixed tool.id: '{current_id}' → '{name_to_id[current_id]}'")
    
    return response_data

@app.post("/query", response_model=QueryResponse)
async def query_tools(request: QueryRequest, request_headers: Request):
    """Query tools using hybrid search with LLM processing."""
    start_time = time.time()
    
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=422,
            detail="Query cannot be empty"
        )
    
    if len(request.query) > 500:
        raise HTTPException(
            status_code=422,
            detail="Query exceeds maximum length of 500 characters"
        )
    
    # Check if limit parameter exists and validate it
    if hasattr(request, 'limit') and request.limit is not None:
        if request.limit <= 0:
            raise HTTPException(
                status_code=422,
                detail="Limit parameter must be greater than 0"
            )
    
    try:
        logger.info(f"Processing query: {request.query}")
        headers = request_headers.headers

        if request.searchFrom:
            logger.info(f"Searching within {len(request.searchFrom)} specified tools")

        # Check cache for non-filtered searches
        cached_response = None
        if not request.searchFrom:
            cached_response = tool_search_cache.get(request.query)
            if cached_response:
                logger.info("Returning cached response")
                return QueryResponse(response=cached_response)
        
        # Check initialization status
        if not app_state.initialization_started:
            error_response = json.dumps({
                "error": "system_not_initialized",
                "message": "The system has not started initialization yet. Please try again later.",
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            return QueryResponse(response=error_response)
        
        if not app_state.index_ready:
            progress = (app_state.vectors_loaded / max(app_state.total_vectors, 1)) * 100
            logger.info(f"Query received during initialization. Progress: {progress:.1f}%")
            initializing_response = json.dumps({
                "status": "initializing",
                "message": f"System is initializing. Progress: {progress:.1f}%",
                "progress_percentage": progress,
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            return QueryResponse(response=initializing_response)
        
        if not app_state.bm25_ready and not app_state.bm25_building:
            logger.info("First search detected - building BM25 index lazily")
            build_bm25_lazy()
        
        # Get vector store
        vector_store = vector_store_manager.get_vector_store(headers, for_query=True)
        total_vectors = vector_store_manager.get_total_vectors()
        logger.info(f"Total vectors in store: {total_vectors}")
        
        # Hybrid search implementation
        try:
            # Setup search filter
            search_filter = None
            if request.searchFrom and len(request.searchFrom) > 0:
                search_filter = {"tool_id": {"$in": request.searchFrom}}
                logger.info(f"Applied search filter: {search_filter}")
            
            # Vector search
            try:
                vector_results_with_scores = vector_store.similarity_search_with_score(
                    request.query,
                    k=Config.VECTOR_SEARCH_K,
                    filter=search_filter
                )
            except Exception as e:
                logger.error(f"Error in vector search: {str(e)}")
                vector_results_with_scores = []
            
            # Process vector results - UPDATED TO PASS ALL METADATA
            processed_vector_results = []
            for doc, score in vector_results_with_scores:
                metadata = doc.metadata
                # Pass ALL metadata without filtering
                result = dict(metadata)  # Copy all metadata fields
                result["score"] = float(1.0 - score)  # Add relevance score
                processed_vector_results.append(result)

            logger.info(f"Vector search returned {len(processed_vector_results)} results")
            
            # BM25 search
            try:
                bm25_results = bm25_index.search(request.query, top_k=Config.BM25_SEARCH_K)
                logger.info(f"DEBUG: BM25 index status - initialized: {bm25_index.is_initialized}, doc_count: {len(bm25_index.tool_data)}")
                logger.info(f"DEBUG: BM25 tokenized query: {bm25_index.preprocess_text(request.query)}")
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
            
            # Fuse results
            hybrid_results = SearchUtils.fuse_search_results(
                processed_vector_results, 
                bm25_results,
                alpha=Config.HYBRID_ALPHA
            )
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            
            # LOG ALL HYBRID RESULTS WITH SCORES
            # logger.info("=== HYBRID SEARCH RESULTS (ALL TOOLS) ===")
            # for i, result in enumerate(hybrid_results[:20]):  # Log top 20 to see the full picture
            #     tool_name = result.get("name", "Unknown")
            #     tool_id = result.get("tool_id", "Unknown")
            #     score = result.get("score", 0)
            #     vector_score = result.get("vector_score", 0)
            #     bm25_score = result.get("bm25_score", 0)
            #     description = result.get("description", "")[:100]  # First 100 chars
                
            #     logger.info(f"Rank {i+1:2d}: '{tool_name}' (ID: {tool_id})")
            #     logger.info(f"         Score: {score:.3f} (V:{vector_score:.3f} + B:{bm25_score:.3f})")
            #     logger.info(f"         Desc: {description}...")
            #     logger.info("-" * 50)
            
            # Handle no results with filter
            if not hybrid_results and request.searchFrom:
                filter_response = json.dumps({
                    "tool_id": [],
                    "tools": [],
                    "message": "No tools found matching your search within the specified tools.",
                    "search_filter_applied": True,
                    "searched_within_tool_ids": request.searchFrom,
                    "timestamp": datetime.now().isoformat()
                })
                return QueryResponse(response=filter_response)
            
            # Use score-based selection and split into LLM vs database processing
            all_selected_results = SearchUtils.select_tools_by_score(
    hybrid_results, 
    min_score=Config.HYBRID_MIN_SCORE,
    max_tools=Config.HYBRID_MAX_TOOLS,
    fallback_count=Config.HYBRID_FALLBACK_COUNT,
    high_individual_threshold=0.9
)

            # Split into LLM and database processing groups
            llm_tools = all_selected_results[:40]  # Top 20 for LLM
            # db_tools = all_selected_results[30:]   # Remaining for database processing

            # logger.info(f"Split results: {len(llm_tools)} tools for LLM, {len(db_tools)} tools for database processing")
            logger.info(f"Using {len(llm_tools)} tools for LLM processing only")
            logger.info(f"Total tools to process: {len(all_selected_results)}")

            # LOG TOOLS SENT TO LLM (basic info for logs)
            logger.info("=== TOOLS SENT TO LLM ===")
            for i, result in enumerate(llm_tools):
                tool_name = result.get("name", "Unknown")
                tool_id = result.get("tool_id", "Unknown")
                score = result.get("score", 0)
                logger.info(f"LLM Tool {i+1}: '{tool_name}' (ID: {tool_id}) - Score: {score:.3f}")
                logger.info("-" * 40)

            logger.info(f"Sending {len(llm_tools)} tools to LLM for processing (score-based selection)")
            
            # Format documents for LLM - UPDATED TO PASS ALL METADATA AS JSON
            # formatted_docs = []
            # for result in llm_tools:  # Only process top 20
            #     # Pass ALL metadata as JSON to LLM - let LLM decide what's relevant
            #     formatted_doc = json.dumps(result, indent=2, ensure_ascii=False)
            #     formatted_docs.append(formatted_doc)
            formatted_docs = []
            for result in llm_tools:
                essential_data = {"tool_id": result.get("tool_id", ""),
                                  "name": result.get("name", ""),
                                  "description": result.get("description", ""),
                                  "category_subcat": result.get("category_subcat", ""),
                                  "pricingType": result.get("pricingType", ""),
                                  "details_speciality": result.get("details_speciality", "") if result.get("details_speciality") else ""
                                  }
                formatted_doc = json.dumps(essential_data, ensure_ascii=False)
                formatted_docs.append(formatted_doc)
            
            logger.info(f"📊 DEBUG: Formatted {len(formatted_docs)} tools for LLM")
            logger.info(f"📊 DEBUG: Sample tool data: {formatted_docs[0] if formatted_docs else 'None'}")
            
            # # LOG ACTUAL DATA SENT TO LLM (first 2 tools for verification)
            # logger.info("=== ACTUAL DATA SENT TO LLM ===")
            # for i, formatted_doc in enumerate(formatted_docs[:2]):  # Log first 2 tools
            #     logger.info(f"Tool {i+1} Complete Data to LLM:")
            #     logger.info(formatted_doc[:500] + "..." if len(formatted_doc) > 500 else formatted_doc)
            #     logger.info("-" * 80)
            
            # Handle no tools case
            if not llm_tools:
                empty_response = json.dumps({
                    "tool_id": [],
                    "tools": [],
                    "message": "No tools found matching your search criteria.",
                    "search_filter_applied": request.searchFrom is not None,
                    "searched_within_tool_ids": request.searchFrom,
                    "timestamp": datetime.now().isoformat()
                })
                return QueryResponse(response=empty_response)
            
            # Prepare LLM processing
            headers = request_headers.headers
            logger.info(f"DEBUG: /query called with headers: {dict(headers)}")
            current_model = ModelUtils.get_current_model(headers)
            logger.info(f"DEBUG: /query using model: {current_model}")
            
            try:
                encoder = tiktoken.encoding_for_model(current_model)
            except Exception:
                encoder = tiktoken.get_encoding("cl100k_base")
            
            context = "\n\n---\n\n".join(formatted_docs)
            cleaned_context = TextCleaner.clean_text(context)

            # Calculate token budget for output management
            tool_count = len(llm_tools)
            tokens_per_tool = min(150, 3000 // max(tool_count, 1)) 
            
            system_text = f"""You are an expert AI tool recommendation assistant who is specialized in keyword overlap, semantic similarity, functional matching, and domain filtering. Your job is to *select and rank all the relevant tools* from a given Available Tools Data if they **match or related** to Query which is: "{request.query}".

##Selection Criteria OR Rules:
- Include all the Query related tools without restricting or limiting to fewer tools following below INCLUSIVE APPROACH.
- **INCLUSIVE APPROACH**: Analyze each tool data and INCLUDE all tools that:
  - Have core functionality addressing the query or related needs
  - Have same or related keywords in the description, metadata or name of the tool.
  - Have semantic similarity to the query.
  - Contain related concepts, or purposes or could be adapted for the use case of the query.
  - Would be useful or helpful for someone with this specific need
  - Are from the same general domain or category
  - Are in related categories that users might consider
- **Ranking Requirements**: Rank them from most to least relevant (1st = most relevant)
- **Query-Specific Conditions**:
  - If query specifies a number ("top 3", "best 5"): return exactly that count
  - If query mentions specific tool name: prioritize and return that tool in 1st position  
- ** EXCLUSION**: Only exclude tools that:
  - Are from completely different domains with zero functional overlap
  - Cannot solve any aspect of the user's problem
  - Would never be considered by someone with this need
  - Have no shared keywords, concepts, or use cases with the query
- **INCLUSION PREFERENCE**: 
  - *Do not duplicate tools - each tool should appear exactly once in the response*
  - *Include all tools that have any relevance to the query*
- **Edge Cases**: If no tools match the query, return: {{ "tool_id": [], "tools": [] }}
- **Count Limits**: *Return all the tools up to 40 tools based on relevance* (prefer more options over fewer)
- **Technical Requirements**: Use the exact "tool_id" field from metadata of each tool
- **Output Specifications**:
  - *Description: Should contain specific relevance by using keywords from Query: "{request.query}"*
  - *Bullets: List features applicable to Query in 2 bullet points*

##JSON FORMAT:
{{
  "tool_id": ["most_relevant_exact_tool_id", "second_most_relevant_exact_tool_id"],
  "tools": [
    {{
      "id": "exact_tool_id_from_data",
      "name": "tool_name",
      "description": "Explain how this tool is relevant to the user query using keywords and intent from the query.",
      "bullets":[
        "Feature that directly solves the task described in the query",
        "Specific capability aligned with the user's intent"
      ]
    }}
  ]
}}

- The "tool_id" array should mirror the ranking order of tools returned in the "tools" list.
- The "id" field inside each object must match its corresponding entry in "tool_id".

##*Important Rules*:
- Return *ONLY valid JSON* in the below given JSON FORMAT *without any PREAMBLE or EXPLANATION.*
- Ensure all brackets and quotes are properly closed."""
            
            prompt_text = f"""Query: "{request.query}"

Available Tools Data:
{cleaned_context}

## Task
Your task is to select and rank all tools that are relevant to the above Query, using the provided Available Tools Data.

## Requirements:
- Apply the Selection Criteria OR Rules as defined above
- Analyze query intent first, then evaluate each tool against that intent
- NO DUPLICATES but include all relevant tools
- Focus on semantic relevance.
- Return only the final JSON result — no extra text or explanations"""

            tier1_system = f"""You are an expert AI tool recommendation assistant specialized in keyword overlap, semantic similarity, functional matching, and domain filtering. Your job is to select and rank all the relevant tools from a given Available Tools Data if they match or related to Query which is: "{request.query}".

#Selection Criteria:
- Include all the Query related tools without restricting or limiting to fewer tools following below INCLUSIVE APPROACH.
- **INCLUSIVE APPROACH**: Include ALL tools that meet ANY ONE of these criteria:
  - Have core functionality addressing the query or related needs 
  - Have same keywords in the description, metadata or name of the tool of the tool data.
  - Have semantic similarity to the query.
IMPORTANT: A tool only needs to satisfy ONE criteria to be included.
- **Ranking Requirements**: Rank them from most to least relevant (1st = most relevant)
- **Query-Specific Conditions**:
  - If query specifies a number ("top 3", "best 5"): return exactly that count
  - If query mentions specific tool name: prioritize and return that tool in 1st position  
- **EXCLUSION**: Exclude the tools that:
  - Are from completely different domains without functional overlap
  - Cannot solve usecase of the user
  - Have no shared keywords, concepts, or use cases with the query
- **INCLUSION PREFERENCE**: 
  - Do not duplicate tools - each tool should appear exactly once in the response
  - Do NOT be selective - Include all tools that have any relevance to the query unless it's completely unrelated.
- **Edge Cases**: If no tools match the query, return: {{ "tool_id": [], "tools": [] }}
- **Count Limits**: *Return all the tools up to 40 tools based on relevance* (prefer more options over fewer)
  - **Technical Requirements**: Use the exact "tool_id" field from metadata of each tool
- **Output Specifications**:
  - Description: Should contain specific relevance using keywords from Query: "{request.query}" WITHOUT bullet points
  - Bullets: Separate array with 2 bullet points listing key features for this query"""
            
            tier1_user = f"""Query: "{request.query}"
Available Tools Data:
{cleaned_context}
Your task is to select and rank up to 40 tools that are relevant to the above query based on the provided Selection Criteria, using the Available Tools Data."""


# Use OpenAI for query tools, fallback to Groq if needed
            openai_api_key = os.getenv("OPENAI_API_KEY")

            if openai_api_key:
                logger.info("🚀 Using OpenAI for query tools processing")
                
                # Call OpenAI
                openai_result = ModelUtils.call_openai_for_query_tools(
                    system_prompt=system_text,
                    user_prompt=prompt_text,
                    tier1_system=tier1_system,
                    tier1_user=tier1_user
                )
                
                if openai_result["success"]:
                    llm_response = openai_result["content"]
                    
                    if openai_result.get("cache_hit"):
                        cache_tokens = openai_result.get("cached_tokens", 0)
                        logger.info(f"✅ OpenAI response received (CACHE HIT - {cache_tokens} tokens cached)")
                    else:
                        logger.info("✅ OpenAI response received (cache miss - building cache)")

                    if openai_result.get("structured"):
                        logger.info("🎯 Using Structured Outputs format")
                    else:
                        logger.info("🎯 Using JSON mode format")
                else:
                    logger.error(f"❌ OpenAI failed: {openai_result['error']}")
                    # Fallback to Groq
                    logger.info("🔄 Falling back to Groq...")
                    
                    llm = ChatGroq(
                        groq_api_key=env.groq_api_key,
                        model_name=current_model,
                        temperature=0.05,
                        model_kwargs={
                            "top_p": 0.5,
                            "frequency_penalty": 0.2,
                            "presence_penalty": 0.0,
                            "response_format": {"type": "json_object"}
                        }
                    )
                    
                    # Add token counting for fallback consistency
                    total_input_text = system_text + prompt_text
                    input_tokens = len(encoder.encode(total_input_text))
                    logger.info(f"🔢 GROQ FALLBACK INPUT TOKENS: {input_tokens}")
                    
                    response = llm.invoke([
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": prompt_text}
                    ])
                    
                    llm_response = response.content
                    
                    # Count output tokens for fallback
                    output_tokens = len(encoder.encode(llm_response))
                    total_tokens = input_tokens + output_tokens
                    
                    logger.info(f"🔢 GROQ FALLBACK OUTPUT TOKENS: {output_tokens}")
                    logger.info(f"🔢 GROQ FALLBACK TOTAL TOKENS: {total_tokens}")
                    logger.info("🔢 GROQ FALLBACK TOKEN BREAKDOWN - Input: %d, Output: %d, Total: %d", input_tokens, output_tokens, total_tokens)
                    logger.info("✅ Groq fallback response received")

            else:
                # No OpenAI key, use Groq directly
                logger.info("🤖 Using Groq for query tools processing (no OpenAI key)")
                
                llm = ChatGroq(
                    groq_api_key=env.groq_api_key,
                    model_name=current_model,
                    temperature=0.05,
                    model_kwargs={
                        "top_p": 0.5,
                        "frequency_penalty": 0.2,
                        "presence_penalty": 0.0,
                        "response_format": {"type": "json_object"}
                    }
                )
                
                # Count input tokens for Groq
                total_input_text = system_text + prompt_text
                input_tokens = len(encoder.encode(total_input_text))
                logger.info(f"🔢 GROQ INPUT TOKENS: {input_tokens}")
                
                response = llm.invoke([
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": prompt_text}
                ])
                
                llm_response = response.content
                
                # Count output tokens for Groq
                output_tokens = len(encoder.encode(llm_response))
                total_tokens = input_tokens + output_tokens
                
                logger.info(f"🔢 GROQ OUTPUT TOKENS: {output_tokens}")
                logger.info(f"🔢 GROQ TOTAL TOKENS: {total_tokens}")
                logger.info("🔢 GROQ TOKEN BREAKDOWN - Input: %d, Output: %d, Total: %d", input_tokens, output_tokens, total_tokens)

            logger.info("✅ LLM response received and processed")
            
            try:
                # Post-process response
                processed_response = QueryProcessor.post_process_llm_response(llm_response)
                logger.info("DEBUG: Raw LLM response before JSON parsing: %s...", str(processed_response[:500]))
                
                # Parse and validate JSON response
                try:
                    response_data = json.loads(processed_response)
                    response_data = validate_and_fix_tool_ids(response_data, llm_tools)
                    
                    # Add search filter info if applied
                    if request.searchFrom:
                        response_data["search_filter_applied"] = True
                        response_data["searched_within_tool_ids"] = request.searchFrom

                    # # Ensure proper tool data structure
                    # if "tools" not in response_data or not response_data["tools"]:
                    #     response_data["tools"] = []
                    #     response_data["tool_id"] = []
                        
                    #     for result in llm_tools:
                    #         bullets = extract_tool_bullets(result)
                    #         tool_data = {
                    #             "id": result.get("tool_id", ""),
                    #             "name": result.get("name", ""),
                    #             "description": result.get("description", ""),
                    #             "bullets": []
                    #         }
                    #         response_data["tools"].append(tool_data)
                    #         response_data["tool_id"].append(result.get("tool_id", ""))
                            
                    #     response_data["message"] = "Structured response incomplete, showing raw search results."
                    
                    # clean_response = json.dumps(response_data, indent=2)
                    # logger.info("Successfully processed valid JSON response")

                    # Ensure proper tool data structure - NO FALLBACKS
                    if "tools" not in response_data or not response_data["tools"]:
                        response_data["tools"] = []
                        response_data["tool_id"] = []
                        response_data["message"] = "No relevant tools found for this query."
                    
                    clean_response = json.dumps(response_data, indent=2)
                    logger.info("Successfully processed valid JSON response")
                    
                    # Process remaining tools from database
                    # if db_tools:
                    #     logger.info(f"Processing {len(db_tools)} additional tools from database")
                        
                    #     for result in db_tools:
                    #         # Extract meaningful bullets from metadata
                    #         bullets = extract_tool_bullets(result)
                            
                    #         tool_data = {
                    #             "id": result.get("tool_id", ""),
                    #             "name": result.get("name", ""),
                    #             "description": result.get("description", ""),
                    #             "bullets": []
                    #         }
                            
                    #         response_data["tools"].append(tool_data)
                    #         response_data["tool_id"].append(result.get("tool_id", ""))
                            
                    #         logger.debug(f"Tool {result.get('name', 'Unknown')}: extracted {len(bullets)} bullets from metadata")
                        
                    #     # Update the JSON response with all tools
                    #     clean_response = json.dumps(response_data, indent=2)
                    #     logger.info(f"Added {len(db_tools)} database-processed tools to LLM results")
                    #     logger.info(f"Final response contains {len(response_data['tools'])} total tools")
                    logger.info(f"Final response contains {len(response_data.get('tools', []))} LLM-only tools")
                    
                except json.JSONDecodeError:
                    fallback_data = {
                        "tool_id": [result.get("tool_id", "") for result in llm_tools],  # Only LLM tools"tools": [],
                        "message": "LLM JSON parsing failed - returning basic LLM tools without processing",
                        "timestamp": datetime.now().isoformat()
                        }
                    logger.warning("LLM JSON parsing failed - using basic LLM tools only")

                    for result in llm_tools:  # Only process LLM tools
                        tool_data = {
                            "id": result.get("tool_id", ""),
                            "name": result.get("name", ""),
                            "description": result.get("description", ""),
                            "bullets": []
                            }
                        fallback_data["tools"].append(tool_data)
                    logger.info(f"Fallback: Created response with {len(fallback_data['tools'])} LLM-only tools")
                    
                    if request.searchFrom:
                        fallback_data["search_filter_applied"] = True
                        fallback_data["searched_within_tool_ids"] = request.searchFrom
                    
                    clean_response = json.dumps(fallback_data, indent=2)
                    logger.warning("Created fallback JSON response")
                
                # Cache result for non-filtered searches
                if not request.searchFrom:
                    tool_search_cache.set(request.query, clean_response)
                
                # Log processing time
                elapsed_time = time.time() - start_time
                logger.info(f"Total processing time: {elapsed_time:.2f}s")
                
                return QueryResponse(response=clean_response)
                
            except Exception as e:
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
            
            fallback_response = json.dumps({
                "error": "search_error",
                "message": f"Failed to process search: {error_msg}",
                "tools": [],
                "timestamp": datetime.now().isoformat()
            })
            
            return QueryResponse(response=fallback_response)
            
    except Exception as e:
        logger.error(f"Error in query_tools: {str(e)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Query processing completed with error in {elapsed_time:.2f}s")
        
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
        vector_store = vector_store_manager.get_vector_store(for_query=False)
        results = []
        skipped_tools = []
        added_tools = []

        # Clean and prepare tools
        cleaned_tools = []
        for tool in bulk_request.tools:
            # Map tool_id from _id if needed
            if hasattr(tool, '_id') and tool._id and not tool.tool_id:
                tool.tool_id = tool._id
                
            # Ensure backward compatibility
            if not tool.category_subcat:
                tool.category_subcat = tool.generate_category_subcat()
                
            if not tool.description:
                tool.description = tool.generate_description()
                
            cleaned_tool = ToolManager.clean_tool_data(tool)
            cleaned_tools.append(cleaned_tool)

        # Check for duplicates
        for tool in cleaned_tools:
            is_duplicate = await check_duplicate_tool(vector_store, tool)
            if is_duplicate:
                skipped_tools.append(tool.tool_id)
                logger.info(f"Skipping duplicate tool: {tool.name} (Tool ID: {tool.tool_id})")
                results.append(ToolResponse(
                    id=f"duplicate-{tool.tool_id}",
                    tool=tool,
                    status="skipped_duplicate"
                ))
            else:
                added_tools.append(tool)
        
        # Add non-duplicate tools
        for tool in added_tools:
            rid = str(uuid4())
            
            # Format tool data
            tool_text = ToolManager.format_tool_for_indexing(tool, rid)
            metadata = create_tool_metadata(tool, rid)
            
            # Add to vector store
            vector_store.add_texts(
                texts=[tool_text],
                metadatas=[metadata],
                ids=[rid]
            )
            
            # Add to BM25 index
            bm25_index.add_tool(tool, rid)
            
            logger.info(f"Added tool: {tool.name} (Tool ID: {tool.tool_id})")
            results.append(ToolResponse(id=rid, tool=tool, status="added"))
        
        # Rebuild BM25 index if tools were added
        if added_tools:
            bm25_index.rebuild_index()
        
        new_count = vector_store_manager.get_total_vectors()
        logger.info(f"Vector count after addition: {new_count}")
        
        if skipped_tools:
            logger.info(f"Skipped {len(skipped_tools)} duplicate tools")
        
        return BulkToolResponse(results=results)
    
    except Exception as e:
        logger.error(f"Error in add_tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add tools: {str(e)}"
        )

@app.delete("/delete-tool/{tool_id}", response_model=DeleteResponse)
async def delete_tool(tool_id: str):
    """Delete a tool by its tool_id."""
    try:
        initial_count = vector_store_manager.get_total_vectors()
        logger.info(f"Current vector count before deletion: {initial_count}")
        
        vector_store = vector_store_manager.get_vector_store()
        
        # Find tool by tool_id
        search_results = vector_store.similarity_search(
            "",
            k=1,
            filter={"tool_id": tool_id}
        )
        
        if not search_results:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        tool_name = search_results[0].metadata.get("name", "Unknown tool")
        rid = search_results[0].metadata.get("rid")
        
        if not rid:
            raise HTTPException(status_code=500, detail="RID not found for the tool")
        
        # Delete from Pinecone
        index = vector_store_manager.get_or_create_index()
        index.delete(ids=[rid])
        
        # Remove from BM25 index
        if rid in bm25_index.tool_data:
            del bm25_index.tool_data[rid]
            bm25_index.is_initialized = False
            logger.info(f"Removed tool from BM25 index: {tool_name} (RID: {rid})")
        
        new_count = vector_store_manager.get_total_vectors()
        logger.info(f"Vector count after deletion: {new_count}")
        logger.info(f"Deleted tool: {tool_name} (Tool ID: {tool_id})")
        
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
        vector_store = vector_store_manager.get_vector_store(for_query=False)
        results = []
        updated_rids = []

        cleaned_tools = [ToolManager.clean_tool_data(tool) for tool in request.tools]
        
        for tool in cleaned_tools:
            try:
                # Find existing tool
                search_results = vector_store.similarity_search(
                    "",
                    k=1,
                    filter={"tool_id": tool.tool_id}
                )
                
                if not search_results:
                    logger.warning(f"Tool not found: {tool.tool_id}")
                    continue
                
                existing_rid = search_results[0].metadata.get("rid")
                if not existing_rid:
                    logger.warning(f"RID not found for tool: {tool.tool_id}")
                    continue
                
                # Update tool data
                tool_text = ToolManager.format_tool_for_indexing(tool, existing_rid)
                metadata = create_tool_metadata(tool, existing_rid)
                
                # Update vector store
                vector_store.add_texts(
                    texts=[tool_text],
                    metadatas=[metadata],
                    ids=[existing_rid]
                )
                
                # Update BM25 index
                bm25_index.add_tool(tool, existing_rid)
                updated_rids.append(existing_rid)
                
                logger.info(f"Updated tool: {tool.name} (Tool ID: {tool.tool_id})")
                results.append(ToolResponse(id=existing_rid, tool=tool, status="updated"))
                
            except Exception as e:
                logger.error(f"Error updating tool {tool.tool_id}: {str(e)}")
                continue
        
        # Rebuild BM25 index if tools were updated
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

@app.post("/reindex-tools")
async def reindex_tools():
    """Re-index all existing tools with task-specific embeddings."""
    try:
        document_vector_store = vector_store_manager.get_vector_store(for_query=False)
        
        total_vectors = vector_store_manager.get_total_vectors()
        logger.info(f"Starting re-indexing of {total_vectors} tools")
        
        if total_vectors == 0:
            return {
                "success": True,
                "message": "No tools to re-index",
                "count": 0
            }
        
        index = vector_store_manager.get_or_create_index()
        
        updated_count = 0
        failed_count = 0
        
        batch_size = Config.DEFAULT_BATCH_SIZE
        total_batches = (total_vectors - 1) // batch_size + 1
        
        pinecone_index = vector_store_manager.pc.Index(env.index_name)
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, total_vectors)
            current_batch_size = end_idx - start_idx
            
            logger.info(f"Processing batch {batch_num+1}/{total_batches} (tools {start_idx+1}-{end_idx})")
            
            results = document_vector_store.similarity_search(
                "",
                k=current_batch_size,
                filter={}
            )
            
            for doc in results:
                try:
                    metadata = doc.metadata
                    rid = metadata.get("rid")
                    
                    if not rid:
                        logger.warning(f"Skipping tool with missing RID")
                        failed_count += 1
                        continue
                    
                    if "tool_id" in metadata:
                        tool = Tool(
                            tool_id=metadata.get("tool_id", ""),
                            name=metadata.get("name", "Unknown"),
                            category_subcat=metadata.get("category_subcat", ""),
                            url=metadata.get("url", "https://example.com"),
                            description=metadata.get("description", ""),
                            image_url=metadata.get("image_url", None),
                            owner=metadata.get("owner", None),
                            status=metadata.get("status", None)
                        )
                        
                        tool_text = ToolManager.format_tool_for_indexing(tool, rid)
                        
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
            
            logger.info(f"Re-indexed batch {batch_num+1}/{total_batches} - {updated_count} tools updated so far")
        
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

@app.post("/clear-index", response_model=Dict[str, Any])
async def clear_index(request: ClearIndexRequest):
    """Delete all vectors in the Pinecone index. Requires API key authorization."""
    try:
        if request.api_key != env.pinecone_api_key:
            logger.warning(f"Unauthorized clear-index attempt with incorrect API key")
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid API key"
            )
        
        initial_count = vector_store_manager.get_total_vectors()
        logger.info(f"Current vector count before clearing: {initial_count}")
        
        if initial_count == 0:
            return {
                "success": True,
                "message": "Index already empty",
                "deleted_count": 0
            }
        
        index = vector_store_manager.get_or_create_index()
        
        # Delete all vectors
        index.delete(delete_all=True)
        
        # Clear BM25 index
        bm25_index.tool_data = {}
        bm25_index.is_initialized = False
        logger.info("Cleared BM25 index")
        
        new_count = vector_store_manager.get_total_vectors()
        
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
        raise he
    except Exception as e:
        logger.error(f"Error clearing index: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear index: {str(e)}"
        )

@app.post("/query-suggestions", response_model=QuerySuggestionsResponse)
async def get_query_suggestions(request: QuerySuggestionsRequest, request_headers: Request):
    """Generate 5 similar/related queries based on user input."""
    try:
        logger.info(f"Generating query suggestions for: {request.query}")
        headers = request_headers.headers
        
        current_model = ModelUtils.get_current_model(headers)
        logger.info(f"Using model: {current_model} for query suggestions")
        
        system_text = """You are a query suggestion assistant for an AI tool search platform. Generate exactly 5 similar, related, or refined queries based on the user's original query.

Guidelines:
- Generate exactly 5 alternative queries related to the original query
- Make suggestions more specific, broader, or explore different angles
- Focus on tool-finding scenarios (AI tools, software, applications)
- Keep suggestions practical and actionable
- Vary suggestions to cover different aspects: pricing, features, use cases, industries
- Each suggestion should be a complete, natural query
- Avoid repeating the exact same query
- Format as a simple JSON array of strings

Output only a JSON array of 5 strings, nothing else."""

        prompt_text = f"""Generate 5 query suggestions for: "{request.query}"

Return only a JSON array of 5 strings."""

        llm = ChatGroq(
            groq_api_key=env.groq_api_key,
            model_name=current_model,
            temperature=0.7,
            max_tokens=200
        )
        
        try:
            response = llm.invoke([
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_text}
            ])
            
            llm_response = response.content.strip()
            logger.info("LLM suggestions response received")
            
            try:
                suggestions = json.loads(llm_response)
                
                if isinstance(suggestions, list) and len(suggestions) >= 5:
                    suggestions = suggestions[:5]
                    suggestions = [str(s).strip() for s in suggestions if str(s).strip()]
                    
                    if len(suggestions) < 5:
                        fallback_suggestions = QueryProcessor.generate_fallback_suggestions(request.query)
                        suggestions.extend(fallback_suggestions[len(suggestions):])
                    
                    return QuerySuggestionsResponse(
                        original_query=request.query,
                        suggestions=suggestions[:5]
                    )
                else:
                    raise ValueError("Invalid suggestions format")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse LLM suggestions: {str(e)}")
                fallback_suggestions = QueryProcessor.generate_fallback_suggestions(request.query)
                return QuerySuggestionsResponse(
                    original_query=request.query,
                    suggestions=fallback_suggestions
                )
        
        except Exception as e:
            logger.error(f"Error in LLM call for suggestions: {str(e)}")
            fallback_suggestions = QueryProcessor.generate_fallback_suggestions(request.query)
            return QuerySuggestionsResponse(
                original_query=request.query,
                suggestions=fallback_suggestions
            )
            
    except Exception as e:
        logger.error(f"Error in get_query_suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate query suggestions: {str(e)}"
        )

@app.post("/popular-by-usecase", response_model=PopularByUseCaseResponse)
async def get_popular_by_usecase(request: PopularByUseCaseRequest):
    """Get the most popular AI tools for a specific use case."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing popular tools request for use case: {request.use_case}")
        
        if not request.use_case or not request.use_case.strip():
            raise HTTPException(
                status_code=400,
                detail="use_case is required and cannot be empty"
            )
        
        use_case = request.use_case.strip().lower()
        
        if not app_state.index_ready:
            raise HTTPException(
                status_code=503,
                detail="System is still initializing. Please try again later."
            )
        
        vector_store = vector_store_manager.get_vector_store(for_query=True)
        
        try:
            # Hybrid search for relevant tools
            vector_results_with_scores = vector_store.similarity_search_with_score(
                request.use_case,
                k=Config.VECTOR_SEARCH_K
            )
            
            processed_vector_results = []
            for doc, score in vector_results_with_scores:
                metadata = doc.metadata
                result = dict(metadata)
                result["score"] = float(1.0 - score)
                processed_vector_results.append(result)
                # processed_vector_results.append({
                #     "rid": metadata["rid"],
                #     "tool_id": metadata["tool_id"],
                #     "name": metadata["name"],
                #     "category_subcat": metadata.get("category_subcat", ""),
                #     "url": metadata.get("url", ""),
                #     "description": metadata.get("description", ""),
                #     "image_url": metadata.get("image_url", ""),
                #     "owner": metadata.get("owner", ""),
                #     "status": metadata.get("status", ""),
                #     "score": float(1.0 - score)
                # })
            
            logger.info(f"Vector search returned {len(processed_vector_results)} results")
            
            bm25_results = bm25_index.search(request.use_case, top_k=Config.BM25_SEARCH_K)
            logger.info(f"BM25 search returned {len(bm25_results)} results")
            
            hybrid_results = SearchUtils.fuse_search_results(
                processed_vector_results, 
                bm25_results,
                alpha=Config.HYBRID_ALPHA
            )
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            
            if not hybrid_results:
                return PopularByUseCaseResponse(
                    use_case=request.use_case,
                    tool_id=[],
                    tools=[],
                    ranking_criteria="No tools found for the specified use case"
                )
            
            # Calculate popularity scores
            scored_tools = []
            for result in hybrid_results:
                try:
                    tool = Tool(
                        tool_id=result.get("tool_id", ""),
                        name=result.get("name", "Unknown"),
                        category_subcat=result.get("category_subcat", ""),
                        url=result.get("url", "https://example.com"),
                        description=result.get("description", ""),
                        image_url=result.get("image_url", None),
                        owner=result.get("owner", None),
                        status=result.get("status", None)
                    )
                    
                    popularity_score = SearchUtils.calculate_popularity_score(tool, use_case)
                    
                    scored_tool = {
                        "id": result.get("tool_id", ""),
                        "name": result.get("name", "Unknown"),
                        "description": result.get("description", ""),
                        "url": result.get("url", ""),
                        "category": result.get("category_subcat", ""),
                        "popularity_score": round(popularity_score, 1),
                        "tool_object": tool
                    }
                    scored_tools.append(scored_tool)
                    
                except Exception as e:
                    logger.warning(f"Error processing tool {result.get('name', 'Unknown')}: {str(e)}")
                    continue
            
            # Sort and ensure diversity
            scored_tools.sort(key=lambda x: x["popularity_score"], reverse=True)
            logger.info(f"Calculated popularity scores for {len(scored_tools)} tools")
            
            diverse_tools = SearchUtils.ensure_tool_diversity(scored_tools)
            top_tools = diverse_tools[:Config.POPULAR_TOOLS_LIMIT]
            
            # Generate final response
            final_tools = []
            tool_ids = []
            
            for tool_data in top_tools:
                tool_obj = tool_data["tool_object"]
                indicators = SearchUtils.generate_popularity_indicators(
                    tool_obj, 
                    use_case, 
                    tool_data["popularity_score"]
                )
                
                final_tool = {
                    "id": tool_data["id"],
                    "name": tool_data["name"],
                    "description": tool_data["description"],
                    "url": tool_data["url"],
                    "category": tool_data["category"],
                    "popularity_score": tool_data["popularity_score"],
                    "popularity_indicators": indicators
                }
                
                final_tools.append(final_tool)
                tool_ids.append(tool_data["id"])
            
            elapsed_time = time.time() - start_time
            logger.info(f"Popular by use case processing completed in {elapsed_time:.2f}s")
            
            return PopularByUseCaseResponse(
                use_case=request.use_case,
                tool_id=tool_ids,
                tools=final_tools,
                ranking_criteria="Based on metadata completeness and use case relevance"
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process search: {str(e)}"
            )
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in get_popular_by_usecase: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get popular tools: {str(e)}"
        )

@app.post("/extract-keywords", response_model=ExtractKeywordsResponse)
async def extract_keywords(request: ExtractKeywordsRequest):
    """Extract keywords from reviews using hybrid statistical + LLM approach."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing keyword extraction for {len(request.reviews)} reviews")
        
        if not request.reviews or len(request.reviews) == 0:
            raise HTTPException(
                status_code=400,
                detail="reviews array is required and cannot be empty"
            )
        
        valid_reviews = [review.strip() for review in request.reviews if review.strip()]
        
        if not valid_reviews:
            raise HTTPException(
                status_code=400,
                detail="No valid reviews found after cleaning"
            )
        
        logger.info(f"Processing {len(valid_reviews)} valid reviews")
        
        # Statistical keyword extraction
        logger.info("Starting statistical keyword extraction")
        statistical_candidates = KeywordExtractor.get_statistical_candidates(valid_reviews)
        
        if not statistical_candidates:
            logger.warning("No statistical candidates found")
            return ExtractKeywordsResponse(keywords=[])
        
        logger.info(f"Found {len(statistical_candidates)} statistical candidates")
        
        # Determine target keyword count
        target_count = min(10, len(statistical_candidates), max(3, len(valid_reviews) * 2))
        logger.info(f"Target keyword count: {target_count}")
        
        # LLM processing
        logger.info("Starting LLM keyword processing")
        try:
            llm_keywords = KeywordExtractor.process_keywords_with_llm(
                statistical_candidates, 
                valid_reviews, 
                target_count
            )
            
            if llm_keywords and len(llm_keywords) > 0:
                logger.info(f"LLM returned {len(llm_keywords)} keywords")
                
                final_keywords = []
                for kw in llm_keywords:
                    final_keywords.append(KeywordItem(
                        keyword=kw["keyword"],
                        sentiment=kw["sentiment"]
                    ))
                
                elapsed_time = time.time() - start_time
                logger.info(f"Keyword extraction completed in {elapsed_time:.2f}s")
                
                return ExtractKeywordsResponse(keywords=final_keywords)
            else:
                logger.warning("LLM returned no valid keywords, falling back to statistical")
                raise Exception("LLM processing failed")
                
        except Exception as e:
            logger.warning(f"LLM processing failed: {str(e)}, falling back to statistical approach")
            
            # Fallback to statistical results
            fallback_candidates = statistical_candidates[:target_count]
            fallback_keywords = KeywordExtractor.apply_basic_sentiment(fallback_candidates)
            
            logger.info(f"Fallback returned {len(fallback_keywords)} keywords")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Keyword extraction completed (fallback) in {elapsed_time:.2f}s")
            
            return ExtractKeywordsResponse(keywords=fallback_keywords[:target_count])
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in extract_keywords: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract keywords: {str(e)}"
        )

# Add this in the API ENDPOINTS section (around line 2000, before if __name__ == "__main__")

@app.get("/related-tools/{tool_id}", response_model=List[str])
async def get_related_tools(tool_id: str):
    """Get related tools for a given tool_id using hybrid search."""
    try:
        logger.info(f"Finding related tools for tool_id: {tool_id}")
        
        # Check if system is ready
        if not app_state.index_ready:
            raise HTTPException(
                status_code=503,
                detail="System is still initializing. Please try again later."
            )
        
        # Get vector store
        vector_store = vector_store_manager.get_vector_store(for_query=True)
        
        # Step 1: Find the source tool
        try:
            source_results = vector_store.similarity_search(
                "",
                k=1,
                filter={"tool_id": tool_id}
            )
            
            if not source_results:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tool with ID '{tool_id}' not found"
                )
            
            source_tool_metadata = source_results[0].metadata
            logger.info(f"Found source tool: {source_tool_metadata.get('name', 'Unknown')}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error finding source tool: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error retrieving source tool: {str(e)}"
            )
        
        # Step 2: Create search query from tool metadata
        search_query = RelatedToolsUtils.create_search_query_from_tool(source_tool_metadata)
        logger.info(f"Created search query: {search_query[:100]}...")
        
        if not search_query.strip():
            logger.warning("Empty search query generated")
            return []
        
        # Step 3: Perform hybrid search (reuse existing logic)
        try:
            # Vector search
            vector_results_with_scores = vector_store.similarity_search_with_score(
                search_query,
                k=Config.VECTOR_SEARCH_K  # 15
            )
            
            # Process vector results
            processed_vector_results = []
            for doc, score in vector_results_with_scores:
                metadata = doc.metadata
                processed_vector_results.append({
                    "rid": metadata.get("rid", ""),
                    "tool_id": metadata.get("tool_id", ""),
                    "name": metadata.get("name", ""),
                    "category_subcat": metadata.get("category_subcat", ""),
                    "url": metadata.get("url", ""),
                    "description": metadata.get("description", ""),
                    "image_url": metadata.get("image_url", ""),
                    "owner": metadata.get("owner", ""),
                    "status": metadata.get("status", ""),
                    "score": float(1.0 - score)  # Convert distance to similarity
                })
            
            logger.info(f"Vector search returned {len(processed_vector_results)} results")
            
            # BM25 search
            bm25_results = []
            try:
                bm25_results = bm25_index.search(search_query, top_k=Config.BM25_SEARCH_K)
                logger.info(f"BM25 search returned {len(bm25_results)} results")
            except Exception as e:
                logger.warning(f"BM25 search failed: {str(e)}, continuing with vector results only")
            
            # Fuse results using existing fusion algorithm
            hybrid_results = SearchUtils.fuse_search_results(
                processed_vector_results,
                bm25_results,
                alpha=Config.HYBRID_ALPHA  # 0.5
            )
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} fused results")
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error performing search: {str(e)}"
            )
        
        # Step 4: Filter and rank results
        related_tool_ids = RelatedToolsUtils.filter_and_rank_results(
            hybrid_results=hybrid_results,
            original_tool_id=tool_id,
            min_score=0.5,  # Adjustable quality threshold
            max_results=6   # Maximum 6 tools
        )
        
        logger.info(f"Returning {len(related_tool_ids)} related tools for {tool_id}")
        return related_tool_ids
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_related_tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/categorize-tool/{tool_id}", response_model=ToolCategorizationResponse)
async def categorize_tool(tool_id: str, request_headers: Request):
    """Categorize a tool based on its metadata using LLM analysis."""
    start_time = time.time()
    
    try:
        logger.info(f"Categorizing tool with ID: {tool_id}")
        
        # Step 1: Fetch tool from MongoDB
        tool_data = await mongodb_manager.get_tool_by_id(tool_id)
        if not tool_data:
            raise HTTPException(
                status_code=404,
                detail=f"Tool with ID '{tool_id}' not found"
            )
        
        tool_name = tool_data.get('name', 'Unknown Tool')
        logger.info(f"Found tool: {tool_name}")
        
        # Step 2: Fetch all categories from MongoDB
        categories = await mongodb_manager.get_all_categories()
        if not categories:
            raise HTTPException(
                status_code=500,
                detail="No categories found in database"
            )
        
        logger.info(f"Fetched {len(categories)} categories for analysis")
        
        # Step 3: Prepare tool metadata for LLM (OPTIMIZED)
        tool_metadata = {
            "name": tool_data.get('name', ''),
            "description": tool_data.get('details', {}).get('introduction', ''),
            "speciality": tool_data.get('details', {}).get('speciality', ''),
            "usage": tool_data.get('details', {}).get('usage', ''),
            "features_pros": tool_data.get('features', {}).get('pros', []),
            "pricingType": tool_data.get('pricingType', '')
        }
        
        # Step 4: Create simple category list for LLM (OPTIMIZED)
        category_names = [cat.get('Category', '') for cat in categories if cat.get('Category', '').strip()]
        category_names = [name for name in category_names if name]  # Remove empty strings
        
        # Step 5: PRINT STATEMENTS - Data sent to LLM
        print("\n" + "="*80)
        print("🔍 TOOL METADATA SENT TO LLM:")
        print("="*80)
        print(json.dumps(tool_metadata, indent=2, ensure_ascii=False))
        
        print("\n" + "="*80)
        print(f"📋 CATEGORY LIST SENT TO LLM ({len(category_names)} categories):")
        print("="*80)
        for i, cat_name in enumerate(category_names, 1):
            print(f"{i:2d}. {cat_name}")
        print("="*80 + "\n")
        
        # Step 6: Prepare LLM prompts (SIMPLIFIED FOR ARRAY RESPONSE)
        system_prompt = """You are an expert AI tool categorization assistant. Your task is to analyze tool metadata and select the most relevant categories from a provided list.

Instructions:
1. Analyze the tool's functionality, features, and use cases
2. Select ONLY the most relevant categories (maximum 25 categories)
3. Be SELECTIVE and PRECISE - only include categories that clearly match
4. Focus on what the tool primarily does, not tangential features
5. Return ONLY a simple JSON array of selected category names: ["Category1", "Category2", "Category3"]
6. NO DUPLICATES - each category should appear exactly once
7. Use exact category names from the provided list

IMPORTANT: Return only a JSON array, nothing else. Be selective, not inclusive."""

        user_prompt = f"""Tool to Categorize:
{json.dumps(tool_metadata, indent=2)}

Available Categories:
{json.dumps(category_names, indent=1)}

Task: Select the most relevant categories for this tool. Return ONLY a JSON array like: ["Category1", "Category2"]"""

        # Step 7: Call LLM with regular Groq model
        try:
            llm = ChatGroq(
                groq_api_key=env.groq_api_key,
                model_name="llama-3.1-8b-instant",  # Reliable model
                temperature=0.1
            )
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            llm_response = response.content.strip()
            logger.info(f"✅ Groq response received")
            
            # Step 8: PRINT STATEMENT - LLM Response
            print("\n" + "="*80)
            print("🤖 LLM RESPONSE:")
            print("="*80)
            print(llm_response)
            print("="*80 + "\n")
            
            # Step 9: Parse simple JSON array response
            try:
                # Clean response if it has code blocks
                if llm_response.startswith("```"):
                    llm_response = llm_response.strip("```json").strip("```").strip()
                
                selected_category_names = json.loads(llm_response)
                
                # Ensure it's a list
                if not isinstance(selected_category_names, list):
                    print(f"❌ Expected list, got {type(selected_category_names)}")
                    selected_category_names = []
                
                # Step 10: PRINT STATEMENT - Parsed Selection
                print("\n" + "="*60)
                print(f"✅ PARSED SELECTED CATEGORIES ({len(selected_category_names)}):")
                print("="*60)
                for i, cat_name in enumerate(selected_category_names, 1):
                    print(f"{i}. {cat_name}")
                print("="*60 + "\n")
                
                # Step 11: Map selected names back to full category data with IDs
                category_matches = []
                category_lookup = {cat.get('Category', ''): cat for cat in categories}
                
                for selected_name in selected_category_names:
                    if selected_name in category_lookup:
                        cat_data = category_lookup[selected_name]
                        category_match = CategoryMatch(
                            id=str(cat_data.get('_id', '')),
                            name=str(cat_data.get('Category', ''))
                        )
                        category_matches.append(category_match)
                        print(f"✓ Matched: '{selected_name}' → ID: {cat_data.get('_id', '')}")
                    else:
                        print(f"✗ No match found for: '{selected_name}'")
                        # Try partial matching as fallback
                        partial_matches = [cat for cat in categories if selected_name.lower() in cat.get('Category', '').lower()]
                        if partial_matches:
                            cat_data = partial_matches[0]  # Take first partial match
                            category_match = CategoryMatch(
                                id=str(cat_data.get('_id', '')),
                                name=str(cat_data.get('Category', ''))
                            )
                            category_matches.append(category_match)
                            print(f"✓ Partial match: '{selected_name}' → '{cat_data.get('Category', '')}' (ID: {cat_data.get('_id', '')})")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Tool categorization completed in {elapsed_time:.2f}s - found {len(category_matches)} matches")
                
                # Step 12: PRINT STATEMENT - Final Results
                print("\n" + "="*70)
                print(f"🎯 FINAL CATEGORIZATION RESULTS:")
                print("="*70)
                print(f"Tool: {tool_name}")
                print(f"Selected Categories: {len(category_matches)}")
                for i, match in enumerate(category_matches, 1):
                    print(f"  {i}. {match.name} (ID: {match.id})")
                print("="*70 + "\n")
                
                return ToolCategorizationResponse(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    matched_categories=category_matches,
                    total_available_categories=len(categories)
                )
                
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON PARSE ERROR: {str(e)}")
                print(f"Raw response was: '{llm_response}'")
                logger.error(f"Failed to parse LLM response: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse categorization results: {str(e)}"
                )
        
        except Exception as e:
            print(f"\n❌ LLM CALL ERROR: {str(e)}")
            logger.error(f"LLM call failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Categorization processing failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in categorize_tool: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool categorization failed: {str(e)}"
        )
        
@app.post("/categorize-tool-usecase/{tool_id}", response_model=ToolUseCaseCategorizationResponse)
async def categorize_tool_usecase(tool_id: str, request_headers: Request):
    """Categorize a tool into use cases based on its metadata using LLM analysis."""
    start_time = time.time()
    
    try:
        logger.info(f"Categorizing tool use cases for ID: {tool_id}")
        
        # Step 1: Fetch tool from MongoDB
        tool_data = await mongodb_manager.get_tool_by_id(tool_id)
        if not tool_data:
            raise HTTPException(
                status_code=404,
                detail=f"Tool with ID '{tool_id}' not found"
            )
        
        tool_name = tool_data.get('name', 'Unknown Tool')
        logger.info(f"Found tool: {tool_name}")
        
        # Step 2: Fetch all use cases from MongoDB
        usecases = await mongodb_manager.get_all_usecases()
        if not usecases:
            raise HTTPException(
                status_code=500,
                detail="No use cases found in database"
            )
        
        logger.info(f"Fetched {len(usecases)} use cases for analysis")
        
        # Step 3: Prepare tool metadata for LLM (OPTIMIZED)
        tool_metadata = {
            "name": tool_data.get('name', ''),
            "description": tool_data.get('details', {}).get('introduction', ''),
            "speciality": tool_data.get('details', {}).get('speciality', ''),
            "usage": tool_data.get('details', {}).get('usage', ''),
            "features_pros": tool_data.get('features', {}).get('pros', []),
            "pricingType": tool_data.get('pricingType', '')
        }
        
        # Step 4: Create simple use case list for LLM (OPTIMIZED)
        usecase_names = [uc.get('UseCase', '') for uc in usecases if uc.get('UseCase', '').strip()]
        usecase_names = [name for name in usecase_names if name]  # Remove empty strings
        
        # Step 5: PRINT STATEMENTS - Data sent to LLM
        print("\n" + "="*80)
        print("🔍 TOOL METADATA SENT TO LLM:")
        print("="*80)
        print(json.dumps(tool_metadata, indent=2, ensure_ascii=False))
        
        print("\n" + "="*80)
        print(f"📋 USE CASE LIST SENT TO LLM ({len(usecase_names)} use cases):")
        print("="*80)
        for i, uc_name in enumerate(usecase_names, 1):
            print(f"{i:2d}. {uc_name}")
        print("="*80 + "\n")
        
        # Step 6: Prepare LLM prompts (SIMPLIFIED FOR ARRAY RESPONSE)
        system_prompt = """You are an expert AI tool use case categorization assistant. Your task is to analyze tool metadata and select the relevant use cases only from a provided Available Use Cases.

Instructions:
1. Analyze the tool's functionality, target audience, and practical applications
2. Select the relevant use cases (maximum 25 use cases) only from the give Available Use Cases.
3. Include use cases where this tool would be useful
4. Focus on the tool's use cases.
5. Return ONLY a simple JSON array of selected use case names: ["Use Case 1", "Use Case 2"]
6. NO DUPLICATES - each use case should appear exactly once
7. Use exact use case names from the provided list

IMPORTANT: Return only a JSON array, nothing else."""

        user_prompt = f"""Tool to Categorize:
{json.dumps(tool_metadata, indent=2)}

Available Use Cases:
{json.dumps(usecase_names, indent=1)}

Task: Select the relevant use cases for this tool. Return ONLY a JSON array like: ["Use Case 1", "Use Case 2"]"""

        # Step 7: Call LLM with regular Groq model
        try:
            llm = ChatGroq(
                groq_api_key=env.groq_api_key,
                model_name="llama-3.1-8b-instant",  # Reliable model
                temperature=0.1
            )
            
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            llm_response = response.content.strip()
            logger.info(f"✅ Groq response received")
            
            # Step 8: PRINT STATEMENT - LLM Response
            print("\n" + "="*80)
            print("🤖 LLM RESPONSE:")
            print("="*80)
            print(llm_response)
            print("="*80 + "\n")
            
            # Step 9: Parse simple JSON array response
            try:
                # Clean response if it has code blocks
                if llm_response.startswith("```"):
                    llm_response = llm_response.strip("```json").strip("```").strip()
                
                selected_usecase_names = json.loads(llm_response)
                
                # Ensure it's a list
                if not isinstance(selected_usecase_names, list):
                    print(f"❌ Expected list, got {type(selected_usecase_names)}")
                    selected_usecase_names = []
                
                # Step 10: PRINT STATEMENT - Parsed Selection
                print("\n" + "="*60)
                print(f"✅ PARSED SELECTED USE CASES ({len(selected_usecase_names)}):")
                print("="*60)
                for i, uc_name in enumerate(selected_usecase_names, 1):
                    print(f"{i}. {uc_name}")
                print("="*60 + "\n")
                
                # Step 11: Map selected names back to full use case data with IDs
                usecase_matches = []
                usecase_lookup = {uc.get('UseCase', ''): uc for uc in usecases}
                
                for selected_name in selected_usecase_names:
                    if selected_name in usecase_lookup:
                        uc_data = usecase_lookup[selected_name]
                        usecase_match = UseCaseMatch(
                            id=str(uc_data.get('_id', '')),
                            name=str(uc_data.get('UseCase', ''))
                        )
                        usecase_matches.append(usecase_match)
                        print(f"✓ Matched: '{selected_name}' → ID: {uc_data.get('_id', '')}")
                    else:
                        print(f"✗ No match found for: '{selected_name}'")
                        # Try partial matching as fallback
                        partial_matches = [uc for uc in usecases if selected_name.lower() in uc.get('UseCase', '').lower()]
                        if partial_matches:
                            uc_data = partial_matches[0]  # Take first partial match
                            usecase_match = UseCaseMatch(
                                id=str(uc_data.get('_id', '')),
                                name=str(uc_data.get('UseCase', ''))
                            )
                            usecase_matches.append(usecase_match)
                            print(f"✓ Partial match: '{selected_name}' → '{uc_data.get('UseCase', '')}' (ID: {uc_data.get('_id', '')})")
                
                elapsed_time = time.time() - start_time
                logger.info(f"Tool use case categorization completed in {elapsed_time:.2f}s - found {len(usecase_matches)} matches")
                
                # Step 12: PRINT STATEMENT - Final Results
                print("\n" + "="*70)
                print(f"🎯 FINAL USE CASE CATEGORIZATION RESULTS:")
                print("="*70)
                print(f"Tool: {tool_name}")
                print(f"Selected Use Cases: {len(usecase_matches)}")
                for i, match in enumerate(usecase_matches, 1):
                    print(f"  {i}. {match.name} (ID: {match.id})")
                print("="*70 + "\n")
                
                return ToolUseCaseCategorizationResponse(
                    tool_id=tool_id,
                    tool_name=tool_name,
                    matched_usecases=usecase_matches,
                    total_available_usecases=len(usecases)
                )
                
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON PARSE ERROR: {str(e)}")
                print(f"Raw response was: '{llm_response}'")
                logger.error(f"Failed to parse LLM response: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse use case categorization results: {str(e)}"
                )
        
        except Exception as e:
            print(f"\n❌ LLM CALL ERROR: {str(e)}")
            logger.error(f"LLM call failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Use case categorization processing failed: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in categorize_tool_usecase: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Tool use case categorization failed: {str(e)}"
        )

@app.get("/debug-env")
async def debug_env():
    """Debug endpoint to check environment variables"""
    return {
        "environment": env.environment,
        "dev_model": env.dev_model,
        "prod_model": env.prod_model,
        "raw_environment": os.getenv("ENVIRONMENT"),
        "raw_dev_model": os.getenv("DEV_MODEL"),
        "raw_prod_model": os.getenv("PROD_MODEL"),
        "current_model_logic": env.dev_model if env.environment == "DEV" else env.prod_model
    }

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )