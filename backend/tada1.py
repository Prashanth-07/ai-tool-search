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
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["embeddings"]
        except Exception as e:
            raise ValueError(f"Error calling Nomic Atlas API: {str(e)}")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single text using the Nomic Atlas API."""
        embeddings = self.embed_documents([text])
        return embeddings[0]
    
def get_vector_store(headers=None):
    """Initialize or return existing vector store using Nomic Atlas API."""
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

@app.post("/query", response_model=QueryResponse)
async def query_tools(request: QueryRequest, request_headers: Request):
    """Query tools based on user input using efficient direct retrieval."""
    start_time = time.time()
    try:
        logger.info(f"Processing query: {request.query}")
        headers = request_headers.headers

        # Try to get cached response
        cached_response = tool_search_cache.get(request.query)
        if cached_response:
            logger.info("Returning cached response")
            return QueryResponse(response=cached_response)
        
        # Get the vector store
        vector_store = get_vector_store(headers)

        # Log total vectors for reference
        total_vectors = get_total_vectors()
        logger.info(f"Total vectors in store: {total_vectors}")
        
        # Direct vector search - simpler and faster approach
        try:
            # prefixed_query = f"query: {request.query}"
            # Simple similarity search
            results = vector_store.similarity_search(
                request.query,
                k=5 # Limit to top 5 results
            )
            
            # Format the documents
            formatted_docs = []
            for doc in results[:5]:  # Only use top 3 for faster processing
                metadata = doc.metadata
                formatted_doc = (
                    f"Tool ID: {metadata.get('tool_id', 'N/A')}\n"
                    f"Name: {metadata.get('name', 'N/A')}\n"
                    f"Description: {metadata.get('description', 'N/A')}\n"
                    f"Categories: {metadata.get('categories', 'N/A')}\n"
                    f"Usage: {metadata.get('usage', 'N/A')}\n"
                    f"Unique Features: {metadata.get('unique_features', 'N/A')}\n"
                    f"Pros: {', '.join(metadata.get('pros', []))}\n"
                    f"Cons: {', '.join(metadata.get('cons', []))}\n"
                    f"Pricing: {metadata.get('pricing', 'N/A')}"
                )
                formatted_docs.append(formatted_doc)
                logger.info(f"=== Document {len(formatted_docs)} ===")
                logger.info(f"Tool ID: {metadata.get('tool_id', 'N/A')}")
                logger.info(f"Formatted doc: {formatted_doc}")
                logger.info("=====================")
            
            # Join the formatted documents with separators
            context = "\n\n---\n\n".join(formatted_docs)
            logger.info("=== Complete Context ===")
            logger.info(context)
            logger.info("=====================")
            
            # Get the model based on headers
            current_model = get_current_model(headers)
            logger.info(f"Using model: {current_model} with Groq API")
            
            # Log the input for debugging
            logger.info("=== LLM Input ===")
            logger.info(f"Question: {request.query}")
            logger.info(f"Context: {context}")
            logger.info("================")

            system_text= f"""You are a tool retrieval assistant tasked with extracting relevant tools from a provided context based on a user query.
            Guidelines:
            1. JSON-only output: Provide the answer as a JSON object with no additional text, markdown, or formatting.
            2. Context-only reasoning: Rely ONLY on the provided context for information. Do not use any outside knowledge.
            3. No hallucinations: Do not invent any tool details or IDs. Use only what is present in the context.
            4. Extract and include the tools even if it is partially related but sort by relevance.
            4. JSON schema adherence: The output JSON must strictly follow this format:
            {{
    "tool_id": ["tool1_id", "tool2_id"],
    "tools": [
        {{
            "id": "tool1_id",
            "name": "Tool Name 1",
            "description": "brief description",
            "relevance": "explanation of relevance to query"
        }}
    ]
}}
Ensure the JSON is valid and includes the tools from the given context only."""
            
            # Create prompt for LangChain
            prompt_text = f"""Identify the all relevant tools from the context below that answer the user's query.

User Query: {request.query}

Tool Data: {context}

Provide the relevant tools as a JSON object following the above schema."""

            # Initialize LangChain's ChatGroq
            llm = ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=current_model,
                temperature=0.1,
                max_tokens=800
            )
            
            # Get response from Groq LLM
            response = llm.invoke([
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt_text}
            ])
            
            # Extract content from response
            llm_response = response.content
            
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
            error_msg = str(e)
            logger.error(f"Error in retrieval or LLM call: {error_msg}")
            
            # Create a fallback JSON response
            fallback_response = json.dumps({
                "error": "processing_error",
                "message": f"Failed to process query: {error_msg}",
                "tools": []
            })
            
            return QueryResponse(response=fallback_response)
            
    except Exception as e:
        logger.error(f"Error in query_tools: {str(e)}")
        
        # Log total processing time even on error
        elapsed_time = time.time() - start_time
        logger.info(f"Query processing completed in {elapsed_time:.2f}s")
        
        # Return a minimal valid response in case of error
        error_response = json.dumps({
            "error": "api_error",
            "message": str(e),
            "tools": []
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
            
            logger.info(f"Added tool: {tool.name} (Tool ID: {tool.tool_id})")
            results.append(ToolResponse(id=rid, tool=tool, status="added"))
        
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
                
                logger.info(f"Updated tool: {tool.name} (Tool ID: {tool.tool_id}, RID: {existing_rid})")
                results.append(ToolResponse(id=existing_rid, tool=tool, status="updated"))
                
            except Exception as e:
                logger.error(f"Error updating tool {tool.tool_id}: {str(e)}")
                continue
        
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
                "description": doc.metadata.get("description", "N/A")[:100] + "...",  # Truncate long descriptions
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