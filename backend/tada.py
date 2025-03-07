import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from uuid import uuid4
import json
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from requests import request

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

# Verify environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

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

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Constants
INDEX_NAME = "ai-tool-search"
DIMENSION = 768  # for nomic-embed-text

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

# def create_rag_chain(retriever):
#     """
#     Creates a RAG chain optimized to return JSON responses in the format:
#     {
#         "tool_id": ["<most relevant tool ID>", "<next relevant tool ID>", ...],
#         "tools": [
#             {"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}
#         ]
#     }
#     The model ranks relevant tools in order of importance and provides their basic details.
#     """
#     prompt = """You are an AI assistant specializing in retrieving and ranking relevant AI tools based on user queries. 
#     Your task is to analyze the user’s question and match it to the most relevant tools from the provided dataset.

#     User Query: {question}

#     Tool Data: {context}

#     **Your Responsibilities:**
#     1. Analyze the user query to understand their need.
#     2. Match the query with relevant tools from the dataset.
#     3. Rank relevant tools based on best match (most relevant first).
#     4. Only return tools that match the query; do NOT include irrelevant tools.
#     5. Strictly output in JSON format.

#     **Response Format (Strict JSON Output Only):**
#     ```json
#     {{
#         "tool_id": ["<most relevant tool ID>", "<next relevant tool ID>", ...],
#         "tools": [
#             {{"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}},
#             {{"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}}
#         ]
#     }}
#     ```

#     - **Do NOT include any explanations or extra text** outside the JSON structure.
#     - **Only include relevant tools** based on the user query.
#     - **Ensure the most relevant tools appear first** in the ranking.
#     """

#     model = ChatOllama(
#         model="deepseek-r1:1.5b",
#         base_url="http://localhost:11434",
#         top_p=0.9,
#         temperature=0.3,  # Lower temp for more deterministic recommendations
#         presence_penalty=0.2,
#         frequency_penalty=0.3,
#         stream=True,
#         max_tokens=250,  # Sufficient for listing relevant tools
#         system_prompt="""You are a retrieval AI assistant. 
#         - Your goal is to find the most relevant tools for a given user query.
#         - Extract key needs from {question}.
#         - Compare it against {context} and return ONLY the most relevant tool IDs and details.
#         - Rank the tools based on best match.
#         - Output strictly in JSON format.
#         - Do not generate text outside of the required JSON response."""
#     )

#     prompt_template = ChatPromptTemplate.from_template(prompt)
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )
    
#     return rag_chain
# Add this function near the top of your file
def get_current_model(headers=None):
    """
    Get the current model based on headers or environment variables
    """
    # Default to environment variable if no headers provided
    if not headers:
        environment = os.getenv("ENVIRONMENT", "DEV")
        if environment == "DEV":
            return os.getenv("DEV_MODEL", "deepseek-r1:1.5b")
        else:
            return os.getenv("PROD_MODEL", "deepseek-r1:7b")
    
    # Use header if provided
    model_choice = headers.get("MODEL_CHOICE", "DEV_MODEL")
    if model_choice == "PROD_MODEL":
        return os.getenv("PROD_MODEL", "deepseek-r1:7b")
    else:
        return os.getenv("DEV_MODEL", "deepseek-r1:1.5b")

def create_rag_chain(retriever, headers=None):
    """
    Creates a RAG chain with debug logging using updated LangChain syntax
    """
    prompt =  """You are an AI assistant specializing in retrieving and ranking relevant AI tools based on user queries. 
    Your task is to analyze the user's question and match it to the most relevant tools from the provided dataset.

    User Query: {question}

    Tool Data: {context}

    **Your Responsibilities:**
    1. Analyze the user query to understand their need.
    2. Match the query with relevant tools from the dataset.
    3. Rank relevant tools based on best match (most relevant first).
    4. Only return tools that match the query; do NOT include irrelevant tools.
    5. For each tool, explain briefly how it relates to the user's query.
    6. Include at least 2-3 tools in the response if available.
    7. Strictly output in JSON format using the tool_id field from the metadata, NOT the rid.

    **Response Format (Strict JSON Output Only):**
    {{
        "tool_id": ["<most relevant tool_id>", "<next relevant tool_id>", ...],
        "tools": [
            {{
                "id": "<tool_id>", 
                "name": "<tool_name>", 
                "description": "<brief description>",
                "relevance": "<brief explanation of how this tool relates to the query>"
            }},
            {{
                "id": "<tool_id>", 
                "name": "<tool_name>", 
                "description": "<brief description>",
                "relevance": "<brief explanation of how this tool relates to the query>"
            }}
        ]
    }}

    - Use the tool_id field (like "github-copilot-001") NOT the rid field
    - Do NOT include any explanations or extra text outside the JSON structure
    - Only include relevant tools based on the user query
    - Ensure the most relevant tools appear first in the ranking
    - Always include at least 2-3 results if possible
    """
    # Get the model based on environment variable
    # environment = os.getenv("ENVIRONMENT", "DEV")
    current_model = get_current_model(headers)
    model = ChatOllama(
        model=current_model,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ai_tool_search_ollama:11434"),
        top_p=0.9,
        temperature=0.3,
        presence_penalty=0.2,
        frequency_penalty=0.3,
        stream=True,
        max_tokens=500,
        system_prompt="""You are a retrieval AI assistant. 
        - Your goal is to find the most relevant tools for a given user query.
        - Extract key needs from {question}.
        - Compare it against {context} and return multiple relevant tool IDs and details.
        - Always include at least 2-3 tools when available.
        - Always use the tool_id field (like "github-copilot-001") NOT the rid field.
        - Rank the tools based on best match.
        - Add a relevance field explaining why each tool matches the query.
        - Output strictly in JSON format.
        - Do not generate text outside of the required JSON response."""
    )

    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    async def format_docs_async(question):
        """Async function to retrieve and format documents with simplified structure"""
        docs = await retriever.aget_relevant_documents(question)
        formatted_docs = []
        
        for doc in docs:
            metadata = doc.metadata
            # Create a single, clean format without duplicates
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
            
            # Debug logging
            logger.info(f"=== Document {len(formatted_docs)} ===")
            logger.info(f"Tool ID: {metadata.get('tool_id', 'N/A')}")
            logger.info(f"Formatted doc: {formatted_doc}")
            logger.info("=====================")
        
        context = "\n\n---\n\n".join(formatted_docs)
        logger.info("=== Complete Context ===")
        logger.info(context)
        logger.info("=====================")
        
        return context

    async def generate_response(question):
        """Main chain execution with debug logging"""
        # Get and format documents
        context = await format_docs_async(question)
        
        # Log the full input going to the LLM
        logger.info("=== LLM Input ===")
        logger.info(f"Question: {question}")
        logger.info(f"Context: {context}")
        logger.info("================")
        
        # Create input dictionary for the prompt
        chain_input = {
            "question": question,
            "context": context
        }
        
        # Execute the chain
        result = await prompt_template.ainvoke(chain_input)
        result = await model.ainvoke(result)
        return await StrOutputParser().ainvoke(result)
    
    return generate_response

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

def get_vector_store():
    """Initialize or return existing vector store"""
    try:
        embeddings = OllamaEmbeddings(
            model='nomic-embed-text',
            base_url=os.getenv("OLLAMA_BASE_URL", "http://ai_tool_search_ollama:11434")
        )
        
        # Initialize the vector store using the new syntax
        vector_store = LangchainPinecone.from_existing_index(
            index_name=INDEX_NAME,
            embedding=embeddings,
            text_key="text"
        )
        
        return vector_store
    except Exception as e:
        print(f"Error in get_vector_store: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize vector store: {str(e)}"
        )

def format_tool_for_indexing(tool: Tool, rid: str) -> str:
    """Format tool data for embedding"""
    return (
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

# API endpoints
@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "active",
        "message": "AI Tool Search API is running",
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

# Define models for bulk addition
class BulkToolRequest(BaseModel):
    tools: List[Tool]

class BulkToolResponse(BaseModel):
    results: List[ToolResponse]


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



#ADD TOOL WITHOUT DUPLICATE CHECK 

# @app.post("/add-tools", response_model=BulkToolResponse)
# async def add_tools(bulk_request: BulkToolRequest):
#     """Add multiple tools to the vector store."""
#     try:
#         vector_store = get_vector_store()
#         results = []
#         for tool in bulk_request.tools:
#             # Generate unique RID for each tool
#             rid = str(uuid4())
#             # Format tool data
#             tool_text = format_tool_for_indexing(tool, rid)
#             # Prepare metadata
#             metadata = {
#                 "rid": rid,
#                 "tool_id": tool.tool_id,
#                 "name": tool.name,
#                 **tool.model_dump()
#             }
#             # Add document to vector store
#             vector_store.add_texts(
#                 texts=[tool_text],
#                 metadatas=[metadata],
#                 ids=[rid]
#             )
#             results.append(ToolResponse(id=rid, tool=tool, status="added"))
        
#         new_count = get_total_vectors()
#         logger.info(f"Vector count after bulk addition: {new_count}")
#         return BulkToolResponse(results=results)
    
#     except Exception as e:
#         logger.error(f"Error in add_tools: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to add tools: {str(e)}"
#         )
# Add this function to your tada.py file
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
            # Find everything between <think> and </think> tags and remove it
            import re
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
            response_text = response_text.strip()
        
        # Check if the response is wrapped in markdown code block
        import re
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

@app.post("/query", response_model=QueryResponse)
async def query_tools(request: QueryRequest, request_headers: Request):
    """Query tools based on user input using multiple retrieval methods."""
    try:
        logger.info(f"Processing query: {request.query}")
        headers = request_headers.headers
        
        # Get the vector store
        vector_store = get_vector_store()

        # Log total vectors for reference
        total_vectors = get_total_vectors()
        logger.info(f"Total vectors in store: {total_vectors}")
        
        # 1. MMR retrieval for diverse but relevant results
        mmr_retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                'k': 10,  # Increased from 5
                'fetch_k': 15,  # Increased from 8
                'lambda_mult': 0.2
            }
        )
        
        # 2. Similarity retrieval for high relevance results
        similarity_retriever = vector_store.as_retriever(
            search_kwargs={'k': 5}
        )
        
        # 3. Create a combined retriever that merges results from both methods
        class CombinedRetriever:
            async def aget_relevant_documents(self, query):
                # Get documents from both retrievers
                mmr_docs = await mmr_retriever.aget_relevant_documents(query)
                similarity_docs = await similarity_retriever.aget_relevant_documents(query)
                
                # Combine and deduplicate results based on tool_id
                all_docs = []
                seen_ids = set()
                
                # Process all documents, avoiding duplicates
                for doc in mmr_docs + similarity_docs:
                    tool_id = doc.metadata.get('tool_id', '')
                    
                    if tool_id and tool_id not in seen_ids:
                        all_docs.append(doc)
                        seen_ids.add(tool_id)
                
                logger.info(f"Combined retriever found {len(all_docs)} unique tools")
                return all_docs
        
        # Create RAG chain with our combined retriever
        rag_chain = create_rag_chain(CombinedRetriever(), headers)
        
        # Execute the chain asynchronously
        response = await rag_chain(request.query)
        
        # Post-process to remove thinking tags
        processed_response = post_process_llm_response(response)
        
        # Try to parse as JSON but don't fail if it's not valid
        try:
            # Parse and validate response
            response_data = json.loads(processed_response)
            
            # Ensure tool_id is used instead of rid
            if "tools" in response_data:
                for tool in response_data["tools"]:
                    if "id" in tool and len(tool["id"]) == 36:  # UUID length
                        # Find corresponding tool_id
                        results = vector_store.similarity_search(
                            "",
                            k=1,
                            filter={"rid": tool["id"]}
                        )
                        if results:
                            tool["id"] = results[0].metadata.get("tool_id")
            
            # Clean response
            clean_response = json.dumps(response_data, indent=2)
            logger.info(f"Processed valid JSON response")
        except json.JSONDecodeError:
            # If not valid JSON, just return the processed response as-is
            clean_response = processed_response
            logger.warning("Response is not valid JSON, returning as-is")
        
        return QueryResponse(response=clean_response)
    
    except Exception as e:
        logger.error(f"Error in query_tools: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )
        # Return the final chain output
        # return QueryResponse(response=response)

    # except Exception as e:
    #     logger.error(f"Error in query_tools: {str(e)}")
    #     raise HTTPException(
    #         status_code=500,
    #         detail=f"Failed to process query: {str(e)}"
    #     )

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

# Update Pydantic models
class BulkUpdateRequest(BaseModel):
    tools: List[Tool]

class BulkUpdateResponse(BaseModel):
    results: List[ToolResponse]

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
    
@app.get("/stats")
async def get_stats():
    """Get vector store statistics and vector metadata"""
    try:
        # Get basic stats
        total_vectors = get_total_vectors()
        index = get_or_create_index()
        stats = index.describe_index_stats()
        
        # Get vector store to fetch metadata
        vector_store = get_vector_store()
        
        # Query all vectors (empty query to get all)
        results = vector_store.similarity_search(
            "",
            k=total_vectors  # Get all vectors
        )
        
        # Extract vector details
        vectors_info = []
        for doc in results:
            vectors_info.append({
                "name": doc.metadata.get("name", "N/A"),
                "tool_id": doc.metadata.get("tool_id", "N/A"),
                "rid": doc.metadata.get("rid", "N/A"),
                "description": doc.metadata.get("description", "N/A"),
                "categories": doc.metadata.get("categories", "N/A"),
                "pricing": doc.metadata.get("pricing", "N/A")
            })
        
        # Create complete stats dictionary
        stats_dict = {
            "total_vectors": total_vectors,
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


# Add the Pydantic model for the API key request
class ClearIndexRequest(BaseModel):
    api_key: str = Field(..., description="API key for authorization to clear the index")

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
        
        # Delete all vectors (Pinecone allows deleting all vectors by passing an empty filter)
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
# import os
# from datetime import datetime
# from typing import List, Optional, Dict, Any
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from pinecone import Pinecone
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from uuid import uuid4
# import json
# import logging
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('vectorstore.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Verify environment variables
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# if not PINECONE_API_KEY:
#     raise ValueError("PINECONE_API_KEY environment variable is not set")

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# def get_total_vectors() -> int:
#     """Get total number of vectors in the index"""
#     try:
#         index = pc.Index(INDEX_NAME)
#         stats = index.describe_index_stats()
#         total_vectors = stats.total_vector_count
#         logger.info(f"Total vectors in index: {total_vectors}")
#         return total_vectors
#     except Exception as e:
#         logger.error(f"Error getting vector count: {str(e)}")
#         return 0

# # Initialize FastAPI app
# app = FastAPI(title="AI Tool Search API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Constants
# INDEX_NAME = "ai-tool-search"
# DIMENSION = 768  # for nomic-embed-text

# # Pydantic models
# class Tool(BaseModel):
#     name: str
#     tool_id: str
#     description: str
#     pros: List[str]
#     cons: List[str]
#     categories: str
#     usage: str
#     unique_features: str
#     pricing: str

# class ToolResponse(BaseModel):
#     id: str
#     tool: Tool
#     status: str = "added"

# class QueryRequest(BaseModel):
#     query: str

# class QueryResponse(BaseModel):
#     response: str

# class DeleteResponse(BaseModel):
#     success: bool
#     deleted_tool: str

# def create_rag_chain(retriever):
#     """
#     Creates a RAG chain optimized to return JSON responses in the format:
#     {
#         "tool_id": ["<most relevant tool ID>", "<next relevant tool ID>", ...],
#         "tools": [
#             {"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}
#         ]
#     }
#     The model ranks relevant tools in order of importance and provides their basic details.
#     """
#     prompt = """You are an AI assistant specializing in retrieving and ranking relevant AI tools based on user queries. 
#     Your task is to analyze the user’s question and match it to the most relevant tools from the provided dataset.

#     User Query: {question}

#     Tool Data: {context}

#     **Your Responsibilities:**
#     1. Analyze the user query to understand their need.
#     2. Match the query with relevant tools from the dataset.
#     3. Rank relevant tools based on best match (most relevant first).
#     4. Only return tools that match the query; do NOT include irrelevant tools.
#     5. Strictly output in JSON format.

#     **Response Format (Strict JSON Output Only):**
#     ```json
#     {{
#         "tool_id": ["<most relevant tool ID>", "<next relevant tool ID>", ...],
#         "tools": [
#             {{"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}},
#             {{"id": "<tool_id>", "name": "<tool_name>", "description": "<brief description>"}}
#         ]
#     }}
#     ```

#     - **Do NOT include any explanations or extra text** outside the JSON structure.
#     - **Only include relevant tools** based on the user query.
#     - **Ensure the most relevant tools appear first** in the ranking.
#     """

#     model = ChatOllama(
#         model="deepseek-r1:1.5b",
#         base_url="http://localhost:11434",
#         top_p=0.9,
#         temperature=0.3,  # Lower temp for more deterministic recommendations
#         presence_penalty=0.2,
#         frequency_penalty=0.3,
#         stream=True,
#         max_tokens=250,  # Sufficient for listing relevant tools
#         system_prompt="""You are a retrieval AI assistant. 
#         - Your goal is to find the most relevant tools for a given user query.
#         - Extract key needs from {question}.
#         - Compare it against {context} and return ONLY the most relevant tool IDs and details.
#         - Rank the tools based on best match.
#         - Output strictly in JSON format.
#         - Do not generate text outside of the required JSON response."""
#     )

#     prompt_template = ChatPromptTemplate.from_template(prompt)
    
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )
    
#     return rag_chain


# def format_docs(docs):
#     """Format retrieved documents for the prompt"""
#     return "\n\n".join(doc.page_content for doc in docs)

# # Vector store initialization
# def get_or_create_index():
#     """Create Pinecone index if it doesn't exist"""
#     try:
#         # Check if index exists
#         indexes = pc.list_indexes()
#         index_names = [index.name for index in indexes]
        
#         if INDEX_NAME not in index_names:
#             print(f"Creating new index: {INDEX_NAME}")
#             pc.create_index(
#                 name=INDEX_NAME,
#                 dimension=DIMENSION,
#                 metric="cosine"
#             )
#             print(f"Index {INDEX_NAME} created successfully")
        
#         return pc.Index(INDEX_NAME)
#     except Exception as e:
#         print(f"Error in get_or_create_index: {str(e)}")
#         raise HTTPException(
#             status_code=500,query_tools
#             detail=f"Failed to initialize Pinecone index: {str(e)}"
#         )

# def get_vector_store():
#     """Initialize or return existing vector store"""
#     try:
#         embeddings = OllamaEmbeddings(
#             model='nomic-embed-text',
#             base_url="http://localhost:11434"
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

# def format_tool_for_indexing(tool: Tool, tool_id: str) -> str:
#     """Format tool data for embedding"""
#     return (
#         f"Tool ID: {tool_id}\n"
#         f"Name: {tool.name}\n"
#         f"Description: {tool.description}\n"
#         f"Pros: {', '.join(tool.pros)}\n"
#         f"Cons: {', '.join(tool.cons)}\n"
#         f"Categories: {tool.categories}\n"
#         f"Usage: {tool.usage}\n"
#         f"Unique Features: {tool.unique_features}\n"
#         f"Pricing: {tool.pricing}"
#     )

# # API endpoints
# @app.get("/")
# async def root():
#     """Root endpoint to verify API is running"""
#     return {
#         "status": "active",
#         "message": "AI Tool Search API is running",
#         "version": "1.0"
#     }

# @app.get("/health")
# async def health_check():
#     """Check if API and Pinecone connection are healthy"""
#     try:
#         # Verify Pinecone connection
#         _ = get_or_create_index()
#         return {
#             "status": "healthy",
#             "api_version": "1.0",
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Service unhealthy: {str(e)}"
#         )
    
# async def check_duplicate_tool(vector_store, tool: Tool) -> bool:
#     """
#     Check if a tool with the same name and tool_id already exists.
#     Returns True if duplicate found, False otherwise.
#     """
#     try:
#         # Search for tools with the same name
#         results = vector_store.similarity_search(
#             tool.name,
#             k=5,  # Get a few results to check through
#             filter={"name": tool.name}
#         )
        
#         # Check if any result matches both name and tool_id
#         for result in results:
#             metadata = result.metadata
#             if (metadata.get("name") == tool.name and 
#                 metadata.get("tool_id") == tool.tool_id):
#                 return True
                
#         return False
    
#     except Exception as e:
#         print(f"Error checking duplicates: {str(e)}")
#         return False

# @app.post("/add-tool", response_model=ToolResponse)
# async def add_tool(tool: Tool):
#     """Add a new tool to the vector store"""
#     try:
#         # Log initial vector count
#         initial_count = get_total_vectors()
#         logger.info(f"Current vector count before addition: {initial_count}")
#         # Get vector store
#         vector_store = get_vector_store()

#         # Check for duplicates
#         is_duplicate = await check_duplicate_tool(vector_store, tool)
#         if is_duplicate:
#             logger.info(f"Skipping duplicate tool: {tool.name} (ID: {tool.tool_id})")
#             return ToolResponse(
#                 id="duplicate_skipped",
#                 tool=tool,
#                 status="skipped"
#             )
#         # Generate unique ID
#         tool_id = str(uuid4())
        
#         # Format tool data
#         tool_text = format_tool_for_indexing(tool, tool_id)

#         # Add document to vector store using model_dump instead of dict
#         metadata = {
#             "id": tool_id,
#             "tool_id": tool.tool_id,
#             "name": tool.name,
#             **tool.model_dump()
#         }
        
#         vector_store.add_texts(
#             texts=[tool_text],
#             metadatas=[metadata],
#             ids=[tool_id]
#         )
#         # Log new vector count
#         new_count = get_total_vectors()
#         logger.info(f"Vector count after addition: {new_count}")
#         logger.info(f"Added tool: {tool.name} (Tool ID: {tool.tool_id}, Vector ID: {tool_id})")
        
#         return ToolResponse(id=tool_id, tool=tool, status="added")
    
#     except Exception as e:
#         print(f"Error in add_tool: {str(e)}")
#         logger.error(f"Error in add_tool: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to add tool: {str(e)}"
#         )
    
# @app.post("/query", response_model=QueryResponse)
# async def query_tools(request: QueryRequest):
#     """Query tools based on user input using MMR retrieval and a RAG chain."""
#     try:
#         logger.info(f"Processing query: {request.query}")

#         # Get the vector store
#         vector_store = get_vector_store()

#         # Log total vectors for reference (optional)
#         total_vectors = get_total_vectors()
#         logger.info(f"Total vectors in store: {total_vectors}")

#         # Create retriever with MMR
#         retriever = vector_store.as_retriever(
#             search_type="mmr",  # Use MMR for diversity
#             search_kwargs={
#                 'k': 5,         # Final number of results
#                 'fetch_k': 8,   # Number of initial candidates
#                 'lambda_mult': 0.2  # Balance between relevance and diversity
#                 # 'filter': {}   # Optional: add metadata-based filters
#             }
#         )

#         # Create and execute the RAG chain with MMR retriever
#         rag_chain = create_rag_chain(retriever)
#         response = rag_chain.invoke(request.query)

#         # Return the final chain output
#         return QueryResponse(response=response)

#     except Exception as e:
#         logger.error(f"Error in query_tools: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to process query: {str(e)}"
#         )


# # @app.post("/query", response_model=QueryResponse)
# # async def query_tools(request: QueryRequest):
# #     """Query tools based on user input"""
# #     try:
# #         logger.info(f"Processing query: {request.query}")
# #         vector_store = get_vector_store()

# #         # Get total vectors before search
# #         total_vectors = get_total_vectors()
# #         logger.info(f"Total vectors in store: {total_vectors}")

# #         # Perform similarity search to get relevant documents
# #         search_results = vector_store.similarity_search(
# #             request.query,
# #             k=5  # Get top 3 matches
# #         )
        
# #         # Log retrieved documents
# #         logger.info(f"Retrieved {len(search_results)} relevant documents")
# #         for i, doc in enumerate(search_results, 1):
# #             metadata = doc.metadata
# #             logger.info(f"Match {i}:")
# #             logger.info(f"  Tool: {metadata.get('name')}")
# #             logger.info(f"  Tool ID: {metadata.get('tool_id')}")
# #             logger.info(f"  Vector ID: {metadata.get('id')}")
# #             logger.info(f"  Score: {getattr(doc, 'score', 'N/A')}")

# #         # # Create and execute RAG chain
# #         # retriever = vector_store.as_retriever(
# #         #     search_type="similarity",
# #         #     search_kwargs={"k": 3}
# #         # )
# #                 # Create retriever with hybrid search
# #         retriever = vector_store.as_retriever(
# #             search_type="mmr",  # Use MMR for diversity
# #             search_kwargs={
# #                 'k': 5,  # Final number of results
# #                 'fetch_k': 8,  # Number of initial candidates
# #                 'lambda_mult': 0.2  # Balance between relevance and diversity
# #                 # 'filter': {  # Optional: Add filters based on query content
# #                 #     # You can add dynamic filters here based on query analysis
# #                 # }
# #             }
# #         )
# #         # Create and execute RAG chain
# #         rag_chain = create_rag_chain(retriever)
# #         response = rag_chain.invoke(request.query)
        
# #         return QueryResponse(response=response)
    
# #     except Exception as e:
# #         print(f"Error in query_tools: {str(e)}")
# #         logger.error(f"Error in query_tools: {str(e)}")
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Failed to process query: {str(e)}"
# #         )

# @app.delete("/delete-tool/{tool_id}", response_model=DeleteResponse)
# async def delete_tool(tool_id: str, tool: Tool):
#     """Delete a tool by ID"""
#     try:
#         # Log initial count
#         initial_count = get_total_vectors()
#         logger.info(f"Current vector count before deletion: {initial_count}")
#         # Get vector store
#         vector_store = get_vector_store()
        
#         # Get tool name before deletion
#         search_results = vector_store.similarity_search(
#             "",
#             k=1,
#             filter={"id": tool_id}
#         )
        
#         if not search_results:
#             raise HTTPException(status_code=404, detail="Tool not found")
        
#         tool_name = search_results[0].metadata.get("name", "Unknown tool")
        
#         # Delete the vector
#         index = get_or_create_index()
#         index.delete(ids=[tool_id])
#         # Log new count
#         new_count = get_total_vectors()
#         logger.info(f"Vector count after deletion: {new_count}")
#         logger.info(f"Deleted tool: {tool_name} (Tool ID: {tool.tool_id}, Vector ID: {tool_id})")
        
#         return DeleteResponse(
#             success=True,
#             deleted_tool=tool_name
#         )
    
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         print(f"Error in delete_tool: {str(e)}")
#         logger.error(f"Error in delete_tool: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to delete tool: {str(e)}"
#         )
    
# # Add endpoint to get vector store stats
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
        
#         # Query all vectors (empty query to get all)
#         results = vector_store.similarity_search(
#             "",
#             k=total_vectors  # Get all vectors
#         )
        
#         # Extract vector details
#         vectors_info = []
#         for doc in results:
#             vectors_info.append({
#                 "name": doc.metadata.get("name", "N/A"),
#                 "tool_id": doc.metadata.get("tool_id", "N/A"),
#                 "vector_id": doc.metadata.get("id", "N/A"),
#                 "description": doc.metadata.get("description", "N/A"),
#                 "categories": doc.metadata.get("categories", "N/A"),
#                 "pricing": doc.metadata.get("pricing", "N/A")
#             })
        
#         # Create complete stats dictionary
#         stats_dict = {
#             "total_vectors": total_vectors,
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
    
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# import os
# from datetime import datetime
# from typing import List, Optional, Dict, Any
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# from pinecone import Pinecone
# from langchain_community.vectorstores import Pinecone as LangchainPinecone
# from langchain_ollama import OllamaEmbeddings, ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from uuid import uuid4
# import json
# from dotenv import load_dotenv
# from fastapi.middleware.cors import CORSMiddleware

# # Load environment variables
# load_dotenv()

# # Verify environment variables
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# if not PINECONE_API_KEY:
#     raise ValueError("PINECONE_API_KEY environment variable is not set")

# # Initialize FastAPI app
# app = FastAPI(title="AI Tool Search API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize Pinecone client
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Constants
# INDEX_NAME = "ai-tool-search"
# DIMENSION = 768  # for nomic-embed-text

# # Pydantic models
# class Tool(BaseModel):
#     name: str
#     tool_id: str
#     description: str
#     pros: List[str]
#     cons: List[str]
#     categories: str
#     usage: str
#     unique_features: str
#     pricing: str

# class ToolResponse(BaseModel):
#     id: str
#     tool: Tool
#     status: str = "added"

# class QueryRequest(BaseModel):
#     query: str

# class QueryResponse(BaseModel):
#     response: str

# class DeleteResponse(BaseModel):
#     success: bool
#     deleted_tool: str

# def create_rag_chain(retriever):
#     """Create RAG chain with custom prompt"""
#     prompt = """You are an AI assistant specialized in recommending AI tools based on user needs.
    
#     User Query: {question}
    
#     Available Tools: {context}
    
#     Please analyze the query and provide relevant tool recommendations. Include:
#     1. Tool name and brief description
#     2. Why it's relevant to the query
#     3. Key features and pricing information
#     4. Any limitations or considerations
    
#     Format the response in a clear, concise manner."""

#     # Initialize the chat model
#     model = ChatOllama(
#         model="deepseek-r1:1.5b",
#         base_url="http://localhost:11434",
#         temperature=0.7
#     )

#     # Create prompt template
#     prompt_template = ChatPromptTemplate.from_template(prompt)
    
#     # Build the RAG chain
#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt_template
#         | model
#         | StrOutputParser()
#     )
    
#     return rag_chain

# def format_docs(docs):
#     """Format retrieved documents for the prompt"""
#     return "\n\n".join(doc.page_content for doc in docs)

# # Vector store initialization
# def get_or_create_index():
#     """Create Pinecone index if it doesn't exist"""
#     try:
#         # Check if index exists
#         indexes = pc.list_indexes()
#         index_names = [index.name for index in indexes]
        
#         if INDEX_NAME not in index_names:
#             print(f"Creating new index: {INDEX_NAME}")
#             pc.create_index(
#                 name=INDEX_NAME,
#                 dimension=DIMENSION,
#                 metric="cosine"
#             )
#             print(f"Index {INDEX_NAME} created successfully")
        
#         return pc.Index(INDEX_NAME)
#     except Exception as e:
#         print(f"Error in get_or_create_index: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to initialize Pinecone index: {str(e)}"
#         )

# def get_vector_store():
#     """Initialize or return existing vector store"""
#     try:
#         embeddings = OllamaEmbeddings(
#             model='nomic-embed-text',
#             base_url="http://localhost:11434"
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

# def format_tool_for_indexing(tool: Tool, tool_id: str) -> str:
#     """Format tool data for embedding"""
#     return (
#         f"Tool ID: {tool_id}\n"
#         f"Name: {tool.name}\n"
#         f"Description: {tool.description}\n"
#         f"Pros: {', '.join(tool.pros)}\n"
#         f"Cons: {', '.join(tool.cons)}\n"
#         f"Categories: {tool.categories}\n"
#         f"Usage: {tool.usage}\n"
#         f"Unique Features: {tool.unique_features}\n"
#         f"Pricing: {tool.pricing}"
#     )

# # API endpoints
# @app.get("/")
# async def root():
#     """Root endpoint to verify API is running"""
#     return {
#         "status": "active",
#         "message": "AI Tool Search API is running",
#         "version": "1.0"
#     }

# @app.get("/health")
# async def health_check():
#     """Check if API and Pinecone connection are healthy"""
#     try:
#         # Verify Pinecone connection
#         _ = get_or_create_index()
#         return {
#             "status": "healthy",
#             "api_version": "1.0",
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Service unhealthy: {str(e)}"
#         )
    
# async def check_duplicate_tool(vector_store, tool: Tool) -> bool:
#     """
#     Check if a tool with the same name and tool_id already exists.
#     Returns True if duplicate found, False otherwise.
#     """
#     try:
#         # Search for tools with the same name
#         results = vector_store.similarity_search(
#             tool.name,
#             k=5,  # Get a few results to check through
#             filter={"name": tool.name}
#         )
        
#         # Check if any result matches both name and tool_id
#         for result in results:
#             metadata = result.metadata
#             if (metadata.get("name") == tool.name and 
#                 metadata.get("tool_id") == tool.tool_id):
#                 return True
                
#         return False
    
#     except Exception as e:
#         print(f"Error checking duplicates: {str(e)}")
#         return False

# @app.post("/add-tool", response_model=ToolResponse)
# async def add_tool(tool: Tool):
#     """Add a new tool to the vector store"""
#     try:
        
#         # Get vector store
#         vector_store = get_vector_store()

#         # Check for duplicates
#         is_duplicate = await check_duplicate_tool(vector_store, tool)
#         if is_duplicate:
#             return ToolResponse(
#                 id="duplicate_skipped",
#                 tool=tool,
#                 status="skipped"
#             )
#         # Generate unique ID
#         tool_id = str(uuid4())
        
#         # Format tool data
#         tool_text = format_tool_for_indexing(tool, tool_id)

#         # Add document to vector store using model_dump instead of dict
#         metadata = {
#             "id": tool_id,
#             "tool_id": tool.tool_id,
#             "name": tool.name,
#             **tool.model_dump()
#         }
        
#         vector_store.add_texts(
#             texts=[tool_text],
#             metadatas=[metadata],
#             ids=[tool_id]
#         )
        
#         return ToolResponse(id=tool_id, tool=tool)
    
#     except Exception as e:
#         print(f"Error in add_tool: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to add tool: {str(e)}"
#         )

# @app.post("/query", response_model=QueryResponse)
# async def query_tools(request: QueryRequest):
#     """Query tools based on user input"""
#     try:
#         vector_store = get_vector_store()
#         retriever = vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 3}
#         )
        
#         # Create and execute RAG chain
#         rag_chain = create_rag_chain(retriever)
#         response = rag_chain.invoke(request.query)
        
#         return QueryResponse(response=response)
    
#     except Exception as e:
#         print(f"Error in query_tools: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to process query: {str(e)}"
#         )

# @app.delete("/delete-tool/{tool_id}", response_model=DeleteResponse)
# async def delete_tool(tool_id: str):
#     """Delete a tool by ID"""
#     try:
#         # Get vector store
#         vector_store = get_vector_store()
        
#         # Get tool name before deletion
#         search_results = vector_store.similarity_search(
#             "",
#             k=1,
#             filter={"id": tool_id}
#         )
        
#         if not search_results:
#             raise HTTPException(status_code=404, detail="Tool not found")
        
#         tool_name = search_results[0].metadata.get("name", "Unknown tool")
        
#         # Delete the vector
#         index = get_or_create_index()
#         index.delete(ids=[tool_id])
        
#         return DeleteResponse(
#             success=True,
#             deleted_tool=tool_name
#         )
    
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         print(f"Error in delete_tool: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to delete tool: {str(e)}"
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)