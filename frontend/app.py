import streamlit as st
import requests
import json
import pandas as pd
from io import StringIO
import time
def remove_html_tags(text):
    """Remove HTML tags from a string."""
    import re
    if not text or not isinstance(text, str):
        return text
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Function to normalize JSON response formats
def normalize_json_response(result_data):
    """
    Normalize different JSON response formats to a consistent structure
    with a "tools" array containing objects
    """
    # If the response already has the expected "tools" array format, return as is
    if "tools" in result_data and isinstance(result_data["tools"], list):
        return result_data
        
    # Handle parallel arrays format
    if "tool_id" in result_data and isinstance(result_data["tool_id"], list):
        # Create a normalized structure
        normalized = {"tools": []}
        
        # Get all arrays
        tool_ids = result_data.get("tool_id", [])
        descriptions = result_data.get("description", [])
        relevances = result_data.get("relevance", [])
        
        # Determine how many tools we have
        num_tools = len(tool_ids)
        
        # Build the tools array
        for i in range(num_tools):
            # Create tool object with all available properties
            tool = {
                "id": tool_ids[i] if i < len(tool_ids) else f"unknown-{i}"
            }
            
            # Set name to id if no separate name field exists
            tool["name"] = tool_ids[i] if i < len(tool_ids) else f"Tool {i+1}"
            
            # Add description if available
            if i < len(descriptions):
                tool["description"] = descriptions[i]
                
            # Add relevance if available
            if i < len(relevances):
                tool["relevance"] = relevances[i]
                
            normalized["tools"].append(tool)
            
        return normalized
    
    # If we can't normalize, return empty result
    return {"tools": []}

# Set page configuration
st.set_page_config(
    page_title="AI Tool Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-bottom: 1rem;
    }
    .search-results {
        color: #FF5252;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .tool-result {
        background-color: #1A1E23;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.3rem;
    }
    .tool-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #4FC3F7;
        margin-bottom: 0.5rem;
    }
    .tool-id {
        font-size: 0.9rem;
        color: #B0BEC5;
        background-color: #263238;
        padding: 0.2rem 0.5rem;
        border-radius: 0.2rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .tool-description {
        color: #E0E0E0;
        font-size: 1rem;
    }
    .tool-relevance {
        color: #E0E0E0;
        font-size: 0.95rem;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    .relevance-label {
        color: #FF9800;
        font-weight: 500;
    }
    .tools-comparison {
        background-color: #0D47A1;
        color: white;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .comparison-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #90CAF9;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .success-banner {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: #d4edda;
        color: #155724;
        margin-bottom: 1rem;
    }
    .error-banner {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: #f8d7da;
        color: #721c24;
        margin-bottom: 1rem;
    }
    .tool-card {
        padding: 1rem;
        border-radius: 0.3rem;
        background-color: white;
        border-left: 4px solid #1E88E5;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://backend:8000"
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'fetched_tool' not in st.session_state:
    st.session_state.fetched_tool = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_key = st.text_input("Pinecone API Key (for admin functions)", 
                                          value=st.session_state.api_key, 
                                          type="password")
    
    # Model selection
    st.divider()
    st.subheader("Model Selection")
    
    # Initialize model choice if not in session state
    if 'model_choice' not in st.session_state:
        st.session_state.model_choice = "DEV_MODEL"
    
    # Radio button for model selection
    st.session_state.model_choice = st.radio(
        "Select Model Environment Variable:",
        options=["DEV_MODEL", "PROD_MODEL"],
        index=0 if st.session_state.model_choice == "DEV_MODEL" else 1,
        horizontal=True
    )
    
    # Display current model information
    if st.session_state.model_choice == "DEV_MODEL":
        st.info("Using DEV_MODEL from .env file (deepseek-r1:1.5b)")
    else:
        st.info("Using PROD_MODEL from .env file (deepseek-r1:7b)")
        
    st.caption("Note: This will tell the backend which environment variable to use for the model selection.")
    
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This interface allows you to interact with the AI Tool Search API.
    
    You can:
    - Add new AI tools (single or bulk)
    - Search for tools based on queries
    - Update existing tools
    - Delete tools
    - View statistics
    """)
    
    st.divider()
    if st.button("Check API Health"):
        try:
            # First try without sending any custom headers
            response = requests.get(f"{st.session_state.api_url}/health")
            if response.status_code == 200:
                st.success("API is healthy! ✅")
                st.json(response.json())
                
                # Display model choice that would be sent
                st.info(f"Selected model: {st.session_state.model_choice}")
                st.caption("Note: Model choice will be applied to other API calls but not used for health check.")
            else:
                st.error(f"API returned status code: {response.status_code}")
                # Try to parse and display the detailed error message
                try:
                    error_detail = response.json().get("detail", response.text)
                    st.error(f"Error detail: {error_detail}")
                except:
                    st.text(f"Response body: {response.text}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Main content
st.markdown('<div class="main-header">AI Tool Search</div>', unsafe_allow_html=True)

# Create tabs for different functionalities
tabs = st.tabs(["🔍 Search", "➕ Add Tools", "🔄 Update Tools", "🗑️ Delete Tools", "📊 Statistics"])

# 1. SEARCH TAB
with tabs[0]:
    st.markdown('<div class="subheader">Search AI Tools</div>', unsafe_allow_html=True)
    
    query = st.text_input("Enter your search query:", 
                          placeholder="e.g., code generation tools for JavaScript",
                          value=st.session_state.last_query)
    
    if st.button("Search", type="primary", key="search_button"):
        st.session_state.last_query = query
        with st.spinner("Searching..."):
            try:
                # Prepare headers with model choice
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json={"query": query},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.session_state.last_result = result
                    
                    # Try to parse the response as JSON
                    try:
                        result_data = json.loads(result["response"])
                        is_valid_json = True
                        
                        # Normalize the JSON structure to handle different formats
                        result_data = normalize_json_response(result_data)
                    except json.JSONDecodeError:
                        # Not valid JSON, will display as text
                        is_valid_json = False
                    
                    # Display the search results
                    st.markdown('<div class="search-results">Search Results</div>', unsafe_allow_html=True)
                    
                    if is_valid_json and "tools" in result_data and len(result_data["tools"]) > 0:
                        # Valid JSON format with tools
                        tool_count = len(result_data["tools"])
                        if tool_count > 1:
                            st.markdown(f"Found {tool_count} tools related to your query, ranked by relevance:")
                        else:
                            st.markdown("Found 1 tool matching your query:")
                        
                        # Create a container for all the results
                        results_container = st.container()
                        
                        # Display each tool with proper formatting
                        for i, tool in enumerate(result_data["tools"]):
                            with results_container:
                                # Create a container for this tool
                                with st.container():
                                    # Tool number and name with proper styling
                                    st.markdown(f"""
                                    <div style="font-size: 1.3rem; font-weight: bold; color: #4FC3F7; margin-bottom: 0.5rem;">
                                        {i+1}. {tool.get('name', 'No Name')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Tool ID with proper styling
                                    st.markdown(f"""
                                    <div style="font-size: 0.9rem; color: #B0BEC5; background-color: #263238; 
                                        padding: 0.2rem 0.5rem; border-radius: 0.2rem; display: inline-block; margin-bottom: 0.5rem;">
                                        ID: {tool.get('id', 'No ID')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Description
                                    st.markdown(f"""
                                    <div style="color: #E0E0E0; font-size: 1rem; margin-bottom: 0.8rem;">
                                        {tool.get('description', 'No description available.')}
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Add a subtle divider
                                    st.markdown("<hr style='margin: 0.8rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
                                    
                                    # Relevance with proper styling
                                    if "relevance" in tool and tool["relevance"]:
                                        st.markdown(f"""
                                        <div style="margin-top: 0.5rem;">
                                            <span style="color: #FF9800; font-weight: 500;">Relevance:</span> 
                                            <span style="color: #E0E0E0;">{tool.get('relevance', 'This tool matches your search criteria.')}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # Add proper spacing between tools
                                if i < len(result_data["tools"]) - 1:
                                    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
                        
                        # Add significant spacing and visual separator before comparison section
                        if tool_count > 1:
                            # Add extra spacing
                            st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                            
                            # Add visual separator
                            st.markdown("<hr style='border-top: 1px solid rgba(255,255,255,0.1); margin: 0;'>", unsafe_allow_html=True)
                            
                            # Add space after separator
                            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                            
                            # Display better styled comparison header
                            st.markdown("""
                            <div style="font-size: 1.5rem; font-weight: 600; color: #4FC3F7; margin-bottom: 1rem;">
                                How These Tools Compare
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Generate comparison text based on the query
                            comparison_text = "These tools offer different approaches to "
                            
                            if "code" in query.lower() or "programming" in query.lower() or "development" in query.lower():
                                comparison_text += "code generation and development assistance. "
                            elif "image" in query.lower() or "visual" in query.lower() or "picture" in query.lower():
                                comparison_text += "image generation and visual content creation. "
                            elif "text" in query.lower() or "write" in query.lower() or "content" in query.lower():
                                comparison_text += "text generation and content writing. "
                            elif "music" in query.lower() or "audio" in query.lower() or "sound" in query.lower():
                                comparison_text += "music generation and audio production. "
                            elif "ai" in query.lower() or "assistant" in query.lower() or "chat" in query.lower():
                                comparison_text += "AI assistant capabilities and conversational abilities. "
                            elif "education" in query.lower() or "learn" in query.lower() or "study" in query.lower() or "tutor" in query.lower():
                                comparison_text += "educational support and learning assistance. "
                            else:
                                comparison_text += f"addressing your needs for '{query}'. "
                                
                            comparison_text += "Consider your specific use case, pricing, and feature requirements when choosing between them."
                            
                            # Display comparison text in styled container
                            st.markdown(f"""
                            <div style="background-color: rgba(13, 71, 161, 0.3); color: #E0E0E0; padding: 1rem; 
                                border-radius: 0.5rem; border-left: 4px solid #1976D2; margin-top: 0.5rem;">
                                {comparison_text}
                            </div>
                            """, unsafe_allow_html=True)
                    elif not is_valid_json:
                        # Not valid JSON, display as plain text
                        st.markdown("### Response from the model")
                        st.markdown("The model provided a text response instead of structured tool recommendations:")
                        st.info(result["response"])
                    else:
                        # Valid JSON but no tools
                        st.info("No matching tools found. Try a different search query or add more tools to the database.")
                    
                    # Show raw response in an expander
                    with st.expander("View Raw Response"):
                        if is_valid_json:
                            st.json(result_data)
                        else:
                            st.text(result["response"])
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# 2. ADD TOOLS TAB
with tabs[1]:
    st.markdown('<div class="subheader">Add New AI Tools</div>', unsafe_allow_html=True)
    
    add_option = st.radio("Choose an option:", ["Add Single Tool", "Bulk Upload"])
    
    if add_option == "Add Single Tool":
        with st.form(key="add_tool_form"):
            st.markdown("### Tool Information")
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Tool Name*", help="The official name of the AI tool")
                tool_id = st.text_input("Tool ID*", help="A unique identifier for this tool (e.g., tool-name-001)")
                categories = st.text_input("Categories", help="Comma-separated list of categories (e.g., Text Generation, Code Assistance)")
                pricing = st.text_input("Pricing", help="Information about pricing tiers (e.g., Free, Freemium, $10/month)")
            
            with col2:
                description = st.text_area("Description*", help="A brief description of what the tool does")
                
                pros_input = st.text_area("Pros (one per line)", 
                                     help="List the advantages of this tool, one per line")
                cons_input = st.text_area("Cons (one per line)", 
                                     help="List limitations or disadvantages, one per line")
            
            st.markdown("### Additional Details")
            usage = st.text_area("Usage Examples", help="How this tool can be used effectively")
            unique_features = st.text_area("Unique Features", help="What makes this tool stand out from others")
            
            submit_button = st.form_submit_button("Add Tool", type="primary")
            
            if submit_button:
                if not name or not tool_id or not description:
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    # Process pros and cons lists
                    pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
                    cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
                    # Create the tool object
                    tool = {
                        "name": name,
                        "tool_id": tool_id,
                        "description": description,
                        "pros": pros,
                        "cons": cons,
                        "categories": categories,
                        "usage": usage,
                        "unique_features": unique_features,
                        "pricing": pricing
                    }
                    
                    # Create the request payload
                    payload = {"tools": [tool]}
                    
                    try:
                        with st.spinner("Adding tool..."):
                            # Prepare headers with model choice
                            headers = {}
                            headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
                            response = requests.post(
                                f"{st.session_state.api_url}/add-tools",
                                json=payload,
                                headers=headers
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Tool '{name}' added successfully!")
                            
                            with st.expander("View Details"):
                                st.json(result)
                        else:
                            st.error(f"Error: API returned status code {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")
    
    else:  # Bulk Upload
        st.markdown("### Bulk Upload Tools")
        
        st.info("""
        Upload a JSON file with multiple tools. The file should have this structure:
        ```json
        {
            "tools": [
                {
                    "name": "Tool Name",
                    "tool_id": "tool-name-001",
                    "description": "Tool description",
                    "pros": ["Pro 1", "Pro 2"],
                    "cons": ["Con 1", "Con 2"],
                    "categories": "Category1, Category2",
                    "usage": "Usage examples",
                    "unique_features": "What makes this tool unique",
                    "pricing": "Pricing information"
                },
                // More tools...
            ]
        }
        ```
        """)
        
        uploaded_file = st.file_uploader("Upload JSON file", type="json")
        
        if uploaded_file is not None:
            try:
                # Load JSON data
                data = json.load(uploaded_file)
                
                # Preview the data
                with st.expander("Preview Upload Data"):
                    st.write(f"Found {len(data.get('tools', []))} tools in the uploaded file.")
                    st.json(data)
                
                if st.button("Process Bulk Upload", type="primary"):
                    with st.spinner("Uploading tools..."):
                        try:
                            # Prepare headers with model choice
                            headers = {}
                            headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
                            response = requests.post(
                                f"{st.session_state.api_url}/add-tools",
                                json=data,
                                headers=headers
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                st.success(f"Successfully processed {len(result['results'])} tools!")
                                
                                # Show results in a table
                                results_data = []
                                for item in result["results"]:
                                    results_data.append({
                                        "Name": item["tool"]["name"],
                                        "ID": item["tool"]["tool_id"],
                                        "Status": item["status"]
                                    })
                                
                                results_df = pd.DataFrame(results_data)
                                st.dataframe(results_df)
                                
                                with st.expander("View Full Response"):
                                    st.json(result)
                            else:
                                st.error(f"Error: API returned status code {response.status_code}")
                                st.text(response.text)
                        except Exception as e:
                            st.error(f"Error connecting to API: {str(e)}")
                
            except json.JSONDecodeError:
                st.error("Invalid JSON file. Please check the format.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# 3. UPDATE TOOLS TAB
with tabs[2]:
    st.markdown('<div class="subheader">Update Existing Tools</div>', unsafe_allow_html=True)
    
    # Step 1: Input tool ID
    tool_id_to_update = st.text_input("Enter Tool ID to update:", 
                                      key="update_tool_id_input",
                                      help="Enter the unique identifier of the tool you want to update")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        fetch_button = st.button("Fetch Tool", key="fetch_tool_button")
    
    if fetch_button and tool_id_to_update:
        with st.spinner("Fetching tool data..."):
            # In a real implementation, you would have an endpoint to fetch a single tool
            # For now, we'll simulate fetching by querying with the tool ID
            try:
                # Prepare headers with model choice
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json={"query": f"tool_id:{tool_id_to_update}"},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    try:
                        result_data = json.loads(result["response"])
                        
                        if "tools" in result_data and len(result_data["tools"]) > 0:
                            # Find the matching tool
                            matching_tool = None
                            for tool in result_data["tools"]:
                                if tool.get("id") == tool_id_to_update:
                                    matching_tool = tool
                                    break
                            
                            if matching_tool:
                                st.session_state.fetched_tool = matching_tool
                                st.success(f"Found tool: {matching_tool.get('name', 'Unnamed Tool')}")
                            else:
                                st.warning(f"Tool with ID '{tool_id_to_update}' not found in search results.")
                                st.session_state.fetched_tool = None
                        else:
                            st.warning(f"No tool found with ID: {tool_id_to_update}")
                            st.session_state.fetched_tool = None
                    except json.JSONDecodeError:
                        st.error("Failed to parse response JSON.")
                        st.session_state.fetched_tool = None
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.session_state.fetched_tool = None
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.session_state.fetched_tool = None
    
    # Step 2: If a tool was fetched, show the update form
    if st.session_state.fetched_tool:
        with st.form(key="update_tool_form"):
            st.markdown("### Update Tool Information")
            col1, col2 = st.columns(2)
            
            # Pre-fill form with existing data
            tool = st.session_state.fetched_tool
            
            with col1:
                name = st.text_input("Tool Name*", 
                                    value=tool.get("name", ""),
                                    help="The official name of the AI tool")
                tool_id = st.text_input("Tool ID*", 
                                       value=tool.get("id", ""),
                                       help="A unique identifier for this tool",
                                       disabled=True)
                categories = st.text_input("Categories", 
                                         value=tool.get("categories", ""),
                                         help="Comma-separated list of categories")
                pricing = st.text_input("Pricing", 
                                      value=tool.get("pricing", ""),
                                      help="Information about pricing tiers")
            
            with col2:
                description = st.text_area("Description*", 
                                         value=tool.get("description", ""),
                                         help="A brief description of what the tool does")
                
                # Join pros and cons with newlines for the text area
                pros_text = "\n".join(tool.get("pros", []))
                cons_text = "\n".join(tool.get("cons", []))
                
                pros_input = st.text_area("Pros (one per line)", 
                                        value=pros_text,
                                        help="List the advantages of this tool, one per line")
                cons_input = st.text_area("Cons (one per line)", 
                                        value=cons_text,
                                        help="List limitations or disadvantages, one per line")
            
            st.markdown("### Additional Details")
            usage = st.text_area("Usage Examples", 
                               value=tool.get("usage", ""),
                               help="How this tool can be used effectively")
            unique_features = st.text_area("Unique Features", 
                                         value=tool.get("unique_features", ""),
                                         help="What makes this tool stand out from others")
            
            update_button = st.form_submit_button("Update Tool", type="primary")
            
            if update_button:
                if not name or not tool_id or not description:
                    st.error("Please fill in all required fields (marked with *).")
                else:
                    # Process pros and cons lists
                    pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
                    cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
                    # Create the updated tool object
                    updated_tool = {
                        "name": name,
                        "tool_id": tool_id,
                        "description": description,
                        "pros": pros,
                        "cons": cons,
                        "categories": categories,
                        "usage": usage,
                        "unique_features": unique_features,
                        "pricing": pricing
                    }
                    
                    # Create the request payload
                    payload = {"tools": [updated_tool]}
                    
                    try:
                        with st.spinner("Updating tool..."):
                            # Prepare headers with model choice
                            headers = {}
                            headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
                            response = requests.put(
                                f"{st.session_state.api_url}/update-tools",
                                json=payload,
                                headers=headers
                            )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"Tool '{name}' updated successfully!")
                            
                            with st.expander("View Details"):
                                st.json(result)
                                
                            # Reset the fetched tool to show the form is complete
                            st.session_state.fetched_tool = None
                            st.rerun()
                        else:
                            st.error(f"Error: API returned status code {response.status_code}")
                            st.text(response.text)
                    except Exception as e:
                        st.error(f"Error connecting to API: {str(e)}")

# 4. DELETE TOOLS TAB
with tabs[3]:
    st.markdown('<div class="subheader">Delete Tools</div>', unsafe_allow_html=True)
    
    st.warning("⚠️ Warning: Deletion is permanent and cannot be undone.")
    
    tool_id_to_delete = st.text_input("Enter Tool ID to delete:", 
                                     key="delete_tool_id_input",
                                     help="Enter the unique identifier of the tool you want to delete")
    
    confirm_delete = st.checkbox("I confirm that I want to delete this tool permanently")
    
    if st.button("Delete Tool", type="primary", disabled=not confirm_delete or not tool_id_to_delete):
        with st.spinner("Deleting tool..."):
            try:
                # Prepare headers with model choice
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.delete(
                    f"{st.session_state.api_url}/delete-tool/{tool_id_to_delete}",
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Tool '{result.get('deleted_tool', 'unknown')}' was deleted successfully!")
                    else:
                        st.error("Deletion failed.")
                elif response.status_code == 404:
                    st.error(f"Tool with ID '{tool_id_to_delete}' not found.")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
    
    st.divider()
    
    st.markdown("### Clear Entire Index")
    st.error("⚠️ DANGER: This will delete ALL tools from the index. This action cannot be undone.")
    
    if not st.session_state.api_key:
        st.info("Please enter your Pinecone API Key in the sidebar to use this function.")
    
    confirm_clear = st.checkbox("I understand that this will delete ALL data from the index permanently")
    
    if st.button("Clear Index", type="primary", disabled=not confirm_clear or not st.session_state.api_key):
        with st.spinner("Clearing index..."):
            try:
                # Prepare headers with model choice
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.delete(
                    f"{st.session_state.api_url}/clear-index",
                    json={"api_key": st.session_state.api_key},
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        st.success(f"Index cleared successfully! Deleted {result.get('deleted_count', 0)} tools.")
                    else:
                        st.error("Operation failed.")
                elif response.status_code == 401:
                    st.error("Unauthorized: Invalid API Key.")
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")

# 5. STATISTICS TAB
with tabs[4]:
    st.markdown('<div class="subheader">Index Statistics</div>', unsafe_allow_html=True)
    
    if st.button("Refresh Statistics", key="refresh_stats"):
        with st.spinner("Fetching statistics..."):
            try:
                # Prepare headers with model choice
                headers = {}
                headers["MODEL_CHOICE"] = st.session_state.model_choice
                
                response = requests.get(
                    f"{st.session_state.api_url}/stats",
                    headers=headers
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tools", stats.get("total_vectors", 0))
                    with col2:
                        st.metric("Vector Dimension", stats.get("dimension", "-"))
                    with col3:
                        st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
                    
                    # Display vector information
                    st.markdown("### Tools in Index")
                    if "vectors" in stats and len(stats["vectors"]) > 0:
                        # Convert to DataFrame for better display
                        vectors_df = pd.DataFrame(stats["vectors"])
                        
                        # Add category counts
                        if "categories" in vectors_df.columns:
                            # Extract categories and count occurrences
                            all_categories = []
                            for cats in vectors_df["categories"]:
                                if cats and cats != "N/A":
                                    categories_list = [c.strip() for c in cats.split(",")]
                                    all_categories.extend(categories_list)
                            
                            category_counts = pd.Series(all_categories).value_counts()
                            
                            # Show category distribution
                            st.markdown("### Category Distribution")
                            st.bar_chart(category_counts)
                        
                        # Show the main table
                        st.dataframe(vectors_df, use_container_width=True)
                    else:
                        st.info("No tools found in the index.")
                    
                    # Show raw JSON for detailed inspection
                    with st.expander("View Raw Statistics"):
                        st.json(stats)
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    st.text(response.text)
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
# import streamlit as st
# import requests
# import json
# import pandas as pd
# from io import StringIO
# import time

# # Set page configuration
# st.set_page_config(
#     page_title="AI Tool Search Interface",
#     page_icon="🔍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for styling
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1E88E5;
#         margin-bottom: 1rem;
#     }
#     .subheader {
#         font-size: 1.5rem;
#         font-weight: bold;
#         color: #333;
#         margin-bottom: 1rem;
#     }
#     .search-results {
#         color: #FF5252;
#         font-size: 1.8rem;
#         font-weight: 600;
#         margin-top: 1.5rem;
#         margin-bottom: 1rem;
#     }
#     .tool-result {
#         background-color: #1A1E23;
#         border-left: 4px solid #1E88E5;
#         padding: 1rem;
#         margin-bottom: 0.5rem;
#         border-radius: 0.3rem;
#     }
#     .tool-title {
#         font-size: 1.3rem;
#         font-weight: bold;
#         color: #4FC3F7;
#         margin-bottom: 0.5rem;
#     }
#     .tool-id {
#         font-size: 0.9rem;
#         color: #B0BEC5;
#         background-color: #263238;
#         padding: 0.2rem 0.5rem;
#         border-radius: 0.2rem;
#         display: inline-block;
#         margin-bottom: 0.5rem;
#     }
#     .tool-description {
#         color: #E0E0E0;
#         font-size: 1rem;
#     }
#     .tool-relevance {
#         color: #E0E0E0;
#         font-size: 0.95rem;
#         margin-top: 0.8rem;
#         padding-top: 0.8rem;
#         border-top: 1px solid rgba(255,255,255,0.1);
#     }
#     .relevance-label {
#         color: #FF9800;
#         font-weight: 500;
#     }
#     .tools-comparison {
#         background-color: #0D47A1;
#         color: white;
#         padding: 1rem;
#         border-radius: 0.3rem;
#         margin-top: 1.5rem;
#         margin-bottom: 1rem;
#     }
#     .comparison-title {
#         font-size: 1.2rem;
#         font-weight: bold;
#         margin-bottom: 0.5rem;
#         color: #90CAF9;
#     }
#     .card {
#         padding: 1.5rem;
#         border-radius: 0.5rem;
#         background-color: #f8f9fa;
#         box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#         margin-bottom: 1rem;
#     }
#     .success-banner {
#         padding: 1rem;
#         border-radius: 0.3rem;
#         background-color: #d4edda;
#         color: #155724;
#         margin-bottom: 1rem;
#     }
#     .error-banner {
#         padding: 1rem;
#         border-radius: 0.3rem;
#         background-color: #f8d7da;
#         color: #721c24;
#         margin-bottom: 1rem;
#     }
#     .tool-card {
#         padding: 1rem;
#         border-radius: 0.3rem;
#         background-color: white;
#         border-left: 4px solid #1E88E5;
#         margin-bottom: 1rem;
#         box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
#     }
#     .stTabs [data-baseweb="tab-list"] {
#         gap: 2rem;
#     }
#     .stTabs [data-baseweb="tab"] {
#         height: 3rem;
#         white-space: pre-wrap;
#         border-radius: 4px 4px 0 0;
#         gap: 1rem;
#         font-weight: 500;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state variables
# if 'api_url' not in st.session_state:
#     st.session_state.api_url = "http://localhost:8000"
# if 'api_key' not in st.session_state:
#     st.session_state.api_key = ""
# if 'last_query' not in st.session_state:
#     st.session_state.last_query = ""
# if 'last_result' not in st.session_state:
#     st.session_state.last_result = None
# if 'fetched_tool' not in st.session_state:
#     st.session_state.fetched_tool = None

# # Sidebar for configuration
# with st.sidebar:
#     st.title("⚙️ Configuration")
#     st.session_state.api_url = st.text_input("API URL", value=st.session_state.api_url)
#     st.session_state.api_key = st.text_input("Pinecone API Key (for admin functions)", 
#                                           value=st.session_state.api_key, 
#                                           type="password")
    
#     # Model selection
#     st.divider()
#     st.subheader("Model Selection")
    
#     # Initialize model choice if not in session state
#     if 'model_choice' not in st.session_state:
#         st.session_state.model_choice = "DEV_MODEL"
    
#     # Radio button for model selection
#     st.session_state.model_choice = st.radio(
#         "Select Model Environment Variable:",
#         options=["DEV_MODEL", "PROD_MODEL"],
#         index=0 if st.session_state.model_choice == "DEV_MODEL" else 1,
#         horizontal=True
#     )
    
#     # Display current model information
#     if st.session_state.model_choice == "DEV_MODEL":
#         st.info("Using DEV_MODEL from .env file (deepseek-r1:1.5b)")
#     else:
#         st.info("Using PROD_MODEL from .env file (deepseek-r1:7b)")
        
#     st.caption("Note: This will tell the backend which environment variable to use for the model selection.")
    
#     st.divider()
#     st.markdown("### About")
#     st.markdown("""
#     This interface allows you to interact with the AI Tool Search API.
    
#     You can:
#     - Add new AI tools (single or bulk)
#     - Search for tools based on queries
#     - Update existing tools
#     - Delete tools
#     - View statistics
#     """)
    
#     st.divider()
#     if st.button("Check API Health"):
#         try:
#             # First try without sending any custom headers
#             response = requests.get(f"{st.session_state.api_url}/health")
#             if response.status_code == 200:
#                 st.success("API is healthy! ✅")
#                 st.json(response.json())
                
#                 # Display model choice that would be sent
#                 st.info(f"Selected model: {st.session_state.model_choice}")
#                 st.caption("Note: Model choice will be applied to other API calls but not used for health check.")
#             else:
#                 st.error(f"API returned status code: {response.status_code}")
#                 # Try to parse and display the detailed error message
#                 try:
#                     error_detail = response.json().get("detail", response.text)
#                     st.error(f"Error detail: {error_detail}")
#                 except:
#                     st.text(f"Response body: {response.text}")
#         except Exception as e:
#             st.error(f"Error connecting to API: {str(e)}")

# # Main content
# st.markdown('<div class="main-header">AI Tool Search</div>', unsafe_allow_html=True)

# # Create tabs for different functionalities
# tabs = st.tabs(["🔍 Search", "➕ Add Tools", "🔄 Update Tools", "🗑️ Delete Tools", "📊 Statistics"])

# # 1. SEARCH TAB
# # 1. SEARCH TAB
# with tabs[0]:
#     st.markdown('<div class="subheader">Search AI Tools</div>', unsafe_allow_html=True)
    
#     query = st.text_input("Enter your search query:", 
#                           placeholder="e.g., code generation tools for JavaScript",
#                           value=st.session_state.last_query)
    
#     if st.button("Search", type="primary", key="search_button"):
#         st.session_state.last_query = query
#         with st.spinner("Searching..."):
#             try:
#                 # Prepare headers with model choice
#                 headers = {}
#                 headers["MODEL_CHOICE"] = st.session_state.model_choice
                
#                 response = requests.post(
#                     f"{st.session_state.api_url}/query",
#                     json={"query": query},
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     st.session_state.last_result = result
                    
#                     # Try to parse the response as JSON
#                     try:
#                         result_data = json.loads(result["response"])
#                         is_valid_json = True
#                     except json.JSONDecodeError:
#                         # Not valid JSON, will display as text
#                         is_valid_json = False
                    
#                     # Display the search results
#                     st.markdown('<div class="search-results">Search Results</div>', unsafe_allow_html=True)
                    
#                     if is_valid_json and "tools" in result_data and len(result_data["tools"]) > 0:
#                         # Valid JSON format with tools
#                         tool_count = len(result_data["tools"])
#                         if tool_count > 1:
#                             st.markdown(f"Found {tool_count} tools related to your query, ranked by relevance:")
#                         else:
#                             st.markdown("Found 1 tool matching your query:")
                        
#                         # Create a container for all the results
#                         results_container = st.container()
                        
#                         # Display each tool with its relevance information
#                         for i, tool in enumerate(result_data["tools"]):
#                             with results_container:
#                                 # Create the tool HTML without the relevance section first
#                                 tool_html = f"""
#                                 <div class="tool-result">
#                                     <div class="tool-title">{i+1}. {tool.get('name', 'No Name')}</div>
#                                     <div class="tool-id">ID: {tool.get('id', 'No ID')}</div>
#                                     <div class="tool-description">{tool.get('description', 'No description available.')}</div>
#                                 """
                                
#                                 # Add relevance section if available
#                                 if "relevance" in tool and tool["relevance"]:
#                                     tool_html += f"""
#                                     <div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);">
#                                         <span style="color: #FF9800; font-weight: 500;">Relevance:</span> 
#                                         {tool.get('relevance', 'This tool matches your search criteria.')}
#                                     </div>
#                                     """
                                
#                                 # Close the div and render the complete HTML
#                                 tool_html += "</div>"
#                                 st.markdown(tool_html, unsafe_allow_html=True)
                                
#                                 # Add spacing between results
#                                 if i < len(result_data["tools"]) - 1:
#                                     st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                        
#                         # Display tools relationship header if multiple tools
#                         if tool_count > 1:
#                             st.markdown("### How These Tools Compare")
                            
#                             # Generate comparison text based on the query
#                             comparison_text = "These tools offer different approaches to "
                            
#                             if "code" in query.lower() or "programming" in query.lower() or "development" in query.lower():
#                                 comparison_text += "code generation and development assistance. "
#                             elif "image" in query.lower() or "visual" in query.lower() or "picture" in query.lower():
#                                 comparison_text += "image generation and visual content creation. "
#                             elif "text" in query.lower() or "write" in query.lower() or "content" in query.lower():
#                                 comparison_text += "text generation and content writing. "
#                             elif "ai" in query.lower() or "assistant" in query.lower() or "chat" in query.lower():
#                                 comparison_text += "AI assistant capabilities and conversational abilities. "
#                             else:
#                                 comparison_text += f"addressing your needs for '{query}'. "
                                
#                             comparison_text += "Consider your specific use case, pricing, and feature requirements when choosing between them."
                            
#                             st.info(comparison_text)
#                     elif not is_valid_json:
#                         # Not valid JSON, display as plain text
#                         st.markdown("### Response from the model")
#                         st.markdown("The model provided a text response instead of structured tool recommendations:")
#                         st.info(result["response"])
#                     else:
#                         # Valid JSON but no tools
#                         st.info("No matching tools found. Try a different search query or add more tools to the database.")
                    
#                     # Show raw response in an expander
#                     with st.expander("View Raw Response"):
#                         if is_valid_json:
#                             st.json(result_data)
#                         else:
#                             st.text(result["response"])
#                 else:
#                     st.error(f"Error: API returned status code {response.status_code}")
#                     st.text(response.text)
#             except Exception as e:
#                 st.error(f"Error connecting to API: {str(e)}")

# # 2. ADD TOOLS TAB
# with tabs[1]:
#     st.markdown('<div class="subheader">Add New AI Tools</div>', unsafe_allow_html=True)
    
#     add_option = st.radio("Choose an option:", ["Add Single Tool", "Bulk Upload"])
    
#     if add_option == "Add Single Tool":
#         with st.form(key="add_tool_form"):
#             st.markdown("### Tool Information")
#             col1, col2 = st.columns(2)
            
#             with col1:
#                 name = st.text_input("Tool Name*", help="The official name of the AI tool")
#                 tool_id = st.text_input("Tool ID*", help="A unique identifier for this tool (e.g., tool-name-001)")
#                 categories = st.text_input("Categories", help="Comma-separated list of categories (e.g., Text Generation, Code Assistance)")
#                 pricing = st.text_input("Pricing", help="Information about pricing tiers (e.g., Free, Freemium, $10/month)")
            
#             with col2:
#                 description = st.text_area("Description*", help="A brief description of what the tool does")
                
#                 pros_input = st.text_area("Pros (one per line)", 
#                                      help="List the advantages of this tool, one per line")
#                 cons_input = st.text_area("Cons (one per line)", 
#                                      help="List limitations or disadvantages, one per line")
            
#             st.markdown("### Additional Details")
#             usage = st.text_area("Usage Examples", help="How this tool can be used effectively")
#             unique_features = st.text_area("Unique Features", help="What makes this tool stand out from others")
            
#             submit_button = st.form_submit_button("Add Tool", type="primary")
            
#             if submit_button:
#                 if not name or not tool_id or not description:
#                     st.error("Please fill in all required fields (marked with *).")
#                 else:
#                     # Process pros and cons lists
#                     pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
#                     cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
#                     # Create the tool object
#                     tool = {
#                         "name": name,
#                         "tool_id": tool_id,
#                         "description": description,
#                         "pros": pros,
#                         "cons": cons,
#                         "categories": categories,
#                         "usage": usage,
#                         "unique_features": unique_features,
#                         "pricing": pricing
#                     }
                    
#                     # Create the request payload
#                     payload = {"tools": [tool]}
                    
#                     try:
#                         with st.spinner("Adding tool..."):
#                             # Prepare headers with model choice
#                             headers = {}
#                             headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
#                             response = requests.post(
#                                 f"{st.session_state.api_url}/add-tools",
#                                 json=payload,
#                                 headers=headers
#                             )
                        
#                         if response.status_code == 200:
#                             result = response.json()
#                             st.success(f"Tool '{name}' added successfully!")
                            
#                             with st.expander("View Details"):
#                                 st.json(result)
#                         else:
#                             st.error(f"Error: API returned status code {response.status_code}")
#                             st.text(response.text)
#                     except Exception as e:
#                         st.error(f"Error connecting to API: {str(e)}")
    
#     else:  # Bulk Upload
#         st.markdown("### Bulk Upload Tools")
        
#         st.info("""
#         Upload a JSON file with multiple tools. The file should have this structure:
#         ```json
#         {
#             "tools": [
#                 {
#                     "name": "Tool Name",
#                     "tool_id": "tool-name-001",
#                     "description": "Tool description",
#                     "pros": ["Pro 1", "Pro 2"],
#                     "cons": ["Con 1", "Con 2"],
#                     "categories": "Category1, Category2",
#                     "usage": "Usage examples",
#                     "unique_features": "What makes this tool unique",
#                     "pricing": "Pricing information"
#                 },
#                 // More tools...
#             ]
#         }
#         ```
#         """)
        
#         uploaded_file = st.file_uploader("Upload JSON file", type="json")
        
#         if uploaded_file is not None:
#             try:
#                 # Load JSON data
#                 data = json.load(uploaded_file)
                
#                 # Preview the data
#                 with st.expander("Preview Upload Data"):
#                     st.write(f"Found {len(data.get('tools', []))} tools in the uploaded file.")
#                     st.json(data)
                
#                 if st.button("Process Bulk Upload", type="primary"):
#                     with st.spinner("Uploading tools..."):
#                         try:
#                             # Prepare headers with model choice
#                             headers = {}
#                             headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
#                             response = requests.post(
#                                 f"{st.session_state.api_url}/add-tools",
#                                 json=data,
#                                 headers=headers
#                             )
                            
#                             if response.status_code == 200:
#                                 result = response.json()
#                                 st.success(f"Successfully processed {len(result['results'])} tools!")
                                
#                                 # Show results in a table
#                                 results_data = []
#                                 for item in result["results"]:
#                                     results_data.append({
#                                         "Name": item["tool"]["name"],
#                                         "ID": item["tool"]["tool_id"],
#                                         "Status": item["status"]
#                                     })
                                
#                                 results_df = pd.DataFrame(results_data)
#                                 st.dataframe(results_df)
                                
#                                 with st.expander("View Full Response"):
#                                     st.json(result)
#                             else:
#                                 st.error(f"Error: API returned status code {response.status_code}")
#                                 st.text(response.text)
#                         except Exception as e:
#                             st.error(f"Error connecting to API: {str(e)}")
                
#             except json.JSONDecodeError:
#                 st.error("Invalid JSON file. Please check the format.")
#             except Exception as e:
#                 st.error(f"Error processing file: {str(e)}")

# # 3. UPDATE TOOLS TAB
# with tabs[2]:
#     st.markdown('<div class="subheader">Update Existing Tools</div>', unsafe_allow_html=True)
    
#     # Step 1: Input tool ID
#     tool_id_to_update = st.text_input("Enter Tool ID to update:", 
#                                       key="update_tool_id_input",
#                                       help="Enter the unique identifier of the tool you want to update")
    
#     col1, col2 = st.columns([1, 5])
#     with col1:
#         fetch_button = st.button("Fetch Tool", key="fetch_tool_button")
    
#     if fetch_button and tool_id_to_update:
#         with st.spinner("Fetching tool data..."):
#             # In a real implementation, you would have an endpoint to fetch a single tool
#             # For now, we'll simulate fetching by querying with the tool ID
#             try:
#                 # Prepare headers with model choice
#                 headers = {}
#                 headers["MODEL_CHOICE"] = st.session_state.model_choice
                
#                 response = requests.post(
#                     f"{st.session_state.api_url}/query",
#                     json={"query": f"tool_id:{tool_id_to_update}"},
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
                    
#                     try:
#                         result_data = json.loads(result["response"])
                        
#                         if "tools" in result_data and len(result_data["tools"]) > 0:
#                             # Find the matching tool
#                             matching_tool = None
#                             for tool in result_data["tools"]:
#                                 if tool.get("id") == tool_id_to_update:
#                                     matching_tool = tool
#                                     break
                            
#                             if matching_tool:
#                                 st.session_state.fetched_tool = matching_tool
#                                 st.success(f"Found tool: {matching_tool.get('name', 'Unnamed Tool')}")
#                             else:
#                                 st.warning(f"Tool with ID '{tool_id_to_update}' not found in search results.")
#                                 st.session_state.fetched_tool = None
#                         else:
#                             st.warning(f"No tool found with ID: {tool_id_to_update}")
#                             st.session_state.fetched_tool = None
#                     except json.JSONDecodeError:
#                         st.error("Failed to parse response JSON.")
#                         st.session_state.fetched_tool = None
#                 else:
#                     st.error(f"Error: API returned status code {response.status_code}")
#                     st.session_state.fetched_tool = None
#             except Exception as e:
#                 st.error(f"Error connecting to API: {str(e)}")
#                 st.session_state.fetched_tool = None
    
#     # Step 2: If a tool was fetched, show the update form
#     if st.session_state.fetched_tool:
#         with st.form(key="update_tool_form"):
#             st.markdown("### Update Tool Information")
#             col1, col2 = st.columns(2)
            
#             # Pre-fill form with existing data
#             tool = st.session_state.fetched_tool
            
#             with col1:
#                 name = st.text_input("Tool Name*", 
#                                     value=tool.get("name", ""),
#                                     help="The official name of the AI tool")
#                 tool_id = st.text_input("Tool ID*", 
#                                        value=tool.get("id", ""),
#                                        help="A unique identifier for this tool",
#                                        disabled=True)
#                 categories = st.text_input("Categories", 
#                                          value=tool.get("categories", ""),
#                                          help="Comma-separated list of categories")
#                 pricing = st.text_input("Pricing", 
#                                       value=tool.get("pricing", ""),
#                                       help="Information about pricing tiers")
            
#             with col2:
#                 description = st.text_area("Description*", 
#                                          value=tool.get("description", ""),
#                                          help="A brief description of what the tool does")
                
#                 # Join pros and cons with newlines for the text area
#                 pros_text = "\n".join(tool.get("pros", []))
#                 cons_text = "\n".join(tool.get("cons", []))
                
#                 pros_input = st.text_area("Pros (one per line)", 
#                                         value=pros_text,
#                                         help="List the advantages of this tool, one per line")
#                 cons_input = st.text_area("Cons (one per line)", 
#                                         value=cons_text,
#                                         help="List limitations or disadvantages, one per line")
            
#             st.markdown("### Additional Details")
#             usage = st.text_area("Usage Examples", 
#                                value=tool.get("usage", ""),
#                                help="How this tool can be used effectively")
#             unique_features = st.text_area("Unique Features", 
#                                          value=tool.get("unique_features", ""),
#                                          help="What makes this tool stand out from others")
            
#             update_button = st.form_submit_button("Update Tool", type="primary")
            
#             if update_button:
#                 if not name or not tool_id or not description:
#                     st.error("Please fill in all required fields (marked with *).")
#                 else:
#                     # Process pros and cons lists
#                     pros = [p.strip() for p in pros_input.split('\n') if p.strip()]
#                     cons = [c.strip() for c in cons_input.split('\n') if c.strip()]
                    
#                     # Create the updated tool object
#                     updated_tool = {
#                         "name": name,
#                         "tool_id": tool_id,
#                         "description": description,
#                         "pros": pros,
#                         "cons": cons,
#                         "categories": categories,
#                         "usage": usage,
#                         "unique_features": unique_features,
#                         "pricing": pricing
#                     }
                    
#                     # Create the request payload
#                     payload = {"tools": [updated_tool]}
                    
#                     try:
#                         with st.spinner("Updating tool..."):
#                             # Prepare headers with model choice
#                             headers = {}
#                             headers["MODEL_CHOICE"] = st.session_state.model_choice
                            
#                             response = requests.put(
#                                 f"{st.session_state.api_url}/update-tools",
#                                 json=payload,
#                                 headers=headers
#                             )
                        
#                         if response.status_code == 200:
#                             result = response.json()
#                             st.success(f"Tool '{name}' updated successfully!")
                            
#                             with st.expander("View Details"):
#                                 st.json(result)
                                
#                             # Reset the fetched tool to show the form is complete
#                             st.session_state.fetched_tool = None
#                             st.rerun()
#                         else:
#                             st.error(f"Error: API returned status code {response.status_code}")
#                             st.text(response.text)
#                     except Exception as e:
#                         st.error(f"Error connecting to API: {str(e)}")

# # 4. DELETE TOOLS TAB
# with tabs[3]:
#     st.markdown('<div class="subheader">Delete Tools</div>', unsafe_allow_html=True)
    
#     st.warning("⚠️ Warning: Deletion is permanent and cannot be undone.")
    
#     tool_id_to_delete = st.text_input("Enter Tool ID to delete:", 
#                                      key="delete_tool_id_input",
#                                      help="Enter the unique identifier of the tool you want to delete")
    
#     confirm_delete = st.checkbox("I confirm that I want to delete this tool permanently")
    
#     if st.button("Delete Tool", type="primary", disabled=not confirm_delete or not tool_id_to_delete):
#         with st.spinner("Deleting tool..."):
#             try:
#                 # Prepare headers with model choice
#                 headers = {}
#                 headers["MODEL_CHOICE"] = st.session_state.model_choice
                
#                 response = requests.delete(
#                     f"{st.session_state.api_url}/delete-tool/{tool_id_to_delete}",
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     if result.get("success"):
#                         st.success(f"Tool '{result.get('deleted_tool', 'unknown')}' was deleted successfully!")
#                     else:
#                         st.error("Deletion failed.")
#                 elif response.status_code == 404:
#                     st.error(f"Tool with ID '{tool_id_to_delete}' not found.")
#                 else:
#                     st.error(f"Error: API returned status code {response.status_code}")
#                     st.text(response.text)
#             except Exception as e:
#                 st.error(f"Error connecting to API: {str(e)}")
    
#     st.divider()
    
#     st.markdown("### Clear Entire Index")
#     st.error("⚠️ DANGER: This will delete ALL tools from the index. This action cannot be undone.")
    
#     if not st.session_state.api_key:
#         st.info("Please enter your Pinecone API Key in the sidebar to use this function.")
    
#     confirm_clear = st.checkbox("I understand that this will delete ALL data from the index permanently")
    
#     if st.button("Clear Index", type="primary", disabled=not confirm_clear or not st.session_state.api_key):
#         with st.spinner("Clearing index..."):
#             try:
#                 # Prepare headers with model choice
#                 headers = {}
#                 headers["MODEL_CHOICE"] = st.session_state.model_choice
                
#                 response = requests.delete(
#                     f"{st.session_state.api_url}/clear-index",
#                     json={"api_key": st.session_state.api_key},
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     if result.get("success"):
#                         st.success(f"Index cleared successfully! Deleted {result.get('deleted_count', 0)} tools.")
#                     else:
#                         st.error("Operation failed.")
#                 elif response.status_code == 401:
#                     st.error("Unauthorized: Invalid API Key.")
#                 else:
#                     st.error(f"Error: API returned status code {response.status_code}")
#                     st.text(response.text)
#             except Exception as e:
#                 st.error(f"Error connecting to API: {str(e)}")

# # 5. STATISTICS TAB
# with tabs[4]:
#     st.markdown('<div class="subheader">Index Statistics</div>', unsafe_allow_html=True)
    
#     if st.button("Refresh Statistics", key="refresh_stats"):
#         with st.spinner("Fetching statistics..."):
#             try:
#                 # Prepare headers with model choice
#                 headers = {}
#                 headers["MODEL_CHOICE"] = st.session_state.model_choice
                
#                 response = requests.get(
#                     f"{st.session_state.api_url}/stats",
#                     headers=headers
#                 )
                
#                 if response.status_code == 200:
#                     stats = response.json()
                    
#                     # Display metrics
#                     col1, col2, col3 = st.columns(3)
#                     with col1:
#                         st.metric("Total Tools", stats.get("total_vectors", 0))
#                     with col2:
#                         st.metric("Vector Dimension", stats.get("dimension", "-"))
#                     with col3:
#                         st.metric("Index Fullness", f"{stats.get('index_fullness', 0):.2%}")
                    
#                     # Display vector information
#                     st.markdown("### Tools in Index")
#                     if "vectors" in stats and len(stats["vectors"]) > 0:
#                         # Convert to DataFrame for better display
#                         vectors_df = pd.DataFrame(stats["vectors"])
                        
#                         # Add category counts
#                         if "categories" in vectors_df.columns:
#                             # Extract categories and count occurrences
#                             all_categories = []
#                             for cats in vectors_df["categories"]:
#                                 if cats and cats != "N/A":
#                                     categories_list = [c.strip() for c in cats.split(",")]
#                                     all_categories.extend(categories_list)
                            
#                             category_counts = pd.Series(all_categories).value_counts()
                            
#                             # Show category distribution
#                             st.markdown("### Category Distribution")
#                             st.bar_chart(category_counts)
                        
#                         # Show the main table
#                         st.dataframe(vectors_df, use_container_width=True)
#                     else:
#                         st.info("No tools found in the index.")
                    
#                     # Show raw JSON for detailed inspection
#                     with st.expander("View Raw Statistics"):
#                         st.json(stats)
#                 else:
#                     st.error(f"Error: API returned status code {response.status_code}")
#                     st.text(response.text)
#             except Exception as e:
#                 st.error(f"Error connecting to API: {str(e)}")