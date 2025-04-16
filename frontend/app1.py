import streamlit as st
import requests
import json
import pandas as pd
from io import StringIO
import time
import os
import urllib3

# Disable SSL warning messages in the UI
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Tool Search Interface",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def remove_html_tags(text):
    """Remove HTML tags from a string."""
    import re
    if not text or not isinstance(text, str):
        return text
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def normalize_json_response(result_data):
    """
    Normalize different JSON response formats to a consistent structure
    with a "tools" array containing objects including names, descriptions, and bullets.
    """
    # If the response already has the expected "tools" array format with all required fields, return as is
    if "tools" in result_data and isinstance(result_data["tools"], list):
        # Check if tools have the required fields
        if all(["id" in tool and "name" in tool and "description" in tool for tool in result_data["tools"]]):
            # Make sure bullets exist (add empty array if not)
            for tool in result_data["tools"]:
                if "bullets" not in tool:
                    tool["bullets"] = []
            return result_data
        
    # Handle parallel arrays format (old format)
    if "tool_id" in result_data and isinstance(result_data["tool_id"], list):
        # Create a normalized structure
        normalized = {"tools": []}
        
        # Get all arrays
        tool_ids = result_data.get("tool_id", [])
        names = result_data.get("tools", [])  # Sometimes the names are in a "tools" array
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
            
            # Set name - try to use the name from "tools" array if available
            if i < len(names) and isinstance(names[i], dict) and "name" in names[i]:
                tool["name"] = names[i]["name"]
            else:
                tool["name"] = tool_ids[i] if i < len(tool_ids) else f"Tool {i+1}"
            
            # Add description if available
            if i < len(descriptions):
                tool["description"] = descriptions[i]
            else:
                tool["description"] = "No description available."
                
            # Add relevance if available
            if i < len(relevances):
                tool["relevance"] = relevances[i]
            
            # Add empty bullets array
            tool["bullets"] = []
                
            normalized["tools"].append(tool)
            
        return normalized
    
    # If we can't normalize, return empty result
    return {"tools": []}

# Add after normalize_json_response function
def check_initialization_status():
    """Check if backend has completed initialization"""
    try:
        response = requests.get(f"{st.session_state.api_url}/initialization-status")
        if response.status_code == 200:
            return response.json()
        else:
            return {"initialized": False, "error": f"Status code: {response.status_code}"}
    except Exception as e:
        return {"initialized": False, "error": str(e)}

# Custom CSS for styling with dark theme
st.markdown("""
<style>
    /* Dark theme base styles */
    .main {
        background-color: #121212;
        color: #E0E0E0;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2196F3;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #E0E0E0;
        margin-bottom: 1rem;
    }
    .search-results {
        color: #FF5252;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1E1E1E;
        color: #E0E0E0;
        border: 1px solid #333;
        border-radius: 4px;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #F44336 !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #D32F2F !important;
        color: white !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E1E1E;
        border-radius: 4px 4px 0 0;
        padding: 0.5rem 1rem;
        margin-right: 2px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
        color: white;
    }
    
    /* Tool styling */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Fix sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A1A1A;
    }
    
    /* Hide checkbox used for tab tracking */
    [data-testid="stCheckbox"] {
        display: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Additional styles for clickable titles */
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        border: none !important;
        color: #4FC3F7 !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        padding: 0 !important;
        text-align: left !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        color: #81D4FA !important;
        text-decoration: underline !important;
    }
    
    /* Improve detail view section headers */
    .detail-section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #64B5F6;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(100, 181, 246, 0.3);
    }
    
    /* Styles for related tools section */
    .related-tools-header {
        color: #FF5252;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid rgba(255, 82, 82, 0.3);
        padding-bottom: 0.5rem;
    }
    
    /* Make related tool buttons more compact */
    .stButton[data-testid^="related_btn_"] > button {
        font-size: 1.1rem !important;
        padding: 0.3rem 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Style for the "More Related Tools" section */
    .more-related-tools {
        background-color: #1A1A1A;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 2rem;
        border-left: 3px solid #FF5252;
    }
    
    /* Make the related tool buttons stand out */
    .stButton[data-testid^="related_btn_"] > button[data-testid="baseButton-secondary"] {
        color: #64B5F6 !important;
        background-color: #263238 !important;
        border-radius: 4px !important;
        padding: 0.5rem 0.8rem !important;
        text-align: center !important;
        margin: 0.3rem !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    
    .stButton[data-testid^="related_btn_"] > button[data-testid="baseButton-secondary"]:hover {
        background-color: #37474F !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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
if 'visible_results' not in st.session_state:
    st.session_state.visible_results = 3
if 'all_tools' not in st.session_state:
    st.session_state.all_tools = []
if 'show_more_clicked' not in st.session_state:
    st.session_state.show_more_clicked = False
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'is_searching' not in st.session_state:
    st.session_state.is_searching = False
if 'search_warning' not in st.session_state:
    st.session_state.search_warning = None
if 'search_error' not in st.session_state:
    st.session_state.search_error = None
if 'is_valid_json' not in st.session_state:
    st.session_state.is_valid_json = False
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'search_query_pending' not in st.session_state:
    st.session_state.search_query_pending = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "DEV_MODEL"
# Add this to the initialization block where other session state variables are defined (around line 170)
if 'selected_tool' not in st.session_state:
    st.session_state.selected_tool = None
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "search"  # Modes: "search" or "detail"
if 'tool_details_cache' not in st.session_state:
    st.session_state.tool_details_cache = {}  # Cache tool details by ID
if 'related_tools' not in st.session_state:
    st.session_state.related_tools = []  # All tools sent to LLM

# Add the search functions
def perform_search(query):
    """Function to perform search and display results with caching for identical queries"""
    # Check if query is empty
    if not query or query.strip() == "":
        st.session_state.search_warning = "Please provide your requirement for AI to do magic! ✨"
        st.session_state.show_results = False
        return
    
    # Clear any previous warnings and errors
    st.session_state.search_warning = None
    st.session_state.search_error = None
    
    # Store the current query in session state
    st.session_state.last_query = query

    # Check initialization status first
    status = check_initialization_status()
    if not status.get("initialized", False):
        # System is still initializing
        progress = status.get("loading_percentage", 0)
        st.session_state.search_warning = f"The system is still initializing. Currently loaded {status.get('vectors_loaded', 0)} of {status.get('total_vectors', 0)} tools ({progress:.1f}%). Please try again in a moment."
        st.session_state.show_results = True
        return
    
    # Check if this is the same query as last processed query (for caching)
    if hasattr(st.session_state, 'last_processed_query') and query == st.session_state.last_processed_query and hasattr(st.session_state, 'last_result') and st.session_state.last_result is not None:
        # Use cached result
        result = st.session_state.last_result
        process_search_results(result, query)
        return
    
    # If not from session state already, set flag for next run
    if not st.session_state.get('executing_search', False):
        st.session_state.is_searching = True
        st.session_state.search_query_pending = query
        st.session_state.executing_search = True
        return
    
    # This only executes if executing_search is True
    try:
        # Show the spinner directly in this function
        with st.spinner("Searching..."):
            # Reset the visible results counter
            st.session_state.visible_results = 3
            st.session_state.last_processed_query = query
            st.session_state.show_more_clicked = False
            
            headers = {
                "MODEL_CHOICE": st.session_state.model_choice
            }
            
            # Introduce a slight delay to ensure spinner is visible
            time.sleep(0.5)
            
            response = requests.post(
                f"{st.session_state.api_url}/query",
                json={"query": query},
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                st.session_state.last_result = result
                process_search_results(result, query)
            else:
                st.session_state.search_error = f"Error: API returned status code {response.status_code}"
                st.session_state.search_error_details = response.text
                st.session_state.show_results = True
    except Exception as e:
        st.session_state.search_error = f"Error connecting to API: {str(e)}"
        st.session_state.show_results = True
    finally:
        # Clear the flags
        st.session_state.is_searching = False
        st.session_state.search_query_pending = None
        st.session_state.executing_search = False

def process_search_results(result, query):
    """Process search results and store them in session state"""
    try:
        result_data = json.loads(result["response"])
        st.session_state.is_valid_json = True
        
        # Store the raw response data
        st.session_state.raw_result_data = result_data
        
        # Extract and store tools from the LLM response
        if "tools" in result_data and isinstance(result_data["tools"], list):
            # The new format already has a "tools" array
            st.session_state.all_tools = result_data.get("tools", [])
        elif "tool_id" in result_data and isinstance(result_data["tool_id"], list):
            # Convert the old format to the new format
            tools = []
            tool_ids = result_data.get("tool_id", [])
            
            for i, tool_id in enumerate(tool_ids):
                # Try to find corresponding data in other arrays
                if i < len(result_data.get("tools", [])):
                    # Use existing tool object
                    tools.append(result_data["tools"][i])
                else:
                    # Create a basic tool object
                    tool = {
                        "id": tool_id,
                        "name": tool_id,
                        "description": "No description available.",
                        "bullets": []
                    }
                    tools.append(tool)
            
            st.session_state.all_tools = tools
        else:
            # Fallback for unexpected format
            st.session_state.all_tools = []
        
        # Check for tools_sent_to_llm in the response
        if "tools_sent_to_llm" in result_data and isinstance(result_data["tools_sent_to_llm"], list):
            # Use the tools_sent_to_llm from the response
            st.session_state.related_tools = result_data.get("tools_sent_to_llm", [])
        else:
            # No tools_sent_to_llm in response, fallback to using all_tools
            st.session_state.related_tools = st.session_state.all_tools.copy()
        
        # Cache all tools from search results for future use
        for tool in st.session_state.all_tools:
            if 'id' in tool:
                st.session_state.tool_details_cache[tool['id']] = tool

        for tool in st.session_state.related_tools:
            if 'id' in tool:
                # Only add to cache if not already there or if this has more info
                if (tool['id'] not in st.session_state.tool_details_cache or 
                    len(tool) > len(st.session_state.tool_details_cache[tool['id']])):
                    st.session_state.tool_details_cache[tool['id']] = tool
        
    except json.JSONDecodeError:
        st.session_state.is_valid_json = False
        st.session_state.raw_response = result["response"]
        st.session_state.all_tools = []
        st.session_state.related_tools = []
    
    st.session_state.show_results = True

def display_tool_detail():
    """Display detailed view of a selected tool with related tools section"""
    if st.session_state.view_mode != "detail" or st.session_state.selected_tool is None:
        return
    
    tool = st.session_state.selected_tool
    tool_id = tool.get('id', '')
    
    # Add back button at the top
    if st.button("← Back to Search Results"):
        st.session_state.view_mode = "search"
        # Ensure we maintain the show_results state to keep search results visible
        st.session_state.show_results = True
        st.rerun()
    
    # Display header and available information immediately
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <div style="font-size: 2.2rem; font-weight: bold; color: #4FC3F7; margin-bottom: 0.5rem;">
            {tool.get('name', 'Tool Details')}
        </div>
        <div style="font-size: 1rem; color: #B0BEC5; background-color: #263238; 
            padding: 0.3rem 0.6rem; border-radius: 0.3rem; display: inline-block; margin-bottom: 1rem;">
            ID: {tool.get('id', 'No ID')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if we need more information for this tool
    # We'll consider a tool to have complete info if it has description and bullets
    has_complete_info = all(key in tool and tool[key] for key in ['description', 'bullets'])
    
    # If not complete, try to find more info from our various data sources
    if not has_complete_info:
        # 1. First check in all_tools
        complete_tool = None
        
        # Check all_tools
        matching_tool = next((t for t in st.session_state.all_tools if t.get('id') == tool_id), None)
        if matching_tool and all(key in matching_tool for key in ['description', 'bullets']):
            complete_tool = matching_tool
        
        # If not found, check related_tools
        if not complete_tool:
            matching_tool = next((t for t in st.session_state.related_tools if t.get('id') == tool_id), None)
            if matching_tool and all(key in matching_tool for key in ['description', 'bullets']):
                complete_tool = matching_tool
        
        # If not found, check cache
        if not complete_tool and tool_id in st.session_state.tool_details_cache:
            cached_tool = st.session_state.tool_details_cache[tool_id]
            if all(key in cached_tool for key in ['description', 'bullets']):
                complete_tool = cached_tool
        
        # If we found a complete version of this tool, use it
        if complete_tool:
            # Update with any additional fields
            for key, value in complete_tool.items():
                if key not in tool or not tool[key]:
                    tool[key] = value
            
            # Cache it for future use
            st.session_state.tool_details_cache[tool_id] = complete_tool
            
            # Update has_complete_info flag
            has_complete_info = True
    
    # Only fetch additional details if we STILL don't have complete information AND we have a valid tool_id
    if not has_complete_info and tool_id:
        with st.spinner("Fetching additional details..."):
            try:
                # Prepare headers with all settings
                headers = {
                    "MODEL_CHOICE": st.session_state.model_choice
                }
                
                # Make API call only if needed - ensure the tool_id is properly included
                response = requests.post(
                    f"{st.session_state.api_url}/query",
                    json={"query": f"tool_id:{tool_id}"},  # Make sure tool_id is properly included
                    headers=headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    try:
                        result_data = json.loads(result["response"])
                        
                        # Check if we got additional details
                        if "tools" in result_data and len(result_data["tools"]) > 0:
                            # Find the matching tool
                            matching_tool = None
                            for t in result_data["tools"]:
                                if t.get("id") == tool_id:
                                    matching_tool = t
                                    break
                            
                            if matching_tool:
                                # Update with any additional fields
                                for key, value in matching_tool.items():
                                    if key not in tool or not tool[key]:
                                        tool[key] = value
                                
                                # Cache the enhanced tool data
                                st.session_state.tool_details_cache[tool_id] = matching_tool
                                
                                # Update the stored tool with complete info
                                st.session_state.selected_tool = tool
                    except json.JSONDecodeError:
                        pass  # Just use what we have
            except Exception as e:
                st.info(f"Using available tool information. ({str(e)})")
    
    # Description section
    st.markdown("### Description")
    st.markdown(f"""
    <div style="color: #E0E0E0; font-size: 1.1rem; margin-bottom: 1.5rem; line-height: 1.6;">
        {tool.get('description', 'No description available.')}
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features section
    st.markdown("### Key Features")
    if 'bullets' in tool and tool['bullets']:
        for i, bullet in enumerate(tool['bullets'], 1):
            st.markdown(f"""
            <div style="margin-bottom: 0.8rem; color: #E0E0E0;">
                <span style="color: #FF9800; font-weight: 600;">{i}.</span> {bullet}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No feature information available for this tool.")
    
    # Pros and Cons section
    st.markdown("### Pros and Cons")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Pros")
        if 'pros' in tool and tool['pros']:
            for pro in tool['pros']:
                st.markdown(f"✅ {pro}")
        else:
            st.info("No pros listed.")
            
    with col2:
        st.markdown("#### Cons")
        if 'cons' in tool and tool['cons']:
            for con in tool['cons']:
                st.markdown(f"⚠️ {con}")
        else:
            st.info("No cons listed.")
    
    # Usage Examples section
    st.markdown("### Usage Examples")
    if 'usage' in tool and tool['usage']:
        st.markdown(tool['usage'])
    else:
        st.info("No usage examples available.")
        
    # Unique Features section
    st.markdown("### Unique Features")
    if 'unique_features' in tool and tool['unique_features']:
        st.markdown(tool['unique_features'])
    else:
        st.info("No unique features listed.")
        
    # Pricing section
    st.markdown("### Pricing")
    if 'pricing' in tool and tool['pricing']:
        st.markdown(f"""
        <div style="background-color: #1E3A5F; padding: 1rem; border-radius: 0.5rem; margin-top: 0.5rem;">
            <div style="font-weight: 600; color: #64B5F6; margin-bottom: 0.5rem;">Pricing Information</div>
            <div style="color: #E0E0E0;">{tool['pricing']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No pricing information available.")
    
    # Tool Metadata section
    st.markdown("### Tool Metadata")
    
    metadata = {
        "Tool ID": tool.get('id', 'N/A'),
        "Name": tool.get('name', 'N/A'),
        "Categories": tool.get('categories', 'N/A'),
        "Owner": tool.get('owner', 'N/A'),
        "Status": tool.get('status', 'N/A')
    }
    
    # Filter out empty or N/A values
    metadata = {k: v for k, v in metadata.items() if v != 'N/A' and v}
    
    if metadata:
        metadata_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['Value'])
        st.dataframe(metadata_df, use_container_width=True)
    else:
        st.info("No metadata available for this tool.")
    
    # URL if available
    if 'url' in tool and tool['url']:
        st.markdown("### Tool URL")
        st.markdown(f"[{tool['url']}]({tool['url']})")
        
    # Image URL if available
    if 'image_url' in tool and tool['image_url']:
        st.markdown("### Tool Image")
        st.image(tool['image_url'], caption=tool.get('name', 'Tool Image'))
    
    # More Related Tools section
    st.markdown("### More Related Tools")
    
    # Get all tools
    all_related_tools = st.session_state.related_tools
    
    # Skip the current tool
    related_tools = [t for t in all_related_tools if t.get('id') != tool_id]
    
    # Get list of tool IDs in LLM response for prioritization
    llm_tool_ids = [t.get('id') for t in st.session_state.all_tools]
    
    # Prioritize tools - first those in LLM response, then others
    prioritized_tools = []
    
    # First add tools that were in LLM response
    for t in related_tools:
        if t.get('id') in llm_tool_ids:
            prioritized_tools.append(t)
    
    # Then add remaining tools
    for t in related_tools:
        if t.get('id') not in llm_tool_ids and t not in prioritized_tools:
            prioritized_tools.append(t)
    
    if not prioritized_tools:
        st.info("No related tools available.")
    else:
        # Create a 2-column layout for related tools
        cols = st.columns(2)
        
        for i, related_tool in enumerate(prioritized_tools):
            with cols[i % 2]:
                # Create a callback for this related tool
                def view_related_tool(selected_tool=related_tool):
                    st.session_state.selected_tool = selected_tool
                    st.session_state.view_mode = "detail"
                
                # Make the title clickable
                st.button(
                    f"{related_tool.get('name', 'Unnamed Tool')}",
                    key=f"related_btn_{related_tool.get('id', i)}",
                    on_click=view_related_tool,
                    type="secondary"
                )
    
    # Add back button at the bottom too
    if st.button("← Back to Search Results", key="back_button_bottom"):
        st.session_state.view_mode = "search"
        # Ensure we maintain the show_results state to keep search results visible
        st.session_state.show_results = True
        st.rerun()

def display_search_results():
    """Display search results from session state"""
    # Initialize raw_response if it doesn't exist
    if "raw_response" not in st.session_state:
        st.session_state.raw_response = ""
    
    # If in detail view mode, don't show search results
    if st.session_state.view_mode == "detail":
        return
    
    if not st.session_state.show_results:
        return
        
    # Check for warnings or errors
    if hasattr(st.session_state, 'search_warning') and st.session_state.search_warning:
        st.warning(st.session_state.search_warning)
        return
        
    # Check if there's an error to display
    if hasattr(st.session_state, 'search_error') and st.session_state.search_error:
        st.error(st.session_state.search_error)
        if hasattr(st.session_state, 'search_error_details') and st.session_state.search_error_details:
            st.text(st.session_state.search_error_details)
        return
    
    st.markdown('<div class="search-results">Search Results</div>', unsafe_allow_html=True)
    
    if st.session_state.is_valid_json and st.session_state.all_tools:
        tools = st.session_state.all_tools
        total_tools = len(tools)
        
        # Display the number of tools found
        st.markdown(f"Found {total_tools} tools related to your query, ranked by relevance:")
        
        # Define how many tools to show
        visible_count = st.session_state.visible_results
        if visible_count > total_tools:
            visible_count = total_tools
        
        # Display visible tools in the new format
        for i, tool in enumerate(tools[:visible_count], start=1):
            # Create a container for each tool
            tool_container = st.container()
            
            with tool_container:
                # Create a clickable title - using a button that looks like text
                def view_details_callback(selected_tool=tool):
                    st.session_state.selected_tool = selected_tool
                    st.session_state.view_mode = "detail"
                
                # Use a button styled as a title
                st.button(
                    f"{i}. {tool.get('name', 'No Name')}",
                    key=f"title_btn_{tool.get('id', i)}",
                    on_click=view_details_callback,
                    type="secondary",
                    use_container_width=True
                )
                
                # Tool ID badge - placed under the clickable title
                
                # Tool ID
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
                
                # Key features/bullets (new)
                if 'bullets' in tool and tool['bullets']:
                    st.markdown("<div style='color: #FF9800; font-weight: 500; margin-top: 0.5rem;'>Key Features:</div>", unsafe_allow_html=True)
                    for bullet in tool['bullets']:
                        st.markdown(f"""
                        <div style="margin-left: 1rem; color: #E0E0E0;">
                            • {bullet}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Add separator between tools
            st.markdown("<hr style='margin: 1rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
        
        # Show More button block using an on_click callback
        if visible_count < total_tools:
            def show_more_callback():
                # Set visible_results to total tools and force show_results to remain True
                st.session_state.visible_results = len(st.session_state.all_tools)
                st.session_state.show_results = True

            st.button("Show More", type="primary", key="show_more_button", on_click=show_more_callback)
            st.markdown(f"Showing {visible_count} of {total_tools} tools")
    
    elif not st.session_state.is_valid_json:
        st.markdown("### Response from the model")
        st.info(st.session_state.raw_response)
    else:
        st.info("No matching tools found. Try a different search query.")
    
    with st.expander("View Raw Response"):
        if st.session_state.is_valid_json:
            st.json(st.session_state.raw_result_data)
        else:
            st.text(st.session_state.raw_response)

# Function to handle search form submission
def handle_form_submit():
    # This ensures the current input value is used, not the last_query from session state
    current_query = st.session_state.query_input
    
    # Reset any previous execution state
    st.session_state.executing_search = False
    
    # Start the search process
    perform_search(current_query)

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    st.session_state.api_url = st.text_input("Backend API URL", value=st.session_state.api_url)
    st.session_state.api_key = st.text_input("Pinecone API Key (for admin functions)",
                                          value=st.session_state.api_key, 
                                          type="password")

    # Add info about Nomic Atlas
    st.info("This application uses Groq API for fast LLM inference and Nomic Atlas for embeddings.")

    # Model selection in sidebar
    st.divider()
    st.subheader("Model Selection")
    
    # Radio button for model selection
    st.session_state.model_choice = st.radio(
        "Select Model Environment:",
        options=["DEV_MODEL", "PROD_MODEL"],
        index=0 if st.session_state.model_choice == "DEV_MODEL" else 1,
        horizontal=True
    )

    # Show current model dynamically
    if st.session_state.model_choice:
        try:
            headers = {"MODEL_CHOICE": st.session_state.model_choice}
            response = requests.get(
                f"{st.session_state.api_url}/model-info",
                headers=headers
            )
            if response.status_code == 200:
                model_info = response.json()
                st.info(f"Using {model_info.get('current_model')} from {model_info.get('provider', 'Groq')}")
        except Exception as e:
            st.info(f"Using {st.session_state.model_choice}")
    
    st.divider()
    if st.button("Check API Health"):
        try:
            response = requests.get(f"{st.session_state.api_url}/health")
            if response.status_code == 200:
                st.success("API is healthy! ✅")
                st.json(response.json())
            else:
                st.error(f"API returned status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Add Test Connection button in main content
if st.button("Test Connection"):
    with st.spinner("Testing connection to backend services..."):
        try:
            # Prepare headers with only the model choice
            headers = {
                "MODEL_CHOICE": st.session_state.model_choice
            }
            
            # Call the test-connection endpoint
            response = requests.get(
                f"{st.session_state.api_url}/test-connection",
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Create a summary of endpoint tests
                endpoint_summary = []
                all_endpoints_ok = True
                
                for endpoint in result.get("endpoints_tested", []):
                    endpoint_ok = endpoint.get("success", False)
                    all_endpoints_ok = all_endpoints_ok and endpoint_ok
                    
                    status_emoji = "✅" if endpoint_ok else "❌"
                    endpoint_name = endpoint.get("endpoint", "unknown")
                    status_code = endpoint.get("status_code", "N/A")
                    
                    endpoint_summary.append(f"{status_emoji} {endpoint_name}: {status_code}")
                
                # Display overall status
                if all_endpoints_ok:
                    st.success("Successfully connected to all endpoints!")
                elif result.get("status") == "partial":
                    st.warning("Partial success: Some endpoints are working, others failed.")
                else:
                    st.error("Failed to connect to endpoints.")
                
                # Show endpoint summary
                st.write("Endpoint Status:")
                for line in endpoint_summary:
                    st.write(line)
                
                # Show details in expander
                with st.expander("Connection Details"):
                    st.json(result)
            else:
                st.error(f"Error: API returned status code {response.status_code}")
                st.text(response.text)
        except Exception as e:
            st.error(f"Error testing connection: {str(e)}")

# Main content
st.markdown('<div class="main-header">AI Tool Search</div>', unsafe_allow_html=True)

# Create tabs directly without visible tracking widgets
tabs = st.tabs(["🔍 Search", "➕ Add Tools", "🔄 Update Tools", "🗑️ Delete Tools", "📊 Statistics"])

# Search tab
with tabs[0]:
    if st.session_state.view_mode == "search":
        st.markdown('<div class="subheader">Search AI Tools</div>', unsafe_allow_html=True)
        
        # Create a form to capture Enter key presses
        with st.form(key="search_form"):
            # Use session state key to track input value properly
            query = st.text_input(
                "Enter your search query:", 
                placeholder="e.g., list all the coding related tools",
                key="query_input"
            )
            
            # Form submit button (triggered by Enter key or click)
            form_submit = st.form_submit_button("Search", type="primary", on_click=handle_form_submit)
        
        # Dedicated search container for the spinner and processing
        search_container = st.container()
        
        # If executing a search from a previous submission, continue the process
        if st.session_state.get('executing_search', False):
            with search_container:
                # Execute the pending search with the same query
                perform_search(st.session_state.search_query_pending)
        
        # Display search results below the search box
        display_search_results()
    else:
        # Display the detailed view of the selected tool
        display_tool_detail()

# 2. ADD TOOLS TAB
with tabs[1]:
    st.markdown('<div class="subheader">Add New AI Tools</div>', unsafe_allow_html=True)
    
    # When we enter this tab, hide any search results
    if st.session_state.active_tab != 1:
        st.session_state.active_tab = 1
        st.session_state.show_results = False
    
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
                            # Prepare headers with all settings
                            headers = {
                                "MODEL_CHOICE": st.session_state.model_choice
                            }
                            
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
                            # Prepare headers with all settings
                            headers = {
                                "MODEL_CHOICE": st.session_state.model_choice
                            }
                            
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
    
    # When we enter this tab, hide any search results
    if st.session_state.active_tab != 2:
        st.session_state.active_tab = 2
        st.session_state.show_results = False
    
    # Step 1: Input tool ID
    tool_id_to_update = st.text_input("Enter Tool ID to update:", 
                                      key="update_tool_id_input",
                                      help="Enter the unique identifier of the tool you want to update")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        fetch_button = st.button("Fetch Tool", key="fetch_tool_button")
    
    if fetch_button and tool_id_to_update:
        with st.spinner("Fetching tool data..."):
            try:
                # Prepare headers with all settings
                headers = {
                    "MODEL_CHOICE": st.session_state.model_choice
                }
                
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
                            # Prepare headers with all settings
                            headers = {
                                "MODEL_CHOICE": st.session_state.model_choice
                            }
                            
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
    
    # When we enter this tab, hide any search results
    if st.session_state.active_tab != 3:
        st.session_state.active_tab = 3
        st.session_state.show_results = False
    
    st.warning("⚠️ Warning: Deletion is permanent and cannot be undone.")
    
    tool_id_to_delete = st.text_input("Enter Tool ID to delete:", 
                                     key="delete_tool_id_input",
                                     help="Enter the unique identifier of the tool you want to delete")
    
    confirm_delete = st.checkbox("I confirm that I want to delete this tool permanently")
    
    if st.button("Delete Tool", type="primary", disabled=not confirm_delete or not tool_id_to_delete):
        with st.spinner("Deleting tool..."):
            try:
                # Prepare headers with all settings
                headers = {
                    "MODEL_CHOICE": st.session_state.model_choice
                }
                
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
                # Prepare headers with all settings
                headers = {
                    "MODEL_CHOICE": st.session_state.model_choice
                }
                
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
# In the statistics tab section of the Streamlit app
with tabs[4]:
    st.markdown('<div class="subheader">Index Statistics</div>', unsafe_allow_html=True)
    
    # Remove the show_all checkbox as it's now always true
    # show_all = st.checkbox("Show All Tools", value=False, 
    #                      help="Show all tools in the index (may be slow if you have many tools)")
    
    refresh_button = st.button("Refresh Statistics", key="refresh_stats")
    
    if refresh_button:
        with st.spinner("Fetching statistics..."):
            try:
                # Prepare headers with all settings
                headers = {
                    "MODEL_CHOICE": st.session_state.model_choice
                }
                
                # Always set show_all to true
                params = {"show_all": "true"}
                
                response = requests.get(
                    f"{st.session_state.api_url}/stats",
                    headers=headers,
                    params=params
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
                        st.metric("Index Fullness", f"{stats.get('index_fullness', 0) * 100:.2f}%")
                    
                    # Remove the conditional info message about limited results
                    # if not show_all and stats.get("total_vectors", 0) > stats.get("vectors_shown", 0):
                    #     st.info(f"Showing {stats.get('vectors_shown', 0)} of {stats.get('total_vectors', 0)} tools. Check 'Show All Tools' to see all.")
                    
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