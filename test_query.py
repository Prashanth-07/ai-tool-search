#!/usr/bin/env python3
"""
AI Tool Search API - Enhanced Debug Query Testing Script with Flow Types

Tests the /query endpoint with different flow types and saves results as individual JSON files.
Captures: input_query, flow_type, tools_sent_to_llm, LLM_answer for analysis.
Now supports all 4 flow types: current, reranker, groq, openai, structured.
"""

import requests
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAPITester:
    """Simple API tester that saves individual JSON files"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                logger.info("✅ API is reachable")
                return True
            else:
                logger.error(f"❌ API returned {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get current model information with debug logging"""
        try:
            logger.info("📡 Getting model info...")
            logger.info(f"DEBUG: Calling {self.base_url}/model-info")
            
            response = self.session.get(f"{self.base_url}/model-info")
            
            if response.status_code == 200:
                data = response.json()
                current_model = data.get("current_groq_model", "Unknown")
                query_tools_llm = data.get("query_tools_llm", "Unknown")
                environment = data.get("environment", "Unknown")
                logger.info(f"✅ Current Groq model: {current_model} (Environment: {environment})")
                logger.info(f"✅ Query tools LLM: {query_tools_llm}")
                logger.info(f"DEBUG: Full model-info response: {data}")
                return data
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return {}
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    def get_stats(self) -> dict:
        """Get API stats and save as tools_database.json"""
        try:
            logger.info("📊 Getting API stats...")
            response = self.session.get(f"{self.base_url}/stats?show_all=true")
            
            if response.status_code == 200:
                data = response.json()
                total_tools = data.get("total_vectors", 0)
                
                # Save stats
                filename = f"tools_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                logger.info(f"✅ Stats saved to {filename} - {total_tools} tools total")
                return data
            else:
                logger.error(f"Failed to get stats: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}
    
    def get_flow_type_from_user(self) -> str:
        """Get flow type from user input."""
        print("\n🔄 Available Flow Types:")
        print("  1. current   - Original flow (OpenAI + Groq fallback)")
        print("  2. reranker  - BGE Reranker + LLM Filter + Database descriptions")
        print("  3. groq      - Groq only (no OpenAI fallback)")
        print("  4. openai    - OpenAI JSON Object Mode only (no Groq fallback)")
        print("  5. structured - Groq Structured Outputs")
        print("  6. default   - Use API default (current)")
        
        while True:
            try:
                choice = input("\nEnter flow type (1-6 or name): ").strip().lower()
                
                # Map choices to flow types
                flow_map = {
                    "1": "current",
                    "2": "reranker", 
                    "3": "groq",
                    "4": "openai",
                    "5": "structured",
                    "6": "default",
                    "current": "current",
                    "reranker": "reranker",
                    "groq": "groq", 
                    "openai": "openai",
                    "structured": "structured",
                    "default": "default"
                }
                
                if choice in flow_map:
                    selected_flow = flow_map[choice]
                    if selected_flow == "default":
                        logger.info("🔄 Using API default flow type")
                        return None  # Will use API default
                    else:
                        logger.info(f"🔄 Selected flow type: {selected_flow}")
                        return selected_flow
                else:
                    print("❌ Invalid choice. Please enter 1-6 or flow type name.")
                    
            except KeyboardInterrupt:
                print("\n👋 Cancelled flow selection")
                return None
    
    def test_query(self, query: str, flow_type: str = None) -> dict:
        """Test a single query and save results with enhanced debugging"""
        logger.info(f"🔍 Testing query: '{query}' with flow_type: {flow_type or 'default'}")
        
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get model info BEFORE making the query
        logger.info("DEBUG: Getting model info before query...")
        model_info_before = self.get_model_info()
        
        # Prepare result structure with enhanced debug info
        result = {
            "timestamp": datetime.now().isoformat(),
            "input_query": query,
            "flow_type": flow_type,
            "model_info_before_query": model_info_before,
            "debug_info": {
                "headers_sent": {},
                "flow_selection_debug": {},
                "api_call_details": {}
            },
            "tools_sent_to_llm": [],
            "llm_answer": {},
            "processing_time_seconds": 0,
            "status": "pending",
            "error": None,
            "raw_api_response": None
        }
        
        try:
            # Prepare headers and log them
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "test-query-script/2.0"
            }
            
            # Log debug info about headers
            logger.info(f"DEBUG: Headers being sent: {headers}")
            result["debug_info"]["headers_sent"] = dict(headers)
            
            # Make API request with flow_type
            payload = {"query": query}
            if flow_type:
                payload["flow_type"] = flow_type
                
            logger.info(f"DEBUG: Making POST request to {self.base_url}/query")
            logger.info(f"DEBUG: Payload: {payload}")
            
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                headers=headers
            )
            
            processing_time = time.time() - start_time
            result["processing_time_seconds"] = round(processing_time, 3)
            
            # Log response details
            logger.info(f"DEBUG: Response status code: {response.status_code}")
            logger.info(f"DEBUG: Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                response_data = response.json()
                result["raw_api_response"] = response_data
                
                # Parse LLM response
                llm_response_text = response_data.get("response", "")
                
                try:
                    # Parse JSON response from LLM
                    if isinstance(llm_response_text, str):
                        llm_parsed = json.loads(llm_response_text)
                    else:
                        llm_parsed = llm_response_text
                    
                    result["llm_answer"] = llm_parsed
                    
                    # Extract tools that were sent to LLM
                    tools_data = llm_parsed.get("tools", [])
                    tool_ids = llm_parsed.get("tool_id", [])
                    actual_flow_type = llm_parsed.get("flow_type", flow_type or "unknown")
                    
                    result["tools_sent_to_llm"] = {
                        "count": len(tools_data),
                        "tool_ids": tool_ids,
                        "tools": tools_data
                    }
                    
                    result["actual_flow_type"] = actual_flow_type
                    result["status"] = "success"
                    
                    # Get model info AFTER the query to compare
                    logger.info("DEBUG: Getting model info after query...")
                    model_info_after = self.get_model_info()
                    result["model_info_after_query"] = model_info_after
                    
                    # Show model info in console
                    current_model = model_info_before.get("current_groq_model", "Unknown")
                    query_tools_llm = model_info_before.get("query_tools_llm", "Unknown")
                    logger.info(f"✅ Success - {len(tools_data)} tools found in {processing_time:.2f}s")
                    logger.info(f"🤖 Used models: Groq={current_model}, Query Tools={query_tools_llm}")
                    logger.info(f"🔄 Flow type: {actual_flow_type}")
                    
                    # Show processing stats if available
                    if "processing_stats" in llm_parsed:
                        stats = llm_parsed["processing_stats"]
                        logger.info(f"📊 Processing stats: {stats}")
                    
                    # Check if model changed between calls
                    if model_info_before.get("current_groq_model") != model_info_after.get("current_groq_model"):
                        logger.warning(f"⚠️ Groq model changed between calls!")
                        logger.warning(f"Before: {model_info_before.get('current_groq_model')}")
                        logger.warning(f"After: {model_info_after.get('current_groq_model')}")
                    
                except json.JSONDecodeError as e:
                    result["error"] = f"JSON parse error: {str(e)}"
                    result["llm_answer"] = {"unparsed_response": llm_response_text}
                    result["status"] = "parse_error"
                    logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
                    
            else:
                result["status"] = "api_error"
                result["error"] = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"❌ API error: {response.status_code}")
                logger.error(f"Response text: {response.text}")
                
        except Exception as e:
            result["status"] = "request_error"
            result["error"] = str(e)
            result["processing_time_seconds"] = time.time() - start_time
            logger.error(f"❌ Request failed: {e}")
        
        # Save result to file
        safe_query = query.replace(' ', '_').replace('/', '_').replace('?', '').replace('!', '').replace('"', '').replace("'", '').replace(':', '').replace('*', '').replace('<', '').replace('>', '').replace('|', '')[:30]
        flow_suffix = f"_{flow_type}" if flow_type else "_default"
        filename = f"query_test_{timestamp}_{safe_query}{flow_suffix}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"💾 Result saved to {filename}")
        return result

def main():
    """Main function with interactive input and enhanced debugging"""
    print("=" * 70)
    print("AI Tool Search API - Enhanced Debug Query Tester with Flow Types")
    print("=" * 70)
    
    # Initialize tester
    tester = SimpleAPITester("http://localhost:8000")  # Change if needed
    
    # Test connectivity
    if not tester.test_connectivity():
        print("❌ Cannot connect to API. Make sure it's running.")
        return
    
    # Get model info first
    model_info = tester.get_model_info()
    if model_info:
        current_model = model_info.get("current_groq_model", "Unknown")
        query_tools_llm = model_info.get("query_tools_llm", "Unknown")
        environment = model_info.get("environment", "Unknown")
        print(f"🤖 Using Groq model: {current_model} (Environment: {environment})")
        print(f"🤖 Query tools LLM: {query_tools_llm}")
    
    # Get stats
    stats = tester.get_stats()
    if stats:
        total_tools = stats.get("total_vectors", 0)
        print(f"📊 API has {total_tools} tools loaded")
    
    # Interactive testing loop
    print("\n🔍 Enter your queries to test (type 'quit' or 'exit' to stop):")
    print("=" * 70)
    
    while True:
        try:
            # Get user input
            query = input("\nEnter your query: ").strip()
            
            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("👋 Goodbye!")
                break
            
            # Get flow type for this query
            flow_type = tester.get_flow_type_from_user()
            
            # Test the query
            print(f"🔍 Testing: '{query}' with flow: {flow_type or 'default'}")
            result = tester.test_query(query, flow_type)
            
            # Show results
            if result["status"] == "success":
                tools_count = result["tools_sent_to_llm"]["count"]
                processing_time = result["processing_time_seconds"]
                actual_flow = result.get("actual_flow_type", flow_type or "unknown")
                tool_names = [tool.get("name", "Unknown") for tool in result["tools_sent_to_llm"]["tools"]]
                
                print(f"✅ Success! Found {tools_count} tools in {processing_time:.2f}s")
                print(f"🔄 Flow used: {actual_flow}")
                print(f"🛠️  Tools found: {', '.join(tool_names[:3])}{'...' if len(tool_names) > 3 else ''}")
                print(f"📁 Saved to JSON file")
                
                # Show flow-specific stats
                if "processing_stats" in result["llm_answer"]:
                    stats = result["llm_answer"]["processing_stats"]
                    print(f"📊 Stats: {stats}")
                
                # Show model debug info
                model_before = result.get("model_info_before_query", {}).get("current_groq_model", "Unknown")
                model_after = result.get("model_info_after_query", {}).get("current_groq_model", "Unknown")
                print(f"🤖 Groq model used: {model_before}")
                if model_before != model_after:
                    print(f"⚠️  Model changed during request: {model_before} → {model_after}")
            else:
                print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("All test results saved as JSON files in current directory.")
    print("Check the 'debug_info' section in JSON files for detailed debugging.")
    print("Flow types tested: current, reranker, groq, openai, structured")
    print("=" * 70)

# Additional helper function for batch testing
def test_multiple_queries():
    """Test multiple queries - call this function separately if needed"""
    queries = [
        "AI writing tools for content creation",
        "free video editing software",
        "project management tools for teams", 
        "AI image generators",
        "code review automation tools"
    ]
    
    flows = ["current", "reranker", "groq", "openai", "structured"]
    
    tester = SimpleAPITester()
    
    print(f"Testing {len(queries)} queries with {len(flows)} flow types...")
    for i, query in enumerate(queries, 1):
        for j, flow in enumerate(flows, 1):
            print(f"\n[{i}.{j}] Testing: {query} (Flow: {flow})")
            result = tester.test_query(query, flow)
            time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# AI Tool Search API - Enhanced Debug Query Testing Script

# Tests the /query endpoint and saves results as individual JSON files.
# Captures: input_query, tools_sent_to_llm, LLM_answer for analysis.
# Now with enhanced debugging for model selection issues.
# """

# import requests
# import json
# import time
# from datetime import datetime
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# class SimpleAPITester:
#     """Simple API tester that saves individual JSON files"""
    
#     def __init__(self, base_url: str = "http://localhost:8000"):
#         self.base_url = base_url.rstrip('/')
#         self.session = requests.Session()
#         self.session.timeout = 30
    
#     def test_connectivity(self) -> bool:
#         """Test basic API connectivity"""
#         try:
#             response = self.session.get(f"{self.base_url}/")
#             if response.status_code == 200:
#                 logger.info("✅ API is reachable")
#                 return True
#             else:
#                 logger.error(f"❌ API returned {response.status_code}")
#                 return False
#         except Exception as e:
#             logger.error(f"❌ Connection failed: {e}")
#             return False
    
#     def get_model_info(self) -> dict:
#         """Get current model information with debug logging"""
#         try:
#             logger.info("📡 Getting model info...")
#             logger.info(f"DEBUG: Calling {self.base_url}/model-info")
            
#             response = self.session.get(f"{self.base_url}/model-info")
            
#             if response.status_code == 200:
#                 data = response.json()
#                 current_model = data.get("current_model", "Unknown")
#                 environment = data.get("environment", "Unknown")
#                 logger.info(f"✅ Current model: {current_model} (Environment: {environment})")
#                 logger.info(f"DEBUG: Full model-info response: {data}")
#                 return data
#             else:
#                 logger.error(f"Failed to get model info: {response.status_code}")
#                 logger.error(f"Response: {response.text}")
#                 return {}
#         except Exception as e:
#             logger.error(f"Error getting model info: {e}")
#             return {}
    
#     def get_stats(self) -> dict:
#         """Get API stats and save as tools_database.json"""
#         try:
#             logger.info("📊 Getting API stats...")
#             response = self.session.get(f"{self.base_url}/stats?show_all=true")
            
#             if response.status_code == 200:
#                 data = response.json()
#                 total_tools = data.get("total_vectors", 0)
                
#                 # Save stats
#                 filename = f"tools_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#                 with open(filename, 'w') as f:
#                     json.dump(data, f, indent=2)
                
#                 logger.info(f"✅ Stats saved to {filename} - {total_tools} tools total")
#                 return data
#             else:
#                 logger.error(f"Failed to get stats: {response.status_code}")
#                 return {}
#         except Exception as e:
#             logger.error(f"Error getting stats: {e}")
#             return {}
    
#     def test_query(self, query: str) -> dict:
#         """Test a single query and save results with enhanced debugging"""
#         logger.info(f"🔍 Testing query: '{query}'")
        
#         start_time = time.time()
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         # Get model info BEFORE making the query
#         logger.info("DEBUG: Getting model info before query...")
#         model_info_before = self.get_model_info()
        
#         # Prepare result structure with enhanced debug info
#         result = {
#             "timestamp": datetime.now().isoformat(),
#             "input_query": query,
#             "model_info_before_query": model_info_before,
#             "debug_info": {
#                 "headers_sent": {},
#                 "model_selection_debug": {},
#                 "api_call_details": {}
#             },
#             "tools_sent_to_llm": [],
#             "llm_answer": {},
#             "processing_time_seconds": 0,
#             "status": "pending",
#             "error": None,
#             "raw_api_response": None
#         }
        
#         try:
#             # Prepare headers and log them
#             headers = {
#                 "Content-Type": "application/json",
#                 "User-Agent": "test-query-script/1.0"
#             }
            
#             # Log debug info about headers
#             logger.info(f"DEBUG: Headers being sent: {headers}")
#             result["debug_info"]["headers_sent"] = dict(headers)
            
#             # Make API request
#             payload = {"query": query}
#             logger.info(f"DEBUG: Making POST request to {self.base_url}/query")
#             logger.info(f"DEBUG: Payload: {payload}")
            
#             response = self.session.post(
#                 f"{self.base_url}/query",
#                 json=payload,
#                 headers=headers
#             )
            
#             processing_time = time.time() - start_time
#             result["processing_time_seconds"] = round(processing_time, 3)
            
#             # Log response details
#             logger.info(f"DEBUG: Response status code: {response.status_code}")
#             logger.info(f"DEBUG: Response headers: {dict(response.headers)}")
            
#             if response.status_code == 200:
#                 response_data = response.json()
#                 result["raw_api_response"] = response_data
                
#                 # Parse LLM response
#                 llm_response_text = response_data.get("response", "")
                
#                 try:
#                     # Parse JSON response from LLM
#                     if isinstance(llm_response_text, str):
#                         llm_parsed = json.loads(llm_response_text)
#                     else:
#                         llm_parsed = llm_response_text
                    
#                     result["llm_answer"] = llm_parsed
                    
#                     # Extract tools that were sent to LLM
#                     tools_data = llm_parsed.get("tools", [])
#                     tool_ids = llm_parsed.get("tool_id", [])
                    
#                     result["tools_sent_to_llm"] = {
#                         "count": len(tools_data),
#                         "tool_ids": tool_ids,
#                         "tools": tools_data
#                     }
                    
#                     result["status"] = "success"
                    
#                     # Get model info AFTER the query to compare
#                     logger.info("DEBUG: Getting model info after query...")
#                     model_info_after = self.get_model_info()
#                     result["model_info_after_query"] = model_info_after
                    
#                     # Show model info in console
#                     current_model = model_info_before.get("current_model", "Unknown")
#                     logger.info(f"✅ Success - {len(tools_data)} tools found in {processing_time:.2f}s using {current_model}")
                    
#                     # Check if model changed between calls
#                     if model_info_before.get("current_model") != model_info_after.get("current_model"):
#                         logger.warning(f"⚠️ Model changed between calls!")
#                         logger.warning(f"Before: {model_info_before.get('current_model')}")
#                         logger.warning(f"After: {model_info_after.get('current_model')}")
                    
#                 except json.JSONDecodeError as e:
#                     result["error"] = f"JSON parse error: {str(e)}"
#                     result["llm_answer"] = {"unparsed_response": llm_response_text}
#                     result["status"] = "parse_error"
#                     logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
                    
#             else:
#                 result["status"] = "api_error"
#                 result["error"] = f"HTTP {response.status_code}: {response.text}"
#                 logger.error(f"❌ API error: {response.status_code}")
#                 logger.error(f"Response text: {response.text}")
                
#         except Exception as e:
#             result["status"] = "request_error"
#             result["error"] = str(e)
#             result["processing_time_seconds"] = time.time() - start_time
#             logger.error(f"❌ Request failed: {e}")
        
#         # Save result to file
#         safe_query = query.replace(' ', '_').replace('/', '_').replace('?', '').replace('!', '').replace('"', '').replace("'", '').replace(':', '').replace('*', '').replace('<', '').replace('>', '').replace('|', '')[:30]
#         filename = f"query_test_{timestamp}_{safe_query}.json"
        
#         with open(filename, 'w') as f:
#             json.dump(result, f, indent=2)
        
#         logger.info(f"💾 Result saved to {filename}")
#         return result

# def main():
#     """Main function with interactive input and enhanced debugging"""
#     print("=" * 60)
#     print("AI Tool Search API - Enhanced Debug Query Tester")
#     print("=" * 60)
    
#     # Initialize tester
#     tester = SimpleAPITester("http://localhost:8000")  # Change if needed
    
#     # Test connectivity
#     if not tester.test_connectivity():
#         print("❌ Cannot connect to API. Make sure it's running.")
#         return
    
#     # Get model info first
#     model_info = tester.get_model_info()
#     if model_info:
#         current_model = model_info.get("current_model", "Unknown")
#         environment = model_info.get("environment", "Unknown")
#         print(f"🤖 Using model: {current_model} (Environment: {environment})")
    
#     # Get stats
#     stats = tester.get_stats()
#     if stats:
#         total_tools = stats.get("total_vectors", 0)
#         print(f"📊 API has {total_tools} tools loaded")
    
#     # Interactive testing loop
#     print("\n🔍 Enter your queries to test (type 'quit' or 'exit' to stop):")
#     print("=" * 60)
    
#     while True:
#         try:
#             # Get user input
#             query = input("\nEnter your query: ").strip()
            
#             # Check for exit commands
#             if query.lower() in ['quit', 'exit', 'q', '']:
#                 print("👋 Goodbye!")
#                 break
            
#             # Test the query
#             print(f"🔍 Testing: '{query}'")
#             result = tester.test_query(query)
            
#             # Show results
#             if result["status"] == "success":
#                 tools_count = result["tools_sent_to_llm"]["count"]
#                 processing_time = result["processing_time_seconds"]
#                 tool_names = [tool.get("name", "Unknown") for tool in result["tools_sent_to_llm"]["tools"]]
                
#                 print(f"✅ Success! Found {tools_count} tools in {processing_time:.2f}s")
#                 print(f"🛠️  Tools found: {', '.join(tool_names[:3])}{'...' if len(tool_names) > 3 else ''}")
#                 print(f"📁 Saved to JSON file")
                
#                 # Show model debug info
#                 model_before = result.get("model_info_before_query", {}).get("current_model", "Unknown")
#                 model_after = result.get("model_info_after_query", {}).get("current_model", "Unknown")
#                 print(f"🤖 Model used: {model_before}")
#                 if model_before != model_after:
#                     print(f"⚠️  Model changed during request: {model_before} → {model_after}")
#             else:
#                 print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            
#             print("-" * 50)
            
#         except KeyboardInterrupt:
#             print("\n👋 Interrupted. Goodbye!")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")
    
#     print("\n" + "=" * 60)
#     print("All test results saved as JSON files in current directory.")
#     print("Check the 'debug_info' section in JSON files for detailed debugging.")
#     print("=" * 60)

# # Additional helper function for batch testing
# def test_multiple_queries():
#     """Test multiple queries - call this function separately if needed"""
#     queries = [
#         "AI writing tools for content creation",
#         "free video editing software",
#         "project management tools for teams", 
#         "AI image generators",
#         "code review automation tools"
#     ]
    
#     tester = SimpleAPITester()
    
#     print(f"Testing {len(queries)} queries...")
#     for i, query in enumerate(queries, 1):
#         print(f"\n[{i}/{len(queries)}] Testing: {query}")
#         result = tester.test_query(query)
#         time.sleep(1)  # Small delay between requests

# if __name__ == "__main__":
#     main()