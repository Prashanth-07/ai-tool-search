#!/usr/bin/env python3
"""
AI Tool Search API - Structured Output Testing Script

Tests the /query endpoint with structured outputs only and saves results as individual JSON files.
Captures: input_query, tools_sent_to_llm, LLM_answer for analysis.
Now focused on structured outputs with openai/gpt-oss-120b model.
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructuredAPITester:
    """API tester focused on structured outputs"""
    
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
    
    def test_query(self, query: str) -> dict:
        """Test a single query and save results with enhanced debugging"""
        logger.info(f"🔍 Testing query: '{query}' with structured outputs")
        
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get model info BEFORE making the query
        logger.info("DEBUG: Getting model info before query...")
        model_info_before = self.get_model_info()
        
        # Prepare result structure with enhanced debug info
        result = {
            "timestamp": datetime.now().isoformat(),
            "input_query": query,
            "flow_type": "structured",
            "model_info_before_query": model_info_before,
            "debug_info": {
                "headers_sent": {},
                "structured_output_debug": {},
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
                "User-Agent": "structured-test-script/1.0"
            }
            
            # Log debug info about headers
            logger.info(f"DEBUG: Headers being sent: {headers}")
            result["debug_info"]["headers_sent"] = dict(headers)
            
            # Make API request (no flow_type parameter needed)
            payload = {"query": query}
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
                    actual_flow_type = llm_parsed.get("flow_type", "structured")
                    model_used = llm_parsed.get("model_used", "Unknown")
                    
                    result["tools_sent_to_llm"] = {
                        "count": len(tools_data),
                        "tool_ids": tool_ids,
                        "tools": tools_data
                    }
                    
                    result["actual_flow_type"] = actual_flow_type
                    result["model_used"] = model_used
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
                    logger.info(f"🎯 Model used for structured outputs: {model_used}")
                    
                    # Show structured output specific info
                    if "processing_stats" in llm_parsed:
                        stats = llm_parsed["processing_stats"]
                        logger.info(f"📊 Processing stats: {stats}")
                    
                    # Check if model changed between calls
                    if model_info_before.get("current_groq_model") != model_info_after.get("current_groq_model"):
                        logger.warning(f"⚠️ Groq model changed between calls!")
                        logger.warning(f"Before: {model_info_before.get('current_groq_model')}")
                        logger.warning(f"After: {model_info_after.get('current_groq_model')}")
                    
                    # Log structured output quality
                    logger.info(f"🏗️ Structured output validation: JSON parsed successfully")
                    if "error" not in llm_parsed:
                        logger.info(f"✅ Structured output quality: No errors detected")
                    
                except json.JSONDecodeError as e:
                    result["error"] = f"JSON parse error: {str(e)}"
                    result["llm_answer"] = {"unparsed_response": llm_response_text}
                    result["status"] = "parse_error"
                    logger.error(f"❌ Failed to parse structured LLM response as JSON: {e}")
                    
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
        filename = f"structured_query_{timestamp}_{safe_query}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"💾 Result saved to {filename}")
        return result

    def test_batch_queries(self, queries: List[str]) -> List[dict]:
        """Test multiple queries in batch"""
        results = []
        
        logger.info(f"🔍 Testing {len(queries)} queries with structured outputs...")
        
        for i, query in enumerate(queries, 1):
            logger.info(f"\n[{i}/{len(queries)}] Testing: {query}")
            result = self.test_query(query)
            results.append(result)
            
            # Small delay between requests
            if i < len(queries):
                time.sleep(1)
        
        # Save batch summary
        batch_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "successful_queries": len([r for r in results if r["status"] == "success"]),
            "failed_queries": len([r for r in results if r["status"] != "success"]),
            "average_processing_time": sum(r["processing_time_seconds"] for r in results) / len(results),
            "queries_tested": queries,
            "results_summary": [
                {
                    "query": r["input_query"],
                    "status": r["status"],
                    "tools_found": r["tools_sent_to_llm"]["count"] if r["status"] == "success" else 0,
                    "processing_time": r["processing_time_seconds"]
                }
                for r in results
            ]
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_filename = f"batch_test_summary_{timestamp}.json"
        
        with open(summary_filename, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        
        logger.info(f"📊 Batch summary saved to {summary_filename}")
        return results

def main():
    """Main function with interactive input and structured output focus"""
    print("=" * 70)
    print("AI Tool Search API - Structured Output Testing")
    print("Using openai/gpt-oss-120b model with Groq structured outputs")
    print("=" * 70)
    
    # Initialize tester
    tester = StructuredAPITester("http://localhost:8000")  # Change if needed
    
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
        print(f"🎯 Structured outputs: openai/gpt-oss-120b")
    
    # Get stats
    stats = tester.get_stats()
    if stats:
        total_tools = stats.get("total_vectors", 0)
        print(f"📊 API has {total_tools} tools loaded")
    
    # Show testing options
    print("\n🔍 Testing Options:")
    print("  1. Interactive testing (enter queries one by one)")
    print("  2. Batch testing (predefined queries)")
    print("  3. Custom batch testing (enter multiple queries)")
    
    while True:
        try:
            choice = input("\nSelect option (1-3) or 'quit' to exit: ").strip()
            
            if choice.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            elif choice == "1":
                # Interactive testing
                print("\n🔍 Interactive Mode - Enter your queries (type 'back' to return to menu):")
                print("=" * 70)
                
                while True:
                    try:
                        query = input("\nEnter your query: ").strip()
                        
                        if query.lower() in ['back', 'menu']:
                            break
                        
                        if query.lower() in ['quit', 'exit', 'q']:
                            print("👋 Goodbye!")
                            return
                        
                        if not query:
                            continue
                        
                        # Test the query
                        print(f"🔍 Testing: '{query}' with structured outputs")
                        result = tester.test_query(query)
                        
                        # Show results
                        if result["status"] == "success":
                            tools_count = result["tools_sent_to_llm"]["count"]
                            processing_time = result["processing_time_seconds"]
                            model_used = result.get("model_used", "Unknown")
                            tool_names = [tool.get("name", "Unknown") for tool in result["tools_sent_to_llm"]["tools"]]
                            
                            print(f"✅ Success! Found {tools_count} tools in {processing_time:.2f}s")
                            print(f"🎯 Model: {model_used}")
                            print(f"🛠️  Tools found: {', '.join(tool_names[:3])}{'...' if len(tool_names) > 3 else ''}")
                            print(f"📁 Saved to JSON file")
                            
                            # Show structured output quality
                            if "error" not in result["llm_answer"]:
                                print(f"🏗️ Structured output: Valid JSON schema")
                            
                        else:
                            print(f"❌ Test failed: {result.get('error', 'Unknown error')}")
                        
                        print("-" * 50)
                        
                    except KeyboardInterrupt:
                        print("\n↩️  Returning to main menu...")
                        break
                    except Exception as e:
                        print(f"❌ Error: {e}")
            
            elif choice == "2":
                # Predefined batch testing
                predefined_queries = [
    "Write PRD for me",
    "coding tools",
    "text to image",
    "AI writing tools for content creation",
    "free video editing software",
    "project management tools for teams",
    "code review automation tools"
]
                
                print(f"\n🚀 Running batch test with {len(predefined_queries)} predefined queries...")
                results = tester.test_batch_queries(predefined_queries)
                
                # Show summary
                successful = len([r for r in results if r["status"] == "success"])
                print(f"\n📊 Batch Test Complete:")
                print(f"   ✅ Successful: {successful}/{len(predefined_queries)}")
                print(f"   ❌ Failed: {len(predefined_queries) - successful}/{len(predefined_queries)}")
                print(f"   📁 Individual results saved as JSON files")
                print(f"   📊 Summary saved as batch_test_summary_*.json")
            
            elif choice == "3":
                # Custom batch testing
                print("\n📝 Enter your queries for batch testing (one per line, empty line to finish):")
                custom_queries = []
                
                while True:
                    query = input(f"Query {len(custom_queries) + 1}: ").strip()
                    if not query:
                        break
                    custom_queries.append(query)
                
                if custom_queries:
                    print(f"\n🚀 Running batch test with {len(custom_queries)} custom queries...")
                    results = tester.test_batch_queries(custom_queries)
                    
                    # Show summary
                    successful = len([r for r in results if r["status"] == "success"])
                    print(f"\n📊 Custom Batch Test Complete:")
                    print(f"   ✅ Successful: {successful}/{len(custom_queries)}")
                    print(f"   ❌ Failed: {len(custom_queries) - successful}/{len(custom_queries)}")
                    print(f"   📁 Individual results saved as JSON files")
                    print(f"   📊 Summary saved as batch_test_summary_*.json")
                else:
                    print("❌ No queries entered.")
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 'quit'.")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("All test results saved as JSON files in current directory.")
    print("Files starting with 'structured_query_' contain individual test results.")
    print("Files starting with 'batch_test_summary_' contain batch test summaries.")
    print("Focus: Structured outputs with openai/gpt-oss-120b model")
    print("=" * 70)

def test_specific_scenarios():
    """Test specific scenarios for structured outputs - call this function separately if needed"""
    tester = StructuredAPITester()
    
    # Test scenarios that might challenge structured outputs
    challenging_queries = [
        "very specific niche query about quantum computing tools",
        "tools with numbers like top 5 best",
        "misspelled querry with typos",
        "extremely long query about comprehensive project management solutions with advanced features for enterprise teams that need collaboration, reporting, analytics, and integration capabilities",
        "single word: analytics",
        "question marks? exclamation points! special characters #$%",
        "mixed case QUERY with Different Capitalizations"
    ]
    
    print("🧪 Testing challenging scenarios for structured outputs...")
    results = tester.test_batch_queries(challenging_queries)
    
    # Analyze structured output quality
    schema_valid = 0
    for result in results:
        if result["status"] == "success" and "error" not in result["llm_answer"]:
            schema_valid += 1
    
    print(f"\n🏗️ Structured Output Quality Analysis:")
    print(f"   📊 Valid schema responses: {schema_valid}/{len(results)}")
    print(f"   📈 Success rate: {(schema_valid/len(results))*100:.1f}%")

if __name__ == "__main__":
    main()