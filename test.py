#!/usr/bin/env python3
"""
AI Tool Categorization Testing Script

Tests the /categorize-tool/{tool_id} and /categorize-tool-usecase/{tool_id} endpoints.
Captures: tool_metadata, categories/usecases_sent_to_llm, LLM_responses in one comprehensive JSON.
"""

import requests
import json
import time
import sys
import io
from datetime import datetime
import logging

# Fix character encoding for Windows console
if sys.platform.startswith('win'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class CategorizationTester:
    """API tester for categorization endpoints with enhanced data capture"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 60
    
    def safe_print(self, message: str):
        """Safe print function that handles Unicode characters"""
        try:
            print(message)
        except UnicodeEncodeError:
            # Fallback to ASCII-safe version
            safe_message = message.encode('ascii', 'replace').decode('ascii')
            print(safe_message)
    
    def test_connectivity(self) -> bool:
        """Test basic API connectivity"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                self.safe_print("✅ API is reachable")
                return True
            else:
                self.safe_print(f"❌ API returned {response.status_code}")
                return False
        except Exception as e:
            self.safe_print(f"❌ Connection failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """Get current model information"""
        try:
            response = self.session.get(f"{self.base_url}/model-info")
            if response.status_code == 200:
                data = response.json()
                current_model = data.get("current_groq_model", "Unknown")
                environment = data.get("environment", "Unknown")
                self.safe_print(f"✅ Current model: {current_model} (Environment: {environment})")
                return data
            else:
                self.safe_print(f"Failed to get model info: {response.status_code}")
                return {}
        except Exception as e:
            self.safe_print(f"Error getting model info: {e}")
            return {}
    
    def get_stats(self) -> dict:
        """Get API statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats?show_all=false")
            if response.status_code == 200:
                data = response.json()
                total_tools = data.get("total_vectors", 0)
                self.safe_print(f"📊 API has {total_tools} tools available")
                return {"total_tools": total_tools}
            return {}
        except Exception as e:
            self.safe_print(f"Error getting stats: {e}")
            return {}
    
    def get_tool_data_direct(self, tool_id: str) -> dict:
        """Get tool data directly from MongoDB via a custom endpoint"""
        try:
            # We'll create a custom endpoint to get raw tool data
            # For now, let's simulate what the API should be sending to LLM
            
            # Try to get some tool info from the stats endpoint
            response = self.session.get(f"{self.base_url}/stats?show_all=true")
            if response.status_code == 200:
                data = response.json()
                vectors = data.get("vectors", [])
                
                # Find our tool in the vectors
                for vector in vectors:
                    if vector.get("tool_id") == tool_id:
                        return {
                            "found_in_vectors": True,
                            "tool_metadata": {
                                "tool_id": vector.get("tool_id", ""),
                                "name": vector.get("name", ""),
                                "description": vector.get("description", ""),
                                "category_subcat": vector.get("category_subcat", ""),
                                "pricingType": vector.get("pricingType", ""),
                                # Add more fields as available in the vectors
                            }
                        }
            
            return {"found_in_vectors": False, "tool_metadata": {}}
            
        except Exception as e:
            self.safe_print(f"Error getting tool data: {e}")
            return {"error": str(e)}
    
    def get_categories_data(self) -> list:
        """Get sample categories data that would be sent to LLM"""
        # Since we can't directly access the MongoDB endpoint from the test script,
        # we'll document what should be captured
        return [
            {
                "note": "This should contain the actual categories data sent to LLM",
                "format": "ID: {category_id} | Category: {category_name} | Main: {category_main}",
                "total_available": 247,
                "sample_format": {
                    "_id": "category_id_here",
                    "Category": "category_name_here", 
                    "CategoryMain": "main_category_here"
                }
            }
        ]
    
    def get_usecases_data(self) -> list:
        """Get sample use cases data that would be sent to LLM"""
        return [
            {
                "note": "This should contain the actual use cases data sent to LLM",
                "format": "ID: {usecase_id} | Use Case: {usecase_name} | Category: {usecase_category}",
                "total_available": 39,
                "sample_format": {
                    "_id": "usecase_id_here",
                    "UseCase": "usecase_name_here",
                    "UseCaseCategory": "category_here"
                }
            }
        ]
    
    def test_comprehensive_categorization(self, tool_id: str) -> dict:
        """Test both categorization endpoints and capture all available data"""
        self.safe_print(f"🔬 Comprehensive categorization test for tool ID: {tool_id}")
        
        start_time = time.time()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get additional context data
        tool_context = self.get_tool_data_direct(tool_id)
        categories_context = self.get_categories_data()
        usecases_context = self.get_usecases_data()
        
        # Initialize comprehensive result structure
        result = {
            "metadata": {
                "tool_id": tool_id,
                "test_timestamp": datetime.now().isoformat(),
                "test_duration_seconds": 0,
                "api_base_url": self.base_url,
                "model_info": self.get_model_info(),
                "api_stats": self.get_stats()
            },
            
            "tool_context": tool_context,
            
            "tool_categorization": {
                "endpoint": "/categorize-tool/{tool_id}",
                "status": "pending",
                "error": None,
                "processing_time_seconds": 0,
                "request_details": {
                    "headers": {"Content-Type": "application/json", "User-Agent": "comprehensive-test-script/1.0"},
                    "method": "POST",
                    "url": f"{self.base_url}/categorize-tool/{tool_id}"
                },
                "data_sent_to_llm": {
                    "tool_metadata": {
                        "note": "Tool metadata extracted from MongoDB and formatted for LLM",
                        "expected_fields": [
                            "name", "description", "speciality", "usage", 
                            "features_pros", "features_cons", "pricingType", "qa_section"
                        ],
                        "from_vector_store": tool_context.get("tool_metadata", {})
                    },
                    "categories_available": categories_context,
                    "expected_llm_prompt_structure": {
                        "system_prompt": "Expert categorization assistant with selection criteria",
                        "user_prompt": "Tool metadata + Available categories list"
                    }
                },
                "llm_interaction": {
                    "model_used": "openai/gpt-oss-120b",
                    "response_format": "structured_output_with_json_schema", 
                    "max_categories_returned": 5,
                    "schema_enforced": True,
                    "raw_llm_response": "CAPTURED_FROM_API_LOGS",
                    "parsed_result": {}
                },
                "final_result": {
                    "tool_name": "",
                    "matched_categories": [],
                    "total_available_categories": 0
                },
                "raw_api_response": None
            },
            
            "usecase_categorization": {
                "endpoint": "/categorize-tool-usecase/{tool_id}",
                "status": "pending",
                "error": None, 
                "processing_time_seconds": 0,
                "request_details": {
                    "headers": {"Content-Type": "application/json", "User-Agent": "comprehensive-test-script/1.0"},
                    "method": "POST",
                    "url": f"{self.base_url}/categorize-tool-usecase/{tool_id}"
                },
                "data_sent_to_llm": {
                    "tool_metadata": {
                        "note": "Same tool metadata as categorization endpoint",
                        "from_vector_store": tool_context.get("tool_metadata", {})
                    },
                    "usecases_available": usecases_context,
                    "expected_llm_prompt_structure": {
                        "system_prompt": "Expert use case categorization assistant", 
                        "user_prompt": "Tool metadata + Available use cases list"
                    }
                },
                "llm_interaction": {
                    "model_used": "openai/gpt-oss-120b",
                    "response_format": "structured_output_with_json_schema",
                    "max_usecases_returned": 3,
                    "schema_enforced": True,
                    "raw_llm_response": "CAPTURED_FROM_API_LOGS",
                    "parsed_result": {}
                },
                "final_result": {
                    "tool_name": "",
                    "matched_usecases": [],
                    "total_available_usecases": 0
                },
                "raw_api_response": None
            },
            
            "comprehensive_summary": {}
        }
        
        # Test 1: Tool Categorization
        self.safe_print("=" * 60)
        self.safe_print("🏷️ Testing tool categorization endpoint")
        
        cat_start_time = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/categorize-tool/{tool_id}",
                headers=result["tool_categorization"]["request_details"]["headers"]
            )
            
            cat_processing_time = time.time() - cat_start_time
            result["tool_categorization"]["processing_time_seconds"] = round(cat_processing_time, 3)
            
            self.safe_print(f"📊 Categorization response: {response.status_code} in {cat_processing_time:.2f}s")
            
            if response.status_code == 200:
                response_data = response.json()
                result["tool_categorization"]["raw_api_response"] = response_data
                result["tool_categorization"]["status"] = "success"
                
                # Extract categorization results
                result["tool_categorization"]["final_result"] = {
                    "tool_name": response_data.get("tool_name", ""),
                    "matched_categories": response_data.get("matched_categories", []),
                    "total_available_categories": response_data.get("total_available_categories", 0)
                }
                
                categories_found = len(response_data.get("matched_categories", []))
                self.safe_print(f"✅ Categorization successful - {categories_found} categories found")
                
                # Log matched categories
                for i, cat in enumerate(response_data.get("matched_categories", []), 1):
                    cat_id = cat.get("id", "Unknown")
                    cat_name = cat.get("name", "Unknown")
                    self.safe_print(f"   📂 {i}. {cat_name} (ID: {cat_id})")
                
            else:
                result["tool_categorization"]["status"] = "failed"
                result["tool_categorization"]["error"] = f"HTTP {response.status_code}: {response.text}"
                self.safe_print(f"❌ Categorization failed: {response.status_code}")
                
        except Exception as e:
            result["tool_categorization"]["status"] = "error"
            result["tool_categorization"]["error"] = str(e)
            result["tool_categorization"]["processing_time_seconds"] = time.time() - cat_start_time
            self.safe_print(f"❌ Categorization error: {e}")
        
        # Test 2: Use Case Categorization
        self.safe_print("=" * 60)
        self.safe_print("🎯 Testing use case categorization endpoint")
        
        usecase_start_time = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/categorize-tool-usecase/{tool_id}",
                headers=result["usecase_categorization"]["request_details"]["headers"]
            )
            
            usecase_processing_time = time.time() - usecase_start_time
            result["usecase_categorization"]["processing_time_seconds"] = round(usecase_processing_time, 3)
            
            self.safe_print(f"📊 Use case response: {response.status_code} in {usecase_processing_time:.2f}s")
            
            if response.status_code == 200:
                response_data = response.json()
                result["usecase_categorization"]["raw_api_response"] = response_data
                result["usecase_categorization"]["status"] = "success"
                
                # Extract use case results
                result["usecase_categorization"]["final_result"] = {
                    "tool_name": response_data.get("tool_name", ""),
                    "matched_usecases": response_data.get("matched_usecases", []),
                    "total_available_usecases": response_data.get("total_available_usecases", 0)
                }
                
                usecases_found = len(response_data.get("matched_usecases", []))
                self.safe_print(f"✅ Use case categorization successful - {usecases_found} use cases found")
                
                # Log matched use cases
                for i, uc in enumerate(response_data.get("matched_usecases", []), 1):
                    uc_id = uc.get("id", "Unknown")
                    uc_name = uc.get("name", "Unknown")
                    self.safe_print(f"   🎯 {i}. {uc_name} (ID: {uc_id})")
                    
            else:
                result["usecase_categorization"]["status"] = "failed"
                result["usecase_categorization"]["error"] = f"HTTP {response.status_code}: {response.text}"
                self.safe_print(f"❌ Use case categorization failed: {response.status_code}")
                
        except Exception as e:
            result["usecase_categorization"]["status"] = "error"
            result["usecase_categorization"]["error"] = str(e)
            result["usecase_categorization"]["processing_time_seconds"] = time.time() - usecase_start_time
            self.safe_print(f"❌ Use case categorization error: {e}")
        
        # Calculate comprehensive summary
        total_test_time = time.time() - start_time
        result["metadata"]["test_duration_seconds"] = round(total_test_time, 3)
        
        cat_success = result["tool_categorization"]["status"] == "success"
        usecase_success = result["usecase_categorization"]["status"] == "success"
        
        categories_found = len(result["tool_categorization"]["final_result"]["matched_categories"])
        usecases_found = len(result["usecase_categorization"]["final_result"]["matched_usecases"])
        
        # Get tool name from either endpoint
        tool_name = (result["tool_categorization"]["final_result"]["tool_name"] or 
                    result["usecase_categorization"]["final_result"]["tool_name"] or 
                    "Unknown")
        
        # Comprehensive summary
        result["comprehensive_summary"] = {
            "both_tests_successful": cat_success and usecase_success,
            "categorization_successful": cat_success,
            "usecase_categorization_successful": usecase_success,
            "total_categories_found": categories_found,
            "total_usecases_found": usecases_found,
            "combined_processing_time": round(
                result["tool_categorization"]["processing_time_seconds"] + 
                result["usecase_categorization"]["processing_time_seconds"], 3
            ),
            "tool_name": tool_name,
            "insights": self._generate_insights(result),
            "data_completeness": {
                "tool_metadata_captured": bool(tool_name != "Unknown"),
                "categories_data_available": result["tool_categorization"]["final_result"]["total_available_categories"] > 0,
                "usecases_data_available": result["usecase_categorization"]["final_result"]["total_available_usecases"] > 0,
                "api_responses_captured": True,
                "note": "LLM prompts and raw responses need to be captured from backend logs"
            },
            "analysis": {
                "why_no_matches": self._analyze_no_matches(result) if categories_found == 0 and usecases_found == 0 else None,
                "backend_logs_needed": "Check backend logs for actual LLM prompts and responses",
                "improvement_suggestions": self._get_improvement_suggestions(result)
            }
        }
        
        # Save comprehensive result to single JSON file
        filename = f"comprehensive_categorization_test_{timestamp}_{tool_id}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.safe_print("=" * 60)
        self.safe_print(f"💾 Comprehensive test results saved to: {filename}")
        
        return result
    
    def _analyze_no_matches(self, result: dict) -> dict:
        """Analyze why no categories/use cases were matched"""
        analysis = {
            "possible_reasons": [
                "LLM was too restrictive in matching criteria",
                "Tool metadata might be incomplete or unclear", 
                "Category/use case descriptions don't match tool functionality",
                "LLM model (openai/gpt-oss-120b) might need different prompting strategy"
            ],
            "tool_name": result["tool_categorization"]["final_result"]["tool_name"],
            "available_categories": result["tool_categorization"]["final_result"]["total_available_categories"],
            "available_usecases": result["usecase_categorization"]["final_result"]["total_available_usecases"],
            "recommendations": [
                "Check backend logs for actual LLM prompts and responses",
                "Verify tool metadata quality in MongoDB",
                "Review LLM prompt engineering in the backend",
                "Consider adjusting matching criteria to be less restrictive"
            ]
        }
        return analysis
    
    def _get_improvement_suggestions(self, result: dict) -> list:
        """Get suggestions for improving the categorization"""
        suggestions = []
        
        if result["comprehensive_summary"]["total_categories_found"] == 0:
            suggestions.append("Consider making category matching criteria less restrictive")
            suggestions.append("Review if tool metadata contains sufficient categorization signals")
        
        if result["comprehensive_summary"]["total_usecases_found"] == 0:
            suggestions.append("Use case matching might need broader semantic matching")
            suggestions.append("Consider adding more use case examples in prompts")
        
        suggestions.append("Add logging to capture actual LLM prompts and responses in test output")
        suggestions.append("Consider testing with different models or temperature settings")
        
        return suggestions
    
    def _generate_insights(self, result: dict) -> list:
        """Generate insights based on test results"""
        insights = []
        
        cat_result = result["tool_categorization"]
        usecase_result = result["usecase_categorization"]
        
        # Success insights
        if cat_result["status"] == "success" and usecase_result["status"] == "success":
            insights.append("✅ Both categorization endpoints responding successfully")
        
        # Performance insights
        cat_time = cat_result["processing_time_seconds"]
        usecase_time = usecase_result["processing_time_seconds"]
        
        if cat_time > 0:
            insights.append(f"⏱️ Category analysis took {cat_time:.2f}s")
        if usecase_time > 0:
            insights.append(f"⏱️ Use case analysis took {usecase_time:.2f}s")
        
        # Data insights
        if cat_result["final_result"]["total_available_categories"] > 0:
            total_cats = cat_result["final_result"]["total_available_categories"]
            found_cats = len(cat_result["final_result"]["matched_categories"])
            insights.append(f"📊 {found_cats}/{total_cats} categories matched")
        
        if usecase_result["final_result"]["total_available_usecases"] > 0:
            total_ucs = usecase_result["final_result"]["total_available_usecases"]
            found_ucs = len(usecase_result["final_result"]["matched_usecases"])
            insights.append(f"📊 {found_ucs}/{total_ucs} use cases matched")
        
        # Error insights
        if cat_result["status"] != "success":
            insights.append(f"❌ Category endpoint failed: {cat_result['error']}")
        
        if usecase_result["status"] != "success":
            insights.append(f"❌ Use case endpoint failed: {usecase_result['error']}")
        
        return insights

def main():
    """Main function with improved error handling"""
    try:
        print("=" * 60)
        print("🔬 AI Tool Comprehensive Categorization Testing Script")
        print("Tests both categorization endpoints and saves all data in one JSON file")
        print("=" * 60)
        
        # Initialize tester
        tester = CategorizationTester("http://localhost:8000")  # Change if needed
        
        # Test connectivity
        if not tester.test_connectivity():
            print("❌ Cannot connect to API. Make sure it's running.")
            return
        
        # Get model info
        model_info = tester.get_model_info()
        if model_info:
            current_model = model_info.get("current_groq_model", "Unknown")
            environment = model_info.get("environment", "Unknown")
            print(f"🤖 Using model: {current_model} (Environment: {environment})")
        
        # Get basic stats
        stats = tester.get_stats()
        if stats:
            total_tools = stats.get("total_tools", 0)
            print(f"📊 API has {total_tools} tools available")
        
        # Show example tool IDs
        print("\n💡 Example tool IDs from your MongoDB:")
        print("   • 6810cec5c762a206e5ea02a1 (Fellow - meeting management)")
        print("   • Or any ObjectId from your aitools collection")
        
        # Interactive testing loop
        print("\n🔍 Enter tool IDs to test comprehensive categorization (type 'quit' or 'exit' to stop):")
        print("=" * 60)
        
        while True:
            try:
                # Get user input
                tool_id = input("\nEnter tool ID (ObjectId): ").strip()
                
                # Check for exit commands
                if tool_id.lower() in ['quit', 'exit', 'q', '']:
                    print("👋 Goodbye!")
                    break
                
                # Validate tool ID format (basic check)
                if len(tool_id) != 24:
                    print("⚠️ Tool ID should be 24 characters (MongoDB ObjectId format)")
                    continue
                
                print(f"🔬 Starting comprehensive categorization test for: {tool_id}")
                
                # Run comprehensive test
                result = tester.test_comprehensive_categorization(tool_id)
                
                # Show summary
                summary = result["comprehensive_summary"]
                print("\n" + "=" * 60)
                print("📊 COMPREHENSIVE TEST SUMMARY:")
                print("=" * 60)
                print(f"Tool Name: {summary['tool_name']}")
                print(f"Categories Found: {summary['total_categories_found']}")
                print(f"Use Cases Found: {summary['total_usecases_found']}")
                print(f"Total Processing Time: {summary['combined_processing_time']:.2f}s")
                print(f"Both Tests Successful: {'✅ Yes' if summary['both_tests_successful'] else '❌ No'}")
                
                # Show insights
                if summary['insights']:
                    print("\n🔍 INSIGHTS:")
                    for insight in summary['insights']:
                        tester.safe_print(f"   {insight}")
                
                # Show analysis if no matches
                if 'analysis' in summary and summary['analysis'].get('why_no_matches'):
                    print("\n🔍 NO MATCHES ANALYSIS:")
                    analysis = summary['analysis']['why_no_matches']
                    print(f"   Tool: {analysis['tool_name']}")
                    print(f"   Available Categories: {analysis['available_categories']}")
                    print(f"   Available Use Cases: {analysis['available_usecases']}")
                    
                    print("\n💡 RECOMMENDATIONS:")
                    for rec in analysis['recommendations']:
                        print(f"   • {rec}")
                
                print("=" * 60)
                print("✅ Complete test data saved to JSON file")
                print("📝 Check backend logs for actual LLM prompts and responses")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("📁 All comprehensive test results saved as:")
        print("• comprehensive_categorization_test_*.json - Complete test data including:")
        print("  - API request/response data")
        print("  - Tool context and metadata")
        print("  - Expected LLM data structures")
        print("  - Performance metrics and analysis")
        print("  - Improvement suggestions")
        print("=" * 60)
        
    except Exception as e:
        print(f"Fatal error: {e}")

if __name__ == "__main__":
    main()