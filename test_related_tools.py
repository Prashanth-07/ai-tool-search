#!/usr/bin/env python3
"""
Test script for Related Tools endpoint
Tests various scenarios and edge cases
Generates JSON log file with input/output data
"""

import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"  # Adjust if your API runs on different port
RELATED_TOOLS_ENDPOINT = f"{BASE_URL}/related-tools"

# JSON logging configuration
LOG_FILE = f"related_tools_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
test_log = {
    "test_session": {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "endpoint": RELATED_TOOLS_ENDPOINT
    },
    "tests": []
}

def save_test_log():
    """Save test log to JSON file."""
    try:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(test_log, f, indent=2, ensure_ascii=False)
        print(f"📁 Test log saved to: {LOG_FILE}")
    except Exception as e:
        print(f"❌ Error saving test log: {e}")

def log_test_result(test_name: str, tool_id: str, request_data: Dict, 
                   response_data: Dict, error: str = None):
    """Log test result to JSON structure."""
    test_entry = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "input": {
            "tool_id": tool_id,
            "endpoint": f"{RELATED_TOOLS_ENDPOINT}/{tool_id}",
            **request_data
        },
        "output": response_data,
        "error": error
    }
    test_log["tests"].append(test_entry)

def test_api_health():
    """Test if API is running and healthy."""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def get_sample_tool_ids():
    """Get some sample tool IDs from the stats endpoint."""
    print("🔍 Getting sample tool IDs...")
    try:
        response = requests.get(f"{BASE_URL}/stats", params={"show_all": False}, timeout=15)
        if response.status_code == 200:
            data = response.json()
            tool_ids = [vector.get("tool_id") for vector in data.get("vectors", [])[:5]]
            tool_ids = [tid for tid in tool_ids if tid and tid != "N/A"]
            print(f"✅ Found {len(tool_ids)} sample tool IDs: {tool_ids}")
            return tool_ids
        else:
            print(f"❌ Failed to get sample tools: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error getting sample tools: {e}")
        return []

def get_tool_details(tool_id: str) -> Dict[str, Any]:
    """Fetch tool details by searching for tool_id in the database."""
    try:
        # Use the existing /stats endpoint to get tool details
        response = requests.get(f"{BASE_URL}/stats", params={"show_all": True}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Find the tool with matching tool_id
            for vector in data.get("vectors", []):
                if vector.get("tool_id") == tool_id:
                    return {
                        "tool_id": vector.get("tool_id", "N/A"),
                        "name": vector.get("name", "N/A"),
                        "description": vector.get("description", "N/A"),
                        "categories": vector.get("categories", "N/A"),
                        "pricing": vector.get("pricing", "N/A")
                    }
            return {"error": f"Tool {tool_id} not found in database"}
        else:
            return {"error": f"Failed to fetch stats: {response.status_code}"}
    except Exception as e:
        return {"error": f"Exception fetching tool details: {str(e)}"}

def analyze_tool_relevance(source_tool: Dict, related_tools_details: List[Dict]) -> Dict:
    """Analyze how related the tools actually are."""
    analysis = {
        "source_tool": {
            "name": source_tool.get("name", "N/A"),
            "categories": source_tool.get("categories", "N/A"),
            "description_snippet": source_tool.get("description", "N/A")[:100] + "..." if source_tool.get("description") else "N/A"
        },
        "related_tools_analysis": [],
        "relevance_summary": {
            "total_related": len(related_tools_details),
            "same_category_count": 0,
            "similar_name_count": 0,
            "quality_score": 0
        }
    }
    
    source_name_words = set(source_tool.get("name", "").lower().split())
    source_categories = source_tool.get("categories", "").lower()
    
    for tool_detail in related_tools_details:
        if "error" in tool_detail:
            continue
            
        # Analyze similarity
        tool_name_words = set(tool_detail.get("name", "").lower().split())
        tool_categories = tool_detail.get("categories", "").lower()
        
        # Check category similarity
        category_match = False
        if source_categories and tool_categories:
            category_match = any(cat.strip() in tool_categories for cat in source_categories.split(","))
        
        # Check name similarity
        name_similarity = len(source_name_words.intersection(tool_name_words)) > 0
        
        # Create analysis entry
        tool_analysis = {
            "tool_id": tool_detail.get("tool_id", "N/A"),
            "name": tool_detail.get("name", "N/A"),
            "categories": tool_detail.get("categories", "N/A"),
            "description_snippet": tool_detail.get("description", "N/A")[:100] + "..." if tool_detail.get("description") else "N/A",
            "relevance_indicators": {
                "category_match": category_match,
                "name_similarity": name_similarity,
                "category_overlap": [cat.strip() for cat in source_categories.split(",") if cat.strip() in tool_categories] if source_categories and tool_categories else []
            }
        }
        
        analysis["related_tools_analysis"].append(tool_analysis)
        
        # Update summary counts
        if category_match:
            analysis["relevance_summary"]["same_category_count"] += 1
        if name_similarity:
            analysis["relevance_summary"]["similar_name_count"] += 1
    
    # Calculate quality score (0-100)
    total = len(related_tools_details)
    if total > 0:
        category_score = (analysis["relevance_summary"]["same_category_count"] / total) * 60
        name_score = (analysis["relevance_summary"]["similar_name_count"] / total) * 40
        analysis["relevance_summary"]["quality_score"] = round(category_score + name_score, 1)
    
    return analysis

def test_related_tools(tool_id: str, test_name: str = None):
    """Test the related tools endpoint with a specific tool ID."""
    test_name = test_name or f"tool_id: {tool_id}"
    print(f"\n🧪 Testing Related Tools - {test_name}")
    
    # Prepare request data for logging
    request_data = {
        "method": "GET",
        "timeout": 15
    }
    
    # Initialize response data
    response_data = {
        "status_code": None,
        "response_time_seconds": None,
        "related_tools": None,
        "related_tools_count": 0,
        "response_format_valid": False,
        "original_tool_excluded": False,
        "within_limit": False,
        "source_tool_details": None,
        "related_tools_details": [],
        "relevance_analysis": None
    }
    
    error_msg = None
    
    try:
        # First, get source tool details
        print("📋 Fetching source tool details...")
        source_tool_details = get_tool_details(tool_id)
        response_data["source_tool_details"] = source_tool_details
        
        if "error" not in source_tool_details:
            print(f"📝 Source Tool: {source_tool_details.get('name', 'Unknown')} | Categories: {source_tool_details.get('categories', 'N/A')}")
        
        # Now test the related tools endpoint
        start_time = time.time()
        response = requests.get(f"{RELATED_TOOLS_ENDPOINT}/{tool_id}", timeout=15)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_data["response_time_seconds"] = round(response_time, 2)
        response_data["status_code"] = response.status_code
        
        print(f"⏱️  Response time: {response_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            related_tools = response.json()
            response_data["related_tools"] = related_tools
            response_data["related_tools_count"] = len(related_tools)
            
            print(f"✅ Success! Found {len(related_tools)} related tools")
            print(f"📝 Related tool IDs: {related_tools}")
            
            # Validate response format
            if isinstance(related_tools, list):
                response_data["response_format_valid"] = True
                print("✅ Response format is correct (list of strings)")
                
                if len(related_tools) <= 6:
                    response_data["within_limit"] = True
                    print("✅ Returned ≤ 6 tools as expected")
                else:
                    print(f"⚠️  Returned {len(related_tools)} tools (expected ≤ 6)")
                
                # Check if original tool_id is not in results
                if tool_id not in related_tools:
                    response_data["original_tool_excluded"] = True
                    print("✅ Original tool not included in results")
                else:
                    print("⚠️  Original tool found in results (should be filtered out)")
                
                # Fetch details for each related tool
                print(f"\n🔍 Fetching details for {len(related_tools)} related tools...")
                related_tools_details = []
                
                for i, related_tool_id in enumerate(related_tools, 1):
                    print(f"  📋 Fetching tool {i}/{len(related_tools)}: {related_tool_id}")
                    tool_details = get_tool_details(related_tool_id)
                    related_tools_details.append(tool_details)
                    
                    if "error" not in tool_details:
                        print(f"    ✅ {tool_details.get('name', 'Unknown')} | Categories: {tool_details.get('categories', 'N/A')}")
                    else:
                        print(f"    ❌ {tool_details.get('error', 'Unknown error')}")
                
                response_data["related_tools_details"] = related_tools_details
                
                # Analyze relevance if source tool details are available
                if "error" not in source_tool_details:
                    print(f"\n🧮 Analyzing tool relevance...")
                    relevance_analysis = analyze_tool_relevance(source_tool_details, related_tools_details)
                    response_data["relevance_analysis"] = relevance_analysis
                    
                    # Print relevance summary
                    summary = relevance_analysis["relevance_summary"]
                    print(f"📊 Relevance Analysis:")
                    print(f"  • Same category: {summary['same_category_count']}/{summary['total_related']} tools")
                    print(f"  • Name similarity: {summary['similar_name_count']}/{summary['total_related']} tools")
                    print(f"  • Quality Score: {summary['quality_score']}/100")
                    
                    # Print individual tool analysis
                    print(f"\n📋 Individual Tool Analysis:")
                    for tool_analysis in relevance_analysis["related_tools_analysis"]:
                        indicators = tool_analysis["relevance_indicators"]
                        relevance_icons = []
                        if indicators["category_match"]:
                            relevance_icons.append("🎯 Category")
                        if indicators["name_similarity"]:
                            relevance_icons.append("📝 Name")
                        if not relevance_icons:
                            relevance_icons.append("❓ Unclear")
                        
                        print(f"  • {tool_analysis['name']} | {' + '.join(relevance_icons)}")
                        if indicators["category_overlap"]:
                            print(f"    Shared categories: {', '.join(indicators['category_overlap'])}")
                
            else:
                print(f"❌ Unexpected response format: {type(related_tools)}")
                response_data["raw_response"] = str(related_tools)
            
        elif response.status_code == 404:
            print(f"❌ Tool not found: {tool_id}")
            error_response = response.json()
            print(f"📝 Error: {error_response}")
            response_data["error_detail"] = error_response
            
        elif response.status_code == 503:
            print("⚠️  System still initializing, try again later")
            error_response = response.json()
            print(f"📝 Message: {error_response}")
            response_data["error_detail"] = error_response
            
        else:
            print(f"❌ Unexpected status: {response.status_code}")
            try:
                error_response = response.json()
                print(f"📝 Error: {error_response}")
                response_data["error_detail"] = error_response
            except:
                print(f"📝 Raw response: {response.text}")
                response_data["raw_response"] = response.text
        
    except requests.exceptions.Timeout:
        error_msg = "Request timed out (API might be processing)"
        print(f"❌ {error_msg}")
        response_data["timeout"] = True
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Request failed: {e}"
        print(f"❌ {error_msg}")
        response_data["request_error"] = str(e)
        
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ {error_msg}")
        response_data["unexpected_error"] = str(e)
    
    # Log the test result
    log_test_result(test_name, tool_id, request_data, response_data, error_msg)
    
    return response_data.get("related_tools")

def test_edge_cases():
    """Test edge cases and error scenarios."""
    print("\n🧪 Testing Edge Cases...")
    
    # Test 1: Non-existent tool ID
    print("\n--- Test: Non-existent tool ID ---")
    test_related_tools("non-existent-tool-123", "Non-existent tool")
    
    # Test 2: Empty tool ID
    print("\n--- Test: Empty tool ID ---")
    try:
        response = requests.get(f"{RELATED_TOOLS_ENDPOINT}/", timeout=10)
        print(f"Empty tool ID status: {response.status_code}")
    except Exception as e:
        print(f"Empty tool ID error: {e}")
    
    # Test 3: Special characters in tool ID
    print("\n--- Test: Special characters ---")
    test_related_tools("tool@#$%", "Special characters")

def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("🚀 Starting Related Tools Endpoint Test Suite")
    print("=" * 60)
    
    # Step 1: Check API health
    if not test_api_health():
        print("❌ Cannot proceed - API is not healthy")
        save_test_log()
        return
    
    # Step 2: Get sample tool IDs
    sample_tools = get_sample_tool_ids()
    if not sample_tools:
        print("⚠️  No sample tools found - testing with manual IDs")
        sample_tools = ["test-tool-1", "example-tool-2"]  # Fallback IDs
    
    # Step 3: Test with sample tools
    print(f"\n🧪 Testing with {len(sample_tools)} sample tools...")
    results = {}
    
    for i, tool_id in enumerate(sample_tools[:3], 1):  # Test first 3 tools
        related_tools = test_related_tools(tool_id, f"Sample tool {i}")
        results[tool_id] = related_tools
        time.sleep(1)  # Small delay between requests
    
    # Step 4: Test edge cases
    test_edge_cases()
    
    # Step 5: Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    successful_tests = sum(1 for result in results.values() if result is not None)
    total_tools_found = sum(len(result) for result in results.values() if result is not None)
    
    print(f"✅ Successful requests: {successful_tests}/{len(sample_tools)}")
    print(f"📈 Total related tools found: {total_tools_found}")
    print(f"📊 Average tools per request: {total_tools_found/max(successful_tests, 1):.1f}")
    
    if successful_tests > 0:
        print("\n🎉 Related Tools endpoint is working!")
        print("\n📝 Sample results:")
        for tool_id, related in results.items():
            if related is not None:
                print(f"  {tool_id} → {len(related)} related tools: {related[:3]}{'...' if len(related) > 3 else ''}")
    else:
        print("\n❌ All tests failed - check your API implementation")
    
    # Add summary to test log
    test_log["summary"] = {
        "total_tests": len(test_log["tests"]),
        "successful_tests": successful_tests,
        "total_tools_found": total_tools_found,
        "average_tools_per_request": round(total_tools_found/max(successful_tests, 1), 1)
    }
    
    # Save JSON log
    save_test_log()

def quick_test():
    """Quick test with a single tool ID (for manual testing)."""
    print("🔥 Quick Test Mode")
    
    # You can manually set a tool_id here if you know one exists
    tool_id = input("Enter a tool_id to test (or press Enter for auto-discovery): ").strip()
    
    if not tool_id:
        sample_tools = get_sample_tool_ids()
        if sample_tools:
            tool_id = sample_tools[0]
            print(f"Using auto-discovered tool: {tool_id}")
        else:
            print("❌ No tools found for testing")
            save_test_log()
            return
    
    test_related_tools(tool_id, "Quick Test")
    
    # Save JSON log
    save_test_log()

if __name__ == "__main__":
    print("Related Tools API Tester")
    print("Choose test mode:")
    print("1. Comprehensive test (recommended)")
    print("2. Quick test with specific tool_id")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        quick_test()
    else:
        run_comprehensive_test()