"""
Complete End-to-End Test Suite for AI Tool Search API
=====================================================

This script tests ALL endpoints with detailed JSON logging:
- Root endpoint (/)
- Health endpoints (/basic-health, /health, /initialization-status)
- Tool management (/add-tools, /update-tools, /delete-tool, /reindex-tools)
- Search endpoints (/query, /query-suggestions, /popular-by-usecase)
- Utility endpoints (/stats, /model-info, /test-connection, /extract-keywords)
- Admin endpoints (/clear-index)

Saves detailed JSON results with input/output for validation.

Usage: python complete_endpoint_test.py
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

# Configuration
BASE_URL = "http://localhost:8000"
TIMEOUT = 45
OUTPUT_FILE = f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Test data using your exact format
SAMPLE_TOOL_DATA = {
    "details": {
        "introduction": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
        "usage": "Boost your productivity with AI coding support.",
        "speciality": "Developer tools and code analysis."
    },
    "features": {
        "pros": [
            "Pro feature 1",
            "Pro feature 2", 
            "Pro feature 3"
        ],
        "cons": [
            "Con issue 1",
            "Con issue 2"
        ]
    },
    "metrics": {
        "functionality": 8.5,
        "innovation": 7.2,
        "performance": 9.1,
        "overall": 8.3,
        "easeOfUse": 9.0,
        "valueForMoney": 7.8
    },
    "tool_id": "test-tool-001",
    "name": "CodeWhiz",
    "categories": [
        {"Category": "Generative AI"},
        {"Category": "AI Marketing Tools"},
        {"Category": "AI Chatbots"},
        {"Category": "AI Image Generation"}
    ],
    "pricing": [
        {
            "planName": "Starter",
            "price": "0",
            "features": ["Feature A", "Feature B"],
            "isPopular": False,
            "_id": "68164f70c7c00fbf66ff98c4"
        },
        {
            "planName": "Pro", 
            "price": "29.99",
            "features": ["Feature A", "Feature B", "Feature C"],
            "isPopular": True,
            "_id": "68164f70c7c00fbf66ff98c5"
        },
        {
            "planName": "Premium",
            "price": "49.99",
            "features": ["Feature 1", "Feature 2", "Feature 3", "Feature 4"],
            "isPopular": False,
            "_id": "68164f70c7c00fbf66ff98c6"
        }
    ],
    "qaSection": [
        {
            "question": "What is this tool used for?",
            "answer": "Boost your productivity with AI coding support."
        },
        {
            "question": "What makes it special?",
            "answer": "Developer tools and code analysis."
        },
        {
            "question": "Does it have a free plan?",
            "answer": "Yes, a free plan is offered."
        },
        {
            "question": "Can I write code using it?",
            "answer": "Yes, you can write code using it."
        }
    ],
    "url": "https://codewhiz.ai",
    "pricingType": "Free",
    "description": "An AI-powered coding assistant that helps developers write better code faster",
    "image_url": "https://example.com/codewhiz.jpg",
    "owner": "CodeWhiz Inc",
    "status": "active"
}

class CompleteAPITester:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.added_tool_ids = []
        self.test_results = {
            "test_execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "base_url": base_url,
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "errors": 0,
                "success_rate": 0.0,
                "total_execution_time": 0.0
            },
            "test_results": []
        }
        self.start_time = time.time()
        
    def log_test_result(self, test_name: str, method: str, endpoint: str, 
                       input_data: Any, response_data: Any, status_code: int, 
                       response_time: float, success: bool, error_message: str = ""):
        """Log detailed test result with input/output"""
        
        result = {
            "test_name": test_name,
            "method": method,
            "endpoint": endpoint,
            "timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "response": {
                "status_code": status_code,
                "data": response_data,
                "response_time_seconds": round(response_time, 3)
            },
            "result": "PASS" if success else "FAIL",
            "error_message": error_message
        }
        
        self.test_results["test_results"].append(result)
        self.test_results["test_execution_summary"]["total_tests"] += 1
        
        if success:
            self.test_results["test_execution_summary"]["passed"] += 1
        else:
            self.test_results["test_execution_summary"]["failed"] += 1
            
        # Console logging
        icon = "✅" if success else "❌"
        print(f"{icon} {test_name}: {'PASS' if success else 'FAIL'}")
        if error_message:
            print(f"   Error: {error_message}")
        if response_time > 0:
            print(f"   Response time: {response_time:.3f}s")
        print()

    def make_request(self, method: str, endpoint: str, data: Any = None, 
                    params: Dict = None, headers: Dict = None) -> tuple:
        """Make HTTP request and return response data and timing"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, headers=headers, timeout=TIMEOUT)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, params=params, headers=headers, timeout=TIMEOUT)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, params=params, headers=headers, timeout=TIMEOUT)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, params=params, headers=headers, timeout=TIMEOUT)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw_response": response.text}
            
            return response_data, response.status_code, response_time, None
            
        except Exception as e:
            response_time = time.time() - start_time
            return None, 0, response_time, str(e)

    # Root and Health Endpoints
    def test_root_endpoint(self):
        """Test GET / endpoint"""
        print("🏠 Testing Root Endpoint")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Root Endpoint", "GET", "/", None, response_data, 
            status_code, response_time, success, error_msg
        )
        return success

    def test_basic_health(self):
        """Test GET /basic-health endpoint"""
        print("💓 Testing Basic Health Check")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/basic-health")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Basic Health Check", "GET", "/basic-health", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_comprehensive_health(self):
        """Test GET /health endpoint"""
        print("🏥 Testing Comprehensive Health Check")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/health")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Comprehensive Health Check", "GET", "/health", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_initialization_status(self):
        """Test GET /initialization-status endpoint"""
        print("🔄 Testing Initialization Status")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/initialization-status")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Initialization Status", "GET", "/initialization-status", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_model_info(self):
        """Test GET /model-info endpoint"""
        print("🧠 Testing Model Info")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/model-info")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Model Info", "GET", "/model-info", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_connection(self):
        """Test GET /test-connection endpoint"""
        print("🔗 Testing Connection")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/test-connection")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Test Connection", "GET", "/test-connection", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    # Stats Endpoints
    def test_stats(self, test_name_suffix: str = ""):
        """Test GET /stats endpoint"""
        test_name = f"Get Statistics{test_name_suffix}"
        print(f"📊 Testing {test_name}")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("GET", "/stats")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            test_name, "GET", "/stats", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_stats_limited(self):
        """Test GET /stats with show_all=false"""
        print("📊 Testing Limited Statistics")
        print("-" * 40)
        
        params = {"show_all": "false"}
        response_data, status_code, response_time, error = self.make_request("GET", "/stats", params=params)
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Invalid response" if not success else "")
        
        self.log_test_result(
            "Limited Statistics", "GET", "/stats?show_all=false", params, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    # Tool Management Endpoints
    def test_add_single_tool(self):
        """Test POST /add-tools with single tool"""
        print("📥 Testing Add Single Tool")
        print("-" * 40)
        
        input_data = {"tools": [SAMPLE_TOOL_DATA]}
        response_data, status_code, response_time, error = self.make_request("POST", "/add-tools", input_data)
        
        success = status_code == 200 and response_data is not None
        if success:
            self.added_tool_ids.append(SAMPLE_TOOL_DATA['tool_id'])
        
        error_msg = error if error else ("Tool addition failed" if not success else "")
        
        self.log_test_result(
            "Add Single Tool", "POST", "/add-tools", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_add_bulk_tools(self):
        """Test POST /add-tools with multiple tools"""
        print("📥 Testing Add Bulk Tools")
        print("-" * 40)
        
        # Create 3 variations
        bulk_tools = []
        for i in range(3):
            tool = SAMPLE_TOOL_DATA.copy()
            tool["tool_id"] = f"bulk-test-{i+1:03d}"
            tool["name"] = f"BulkTool {i+1}"
            tool["url"] = f"https://bulktool{i+1}.ai"
            tool["description"] = f"Bulk test tool number {i+1}"
            bulk_tools.append(tool)
        
        input_data = {"tools": bulk_tools}
        response_data, status_code, response_time, error = self.make_request("POST", "/add-tools", input_data)
        
        success = status_code == 200 and response_data is not None
        if success:
            self.added_tool_ids.extend([tool["tool_id"] for tool in bulk_tools])
        
        error_msg = error if error else ("Bulk addition failed" if not success else "")
        
        self.log_test_result(
            "Add Bulk Tools", "POST", "/add-tools", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_update_tools(self):
        """Test PUT /update-tools endpoint"""
        print("✏️ Testing Update Tools")
        print("-" * 40)
        
        if not self.added_tool_ids:
            error_msg = "No tools available to update"
            self.log_test_result(
                "Update Tools", "PUT", "/update-tools", None, None,
                0, 0, False, error_msg
            )
            return False
        
        # Update the first added tool
        updated_tool = SAMPLE_TOOL_DATA.copy()
        updated_tool["tool_id"] = self.added_tool_ids[0]
        updated_tool["name"] = "CodeWhiz Updated"
        updated_tool["description"] = "Updated AI-powered coding assistant"
        updated_tool["pricingType"] = "Freemium"
        
        input_data = {"tools": [updated_tool]}
        response_data, status_code, response_time, error = self.make_request("PUT", "/update-tools", input_data)
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Update failed" if not success else "")
        
        self.log_test_result(
            "Update Tools", "PUT", "/update-tools", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_delete_tool(self):
        """Test DELETE /delete-tool/{tool_id} endpoint"""
        print("🗑️ Testing Delete Tool")
        print("-" * 40)
        
        if not self.added_tool_ids:
            error_msg = "No tools available to delete"
            self.log_test_result(
                "Delete Tool", "DELETE", "/delete-tool/{tool_id}", None, None,
                0, 0, False, error_msg
            )
            return False
        
        tool_id_to_delete = self.added_tool_ids[-1]
        endpoint = f"/delete-tool/{tool_id_to_delete}"
        
        response_data, status_code, response_time, error = self.make_request("DELETE", endpoint)
        
        success = status_code == 200 and response_data is not None
        if success:
            self.added_tool_ids.remove(tool_id_to_delete)
        
        error_msg = error if error else ("Delete failed" if not success else "")
        
        self.log_test_result(
            "Delete Tool", "DELETE", endpoint, {"tool_id": tool_id_to_delete}, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_reindex_tools(self):
        """Test POST /reindex-tools endpoint"""
        print("🔄 Testing Reindex Tools")
        print("-" * 40)
        
        response_data, status_code, response_time, error = self.make_request("POST", "/reindex-tools")
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Reindex failed" if not success else "")
        
        self.log_test_result(
            "Reindex Tools", "POST", "/reindex-tools", None, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    # Search Endpoints
    def test_query_tools(self):
        """Test POST /query endpoint with multiple queries"""
        print("🔍 Testing Query Tools")
        print("-" * 40)
        
        queries = [
            {"query": "AI coding assistant"},
            {"query": "generative AI tools"},
            {"query": "developer productivity tools"}
        ]
        
        all_success = True
        for i, query_data in enumerate(queries):
            response_data, status_code, response_time, error = self.make_request("POST", "/query", query_data)
            
            success = status_code == 200 and response_data is not None
            if not success:
                all_success = False
            
            error_msg = error if error else ("Query failed" if not success else "")
            
            self.log_test_result(
                f"Query Tools - Test {i+1}", "POST", "/query", query_data, response_data,
                status_code, response_time, success, error_msg
            )
        
        return all_success

    def test_filtered_search(self):
        """Test POST /query with searchFrom filter"""
        print("🔍 Testing Filtered Search")
        print("-" * 40)
        
        input_data = {
            "query": "AI tool",
            "searchFrom": self.added_tool_ids[:2] if len(self.added_tool_ids) >= 2 else self.added_tool_ids
        }
        
        response_data, status_code, response_time, error = self.make_request("POST", "/query", input_data)
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Filtered search failed" if not success else "")
        
        self.log_test_result(
            "Filtered Search", "POST", "/query", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    def test_query_suggestions(self):
        """Test POST /query-suggestions endpoint"""
        print("💡 Testing Query Suggestions")
        print("-" * 40)
        
        queries = [
            {"query": "AI coding"},
            {"query": "developer tools"},
            {"query": "automation"}
        ]
        
        all_success = True
        for i, query_data in enumerate(queries):
            response_data, status_code, response_time, error = self.make_request("POST", "/query-suggestions", query_data)
            
            success = status_code == 200 and response_data is not None
            if not success:
                all_success = False
            
            error_msg = error if error else ("Suggestions failed" if not success else "")
            
            self.log_test_result(
                f"Query Suggestions - Test {i+1}", "POST", "/query-suggestions", query_data, response_data,
                status_code, response_time, success, error_msg
            )
        
        return all_success

    def test_popular_by_usecase(self):
        """Test POST /popular-by-usecase endpoint"""
        print("🌟 Testing Popular by Use Case")
        print("-" * 40)
        
        use_cases = [
            {"use_case": "AI development"},
            {"use_case": "Code generation"},
            {"use_case": "Developer tools"}
        ]
        
        all_success = True
        for i, use_case_data in enumerate(use_cases):
            response_data, status_code, response_time, error = self.make_request("POST", "/popular-by-usecase", use_case_data)
            
            success = status_code == 200 and response_data is not None
            if not success:
                all_success = False
            
            error_msg = error if error else ("Popular by use case failed" if not success else "")
            
            self.log_test_result(
                f"Popular by Use Case - Test {i+1}", "POST", "/popular-by-usecase", use_case_data, response_data,
                status_code, response_time, success, error_msg
            )
        
        return all_success

    # Utility Endpoints
    def test_extract_keywords(self):
        """Test POST /extract-keywords endpoint"""
        print("🔑 Testing Extract Keywords")
        print("-" * 40)
        
        input_data = {
            "reviews": [
                "This tool is amazing and very user-friendly. Great for coding!",
                "Excellent AI assistant, but sometimes slow. Overall good experience.",
                "Love the features, very innovative. Could be more affordable though.",
                "Perfect for developers. Easy to use and powerful functionality.",
                "Great tool but the pricing is a bit expensive for small teams."
            ]
        }
        
        response_data, status_code, response_time, error = self.make_request("POST", "/extract-keywords", input_data)
        
        success = status_code == 200 and response_data is not None
        error_msg = error if error else ("Keyword extraction failed" if not success else "")
        
        self.log_test_result(
            "Extract Keywords", "POST", "/extract-keywords", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    # Admin Endpoints
    def test_clear_index(self):
        """Test POST /clear-index endpoint (with wrong API key for safety)"""
        print("🗑️ Testing Clear Index (with wrong key)")
        print("-" * 40)
        
        input_data = {"api_key": "wrong-api-key-for-testing"}
        response_data, status_code, response_time, error = self.make_request("POST", "/clear-index", input_data)
        
        # We expect this to fail with 401 for security
        success = status_code == 401
        error_msg = "Expected 401 unauthorized" if status_code != 401 else ""
        
        self.log_test_result(
            "Clear Index (Unauthorized Test)", "POST", "/clear-index", input_data, response_data,
            status_code, response_time, success, error_msg
        )
        return success

    # Error Handling Tests
    def test_error_scenarios(self):
        """Test various error scenarios"""
        print("⚠️ Testing Error Scenarios")
        print("-" * 40)
        
        # Test empty query
        empty_query = {"query": ""}
        response_data, status_code, response_time, error = self.make_request("POST", "/query", empty_query)
        success = status_code == 422  # Should return validation error
        
        self.log_test_result(
            "Empty Query Validation", "POST", "/query", empty_query, response_data,
            status_code, response_time, success, "Expected 422 validation error"
        )
        
        # Test invalid tool ID delete
        invalid_endpoint = "/delete-tool/non-existent-tool-id"
        response_data, status_code, response_time, error = self.make_request("DELETE", invalid_endpoint)
        success = status_code == 404  # Should return not found
        
        self.log_test_result(
            "Invalid Tool ID Delete", "DELETE", invalid_endpoint, {"tool_id": "non-existent-tool-id"}, response_data,
            status_code, response_time, success, "Expected 404 not found"
        )
        
        return True

    def cleanup_remaining_tools(self):
        """Clean up any remaining test tools"""
        print("🧹 Cleaning up remaining test tools...")
        cleanup_results = []
        
        for tool_id in self.added_tool_ids[:]:
            try:
                response_data, status_code, response_time, error = self.make_request("DELETE", f"/delete-tool/{tool_id}")
                
                if status_code == 200:
                    print(f"   ✅ Cleaned up tool: {tool_id}")
                    self.added_tool_ids.remove(tool_id)
                    cleanup_results.append({"tool_id": tool_id, "status": "success"})
                else:
                    print(f"   ❌ Failed to clean up tool: {tool_id}")
                    cleanup_results.append({"tool_id": tool_id, "status": "failed", "error": f"Status: {status_code}"})
                    
            except Exception as e:
                print(f"   ❌ Error cleaning up tool {tool_id}: {str(e)}")
                cleanup_results.append({"tool_id": tool_id, "status": "error", "error": str(e)})
        
        return cleanup_results

    def save_results_to_file(self):
        """Save test results to JSON file"""
        total_time = time.time() - self.start_time
        
        # Update summary
        summary = self.test_results["test_execution_summary"]
        summary["total_execution_time"] = round(total_time, 2)
        if summary["total_tests"] > 0:
            summary["success_rate"] = round((summary["passed"] / summary["total_tests"]) * 100, 2)
        
        # Add cleanup results
        cleanup_results = self.cleanup_remaining_tools()
        summary["cleanup_results"] = cleanup_results
        
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2, ensure_ascii=False)
            print(f"📄 Test results saved to: {OUTPUT_FILE}")
            return True
        except Exception as e:
            print(f"❌ Failed to save results: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all endpoint tests in logical order"""
        print("🚀 Starting Complete API Test Suite")
        print("=" * 70)
        print(f"🎯 Target URL: {self.base_url}")
        print(f"📄 Results will be saved to: {OUTPUT_FILE}")
        print()
        
        # Test execution order (dependencies considered)
        test_groups = [
            ("System Health", [
                ("Root Endpoint", self.test_root_endpoint),
                ("Basic Health Check", self.test_basic_health),
                ("Comprehensive Health Check", self.test_comprehensive_health),
                ("Initialization Status", self.test_initialization_status),
                ("Model Info", self.test_model_info),
                ("Test Connection", self.test_connection),
            ]),
            ("Statistics (Before)", [
                ("Initial Statistics", lambda: self.test_stats(" (Initial)")),
                ("Limited Statistics", self.test_stats_limited),
            ]),
            ("Tool Management", [
                ("Add Single Tool", self.test_add_single_tool),
                ("Add Bulk Tools", self.test_add_bulk_tools),
                ("Update Tools", self.test_update_tools),
            ]),
            ("Search & Query", [
                ("Query Tools", self.test_query_tools),
                ("Filtered Search", self.test_filtered_search),
                ("Query Suggestions", self.test_query_suggestions),
                ("Popular by Use Case", self.test_popular_by_usecase),
            ]),
            ("Utility Features", [
                ("Extract Keywords", self.test_extract_keywords),
            ]),
            ("Statistics (After)", [
                ("Final Statistics", lambda: self.test_stats(" (Final)")),
            ]),
            ("Tool Operations", [
                ("Delete Tool", self.test_delete_tool),
                ("Reindex Tools", self.test_reindex_tools),
            ]),
            ("Admin & Security", [
                ("Clear Index (Unauthorized)", self.test_clear_index),
            ]),
            ("Error Handling", [
                ("Error Scenarios", self.test_error_scenarios),
            ]),
        ]
        
        for group_name, tests in test_groups:
            print(f"📋 Running {group_name} Tests")
            print("=" * 50)
            
            for test_name, test_func in tests:
                print(f"⏳ {test_name}")
                try:
                    test_func()
                except Exception as e:
                    self.test_results["test_execution_summary"]["errors"] += 1
                    self.log_test_result(
                        test_name, "UNKNOWN", "UNKNOWN", None, None,
                        0, 0, False, f"Unexpected error: {str(e)}"
                    )
                    print(f"❌ {test_name}: ERROR - {str(e)}")
                    print()
            
            print()
        
        # Generate final summary
        self.generate_console_summary()
        
        # Save results to file
        self.save_results_to_file()
        
        # Return success status
        summary = self.test_results["test_execution_summary"]
        return summary["failed"] == 0 and summary["errors"] == 0

    def generate_console_summary(self):
        """Generate and display console summary"""
        summary = self.test_results["test_execution_summary"]
        total_time = time.time() - self.start_time
        
        print("=" * 70)
        print("📊 TEST EXECUTION SUMMARY")
        print("=" * 70)
        print(f"Total Tests:     {summary['total_tests']}")
        print(f"✅ Passed:       {summary['passed']}")
        print(f"❌ Failed:       {summary['failed']}")
        print(f"🔥 Errors:       {summary['errors']}")
        print(f"Success Rate:    {summary.get('success_rate', 0):.1f}%")
        print(f"Total Time:      {total_time:.2f}s")
        print()
        
        # Show failed/error tests
        failed_tests = [test for test in self.test_results["test_results"] if test["result"] == "FAIL"]
        if failed_tests:
            print("⚠️  FAILED TESTS:")
            print("-" * 40)
            for test in failed_tests:
                print(f"❌ {test['test_name']}")
                print(f"   Endpoint: {test['method']} {test['endpoint']}")
                print(f"   Status: {test['response']['status_code']}")
                if test['error_message']:
                    print(f"   Error: {test['error_message']}")
                print()
        
        # Show performance insights
        all_tests = self.test_results["test_results"]
        slowest_tests = sorted([t for t in all_tests if t['response']['response_time_seconds'] > 0], 
                              key=lambda x: x['response']['response_time_seconds'], reverse=True)[:5]
        
        if slowest_tests:
            print("📈 PERFORMANCE INSIGHTS:")
            print("-" * 40)
            for i, test in enumerate(slowest_tests, 1):
                print(f"{i}. {test['test_name']}: {test['response']['response_time_seconds']:.3f}s")
            print()
        
        print("✅ Test execution completed!")


def main():
    """Main function to run all tests"""
    try:
        # Test API connectivity first
        print("🔗 Testing API connectivity...")
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"❌ API not reachable at {BASE_URL}")
            return False
        print("✅ API is reachable")
        print()
        
        # Run comprehensive tests
        tester = CompleteAPITester(BASE_URL)
        success = tester.run_all_tests()
        
        print(f"\n📄 Detailed results saved to: {OUTPUT_FILE}")
        print("📝 The JSON file contains:")
        print("   - Complete input/output for each test")
        print("   - Response times and status codes")
        print("   - Error messages and validation details")
        print("   - Test execution summary and statistics")
        
        return success
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API at {BASE_URL}: {str(e)}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)