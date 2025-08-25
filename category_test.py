#!/usr/bin/env python3
"""
Test script for the categorize-tool endpoint.
This script takes a tool ID as input and hits the API endpoint to show categorization results.
"""

import requests
import json
import sys
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this if your API is hosted elsewhere
CATEGORIZE_ENDPOINT = "/categorize-tool"

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_subheader(title):
    """Print a formatted subheader."""
    print("\n" + "-"*60)
    print(f" {title}")
    print("-"*60)

def print_error(message):
    """Print an error message."""
    print(f"\n❌ ERROR: {message}")

def print_success(message):
    """Print a success message."""
    print(f"\n✅ {message}")

def test_categorize_tool(tool_id):
    """Test the categorize-tool endpoint with the given tool ID."""
    
    print_header(f"TESTING CATEGORIZE TOOL ENDPOINT")
    print(f"🎯 Tool ID: {tool_id}")
    print(f"🌐 API URL: {API_BASE_URL}")
    print(f"📅 Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Construct the full URL
    url = f"{API_BASE_URL}{CATEGORIZE_ENDPOINT}/{tool_id}"
    print(f"🔗 Full URL: {url}")
    
    try:
        # Record start time
        start_time = time.time()
        
        print_subheader("MAKING API REQUEST")
        print("⏳ Sending request...")
        
        # Make the API request
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            timeout=30  # 30 second timeout
        )
        
        # Record end time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"⏱️  Response time: {elapsed_time:.2f} seconds")
        print(f"📊 Status Code: {response.status_code}")
        
        # Handle different response status codes
        if response.status_code == 200:
            print_success("API request successful!")
            
            try:
                # Parse JSON response
                response_data = response.json()
                
                print_subheader("CATEGORIZATION RESULTS")
                
                # Display tool information
                tool_name = response_data.get('tool_name', 'Unknown')
                matched_categories = response_data.get('matched_categories', [])
                total_available = response_data.get('total_available_categories', 0)
                
                print(f"🔧 Tool Name: {tool_name}")
                print(f"📋 Total Available Categories: {total_available}")
                print(f"🎯 Matched Categories: {len(matched_categories)}")
                
                if matched_categories:
                    print("\n📂 SELECTED CATEGORIES:")
                    for i, category in enumerate(matched_categories, 1):
                        cat_id = category.get('id', 'N/A')
                        cat_name = category.get('name', 'N/A')
                        print(f"  {i:2d}. {cat_name}")
                        print(f"      ID: {cat_id}")
                        
                        # Add separator for readability
                        if i < len(matched_categories):
                            print(f"      {'-'*40}")
                else:
                    print("\n⚠️  No categories were matched for this tool.")
                
                # Display raw JSON response
                print_subheader("FULL JSON RESPONSE")
                print(json.dumps(response_data, indent=2, ensure_ascii=False))
                
            except json.JSONDecodeError as e:
                print_error(f"Failed to parse JSON response: {str(e)}")
                print_subheader("RAW RESPONSE")
                print(response.text)
                
        elif response.status_code == 404:
            print_error(f"Tool with ID '{tool_id}' not found")
            try:
                error_data = response.json()
                print(f"Details: {error_data.get('detail', 'No additional details')}")
            except:
                print(f"Raw response: {response.text}")
                
        elif response.status_code == 500:
            print_error("Internal server error")
            try:
                error_data = response.json()
                print(f"Details: {error_data.get('detail', 'No additional details')}")
            except:
                print(f"Raw response: {response.text}")
                
        elif response.status_code == 503:
            print_error("Service unavailable - system may be initializing")
            try:
                error_data = response.json()
                print(f"Details: {error_data.get('detail', 'No additional details')}")
            except:
                print(f"Raw response: {response.text}")
                
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            print_subheader("RAW RESPONSE")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print_error("Failed to connect to the API server")
        print("💡 Make sure the API server is running and accessible")
        print(f"   Check if {API_BASE_URL} is reachable")
        
    except requests.exceptions.Timeout:
        print_error("Request timed out")
        print("💡 The server might be processing a large request or is overloaded")
        
    except requests.exceptions.RequestException as e:
        print_error(f"Request failed: {str(e)}")
        
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")

def get_tool_id():
    """Get tool ID from command line arguments or user input."""
    
    # Check if tool ID is provided as command line argument
    if len(sys.argv) > 1:
        tool_id = sys.argv[1].strip()
        if tool_id:
            return tool_id
    
    # If not provided via command line, ask for user input
    print_header("CATEGORIZE TOOL TESTER")
    print("This script will test the categorize-tool endpoint")
    print(f"API Server: {API_BASE_URL}")
    
    while True:
        tool_id = input("\n🔧 Enter Tool ID to categorize: ").strip()
        
        if tool_id:
            return tool_id
        else:
            print("❌ Tool ID cannot be empty. Please try again.")

def check_api_health():
    """Check if the API server is accessible."""
    try:
        health_url = f"{API_BASE_URL}/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            print_success(f"API server is accessible at {API_BASE_URL}")
            return True
        else:
            print_error(f"API health check failed with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error(f"Cannot connect to API server at {API_BASE_URL}")
        print("💡 Please make sure the server is running")
        return False
        
    except Exception as e:
        print_error(f"Health check failed: {str(e)}")
        return False

def main():
    """Main function to run the test."""
    
    try:
        # Check API health first
        print_header("API SERVER HEALTH CHECK")
        if not check_api_health():
            print("\n⚠️  Continuing with test anyway...")
        
        # Get tool ID
        tool_id = get_tool_id()
        
        # Validate tool ID format (basic check)
        if len(tool_id) < 10:
            print(f"\n⚠️  Warning: Tool ID '{tool_id}' seems short. Are you sure it's correct?")
            confirm = input("Continue anyway? (y/N): ").strip().lower()
            if confirm != 'y':
                print("Test cancelled.")
                return
        
        # Run the test
        test_categorize_tool(tool_id)
        
        # Ask if user wants to test another tool
        print_subheader("TEST COMPLETE")
        another_test = input("\n🔄 Test another tool? (y/N): ").strip().lower()
        if another_test == 'y':
            main()  # Recursive call for another test
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test cancelled by user.")
        
    except Exception as e:
        print_error(f"Unexpected error in main: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Category Tool Test Script...")
    main()
    print("\n👋 Test script finished. Goodbye!")