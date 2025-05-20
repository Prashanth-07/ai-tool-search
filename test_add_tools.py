import requests
import json
import uuid
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change if your API is running elsewhere
ADD_TOOLS_ENDPOINT = f"{API_BASE_URL}/add-tools"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"

def test_search_filtering():
    """Test the search filtering feature with searchFrom parameter"""
    print("=== Testing Search Filtering with searchFrom parameter ===\n")
    
    # First, add some test tools to search from
    test_data = {
        "tools": [
            {
                "tool_id": str(uuid.uuid4()),
                "name": "TEST_SEARCH_ImageGenerator",
                "url": "https://imagegen.test",
                "description": "AI-powered image generation tool for creating custom artwork",
                "category_subcat": "AI Image Generation, Graphics",
                "pricingType": "Freemium"
            },
            {
                "tool_id": str(uuid.uuid4()),
                "name": "TEST_SEARCH_CodeAssistant",
                "url": "https://codeassist.test",
                "description": "AI coding assistant that helps developers write better code",
                "category_subcat": "Developer Tools, AI Coding",
                "pricingType": "Subscription"
            },
            {
                "tool_id": str(uuid.uuid4()),
                "name": "TEST_SEARCH_MarketingHelper",
                "url": "https://marketingai.test",
                "description": "AI marketing tool for content creation and campaign optimization",
                "category_subcat": "Marketing, AI Content",
                "pricingType": "Free"
            }
        ]
    }
    
    print("Adding test tools for search filtering test...")
    add_response = requests.post(
        ADD_TOOLS_ENDPOINT,
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    
    if add_response.status_code != 200:
        print(f"Error adding test tools: {add_response.text}")
        return
    
    # Get the tool IDs from the response
    add_result = add_response.json()
    tool_ids = []
    tool_names = {}
    
    for item in add_result["results"]:
        if item["status"] == "added":
            tool_id = item["tool"]["tool_id"]
            tool_ids.append(tool_id)
            tool_names[tool_id] = item["tool"]["name"]
    
    if not tool_ids:
        print("No tools were successfully added. Cannot continue with test.")
        return
    
    print(f"Successfully added {len(tool_ids)} test tools:")
    for tool_id, name in tool_names.items():
        print(f"  - {name} (ID: {tool_id})")
    
    # Wait for indexing to complete
    print("\nWaiting for indexing to complete...")
    time.sleep(3)
    
    # Now test various search filtering scenarios
    print("\n=== Testing Different Search Scenarios ===")
    
    # Helper function to safely parse JSON
    def safely_parse_json(response):
        try:
            result_json = response.json()
            if "response" in result_json:
                try:
                    return json.loads(result_json["response"])
                except json.JSONDecodeError:
                    print(f"  Failed to parse JSON in response: {result_json['response'][:100]}...")
                    return None
            return result_json
        except Exception as e:
            print(f"  Error parsing response: {str(e)}")
            return None
    
    # Scenario 1: Search for "image" across all tools
    print("\nScenario 1: Search for 'image' across all tools")
    response1 = requests.post(
        QUERY_ENDPOINT,
        json={"query": "image generation tool"},
        headers={"Content-Type": "application/json"}
    )
    
    if response1.status_code == 200:
        print("  Search successful (status 200)")
        result1 = safely_parse_json(response1)
        
        if result1 and "tools" in result1 and result1["tools"]:
            print(f"  Found {len(result1['tools'])} tools in full search")
        else:
            print("  No tools found in full search or couldn't parse response")
    else:
        print(f"  Error in full search: {response1.text}")
    
    # Scenario 2: Filter search to only include the ImageGenerator tool
    print("\nScenario 2: Filter search for 'image' to only include the ImageGenerator tool")
    image_tool_id = next((id for id, name in tool_names.items() if "Image" in name), None)
    
    if image_tool_id:
        response2 = requests.post(
            QUERY_ENDPOINT,
            json={
                "query": "generation tool",
                "searchFrom": [image_tool_id]
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response2.status_code == 200:
            print("  Filtered search successful (status 200)")
            # Print the raw response for debugging
            print(f"  Raw response: {response2.text[:150]}...")
            
            result2 = safely_parse_json(response2)
            
            if result2 and "tools" in result2 and result2["tools"]:
                print(f"  Found {len(result2['tools'])} tools in filtered search")
                
                # Check if search_filter_applied field is present
                if "search_filter_applied" in result2:
                    print(f"  Filter was applied: {result2['search_filter_applied']}")
                    
                # Print the found tools
                for tool in result2["tools"]:
                    print(f"    - {tool.get('name', 'Unknown')}")
                    
                # Verify filter was applied correctly
                if all(tool.get('id') == image_tool_id for tool in result2["tools"]):
                    print("  ✅ Filter appears to be correctly applied")
                else:
                    print("  ❌ Filter may not have been applied correctly")
            else:
                print("  No tools found in filtered search or couldn't parse response")
        else:
            print(f"  Error in filtered search: {response2.text}")
    else:
        print("  Could not find the ImageGenerator tool for testing")
    
    # Scenario 3: Filter search to include only coding and marketing tools
    print("\nScenario 3: Filter search for 'AI assistant' to only include coding and marketing tools")
    other_tool_ids = [id for id, name in tool_names.items() if "Image" not in name]
    
    if other_tool_ids:
        response3 = requests.post(
            QUERY_ENDPOINT,
            json={
                "query": "AI assistant",
                "searchFrom": other_tool_ids
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response3.status_code == 200:
            print("  Multi-tool filtered search successful (status 200)")
            result3 = safely_parse_json(response3)
            
            if result3 and "tools" in result3 and result3["tools"]:
                print(f"  Found {len(result3['tools'])} tools in multi-tool filtered search")
                
                # Check if search_filter_applied field is present
                if "search_filter_applied" in result3:
                    print(f"  Filter was applied: {result3['search_filter_applied']}")
                    
                # Print the found tools
                for tool in result3["tools"]:
                    print(f"    - {tool.get('name', 'Unknown')}")
            else:
                print("  No tools found in multi-tool filtered search or couldn't parse response")
        else:
            print(f"  Error in multi-tool filtered search: {response3.text}")
    else:
        print("  Could not find other tools for testing")
    
    print("\nSearch filtering test completed")

if __name__ == "__main__":
    test_search_filtering()