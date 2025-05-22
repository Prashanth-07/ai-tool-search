import requests
import json
import uuid
import time

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change if your API is running elsewhere
ADD_TOOLS_ENDPOINT = f"{API_BASE_URL}/add-tools"
UPDATE_TOOLS_ENDPOINT = f"{API_BASE_URL}/update-tools"
QUERY_ENDPOINT = f"{API_BASE_URL}/query"
DELETE_TOOL_ENDPOINT = f"{API_BASE_URL}/delete-tool"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"

class APITester:
    def __init__(self):
        self.test_tool_ids = []
        self.test_tools_data = {}
    
    def safely_parse_json(self, response):
        """Helper function to safely parse JSON responses"""
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
    
    def generate_test_tools(self):
        """Generate comprehensive test tools with new schema format"""
        tool_ids = [str(uuid.uuid4()) for _ in range(3)]
        
        print(f"Generated tool IDs for testing:")
        for i, tool_id in enumerate(tool_ids):
            print(f"  Tool {i+1}: {tool_id}")
        
        return {
            "tools": [
                {
                    "tool_id": tool_ids[0],
                    "name": "COMPREHENSIVE_TEST_AIWritingAssistant",
                    "url": "https://aiwritingassistant.test",
                    "description": "Advanced AI-powered writing assistant for content creation",
                    "category_subcat": "AI Writing Tools, Content Creation",
                    "pricingType": "Freemium",
                    "details": {
                        "introduction": "AIWritingAssistant is a comprehensive tool that helps writers create better content faster using advanced AI technology.",
                        "usage": "Perfect for bloggers, marketers, and content creators who need high-quality writing assistance.",
                        "speciality": "Specialized in creative writing, technical documentation, and marketing copy."
                    },
                    "features": {
                        "pros": [
                            "Real-time grammar and style checking",
                            "AI-powered content suggestions",
                            "Multiple writing styles and tones",
                            "Plagiarism detection"
                        ],
                        "cons": [
                            "Limited features in free tier",
                            "Requires internet connection"
                        ]
                    },
                    "metrics": {
                        "functionality": 4.8,
                        "innovation": 4.6,
                        "performance": 4.7,
                        "overall": 4.7,
                        "easeOfUse": 4.9,
                        "valueForMoney": 4.5
                    },
                    "categories": [
                        {
                            "Category": "AI Writing Tools"
                        },
                        {
                            "Category": "Content Creation"
                        },
                        {
                            "Category": "Productivity"
                        }
                    ],
                    "pricing": [
                        {
                            "planName": "Free",
                            "price": "0",
                            "features": [
                                "Basic grammar checking",
                                "Up to 1000 words per month",
                                "Limited AI suggestions"
                            ],
                            "isPopular": False,
                            "_id": str(uuid.uuid4())
                        },
                        {
                            "planName": "Premium",
                            "price": "19.99",
                            "features": [
                                "Advanced grammar and style checking",
                                "Unlimited word count",
                                "Full AI assistance",
                                "Plagiarism detection"
                            ],
                            "isPopular": True,
                            "_id": str(uuid.uuid4())
                        }
                    ],
                    "qaSection": [
                        {
                            "question": "Can it help with different writing styles?",
                            "answer": "Yes, it supports academic, business, creative, and technical writing styles."
                        },
                        {
                            "question": "Does it work offline?",
                            "answer": "No, an internet connection is required for AI features to function."
                        }
                    ],
                    "image_url": "https://example.com/aiwriting-logo.png"
                },
                {
                    "tool_id": tool_ids[1],
                    "name": "COMPREHENSIVE_TEST_DataVisualizer",
                    "url": "https://datavisualizer.test",
                    "description": "Professional data visualization tool for creating interactive charts and graphs",
                    "category_subcat": "Data Visualization, Analytics",
                    "pricingType": "Subscription",
                    "details": {
                        "introduction": "DataVisualizer transforms complex data into beautiful, interactive visualizations that tell compelling stories.",
                        "usage": "Ideal for data analysts, business intelligence professionals, and researchers.",
                        "speciality": "Advanced statistical charts, real-time dashboards, and interactive data exploration."
                    },
                    "features": {
                        "pros": [
                            "150+ chart types and templates",
                            "Real-time data connectivity",
                            "Interactive dashboards",
                            "Export in multiple formats"
                        ],
                        "cons": [
                            "Steep learning curve for advanced features",
                            "Expensive for small teams"
                        ]
                    },
                    "metrics": {
                        "functionality": 4.9,
                        "innovation": 4.7,
                        "performance": 4.6,
                        "overall": 4.8,
                        "easeOfUse": 4.3,
                        "valueForMoney": 4.2
                    },
                    "categories": [
                        {
                            "Category": "Data Visualization"
                        },
                        {
                            "Category": "Analytics"
                        },
                        {
                            "Category": "Business Intelligence"
                        }
                    ],
                    "pricing": [
                        {
                            "planName": "Professional",
                            "price": "49.99",
                            "features": [
                                "All chart types",
                                "Real-time data connections",
                                "Interactive dashboards",
                                "Basic collaboration"
                            ],
                            "isPopular": True,
                            "_id": str(uuid.uuid4())
                        },
                        {
                            "planName": "Enterprise",
                            "price": "99.99",
                            "features": [
                                "Everything in Professional",
                                "Advanced analytics",
                                "Custom branding",
                                "Dedicated support"
                            ],
                            "isPopular": False,
                            "_id": str(uuid.uuid4())
                        }
                    ],
                    "qaSection": [
                        {
                            "question": "Can it connect to databases?",
                            "answer": "Yes, it supports connections to MySQL, PostgreSQL, MongoDB, and many other databases."
                        },
                        {
                            "question": "Is there a free trial?",
                            "answer": "Yes, we offer a 14-day free trial with full access to all features."
                        }
                    ],
                    "image_url": "https://example.com/datavis-logo.png"
                },
                {
                    "tool_id": tool_ids[2],
                    "name": "COMPREHENSIVE_TEST_ProjectManager",
                    "url": "https://projectmanager.test",
                    "description": "Comprehensive project management tool with AI-powered insights",
                    "category_subcat": "Project Management, Team Collaboration",
                    "pricingType": "Free",
                    "details": {
                        "introduction": "ProjectManager combines traditional project management with AI insights to help teams deliver projects on time and within budget.",
                        "usage": "Perfect for project managers, team leads, and organizations of all sizes.",
                        "speciality": "AI-powered project predictions, resource optimization, and team performance analytics."
                    },
                    "features": {
                        "pros": [
                            "AI-powered project insights",
                            "Gantt charts and Kanban boards",
                            "Time tracking and reporting",
                            "Team collaboration tools"
                        ],
                        "cons": [
                            "Mobile app could be better",
                            "Advanced features require learning"
                        ]
                    },
                    "metrics": {
                        "functionality": 4.7,
                        "innovation": 4.5,
                        "performance": 4.6,
                        "overall": 4.6,
                        "easeOfUse": 4.7,
                        "valueForMoney": 4.8
                    },
                    "categories": [
                        {
                            "Category": "Project Management"
                        },
                        {
                            "Category": "Team Collaboration"
                        },
                        {
                            "Category": "Productivity"
                        }
                    ],
                    "pricing": [
                        {
                            "planName": "Free",
                            "price": "0",
                            "features": [
                                "Up to 5 projects",
                                "Basic Kanban boards",
                                "5 team members",
                                "Basic reporting"
                            ],
                            "isPopular": True,
                            "_id": str(uuid.uuid4())
                        }
                    ],
                    "qaSection": [
                        {
                            "question": "How does the AI prediction work?",
                            "answer": "Our AI analyzes project patterns, team velocity, and historical data to predict project completion dates and identify potential risks."
                        },
                        {
                            "question": "Can it integrate with other tools?",
                            "answer": "Yes, it integrates with popular tools like Slack, Google Workspace, Microsoft Teams, and many others."
                        }
                    ],
                    "image_url": "https://example.com/projectmgr-logo.png"
                }
            ]
        }
    
    def test_add_tools(self):
        """Test the add-tools endpoint"""
        print("=" * 60)
        print("TESTING ADD TOOLS ENDPOINT")
        print("=" * 60)
        
        # Generate test tools
        test_data = self.generate_test_tools()
        print(f"Generated {len(test_data['tools'])} test tools with comprehensive schema")
        
        # Send request to add-tools endpoint
        print("\nSending request to add-tools endpoint...")
        response = requests.post(
            ADD_TOOLS_ENDPOINT,
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        # Check response
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\nSuccessfully added tools:")
            
            for i, item in enumerate(result["results"]):
                print(f"  {i+1}. Tool: {item['tool']['name']}")
                print(f"     Status: {item['status']}")
                print(f"     ID: {item['id']}")
                print(f"     Tool ID: {item['tool']['tool_id']}")
                print()
                
                if item['status'] == "added":
                    self.test_tool_ids.append(item['tool']['tool_id'])
                    self.test_tools_data[item['tool']['tool_id']] = {
                        'rid': item['id'],
                        'name': item['tool']['name'],
                        'original_data': item['tool']
                    }
            
            print(f"✅ ADD TOOLS TEST: Successfully added {len(self.test_tool_ids)} tools")
            return True
        else:
            print(f"❌ ADD TOOLS TEST: Error - {response.text}")
            return False
    
    def test_query_tools(self):
        """Test the query endpoint with different scenarios"""
        print("=" * 60)
        print("TESTING QUERY ENDPOINT")
        print("=" * 60)
        
        if not self.test_tool_ids:
            print("❌ QUERY TEST: No tools available for testing")
            return False
        
        # Wait for indexing to complete
        print("Waiting for indexing to complete...")
        time.sleep(3)
        
        test_scenarios = [
            {
                "name": "General search",
                "query": "AI assistant tool",
                "searchFrom": None
            },
            {
                "name": "Writing tool search", 
                "query": "writing assistant",
                "searchFrom": None
            },
            {
                "name": "Filtered search - single tool",
                "query": "data visualization",
                "searchFrom": [self.test_tool_ids[1]] if len(self.test_tool_ids) > 1 else None
            },
            {
                "name": "Filtered search - multiple tools",
                "query": "productivity tool",
                "searchFrom": self.test_tool_ids[:2] if len(self.test_tool_ids) > 1 else None
            }
        ]
        
        success_count = 0
        for i, scenario in enumerate(test_scenarios):
            print(f"\nScenario {i+1}: {scenario['name']}")
            
            # Prepare request
            request_data = {"query": scenario["query"]}
            if scenario["searchFrom"]:
                request_data["searchFrom"] = scenario["searchFrom"]
                print(f"  Filtering within {len(scenario['searchFrom'])} tool(s)")
            
            # Send request
            response = requests.post(
                QUERY_ENDPOINT,
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = self.safely_parse_json(response)
                
                if result and "tools" in result:
                    print(f"  ✅ Found {len(result['tools'])} tools")
                    
                    # Check if filter was applied correctly
                    if scenario["searchFrom"] and "search_filter_applied" in result:
                        print(f"  Filter applied: {result['search_filter_applied']}")
                    
                    # Show found tools
                    for tool in result["tools"]:
                        print(f"    - {tool.get('name', 'Unknown')}")
                    
                    success_count += 1
                else:
                    print(f"  ⚠️ No tools found or couldn't parse response")
            else:
                print(f"  ❌ Error: {response.text}")
        
        print(f"\n✅ QUERY TEST: {success_count}/{len(test_scenarios)} scenarios successful")
        return success_count == len(test_scenarios)
    
    def test_update_tools(self):
        """Test the update-tools endpoint"""
        print("=" * 60)
        print("TESTING UPDATE TOOLS ENDPOINT")
        print("=" * 60)
        
        if not self.test_tool_ids:
            print("❌ UPDATE TEST: No tools available for testing")
            return False
        
        # Update the first tool
        tool_id_to_update = self.test_tool_ids[0]
        original_tool = self.test_tools_data[tool_id_to_update]['original_data']
        
        # Create updated version
        updated_tool = original_tool.copy()
        updated_tool['name'] = f"{original_tool['name']}_UPDATED"
        updated_tool['description'] = f"UPDATED: {original_tool['description']}"
        updated_tool['pricingType'] = "Premium"
        
        # Update metrics
        if 'metrics' in updated_tool:
            updated_tool['metrics']['overall'] = 5.0
            updated_tool['metrics']['functionality'] = 5.0
        
        # Add new feature
        if 'features' in updated_tool and 'pros' in updated_tool['features']:
            updated_tool['features']['pros'].append("UPDATED: New advanced feature")
        
        update_data = {"tools": [updated_tool]}
        
        print(f"Updating tool: {original_tool['name']} -> {updated_tool['name']}")
        
        # Send update request
        response = requests.put(
            UPDATE_TOOLS_ENDPOINT,
            json=update_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("\nUpdate results:")
            
            for item in result["results"]:
                print(f"  Tool: {item['tool']['name']}")
                print(f"  Status: {item['status']}")
                print(f"  ID: {item['id']}")
                
                if item['status'] == "updated":
                    # Update our tracking data
                    self.test_tools_data[tool_id_to_update]['name'] = updated_tool['name']
            
            print("✅ UPDATE TEST: Tool successfully updated")
            return True
        else:
            print(f"❌ UPDATE TEST: Error - {response.text}")
            return False
    
    def test_delete_tool(self):
        """Test the delete-tool endpoint"""
        print("=" * 60)
        print("TESTING DELETE TOOL ENDPOINT")
        print("=" * 60)
        
        if not self.test_tool_ids:
            print("❌ DELETE TEST: No tools available for testing")
            return False
        
        # Delete the last tool
        tool_id_to_delete = self.test_tool_ids[-1]
        tool_name = self.test_tools_data[tool_id_to_delete]['name']
        
        print(f"Deleting tool: {tool_name} (ID: {tool_id_to_delete})")
        
        # Send delete request
        response = requests.delete(
            f"{DELETE_TOOL_ENDPOINT}/{tool_id_to_delete}",
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ DELETE TEST: {result['deleted_tool']} successfully deleted")
            
            # Remove from our tracking
            self.test_tool_ids.remove(tool_id_to_delete)
            del self.test_tools_data[tool_id_to_delete]
            
            return True
        else:
            print(f"❌ DELETE TEST: Error - {response.text}")
            return False
    
    def verify_final_state(self):
        """Verify the final state by checking stats"""
        print("=" * 60)
        print("VERIFYING FINAL STATE")
        print("=" * 60)
        
        # Get current stats
        response = requests.get(f"{STATS_ENDPOINT}?show_all=true")
        
        if response.status_code == 200:
            stats = response.json()
            print(f"Total vectors in store: {stats['total_vectors']}")
            
            # Check if remaining tools are still there
            remaining_tools_found = 0
            for vector in stats['vectors']:
                if vector.get('tool_id') in self.test_tool_ids:
                    remaining_tools_found += 1
                    print(f"  Found: {vector.get('name', 'Unknown')}")
            
            print(f"\n✅ VERIFICATION: Found {remaining_tools_found}/{len(self.test_tool_ids)} remaining test tools")
            return remaining_tools_found == len(self.test_tool_ids)
        else:
            print(f"❌ VERIFICATION: Error getting stats - {response.text}")
            return False
    
    def cleanup_remaining_tools(self):
        """Clean up any remaining test tools"""
        print("=" * 60)
        print("CLEANING UP REMAINING TEST TOOLS")
        print("=" * 60)
        
        if not self.test_tool_ids:
            print("No tools to clean up")
            return
        
        for tool_id in self.test_tool_ids.copy():
            tool_name = self.test_tools_data[tool_id]['name']
            print(f"Cleaning up: {tool_name}")
            
            response = requests.delete(
                f"{DELETE_TOOL_ENDPOINT}/{tool_id}",
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"  ✅ Deleted successfully")
                self.test_tool_ids.remove(tool_id)
                del self.test_tools_data[tool_id]
            else:
                print(f"  ❌ Failed to delete: {response.text}")
        
        print(f"Cleanup complete. Remaining tools: {len(self.test_tool_ids)}")

def main():
    """Main function to run comprehensive API tests"""
    print("🚀 STARTING COMPREHENSIVE API TESTING")
    print("=" * 60)
    
    # Check if API is running
    try:
        health_check = requests.get(f"{API_BASE_URL}/basic-health")
        if health_check.status_code != 200:
            print(f"❌ API is not healthy. Status code: {health_check.status_code}")
            return
        print("✅ API is running and healthy")
    except requests.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
        print(f"Make sure the API is running at {API_BASE_URL}")
        return
    
    # Initialize tester
    tester = APITester()
    
    # Run tests in sequence
    test_results = {
        "add_tools": False,
        "query_tools": False, 
        "update_tools": False,
        "delete_tool": False,
        "verification": False
    }
    
    try:
        # Test add tools
        test_results["add_tools"] = tester.test_add_tools()
        
        # Test query functionality
        if test_results["add_tools"]:
            test_results["query_tools"] = tester.test_query_tools()
        
        # Test update functionality
        if test_results["add_tools"]:
            test_results["update_tools"] = tester.test_update_tools()
        
        # Test delete functionality
        if test_results["add_tools"]:
            test_results["delete_tool"] = tester.test_delete_tool()
        
        # Verify final state
        test_results["verification"] = tester.verify_final_state()
        
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {str(e)}")
    finally:
        # Clean up remaining tools
        tester.cleanup_remaining_tools()
    
    # Print final results
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests completed successfully!")
    else:
        print("⚠️ Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    main()