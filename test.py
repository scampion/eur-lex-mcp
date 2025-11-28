#!/usr/bin/env python3
"""
Enhanced test script for EUR-Lex MCP Server with resource template support
Run with: python test_mcp.py
"""

import requests
import json
import time

SERVER_URL = "http://127.0.0.1:8000/mcp"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}


def send_request(method, params, session_id=None, timeout=60):
    """Send a JSON-RPC request to the MCP server"""
    payload = {
        "jsonrpc": "2.0",
        "id": method,
        "method": method,
        "params": params
    }

    headers = HEADERS.copy()
    if session_id:
        headers["Mcp-Session-Id"] = session_id

    print(f"Sending request: {method}", end="", flush=True)
    response = requests.post(SERVER_URL, json=payload, headers=headers, timeout=timeout)
    print(" ✓")

    # Handle Server-Sent Events response
    if "event:" in response.text:
        # Extract JSON from SSE format
        for line in response.text.split('\n'):
            if line.startswith('data: '):
                return json.loads(line[6:]), response.headers.get("Mcp-Session-Id")

    return response.json(), response.headers.get("Mcp-Session-Id")


def main():
    print("Testing EUR-Lex MCP Server")
    print("=" * 50)

    # Step 1: Initialize
    print("\n1. Initializing connection...")
    init_params = {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "resources": {}  # Explicitly request resource support
        },
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }

    result, session_id = send_request("initialize", init_params)
    print(f"✓ Initialized. Session ID: {session_id}")
    print(f"  Server: {result['result']['serverInfo']['name']} v{result['result']['serverInfo']['version']}")

    # Check server capabilities
    if 'capabilities' in result['result']:
        print(f"  Server capabilities: {json.dumps(result['result']['capabilities'], indent=2)}")

    # Step 2: List static resources
    print("\n2. Listing static resources...")
    result, _ = send_request("resources/list", {}, session_id)

    if "result" in result and "resources" in result["result"]:
        resources = result["result"]["resources"]
        print(f"✓ Found {len(resources)} static resource(s):")
        for resource in resources:
            print(f"  - URI: {resource['uri']}")
            print(f"    Name: {resource.get('name', 'N/A')}")
            print()
    else:
        print("  No static resources found (this is normal if only using templates)")

    # Step 3: List resource TEMPLATES (this is what you need!)
    print("\n3. Listing resource templates...")
    result, _ = send_request("resources/templates/list", {}, session_id)

    print(f"Full response: {json.dumps(result, indent=2)}")

    if "result" in result and "resourceTemplates" in result["result"]:
        templates = result["result"]["resourceTemplates"]
        print(f"✓ Found {len(templates)} resource template(s):")
        for template in templates:
            print(f"  - URI Template: {template['uriTemplate']}")
            print(f"    Name: {template.get('name', 'N/A')}")
            print(f"    Description: {template.get('description', 'N/A')}")
            print()
    else:
        print("  Could not retrieve resource templates")
        print(f"  Error: {result.get('error', 'Unknown')}")

    # Step 4: Try reading a specific resource using the template
    print("\n4. Testing resource read for GDPR (32016R0679)...")
    celex = "32016R0679"
    resource_uri = f"resource://eurlex/document/{celex}"

    read_params = {
        "uri": resource_uri
    }

    try:
        result, _ = send_request("resources/read", read_params, session_id, timeout=120)

        if "result" in result:
            print("✓ Resource read successful!")
            contents = result["result"]["contents"]
            for content in contents:
                print(f"\n  Content type: {content.get('mimeType', 'unknown')}")
                print(f"  URI: {content.get('uri', 'N/A')}")
                text = content.get('text', '')
                if text:
                    print(f"  Content length: {len(text)} characters")
                    print(f"  Content preview (first 500 chars):")
                    print(f"  {'-' * 45}")
                    print(f"  {text[:500]}...")
                else:
                    print(f"  (No text content)")
        else:
            print(f"✗ Error reading resource: {result.get('error', 'Unknown error')}")
            if 'error' in result:
                print(f"  Details: {json.dumps(result['error'], indent=2)}")
    except Exception as e:
        print(f"✗ Exception while reading resource: {e}")

    # Step 5: List tools
    print("\n5. Listing available tools...")
    result, _ = send_request("tools/list", {}, session_id)

    if "result" in result and "tools" in result["result"]:
        tools = result["result"]["tools"]
        print(f"✓ Found {len(tools)} tool(s):")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

    # Step 6: Test expert_search
    print("\n6. Testing expert_search tool with GDPR query...")
    tool_params = {
        "name": "expert_search",
        "arguments": {
            "query": "DN = 32016R0679",
            "page_size": 3
        }
    }

    result, _ = send_request("tools/call", tool_params, session_id)

    if "result" in result:
        print("✓ Search successful!")
        content = result["result"]["content"]
        for item in content:
            if item["type"] == "text":
                print(f"\n  Search Results:")
                print(f"  {'-' * 45}")
                text = item['text']
                for line in text.split('\n')[:5]:
                    if line.strip():
                        print(f"  {line}")
                break


    time.sleep(1)  # Small delay for server constraints
    # Step 7: Test search for cheese transport legislation
    print("\n7. Testing expert_search for EU cheese transport legislation...")
    tool_params = {
        "name": "expert_search",
        "arguments": {
            # EUR-Lex expert search requires: Field ~ SearchTerm
            # Using Text field to search in document text, with NEAR10 operator
            "query": "Text ~ cheese NEAR10 transport",
            "page_size": 10
        }
    }

    result, _ = send_request("tools/call", tool_params, session_id)

    if "result" in result:
        print("✓ Search successful!")
        content = result["result"]["content"]
        for item in content:
            if item["type"] == "text":
                print(f"\n  Search Results for Cheese Transport:")
                print(f"  {'-' * 45}")
                text = item['text']
                lines = text.split('\n')
                for i, line in enumerate(lines[:10], 1):  # Show first 10 results
                    if line.strip():
                        print(f"  {i}. {line}")
                if len(lines) > 10:
                    print(f"  ... and {len(lines) - 10} more results")
                break
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 50)
    print("Test complete!")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to server at http://127.0.0.1:8000/mcp")
        print("  Make sure the server is running: python server.py --transport http")
    except requests.exceptions.Timeout:
        print("✗ Error: Request timed out")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()