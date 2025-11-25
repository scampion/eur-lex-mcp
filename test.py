#!/usr/bin/env python3
"""
Minimal test script for EUR-Lex MCP Server
Run with: python test_mcp.py
"""

import requests
import json

SERVER_URL = "http://127.0.0.1:8000/mcp"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}


def send_request(method, params, session_id=None):
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

    response = requests.post(SERVER_URL, json=payload, headers=headers)

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
        "capabilities": {},
        "clientInfo": {
            "name": "test-client",
            "version": "1.0.0"
        }
    }

    result, session_id = send_request("initialize", init_params)
    print(f"✓ Initialized. Session ID: {session_id}")
    print(f"  Server: {result['result']['serverInfo']['name']} v{result['result']['serverInfo']['version']}")

    # Step 2: List tools
    print("\n2. Listing available tools...")
    result, _ = send_request("tools/list", {}, session_id)

    if "result" in result and "tools" in result["result"]:
        tools = result["result"]["tools"]
        print(f"✓ Found {len(tools)} tool(s):")
        for tool in tools:
            print(f"  - {tool['name']}: {tool['description']}")

    # Step 3: List resources
    print("\n3. Listing available resources...")
    result, _ = send_request("resources/list", {}, session_id)

    if "result" in result and "resources" in result["result"]:
        resources = result["result"]["resources"]
        print(f"✓ Found {len(resources)} resource(s):")
        for resource in resources:
            print(f"  - {resource['uri']}: {resource.get('name', 'N/A')}")

    # Step 4: Test a tool call (expert_search)
    print("\n4. Testing expert_search tool with GDPR query...")
    tool_params = {
        "name": "expert_search",
        "arguments": {
            "query": "data protection GDPR",
            "page_size": 3,
            "page": 1
        }
    }

    result, _ = send_request("tools/call", tool_params, session_id)

    if "result" in result:
        print("✓ Search successful!")
        content = result["result"]["content"]
        for item in content:
            if item["type"] == "text":
                # Parse and display results nicely
                text = item['text']
                print(f"\n  Search Results:")
                print(f"  {'-' * 45}")
                if "CELEX:" in text:
                    # Try to extract individual results
                    lines = text.split('\n')
                    for line in lines[:10]:  # Show first 10 lines
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"  {text[:300]}...")
                break
    else:
        print(f"✗ Error: {result.get('error', 'Unknown error')}")

    # Step 5: Test another search - EU directives on cybersecurity
    print("\n5. Testing expert_search with cybersecurity directive query...")
    tool_params = {
        "name": "expert_search",
        "arguments": {
            "query": "cybersecurity directive NIS2",
            "page_size": 2
        }
    }

    result, _ = send_request("tools/call", tool_params, session_id)

    if "result" in result:
        print("✓ Search successful!")
        content = result["result"]["content"]
        for item in content:
            if item["type"] == "text":
                print(f"\n  Results preview:")
                print(f"  {item['text'][:250]}...")
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
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()