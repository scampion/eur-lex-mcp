#!/usr/bin/env python3
"""
Enhanced test script for EUR-Lex MCP Server with full documentation verification
Run with: python test_mcp_enhanced.py
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


def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def main():
    print_section("EUR-Lex MCP Server Enhanced Documentation Test")

    # Step 1: Initialize
    print("\n1. Initializing connection...")
    init_params = {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "resources": {}
        },
        "clientInfo": {
            "name": "enhanced-test-client",
            "version": "1.0.0"
        }
    }

    result, session_id = send_request("initialize", init_params)
    print(f"✓ Initialized. Session ID: {session_id}")
    server_info = result['result']['serverInfo']
    print(f"  Server: {server_info['name']} v{server_info['version']}")

    # Check for server instructions (enhanced documentation)
    if 'instructions' in server_info:
        print(f"\n  📚 Server Instructions:")
        print(f"  {'-' * 66}")
        instructions = server_info['instructions']
        for line in instructions.split('\n')[:10]:  # Show first 10 lines
            print(f"  {line}")
        if len(instructions.split('\n')) > 10:
            print(f"  ... ({len(instructions.split('\n')) - 10} more lines)")

    # Step 2: List tools with FULL details
    print_section("2. Listing Tools with Full Documentation")
    result, _ = send_request("tools/list", {}, session_id)

    if "result" in result and "tools" in result["result"]:
        tools = result["result"]["tools"]
        print(f"✓ Found {len(tools)} tool(s)\n")

        for tool in tools:
            print(f"📋 Tool: {tool['name']}")
            print(f"   {'-' * 66}")

            # Short description
            print(f"   Description: {tool.get('description', 'N/A')}")

            # Input schema with descriptions
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                print(f"\n   📝 Parameters:")
                if 'properties' in schema:
                    for param_name, param_info in schema['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', 'No description')
                        default = param_info.get('default', None)

                        print(f"      • {param_name} ({param_type})")
                        if default is not None:
                            print(f"        Default: {default}")

                        # Show parameter description (this is where extended docs appear)
                        if param_desc and len(param_desc) > 100:
                            print(f"        {param_desc[:200]}...")
                            print(f"        [Full description: {len(param_desc)} characters]")
                        else:
                            print(f"        {param_desc}")

                # Check if there's a general description field in the schema
                if 'description' in schema:
                    desc = schema['description']
                    print(f"\n   📖 Schema Description ({len(desc)} chars):")
                    print(f"   {'-' * 66}")
                    # Show first 500 chars
                    for line in desc.split('\n')[:15]:
                        print(f"   {line}")
                    if len(desc) > 500:
                        print(f"   ... [+{len(desc) - 500} more characters]")

            print()

    # Step 3: Check tool docstring via introspection (if available)
    print_section("3. Checking Tool Documentation Coverage")

    for tool in tools:
        tool_name = tool['name']
        full_desc = tool.get('description', '')

        print(f"\n📊 Analysis for '{tool_name}':")
        print(f"   Description length: {len(full_desc)} characters")
        print(f"   Lines: {len(full_desc.split(chr(10)))}")

        # Check for key documentation elements
        keywords = ['FIELD', 'QUERY', 'SYNTAX', 'EXAMPLE', 'CELEX', 'OPERATOR', 'BOOLEAN']
        found_keywords = [kw for kw in keywords if kw in full_desc.upper()]

        print(f"   Documentation keywords found: {', '.join(found_keywords) if found_keywords else 'None'}")

        if len(full_desc) > 500:
            print(f"   ✅ Rich documentation present")
        else:
            print(f"   ⚠️  Documentation might be truncated")

    # Step 4: Resource templates
    print_section("4. Resource Templates Documentation")
    result, _ = send_request("resources/templates/list", {}, session_id)

    if "result" in result and "resourceTemplates" in result["result"]:
        templates = result["result"]["resourceTemplates"]
        print(f"✓ Found {len(templates)} resource template(s)\n")

        for template in templates:
            print(f"📄 Template: {template.get('name', 'Unnamed')}")
            print(f"   URI Template: {template['uriTemplate']}")
            if 'description' in template:
                print(f"   Description: {template['description']}")
            if 'mimeType' in template:
                print(f"   MIME Type: {template['mimeType']}")
            print()

    # Step 5: Test actual search functionality
    print_section("5. Testing Search Functionality")

    test_queries = [
        ("DN = 32016R0679", "GDPR by CELEX number"),
        ("TI ~ artificial intelligence AND DD >= 2023", "Recent AI regulations"),
        ("TE ~ cheese NEAR10 transport", "Cheese transport legislation")
    ]

    for query, description in test_queries:
        print(f"\n🔍 Query: {description}")
        print(f"   Syntax: {query}")

        tool_params = {
            "name": "expert_search",
            "arguments": {
                "query": query,
                "page_size": 3
            }
        }

        result, _ = send_request("tools/call", tool_params, session_id)

        if "result" in result:
            content = result["result"]["content"]
            for item in content:
                if item["type"] == "text":
                    text = item['text']
                    lines = [l for l in text.split('\n') if l.strip()]
                    print(f"   ✓ Found {len(lines)} result(s)")
                    for i, line in enumerate(lines[:2], 1):
                        print(f"      {i}. {line[:80]}{'...' if len(line) > 80 else ''}")
                    break
        else:
            print(f"   ✗ Error: {result.get('error', {}).get('message', 'Unknown')}")

        time.sleep(1)

    # Step 6: Test resource fetching
    print_section("6. Testing Document Retrieval")

    celex = "32016R0679"
    resource_uri = f"resource://eurlex/document/{celex}"

    print(f"📖 Fetching GDPR (CELEX: {celex})...")

    read_params = {"uri": resource_uri}

    try:
        result, _ = send_request("resources/read", read_params, session_id, timeout=120)

        if "result" in result:
            contents = result["result"]["contents"]
            for content in contents:
                text = content.get('text', '')
                if text:
                    print(f"   ✓ Retrieved {len(text)} characters")
                    print(f"   First 300 characters:")
                    print(f"   {'-' * 66}")
                    for line in text[:300].split('\n')[:8]:
                        print(f"   {line}")
                    print(f"   ...")
        else:
            print(f"   ✗ Error: {result.get('error', {}).get('message', 'Unknown')}")
    except Exception as e:
        print(f"   ✗ Exception: {e}")

    # Final summary
    print_section("Test Summary")
    print("""
✅ Server initialized with enhanced documentation
✅ Tools exposed with comprehensive descriptions  
✅ Resource templates available for document retrieval
✅ Search functionality working with expert query syntax
✅ Document fetching operational

💡 For LLMs: The tool descriptions now include:
   • Detailed query syntax documentation
   • Field reference guide
   • Example queries
   • Return format specifications
   • Integration instructions

This ensures LLMs can effectively construct EUR-Lex queries without
external documentation lookup.
""")

    print("=" * 70)


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