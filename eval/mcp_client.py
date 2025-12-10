"""
MCP Client for EUR-Lex Server Interaction.

This module provides a client wrapper for interacting with the EUR-Lex MCP server
during evaluation.
"""
import asyncio
import json
import logging
import os

import aiohttp
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import mcp_client_config, query_llm_config

logger = logging.getLogger(__name__)


try:
    from phoenix.otel import register
    tracer_provider = register(
        project_name="eval-eurlex-mcp-client",
        auto_instrument=True,
    )
    tracer = tracer_provider.get_tracer(__name__)
except ImportError:
    tracer_provider = None
    tracer = None
    logger.debug("Phoenix tracing not available")


@dataclass
class SearchResult:
    """A search result from the MCP server."""

    celex: str
    title: str
    text: str
    score: float
    metadata: Dict[str, Any]


class MCPClient:
    """Client for interacting with the EUR-Lex MCP server."""

    def __init__(self, config=mcp_client_config):
        self.config = config
        self._process = None
        self._reader = None
        self._writer = None
        self._request_id = 0

    async def connect(self):
        """Start the MCP server process and establish connection."""
        if self._process is not None:
            return

        logger.info("Starting MCP server...")

        # Start the MCP server as a subprocess
        self._process = await asyncio.create_subprocess_exec(
            self.config.server_command,
            *self.config.server_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._reader = self._process.stdout
        self._writer = self._process.stdin

        # Wait for server to be ready (read initial message if any)
        await asyncio.sleep(1)

        logger.info("MCP server started")

    async def disconnect(self):
        """Stop the MCP server process."""
        if self._process is not None:
            self._process.terminate()
            await self._process.wait()
            self._process = None
            self._reader = None
            self._writer = None
            logger.info("MCP server stopped")

    async def _send_request(self, method: str, params: Dict) -> Dict:
        """Send a JSON-RPC request to the MCP server."""
        if self._writer is None:
            await self.connect()

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        request_line = json.dumps(request) + "\n"
        self._writer.write(request_line.encode())
        await self._writer.drain()

        # Read response
        response_line = await asyncio.wait_for(
            self._reader.readline(),
            timeout=self.config.timeout_seconds,
        )

        if not response_line:
            raise RuntimeError("No response from MCP server")

        response = json.loads(response_line.decode())

        if "error" in response:
            raise RuntimeError(f"MCP error: {response['error']}")

        return response.get("result", {})

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Search the EUR-Lex database.

        Args:
            query: Search query text
            limit: Maximum number of results
            filters: Optional filters (doc_type, date_range, etc.)

        Returns:
            List of search results
        """
        params = {
            "query": query,
            "limit": limit,
        }
        if filters:
            params["filters"] = filters

        try:
            result = await self._send_request("search", params)
            return result.get("results", [])
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_document(self, celex: str) -> Optional[Dict]:
        """
        Retrieve a specific document by CELEX number.

        Args:
            celex: CELEX document identifier

        Returns:
            Document data or None if not found
        """
        try:
            result = await self._send_request("get_document", {"celex": celex})
            return result
        except Exception as e:
            logger.error(f"Get document failed: {e}")
            return None

    async def list_tools(self) -> List[Dict]:
        """List available MCP tools."""
        try:
            result = await self._send_request("tools/list", {})
            return result.get("tools", [])
        except Exception as e:
            logger.error(f"List tools failed: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        try:
            result = await self._send_request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )
            return result
        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return {}


class MCPClientSync:
    """Synchronous wrapper for MCPClient."""

    def __init__(self, config=mcp_client_config):
        self._async_client = MCPClient(config)
        self._loop = None

    def _get_loop(self):
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop

    def search(self, query: str, limit: int = 10, filters: Optional[Dict] = None) -> List[Dict]:
        """Synchronous search."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self._async_client.search(query, limit, filters)
        )

    def get_document(self, celex: str) -> Optional[Dict]:
        """Synchronous get document."""
        loop = self._get_loop()
        return loop.run_until_complete(self._async_client.get_document(celex))

    def close(self):
        """Close the client."""
        if self._loop:
            self._loop.run_until_complete(self._async_client.disconnect())
            self._loop.close()


class MockMCPClient:
    """Mock MCP client for testing without a running server."""

    def __init__(self, corpus: Optional[Dict] = None):
        self.corpus = corpus or {}
        self._search_index = None

    def _build_index(self):
        """Build simple search index from corpus."""
        if self._search_index is not None:
            return

        self._search_index = {}
        for celex, doc in self.corpus.items():
            # Simple term indexing
            text = f"{doc.get('title', '')} {doc.get('text', '')}".lower()
            terms = set(text.split())
            for term in terms:
                if term not in self._search_index:
                    self._search_index[term] = []
                self._search_index[term].append(celex)

    async def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict]:
        """Mock search using simple term matching."""
        self._build_index()

        query_terms = query.lower().split()
        scores = {}

        for term in query_terms:
            if term in self._search_index:
                for celex in self._search_index[term]:
                    scores[celex] = scores.get(celex, 0) + 1

        # Sort by score and return top results
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        results = []
        for celex, score in ranked[:limit]:
            if celex in self.corpus:
                doc = self.corpus[celex]
                results.append({
                    "celex": celex,
                    "title": doc.get("title", ""),
                    "text": doc.get("text", "")[:500],
                    "score": score / len(query_terms),
                })

        return results

    async def get_document(self, celex: str) -> Optional[Dict]:
        """Mock get document."""
        return self.corpus.get(celex)

    async def connect(self):
        """No-op for mock client."""
        pass

    async def disconnect(self):
        """No-op for mock client."""
        pass


class FastMCPClient:
    """HTTP client for FastMCP servers using streamable-http transport.

    Compatible with EUR-Lex MCP server running FastMCP.
    """

    def __init__(
        self,
        base_url: str = None,
        timeout: int = None,
        llm_client=None,
        llm_config=None,
    ):
        # Use config defaults if not specified
        self.base_url = (base_url or mcp_client_config.server_base_url).rstrip("/")
        self.timeout = timeout or mcp_client_config.timeout_seconds
        self._session = None
        self._session_id = None
        self._request_id = 0
        self._llm_client = llm_client
        self._llm_config = llm_config or query_llm_config

    async def connect(self):
        """Initialize HTTP session and get MCP session ID."""
        if self._session is not None and self._session_id is not None:
            return

        self._session = aiohttp.ClientSession()

        # Initialize MCP session by sending 'initialize' request
        url = f"{self.base_url}/mcp"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

        self._request_id += 1
        init_request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "eurlex-eval-client",
                    "version": "1.0.0"
                }
            }
        }

        try:
            async with self._session.post(
                url,
                json=init_request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                # Get session ID from response header
                self._session_id = resp.headers.get("mcp-session-id")
                if resp.status == 200:
                    content_type = resp.headers.get("Content-Type", "")
                    if "text/event-stream" in content_type:
                        response = await self._parse_sse_response_init(resp)
                    else:
                        response = await resp.json()
                    logger.info(f"MCP session initialized: {self._session_id}")
                    logger.debug(f"Server capabilities: {response.get('result', {})}")
                else:
                    text = await resp.text()
                    logger.error(f"Failed to initialize MCP session: {resp.status} - {text}")
                    raise RuntimeError(f"MCP initialization failed: {text}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise

        logger.info(f"FastMCP client connected to {self.base_url} (session: {self._session_id})")

    async def _parse_sse_response_init(self, resp) -> Dict:
        """Parse SSE response during initialization (before _parse_sse_response is available)."""
        result = {}
        current_data = []

        async for line in resp.content:
            line = line.decode("utf-8").strip()

            if not line:
                if current_data:
                    data_str = "".join(current_data)
                    try:
                        parsed = json.loads(data_str)
                        if "result" in parsed or "error" in parsed:
                            result = parsed
                    except json.JSONDecodeError:
                        pass
                    current_data = []
                continue

            if line.startswith("data:"):
                current_data.append(line[5:].strip())

        if current_data:
            data_str = "".join(current_data)
            try:
                parsed = json.loads(data_str)
                if "result" in parsed or "error" in parsed:
                    result = parsed
            except json.JSONDecodeError:
                pass

        return result

    async def disconnect(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None
            self._session_id = None
            logger.info("FastMCP client disconnected")

    async def _send_request(self, method: str, params: Dict = None) -> Dict:
        """Send a JSON-RPC request to the FastMCP server."""
        if self._session is None or self._session_id is None:
            await self.connect()

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params:
            request["params"] = params

        # FastMCP streamable-http uses /mcp endpoint
        url = f"{self.base_url}/mcp"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self._session_id,
        }

        try:
            async with self._session.post(
                url,
                json=request,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get("Content-Type", "")

                    if "text/event-stream" in content_type:
                        # Parse SSE response
                        response = await self._parse_sse_response(resp)
                    else:
                        # Regular JSON response
                        response = await resp.json()

                    if "error" in response:
                        raise RuntimeError(f"MCP error: {response['error']}")
                    return response.get("result", {})
                else:
                    text = await resp.text()
                    raise RuntimeError(f"HTTP error {resp.status}: {text}")
        except aiohttp.ClientError as e:
            logger.error(f"MCP request failed: {e}")
            raise

    async def _parse_sse_response(self, resp) -> Dict:
        """Parse Server-Sent Events response from FastMCP.

        SSE format:
        event: message
        data: {"jsonrpc": "2.0", ...}

        """
        result = {}
        current_event = None
        current_data = []

        async for line in resp.content:
            line = line.decode("utf-8").strip()

            if not line:
                # Empty line marks end of event
                if current_data:
                    data_str = "".join(current_data)
                    try:
                        parsed = json.loads(data_str)
                        # Look for the final result message
                        if "result" in parsed:
                            result = parsed
                        elif "error" in parsed:
                            result = parsed
                    except json.JSONDecodeError:
                        pass
                    current_data = []
                    current_event = None
                continue

            if line.startswith("event:"):
                current_event = line[6:].strip()
            elif line.startswith("data:"):
                current_data.append(line[5:].strip())

        # Handle any remaining data
        if current_data:
            data_str = "".join(current_data)
            try:
                parsed = json.loads(data_str)
                if "result" in parsed or "error" in parsed:
                    result = parsed
            except json.JSONDecodeError:
                pass

        return result

    async def call_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Call an MCP tool."""
        result = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )
        return result

    async def _get_agent_tools_schema_async(self) -> List[Dict]:
        """Return OpenAI-compatible tool schemas fetched from the MCP server."""
        try:
            # Get tools from MCP server
            mcp_tools = await self.list_tools()

            # Convert MCP tool format to OpenAI format
            openai_tools = []
            for tool in mcp_tools:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {
                            "type": "object",
                            "properties": {}
                        })
                    }
                }
                openai_tools.append(openai_tool)

            # ============= Add MCP resources as tools =============
            try:
                # List static resources (no URI parameters)
                resources_result = await self.list_resources()

                # The result structure is: {"resources": [{"uri": "...", "name": "...", "description": "..."}]}
                if hasattr(resources_result, 'resources'):
                    resources_list = resources_result.resources
                else:
                    resources_list = resources_result.get('resources', [])

                for resource in resources_list:
                    # Extract info from resource
                    resource_uri = resource.get('uri') or (resource.uri if hasattr(resource, 'uri') else '')
                    resource_name = resource.get('name') or (resource.name if hasattr(resource, 'name') else '')
                    resource_description = resource.get('description') or (
                        resource.description if hasattr(resource, 'description') else '')

                    if not resource_uri:
                        continue

                    # Skip parameterized resources (like document/{celex_number})
                    if '{' in resource_uri:
                        continue

                    # Create function name from URI
                    # e.g., "resource://eurlex/expert-search-documentation" -> "get_expert_search_documentation"
                    uri_path = resource_uri.split('/')[-1]  # Get last part of URI
                    # Convert hyphens to underscores for valid Python/JSON function name
                    function_name = f"get_{uri_path.replace('-', '_')}"

                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "description": f"Read resource: {resource_description or resource_name or uri_path}",
                            "parameters": {
                                "type": "object",
                                "properties": {},  # Static resources have no parameters
                                "required": []
                            },
                            # Store the URI for later use
                            "_mcp_resource_uri": resource_uri
                        }
                    })
                    logger.debug(f"Added resource as tool: {function_name} -> {resource_uri}")

                logger.info(f"Added {len(resources_list)} resources as tools")

            except AttributeError as e:
                logger.warning(f"Client doesn't support list_resources: {e}")
            except Exception as e:
                logger.warning(f"Could not fetch resources from MCP server: {e}")
                import traceback
                traceback.print_exc()
            # ============= END NEW CODE =============
            #
            # # Add the special submit_final_query tool for the agent
            # openai_tools.append({
            #     "type": "function",
            #     "function": {
            #         "name": "submit_final_query",
            #         "description": "Submit the final EUR-Lex expert query. Call this when you have a valid query ready.",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "query": {
            #                     "type": "string",
            #                     "description": "The final EUR-Lex expert query to use for searching"
            #                 }
            #             },
            #             "required": ["query"]
            #         }
            #     }
            # })

            logger.info(f"Loaded {len(openai_tools)} tools total (including resources)")
            return openai_tools

        except Exception as e:
            logger.error(f"Failed to get tools from MCP server: {e}")
            return None

    # ============= Handler for resource tool calls =============
    async def _handle_tool_call(self, tool_name: str, arguments: dict, tool_schema: dict = None) -> str:
        """Handle tool calls from the agent, including resource access."""

        # Check if this tool has a stored MCP resource URI (from schema generation)
        if tool_schema and "_mcp_resource_uri" in tool_schema.get("function", {}):
            resource_uri = tool_schema["function"]["_mcp_resource_uri"]

            try:
                # Read the resource from MCP server
                content = await self.read_resource(resource_uri)
                return content if content else f"Resource {resource_uri} returned empty"

            except Exception as e:
                logger.error(f"Error accessing resource {resource_uri}: {e}")
                return f"Error accessing resource {resource_uri}: {e}"

        # Check if this is a resource by naming convention (fallback)
        if tool_name.startswith("read_") or tool_name.startswith("get_"):
            # Try to construct the resource URI
            resource_name = tool_name.replace("read_", "").replace("get_", "").replace("_", "-")
            resource_uri = f"resource://eurlex/{resource_name}"

            try:
                content = await self.read_resource(resource_uri)
                return content if content else f"Resource {resource_uri} returned empty"

            except Exception as e:
                # If resource read fails, try as normal tool
                logger.warning(f"Resource read failed for {resource_uri}, trying as tool: {e}")

        # Otherwise, handle as normal tool call
        try:
            result = await self.call_tool(tool_name, arguments)
            return str(result) if result else "Tool returned empty result"
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return f"Error calling tool {tool_name}: {e}"

    async def _execute_agent_tool(self, tool_name: str, arguments: Dict, tools_cache: List[Dict] = None) -> str:
        """Execute a tool call from the agent and return result as string.

        Handles:
        1. Special agent tools (submit_final_query)
        2. MCP tools (called via call_tool)
        3. MCP resources (read via read_resource)
        """
        try:
            # Special tool: submit the final query
            # if tool_name == "submit_final_query":
            #     return f"FINAL_QUERY:{arguments.get('query', '')}"

            # Check if this is a resource read (tools that start with read_ or get_ for resources)
            # First, look in the tools cache for resource URIs
            if tools_cache:
                for tool in tools_cache:
                    func = tool.get("function", {})
                    if func.get("name") == tool_name:
                        # Check if this tool has a resource URI attached
                        resource_uri = func.get("_mcp_resource_uri")
                        if resource_uri:
                            logger.info(f"Reading resource: {resource_uri}")
                            content = await self.read_resource(resource_uri)
                            if content:
                                return content[:4000]  # Truncate if too long
                            return f"Failed to read resource: {resource_uri}"
                        break

            # Check if tool name looks like a resource accessor (get_* or read_*)
            # Try resource first for these patterns before calling as tool
            if tool_name.startswith("read_") or tool_name.startswith("get_"):
                # Convert tool name to resource URI
                # e.g., "get_expert_search_documentation" -> "resource://eurlex/expert-search-documentation"
                resource_name = tool_name.replace("read_", "").replace("get_", "")
                resource_name = resource_name.replace("_", "-")
                resource_uri = f"resource://eurlex/{resource_name}"

                logger.info(f"Trying as resource first: {resource_uri}")
                content = await self.read_resource(resource_uri)
                if content:
                    return content[:4000]
                logger.warning(f"Resource read returned empty for {resource_uri}, trying as tool")

            # Try to call it as an MCP tool
            try:
                result = await self.call_tool(tool_name, arguments)

                # Parse the result based on MCP response format
                if isinstance(result, dict) and "content" in result:
                    # Extract text content from MCP tool response
                    text_parts = []
                    for item in result["content"]:
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)[:4000]  # Truncate if too long
                    return str(result)
                elif isinstance(result, str):
                    return result
                else:
                    return str(result)

            except Exception as e:
                logger.warning(f"MCP tool call failed for {tool_name}: {e}")
                return f"Error calling tool {tool_name}: {str(e)}"

        except Exception as e:
            logger.error(f"Agent tool execution failed: {tool_name} - {e}")
            import traceback
            traceback.print_exc()
            return f"Error executing {tool_name}: {str(e)}"

    async def natural_language_to_query(self, question: str, max_iterations: int = 15) -> str:
        """Convert a natural language question to EUR-Lex expert query syntax using an agentic LLM loop.

        The agent has access to MCP tools to build, validate, and refine the query.

        Args:
            question: Natural language question about EU law
            max_iterations: Maximum number of agent iterations (default: 5)

        Returns:
            EUR-Lex expert query syntax string
        """
        config = self._llm_config

        # Initialize LLM client if needed
        if self._llm_client is None:
            try:
                if config.llm_provider == "openai":
                    from openai import AsyncOpenAI
                    if config.llm_base_url:
                        self._llm_client = AsyncOpenAI(base_url=config.llm_base_url, api_key=os.environ.get("OPENAI_API_KEY"))
                    else:
                        self._llm_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
                    # add API key from environment or default location
                elif config.llm_provider == "ollama":
                    from openai import AsyncOpenAI
                    self._llm_client = AsyncOpenAI(
                        base_url=f"{config.ollama_base_url}/v1",
                        api_key="ollama"
                    )
                elif config.llm_provider == "custom":
                    from openai import AsyncOpenAI
                    self._llm_client = AsyncOpenAI(base_url=config.llm_base_url)
                else:
                    raise ValueError(f"Unknown LLM provider: {config.llm_provider}")
            except ImportError:
                logger.warning("OpenAI package not installed, falling back to build_eurlex_query tool")
                #result = await self.build_query("text", question)
                return None

        # Select model
        model = config.llm_model if config.llm_provider in ("openai", "custom") else config.ollama_model

        # System prompt for the agent
        system_prompt = f"""{config.system_prompt}
You have access to tools and resources to help you build the query. 

You must read the documentation ressource : get_expert_search_documentation  resource://eurlex/expert-search-documentation  
and the query examples resource : get_query_examples
Use the tools as needed to construct and validate the query.

When you have a valid EUR-Lex expert query, return it between <QUERY> and </QUERY> tags.

"""

        # Initial message
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"\nConvert this question to a EUR-Lex expert query:\n\n{question}"}
        ]

        tools = await self._get_agent_tools_schema_async()

        # Agentic loop
        for iteration in range(max_iterations):
            logger.debug(f"Agent iteration {iteration + 1}/{max_iterations}")

            try:
                response = await self._llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=config.temperature
                )

                # Get assistant message
                assistant_message = response.choices[0].message

                # Check if we got tool calls
                if assistant_message.tool_calls:
                    # Add assistant message to history
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })

                    # Process each tool call
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        logger.info(f"Agent calling tool: {tool_name}({arguments})")

                        # Execute the tool (pass tools cache for resource URI lookup)
                        result = await self._execute_agent_tool(tool_name, arguments, tools_cache=tools)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                else:
                    # No tool calls - check if there's a query in the response
                    content = assistant_message.content or ""
                    logger.debug(f"Agent response without tool calls: {content[:200]}")

                    # extract query if present
                    print(content)
                    start_tag = "<QUERY>"
                    end_tag = "</QUERY>"
                    start_idx = content.find(start_tag)
                    end_idx = content.find(end_tag)
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        query = content[start_idx + len(start_tag):end_idx].strip()
                        logger.info(f"Agent produced final query: {query}")
                        return query

                    # Add response to continue the conversation
                    #messages.append({"role": "assistant", "content": content})
                    #messages.append({
                    #    "role": "user",
                    #    "content": "Please use the submit_final_query tool to submit your final EUR-Lex query."
                    #})

            except Exception as e:
                logger.error(f"Agent iteration failed: {e}")
                import traceback
                traceback.print_exc()
                break

        # Fallback if agent loop didn't produce a query
        logger.warning("Agent loop completed without producing a query, using fallback")
        #result = await self.build_query("text", question)
        assert result is not None
        return result #if result else f'TE ~ "{question}"'

    async def list_tools(self) -> List[Dict]:
        """List available MCP tools."""
        result = await self._send_request("tools/list", {})
        return result.get("tools", [])

    async def list_resources(self) -> Dict:
        """List available MCP resources."""
        result = await self._send_request("resources/list", {})
        return result

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource by URI."""
        try:
            result = await self._send_request(
                "resources/read",
                {"uri": uri},
            )
            # Resources return contents array
            if isinstance(result, dict) and "contents" in result:
                for content in result["contents"]:
                    if "text" in content:
                        return content["text"]
            return None
        except Exception as e:
            logger.error(f"Read resource failed: {e}")
            return None

    def _parse_search_results(self, content: List[Dict]) -> List[Dict]:
        """Parse EUR-Lex search results from MCP tool response.

        Server returns: [{"type": "text", "text": "CELEX: xxx ; TITLE: yyy\\n..."}]
        """
        results = []
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                # Check for error messages
                if text.startswith("An error occurred:") or text == "No results found.":
                    logger.warning(f"Search returned: {text}")
                    return []
                # Parse lines in format "CELEX: xxx ; TITLE: yyy"
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    result = {}
                    if "CELEX:" in line:
                        parts = line.split(";")
                        for part in parts:
                            part = part.strip()
                            if part.startswith("CELEX:"):
                                result["celex"] = part[6:].strip()
                            elif part.startswith("TITLE:"):
                                result["title"] = part[6:].strip()
                        if result:
                            results.append(result)
        return results

    async def search(
        self,
        query: str,
        limit: int = 10,
        language: str = "en",
        use_llm: bool = True,
        **kwargs
    ) -> List[Dict]:
        """Search EUR-Lex using expert_search tool.

        Args:
            query: Search query - can be natural language if use_llm=True,
                   or EUR-Lex expert syntax if use_llm=False
            limit: Maximum results (page_size)
            language: Search language (default: "en")
            use_llm: If True, convert natural language to expert syntax using LLM.
                     If False, use query as-is (must be valid expert syntax).

        Returns:
            List of results with 'celex' and 'title' keys
        """
        try:
            # Convert natural language to expert query syntax if needed
            if use_llm:
                expert_query = await self.natural_language_to_query(query)
                logger.info(f"Using expert query: {expert_query}")
            else:
                expert_query = query

            if not expert_query:
                logger.warning("No expert query generated, returning empty results")
                return []

            result = await self.call_tool(
                "expert_search",
                {
                    "query": expert_query,
                    "language": language,
                    "page": 1,
                    "page_size": limit,
                },
            )
            print(result)
            # Parse the result - MCP tools return content array
            if isinstance(result, dict) and "content" in result:
                return self._parse_search_results(result["content"])
            elif isinstance(result, list):
                return self._parse_search_results(result)
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    async def get_document(self, celex: str) -> Optional[Dict]:
        """Get document by CELEX number using resource.

        Args:
            celex: CELEX document identifier (e.g., "32016R0679")

        Returns:
            Dict with 'text' key containing document content
        """
        try:
            uri = f"resource://eurlex/document/{celex}"
            text = await self.read_resource(uri)
            if text:
                return {"celex": celex, "text": text}
            return None
        except Exception as e:
            logger.error(f"Get document failed: {e}")
            return None

    async def validate_query(self, query: str) -> Dict:
        """Validate a query using validate_eurlex_query tool.

        Returns:
            Dict with 'valid' bool and 'message' string
        """
        try:
            result = await self.call_tool(
                "validate_eurlex_query",
                {"query": query},
            )
            if isinstance(result, dict) and "content" in result:
                for item in result["content"]:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        is_valid = "✅" in text or "Valid" in text
                        return {"valid": is_valid, "message": text}
            return {"valid": False, "message": "Unknown validation result"}
        except Exception as e:
            logger.error(f"Validate query failed: {e}")
            return {"valid": False, "message": str(e)}


# Aliases for compatibility
HTTPSSEMCPClient = FastMCPClient
HTTPMCPClient = FastMCPClient


def create_client(
    client_type: str = "http",
    **kwargs,
):
    """
    Factory function to create appropriate client.

    Args:
        client_type: One of "http", "mcp", "mock"
        **kwargs: Additional arguments for client initialization
            - base_url: Server URL (default from mcp_client_config.server_base_url)
            - timeout: Request timeout (default from mcp_client_config.timeout_seconds)
            - llm_client: Pre-configured LLM client instance
            - llm_config: QueryConversionLLMConfig instance (default: query_llm_config)

    Returns:
        Client instance
    """
    if client_type == "http":
        return FastMCPClient(**kwargs)
    elif client_type == "mcp":
        return MCPClient(**kwargs)
    elif client_type == "mock":
        return MockMCPClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


async def main():
    """Test MCP client connectivity with EUR-Lex FastMCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Use FastMCP client with settings from config
    print(f"Connecting to MCP server at: {mcp_client_config.server_base_url}")
    print(f"LLM provider: {query_llm_config.llm_provider} ({query_llm_config.llm_model})")
    client = create_client("http")

    try:
        await client.connect()

        # List tools
        print("\n--- Listing Tools ---")
        tools = await client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")

        # Test building a query
        print("\n--- Building Query ---")
        #query = await client.build_query("text", "data protection")
        query = "cheese transport"
        print(f"Built query: {query}")

        # Test validation
        if query:
            print("\n--- Validating Query ---")
            validation = await client.validate_query(query)
            print(f"Validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")

        # Test search with proper EUR-Lex syntax (no LLM)
        print("\n--- Searching (expert syntax, no LLM) ---")
        results = await client.search("TE ~ GDPR", limit=5, use_llm=False)
        print(f"Search results for 'TE ~ GDPR':")
        for r in results:
            print(f"  - {r.get('celex', 'N/A')}: {r.get('title', '')[:60]}...")

        # Test search with natural language (using LLM)
        print("\n--- Searching (natural language, with LLM) ---")
        natural_query = "What was the directive about maximum permitted levels for undesirable substances in feedingstuffs?"
        print(f"Natural language query: {natural_query}")
        results_nl = await client.search(natural_query, limit=5, use_llm=True)
        print(f"Search results:")
        for r in results_nl:
            print(f"  - {r.get('celex', 'N/A')}: {r.get('title', '')[:60]}...")

        # Test getting a document (GDPR)
        if results:
            print("\n--- Fetching Document ---")
            celex = results[0].get("celex")
            if celex:
                doc = await client.get_document(celex)
                if doc:
                    text = doc.get("text", "")
                    print(f"Document {celex}: {len(text)} characters")
                    print(f"Preview: {text[:200]}...")

    except Exception as e:
        logger.error(f"Client test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
