"""
MCP Client for EUR-Lex Server Interaction.

This module provides a client wrapper for interacting with the EUR-Lex MCP server
during evaluation.
"""
import asyncio
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import mcp_client_config

logger = logging.getLogger(__name__)


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


class HTTPMCPClient:
    """HTTP-based MCP client for servers with HTTP transport."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict]:
        """Search via HTTP endpoint."""
        import aiohttp

        url = f"{self.base_url}/search"
        params = {"query": query, "limit": limit}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=self.timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("results", [])
                else:
                    logger.error(f"HTTP search failed: {resp.status}")
                    return []

    async def get_document(self, celex: str) -> Optional[Dict]:
        """Get document via HTTP endpoint."""
        import aiohttp

        url = f"{self.base_url}/documents/{celex}"

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=self.timeout) as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"HTTP get document failed: {resp.status}")
                    return None


def create_client(
    client_type: str = "mcp",
    **kwargs,
) -> MCPClient:
    """
    Factory function to create appropriate client.

    Args:
        client_type: One of "mcp", "http", "mock"
        **kwargs: Additional arguments for client initialization

    Returns:
        Client instance
    """
    if client_type == "mcp":
        return MCPClient(**kwargs)
    elif client_type == "http":
        return HTTPMCPClient(**kwargs)
    elif client_type == "mock":
        return MockMCPClient(**kwargs)
    else:
        raise ValueError(f"Unknown client type: {client_type}")


async def main():
    """Test MCP client connectivity."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    client = MCPClient()

    try:
        await client.connect()

        # List tools
        tools = await client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")

        # Test search
        results = await client.search("GDPR data protection", limit=5)
        print(f"\nSearch results for 'GDPR data protection':")
        for r in results:
            print(f"  - {r.get('celex')}: {r.get('title', '')[:60]}...")

    except Exception as e:
        logger.error(f"Client test failed: {e}")

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
