import asyncio
import os
import sys
import argparse
from dotenv import load_dotenv

from fastmcp import FastMCP

from client import EurLexClient


def _format_search_results(raw_results: dict) -> list[dict]:
    """
    Parses the verbose SOAP/Zeep response into a clean list of document dictionaries.
    """
    if not raw_results or raw_results.get("error"):
        return [raw_results] if raw_results else []

    try:
        results_list = raw_results["body"]["searchResults"]["result"]
        if not isinstance(results_list, list):
            results_list = [results_list]
    except (KeyError, TypeError):
        return []

    formatted_docs = []
    for item in results_list:
        try:
            # Extract main content and links, with robust handling for missing fields
            content = item.get("content", {})
            links = item.get("document_link", [])
            if not isinstance(links, list):
                links = [links]

            # Find the HTML link from the list of document manifestations
            html_link = next((link["URL"] for link in links if link.get("TYPE") == "html"), None)

            # Safely extract CELEX and Title using .get() to avoid KeyErrors
            celex = content.get("NOTICE", {}).get("ID_CELEX", {}).get("VALUE")
            title = (
                content.get("NOTICE", {})
                .get("EXPRESSION", [{}])[0]
                .get("EXPRESSION_TITLE", [{}])[0]
                .get("VALUE")
            )

            if celex and title:
                formatted_docs.append(
                    {
                        "celex": celex,
                        "title": title.strip(),
                        "url": html_link,
                    }
                )
        except (KeyError, TypeError, IndexError):
            # If a single item is malformed, skip it and continue
            print(f"Skipping malformed search result item.", file=sys.stderr)
            continue

    return formatted_docs


# --- Main Server Logic ---

load_dotenv()
EURLEX_USERNAME = os.environ.get("EURLEX_USERNAME")
EURLEX_PASSWORD = os.environ.get("EURLEX_PASSWORD")

if not EURLEX_USERNAME or not EURLEX_PASSWORD:
    print(
        "Error: EURLEX_USERNAME and EURLEX_PASSWORD must be set in your environment or a .env file.",
        file=sys.stderr,
    )
    exit(1)

eurlex_client = EurLexClient(EURLEX_USERNAME, EURLEX_PASSWORD)

mcp = FastMCP(
    name="EurLexMCPServer",
    instructions="A server for searching European Union law. Use the `expert_search` tool for broad searches or access a specific document via the `resource://eurlex/document/{celex_number}` resource.",
    dependencies=["zeep", "python-dotenv"],
)


# --- Tool Definition ---
@mcp.tool
async def expert_search(
    query: str, language: str = "en", page: int = 1, page_size: int = 10
) -> list[dict]:
    """
    Performs an expert search for EU legal documents and returns a clean list of results.
    The query must be written using EUR-Lex expert query syntax.
    Example query: 'DN = 32024R*' to find all regulations from 2024.
    """
    print(f"Performing EUR-LEX search for query: '{query}'", file=sys.stderr)
    raw_results = await eurlex_client.search(
        query=query, language=language, page=page, page_size=page_size
    )
    return _format_search_results(raw_results)


# --- Resource Template Definition ---
@mcp.resource("resource://eurlex/document/{celex_number}")
async def get_document_by_celex(celex_number: str) -> dict:
    """
    Fetches a single EU legal document by its unique CELEX number.
    """
    print(f"Fetching document with CELEX: '{celex_number}'", file=sys.stderr)
    # The expert query syntax for an exact CELEX match is DN = '...'
    query = f"DN = '{celex_number}'"
    raw_results = await eurlex_client.search(query=query, page_size=1)

    formatted_results = _format_search_results(raw_results)

    if formatted_results:
        return formatted_results[0]
    else:
        return {"error": "Document not found", "celex_number": celex_number}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUR-LEX MCP Server: Runs in STDIO or HTTP mode.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="The transport protocol to use. Defaults to 'stdio' for local clients.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host for HTTP transport. Defaults to '127.0.0.1'."
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for HTTP transport. Defaults to 8000."
    )
    args = parser.parse_args()

    if args.transport == "http":
        print(f"Starting server in HTTP mode on http://{args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        print("Starting server in STDIO mode.", file=sys.stderr)
        mcp.run(transport="stdio")
