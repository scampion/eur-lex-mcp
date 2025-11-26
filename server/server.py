import asyncio
import os
import sys
import argparse
import httpx
from lxml import etree
from dotenv import load_dotenv
from fastmcp import FastMCP

# --- Main Server Logic ---

load_dotenv()
EURLEX_USERNAME = os.environ.get("EURLEX_USERNAME")
EURLEX_PASSWORD = os.environ.get("EURLEX_PASSWORD")

if not EURLEX_USERNAME or not EURLEX_PASSWORD:
    print(
        "Error: EURLEX_USERNAME and EURLEX_PASSWORD must be set in your environment or a .env file.",
        file=sys.stderr,
    )
    sys.exit(1)


class EurLexClient:
    """An async client for the EUR-LEX SOAP Webservice using httpx and lxml."""

    ENDPOINT_URL = "https://eur-lex.europa.eu/EURLexWebService"
    # Namespaces for SOAP 1.2
    NS_SOAP = "http://www.w3.org/2003/05/soap-envelope"
    NS_WSSE = "http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-wssecurity-secext-1.0.xsd"
    NS_SEARCH = "http://eur-lex.europa.eu/search"

    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.client = httpx.AsyncClient(timeout=45.0)

    def _build_soap_request(self, query: str, language: str, page: int, page_size: int) -> bytes:
        """Constructs the SOAP 1.2 XML payload for the searchRequest operation."""
        # Namespace map for lxml
        nsmap = {
            'soap': self.NS_SOAP,
            'sear': self.NS_SEARCH,
            'wsse': self.NS_WSSE
        }
        # Root <soap:Envelope>
        root = etree.Element(f"{{{self.NS_SOAP}}}Envelope", nsmap=nsmap)
        
        # <soap:Header> with WS-Security
        header = etree.SubElement(root, f"{{{self.NS_SOAP}}}Header")
        security = etree.SubElement(
            header,
            f"{{{self.NS_WSSE}}}Security",
            attrib={f"{{{self.NS_SOAP}}}mustUnderstand": "true"}
        )
        username_token = etree.SubElement(security, f"{{{self.NS_WSSE}}}UsernameToken")
        etree.SubElement(username_token, f"{{{self.NS_WSSE}}}Username").text = self.username
        etree.SubElement(
            username_token,
            f"{{{self.NS_WSSE}}}Password",
            Type="http://docs.oasis-open.org/wss/2004/01/oasis-200401-wss-username-token-profile-1.0#PasswordText"
        ).text = self.password

        # <soap:Body> with the search query
        body = etree.SubElement(root, f"{{{self.NS_SOAP}}}Body")
        search_request = etree.SubElement(body, f"{{{self.NS_SEARCH}}}searchRequest")
        etree.SubElement(search_request, f"{{{self.NS_SEARCH}}}expertQuery").text = query
        etree.SubElement(search_request, f"{{{self.NS_SEARCH}}}page").text = str(page)
        etree.SubElement(search_request, f"{{{self.NS_SEARCH}}}pageSize").text = str(page_size)
        etree.SubElement(search_request, f"{{{self.NS_SEARCH}}}searchLanguage").text = language

        return etree.tostring(root, xml_declaration=True, encoding="UTF-8")

    async def search(self, query: str, language: str = "en", page: int = 1, page_size: int = 10) -> dict:
        """Performs an expert search by sending a SOAP request."""
        soap_request_body = self._build_soap_request(query, language, page, page_size)
        headers = {
            "Content-Type": "application/soap+xml; charset=utf-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
            "Connection": "close",
        }

        try:
            print(f"Sending SOAP request for query: '{query}'", file=sys.stderr)
            response = await self.client.post(self.ENDPOINT_URL, content=soap_request_body, headers=headers)
            response.raise_for_status()
            print("SOAP response received successfully.", file=sys.stderr)
            return self._parse_soap_response(response.content)
        except httpx.HTTPStatusError as e:
            print(f"HTTP error occurred: {e.response.status_code}\nResponse body: {e.response.text}", file=sys.stderr)
            return {"error": "HTTP request failed", "details": e.response.text}
        except Exception as e:
            print(f"An unexpected error occurred while querying EUR-LEX: {e}", file=sys.stderr)
            return {"error": str(e)}

    def _parse_soap_response(self, xml_content: bytes) -> dict:
        """Parses the SOAP XML response into a dictionary."""
        try:
            # Parse the XML, removing blank text nodes
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.fromstring(xml_content, parser)
            
            # Find all 'result' nodes in the response
            search_ns = {'sear': 'http://eur-lex.europa.eu/search'}
            results_nodes = root.xpath("//sear:result", namespaces=search_ns)
            
            results = []
            for node in results_nodes:
                # Values are nested in a <VALUE> tag in the SOAP 1.2 response
                celex = node.findtext('.//sear:ID_CELEX/sear:VALUE', namespaces=search_ns)
                title = node.findtext('.//sear:EXPRESSION_TITLE/sear:VALUE', namespaces=search_ns)

                titles = node.xpath('.//sear:EXPRESSION_TITLE/sear:VALUE/text() | .//sear:WORK_TITLE/sear:VALUE/text()', namespaces=search_ns)
                if titles:
                    title = titles[0]

                # Find the HTML link specifically
                html_link_node = node.find(".//sear:document_link[@type='html']", namespaces=search_ns)
                url = html_link_node.text if html_link_node is not None else None

                if celex and title:
                    results.append({"celex": celex, "title": title.strip(), "url": url})
            
            return {"results": results}
        except etree.XMLSyntaxError as e:
            print(f"Error parsing SOAP XML response: {e}", file=sys.stderr)
            return {"error": "Failed to parse SOAP response"}


def _format_search_results_for_mcp(parsed_results: dict) -> list[dict]:
    """Formats the parsed SOAP data into the text format expected by the MCP client."""
    if "error" in parsed_results:
        return [{"type": "text", "text": f"An error occurred: {parsed_results.get('details', parsed_results['error'])}"}]

    if not parsed_results.get("results"):
        return [{"type": "text", "text": "No results found."}]
    
    # Format into a readable string for the test client
    header = f"CELEX{'':<25} | Title"
    lines = [header, "-" * (len(header) + 20)]
    for item in parsed_results["results"]:
        title_preview = item['title'][:70] + '...' if len(item['title']) > 70 else item['title']
        lines.append(f"{item['celex']:<30} | {title_preview}")
    
    return [{"type": "text", "text": "\n".join(lines)}]

eurlex_client = EurLexClient(EURLEX_USERNAME, EURLEX_PASSWORD)

mcp = FastMCP(
    name="EurLexMCPServer",
    instructions="A server for searching European Union law.",
    dependencies=["httpx", "lxml", "python-dotenv"],
)

@mcp.tool
async def expert_search(query: str, language: str = "en", page: int = 1, page_size: int = 10) -> list[dict]:
    """
    Performs an expert search for EU legal documents.
    Example query: 'DN = 32024R*' for all regulations from 2024.
    """
    print(f"Executing tool 'expert_search' with query: '{query}'", file=sys.stderr)
    parsed_results = await eurlex_client.search(query, language, page, page_size)
    return _format_search_results_for_mcp(parsed_results)

@mcp.resource("resource://eurlex/document/{celex_number}")
async def get_document_by_celex(celex_number: str) -> dict:
    """Fetches a single EU legal document by its CELEX number."""
    print(f"Executing resource 'get_document_by_celex' for: '{celex_number}'", file=sys.stderr)
    query = f"DN = '{celex_number}'"
    parsed_results = await eurlex_client.search(query, page_size=1)
    
    if "error" in parsed_results:
        return {"error": parsed_results.get('details', parsed_results['error'])}
    
    results = parsed_results.get("results")
    return results[0] if results else {"error": "Document not found"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUR-LEX MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "http":
        print(f"Starting server in HTTP mode on http://{args.host}:{args.port}", file=sys.stderr)
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    else:
        print("Starting server in STDIO mode.", file=sys.stderr)
        mcp.run(transport="stdio")