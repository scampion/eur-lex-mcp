import re
import asyncio
import os
import sys
import argparse
import httpx
from lxml import etree
from dotenv import load_dotenv
from bs4 import BeautifulSoup
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
            # print traceback for debugging
            import traceback
            traceback.print_exc()
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

                titles = node.xpath('.//sear:EXPRESSION_TITLE/sear:VALUE/text() | .//sear:WORK_TITLE/sear:VALUE/text()',
                                    namespaces=search_ns)
                if titles:
                    title = titles[0]

                if celex and title:
                    results.append({"celex": celex, "title": title.strip()})

            return {"results": results}
        except etree.XMLSyntaxError as e:
            print(f"Error parsing SOAP XML response: {e}", file=sys.stderr)
            return {"error": "Failed to parse SOAP response"}


def _node_to_md(el):
    parts = []
    for child in el.children:
        if getattr(child, "name", None) is None:
            parts.append(str(child))
            continue
        tag = child.name.lower()
        inner = _node_to_md(child).strip()
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            parts.append("\n" + "#" * level + " " + inner + "\n\n")
        elif tag == "p":
            parts.append(inner + "\n\n")
        elif tag in ("strong", "b"):
            parts.append("**" + inner + "**")
        elif tag in ("em", "i"):
            parts.append("*" + inner + "*")
        elif tag == "a":
            href = child.get("href", "")
            parts.append(f"[{inner}]({href})")
        elif tag in ("ul", "ol"):
            for idx, li in enumerate(child.find_all("li", recursive=False), start=1):
                li_text = _node_to_md(li).strip()
                if tag == "ul":
                    parts.append(f"- {li_text}\n")
                else:
                    parts.append(f"{idx}. {li_text}\n")
            parts.append("\n")
        elif tag == "li":
            parts.append(inner)
        else:
            parts.append(inner)
    return "".join(parts)


def _format_search_results_for_mcp(parsed_results: dict) -> list[dict]:
    """Formats the parsed SOAP data into the text format expected by the MCP client."""
    if "error" in parsed_results:
        return [
            {"type": "text", "text": f"An error occurred: {parsed_results.get('details', parsed_results['error'])}"}]

    if not parsed_results.get("results"):
        return [{"type": "text", "text": "No results found."}]

    # Format en "CELEX: XXXXX ; TITLE: abcdef..."
    lines = []
    for item in parsed_results["results"]:
        celex = item.get("celex", "").strip()
        title = item.get("title", "").strip()
        if celex and title:
            lines.append(f"CELEX: {celex} ; TITLE: {title}")
        elif celex:
            lines.append(f"CELEX: {celex}")
        elif title:
            lines.append(f"TITLE: {title}")

    return [{"type": "text", "text": "\n".join(lines)}]


async def _get_text_from_celex_async(celex_number: str) -> str:
    """Async version: Fetches and extracts text content from a EUR-Lex document by CELEX number."""
    url = f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex_number}"

    # Use the existing async client from EurLexClient or create a new one
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception as e:
            print(f"Error fetching document {celex_number}: {e}", file=sys.stderr)
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')
        content_node = soup.find(id="PP4Contents")
        if content_node:
            content_md = _node_to_md(content_node).strip()
            content = re.sub(r"\n{3,}", "\n\n", content_md)

            if "DEBUG_EURLEX" in os.environ:
                with open(f"eurlex_{celex_number}.txt", "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Saved debug TXT content to eurlex_{celex_number}.txt", file=sys.stderr)

            return content
        return ""


eurlex_client = EurLexClient(EURLEX_USERNAME, EURLEX_PASSWORD)

_EXPERT_SEARCH_DOCS = ""
try:
    md_path = os.path.join(os.path.dirname(__file__), "..", "eurlex_expert_search.md")
    with open(md_path, "r", encoding="utf-8") as _f:
        _EXPERT_SEARCH_DOCS = _f.read()
    print(f"Loaded expert search documentation ({len(_EXPERT_SEARCH_DOCS)} chars)", file=sys.stderr)
except Exception as e:
    print(f"Warning: Could not load eurlex_expert_search.md: {e}", file=sys.stderr)


_KEY_SYNTAX = """
⚠️ EUR-LEX REQUIRES SPECIFIC QUERY SYNTAX - DO NOT USE PLAIN TEXT ⚠️

REQUIRED FORMAT: FieldName OPERATOR SearchValue

OPERATORS: ~ (contains), = (exact), >=, <=, >, <

KEY FIELDS:
  TE  = Full text content
  TI  = Document title  
  DN  = CELEX number (document ID)
  DD  = Date (YYYY-MM-DD format)
  AU  = Author (Commission, Parliament, Council)
  CT  = Document type (Regulation, Directive, Decision)
  FM  = Legal force (INFORCE, REPEALED)

BOOLEAN: AND, OR, NOT (UPPERCASE only)
PROXIMITY: NEAR, NEAR5, NEAR10, NEAR20
WILDCARDS: * (multiple chars), ? (single char)

MANDATORY EXAMPLES - COPY THESE PATTERNS:
  ✅ TE ~ cheese NEAR10 transport
  ✅ DN = 32016R0679  
  ✅ TI ~ "data protection" AND DD >= 2020-01-01
  ✅ (TI ~ AI OR TI ~ "artificial intelligence") AND CT = Regulation

  ❌ cheese transport (WRONG!)
  ❌ find GDPR (WRONG!)
  ❌ search for regulations (WRONG!)

ALWAYS use field syntax. Read resource://eurlex/expert-search-documentation for details.
"""


mcp = FastMCP(
    name="EurLexMCPServer",
    instructions=f"""A server for searching European Union legal documents using EUR-Lex.

{_KEY_SYNTAX}

AVAILABLE TOOLS:
1. expert_search - Search EUR-Lex (REQUIRES FIELD SYNTAX - see above!)
2. build_eurlex_query - Helper to construct valid queries (USE THIS if unsure!)
3. validate_eurlex_query - Check query syntax before searching

RESOURCES:
- resource://eurlex/document/{{celex}} - Fetch full document text
- resource://eurlex/expert-search-documentation - Complete syntax guide

WORKFLOW FOR SEARCHING:
1. If user asks in natural language, use build_eurlex_query first
2. Or construct query following the syntax rules above
3. Optionally validate with validate_eurlex_query
4. Execute expert_search with properly formatted query
5. Use document resource to fetch full text of results

REMEMBER: Plain text queries will FAIL. Always use field syntax!
""",
    dependencies=["httpx", "lxml", "python-dotenv", "beautifulsoup4", "fastmcp"],
)

@mcp.tool
async def expert_search(query: str, language: str = "en", page: int = 1, page_size: int = 10) -> list[dict]:
    """Search EU legal documents using EUR-Lex expert query syntax.

    The expert query syntax enables precise searches with field-specific queries and boolean operators.

    IMPORTANT QUERY SYNTAX:
    - Field searches: FieldName ~ SearchTerm  or  FieldName = ExactValue
    - Boolean: AND, OR, NOT
    - Proximity: NEAR (default 10 words), NEAR5, NEAR20, etc.
    - Wildcards: * (multiple chars), ? (single char)
    - Phrases: Use quotes "exact phrase"

    COMMON FIELDS:
    - DN: Document Number (CELEX number) - e.g., "DN = 32016R0679"
    - TI: Title - e.g., "TI ~ data protection"
    - TE: Full text content - e.g., "TE ~ GDPR"
    - DD: Document Date - e.g., "DD >= 2020-01-01"
    - AU: Author/Institution - e.g., "AU = Parliament"
    - CT: Document Type - e.g., "CT = Regulation"

    EXAMPLE QUERIES:
    1. Find GDPR: "DN = 32016R0679"
    2. Recent AI regulations: "TI ~ artificial intelligence AND DD >= 2023"
    3. Cheese transport laws: "TE ~ cheese NEAR10 transport"
    4. Commission directives: "AU = Commission AND CT = Directive"
    5. Privacy and data together: "TE ~ (privacy AND data)"

    RETURNS:
    Results as text with one result per line in format:
    CELEX: [number] ; TITLE: [document title]

    After getting results, use the resource template "resource://eurlex/document/{celex_number}" 
    to fetch the full document text.
    """ + ("\n\n" + "DETAILED DOCUMENTATION:\n" + "=" * 50 + "\n" + _EXPERT_SEARCH_DOCS if _EXPERT_SEARCH_DOCS else "")

    print(f"Executing tool 'expert_search' with query: '{query}'", file=sys.stderr)
    parsed_results = await eurlex_client.search(query, language, page, page_size)
    return _format_search_results_for_mcp(parsed_results)


@mcp.resource("resource://eurlex/document/{celex_number}")
async def get_document_by_celex(celex_number: str) -> str:
    """Fetches the complete text of an EU legal document by its CELEX number.

    CELEX is the unique identifier for EU documents. After performing an expert_search,
    use the CELEX numbers from the results to retrieve full document text.

    Examples:
    - resource://eurlex/document/32016R0679 (GDPR)
    - resource://eurlex/document/32022R0868 (Data Governance Act)
    - resource://eurlex/document/62021CJ0252 (Court case)

    The resource returns the complete document text in markdown format, including:
    - Full legal text
    - Preambles and recitals
    - Articles and provisions
    - Annexes
    """
    content = await _get_text_from_celex_async(celex_number)
    if not content:
        return f"Document with CELEX number {celex_number} not found or could not be retrieved."
    return content


@mcp.resource("resource://eurlex/expert-search-documentation")
async def get_expert_search_documentation() -> str:
    """Complete EUR-Lex expert search syntax documentation.

    Read this resource when you need detailed information about:
    - All available search fields and their descriptions
    - Query operators and syntax rules
    - Advanced search techniques (proximity, wildcards, boolean logic)
    - Comprehensive examples for various search scenarios
    - Field-specific syntax and value formats

    This is the authoritative reference for constructing EUR-Lex queries.
    """
    if not _EXPERT_SEARCH_DOCS:
        return "Documentation file not available. Using built-in syntax guide:\n\n" + _KEY_SYNTAX

    return f"""EUR-LEX EXPERT SEARCH - COMPLETE DOCUMENTATION
{'=' * 60}

{_EXPERT_SEARCH_DOCS}

{'=' * 60}
QUICK REFERENCE:
{_KEY_SYNTAX}
"""

@mcp.prompt()
async def search_eurlex_help() -> str:
    """Provides guidance on using EUR-Lex expert search with full documentation."""
    return f"""You are helping a user search EUR-Lex legal documents. 

CRITICAL: EUR-Lex requires SPECIFIC QUERY SYNTAX. You MUST NOT send plain text queries.

Before constructing any query, read the complete documentation:
resource://eurlex/expert-search-documentation

Key Syntax Rules:
- Field searches: FieldName ~ SearchTerm or FieldName = ExactValue
- Common fields: TE (text), TI (title), DN (CELEX number), DD (date)
- Boolean: AND, OR, NOT
- Proximity: NEAR, NEAR5, NEAR10, etc.

Example: Instead of "cheese transport", use "TE ~ cheese NEAR10 transport"

Always construct queries using proper field syntax based on the documentation."""


@mcp.tool
async def build_eurlex_query(
        search_type: str,
        search_terms: str,
        filters: dict = None
) -> list[dict]:
    """Helper tool to build properly formatted EUR-Lex expert search queries.

    This tool helps construct valid EUR-Lex queries by providing a structured interface.
    Use this instead of trying to write query syntax manually.

    Args:
        search_type: What to search in. Options:
            - "text" → searches full document text (TE field)
            - "title" → searches document titles (TI field)
            - "celex" → finds specific document by CELEX number (DN field)
            - "combined" → searches multiple fields

        search_terms: The terms to search for. Examples:
            - "cheese transport" → will be converted to proximity search
            - "GDPR" → single term
            - "data protection" → will be treated as phrase

        filters: Optional dictionary with:
            - "date_from": "YYYY-MM-DD" → documents from this date onwards
            - "date_to": "YYYY-MM-DD" → documents up to this date
            - "author": "Commission" | "Parliament" | "Council"
            - "doc_type": "Regulation" | "Directive" | "Decision"
            - "in_force": True → only documents currently in force

    Returns:
        A formatted query string ready for expert_search, plus explanation

    Examples:
        build_eurlex_query("text", "cheese transport")
        → Returns: "TE ~ cheese NEAR10 transport"

        build_eurlex_query("title", "data protection", {"date_from": "2020-01-01"})
        → Returns: "TI ~ \"data protection\" AND DD >= 2020-01-01"

        build_eurlex_query("celex", "32016R0679")
        → Returns: "DN = 32016R0679"
    """

    query_parts = []

    # Build main search part
    if search_type == "text":
        # Check if multiple words - use proximity
        words = search_terms.strip().split()
        if len(words) > 1:
            query_parts.append(f"TE ~ {words[0]} NEAR10 {' NEAR10 '.join(words[1:])}")
        else:
            query_parts.append(f"TE ~ {search_terms}")

    elif search_type == "title":
        # Use phrase search for titles
        if " " in search_terms:
            query_parts.append(f'TI ~ "{search_terms}"')
        else:
            query_parts.append(f"TI ~ {search_terms}")

    elif search_type == "celex":
        query_parts.append(f"DN = {search_terms}")

    elif search_type == "combined":
        # Search in both title and text
        if " " in search_terms:
            query_parts.append(f'(TI ~ "{search_terms}" OR TE ~ {search_terms.split()[0]})')
        else:
            query_parts.append(f"(TI ~ {search_terms} OR TE ~ {search_terms})")

    # Add filters
    if filters:
        if "date_from" in filters:
            query_parts.append(f"DD >= {filters['date_from']}")
        if "date_to" in filters:
            query_parts.append(f"DD <= {filters['date_to']}")
        if "author" in filters:
            query_parts.append(f"AU = {filters['author']}")
        if "doc_type" in filters:
            query_parts.append(f"CT = {filters['doc_type']}")
        if filters.get("in_force"):
            query_parts.append("FM = INFORCE")

    # Combine with AND
    final_query = " AND ".join(query_parts)

    explanation = f"""
Built EUR-Lex Expert Query:
==========================
Query: {final_query}

Explanation:
- Search type: {search_type}
- Terms: {search_terms}
- Filters: {filters or 'None'}

This query is now ready to use with expert_search tool.
"""

    return [
        {"type": "text", "text": f"QUERY: {final_query}"},
        {"type": "text", "text": explanation}
    ]


@mcp.tool
async def validate_eurlex_query(query: str) -> list[dict]:
    """Validates if a EUR-Lex query uses correct syntax before executing it.

    Use this to check your query before running expert_search.

    Args:
        query: The expert search query to validate

    Returns:
        Validation results with suggestions if issues found
    """

    issues = []
    suggestions = []

    # Check for field operators
    has_tilde = '~' in query
    has_equals = '=' in query

    if not (has_tilde or has_equals):
        issues.append("❌ No field operator found (~ or =)")
        suggestions.append("Add a field operator like 'TE ~' or 'DN ='")

    # Check for field names
    common_fields = ['TE', 'TI', 'DN', 'DD', 'AU', 'CT', 'FM']
    has_field = any(field in query.upper() for field in common_fields)

    if not has_field:
        issues.append("❌ No recognized field name found")
        suggestions.append(f"Use a field name: {', '.join(common_fields)}")

    # Check for proper boolean operators
    if any(op in query.lower() for op in [' and ', ' or ', ' not ']):
        issues.append("⚠️ Boolean operators should be UPPERCASE (AND, OR, NOT)")

    # Check for unquoted phrases
    words = query.split()
    if len([w for w in words if not w.isupper() and w not in ['~', '=', '(', ')']]) > 3:
        if '"' not in query:
            suggestions.append("💡 Consider using quotes for exact phrases")

    if not issues:
        validation_result = f"""
✅ Query Syntax Valid: {query}

The query appears to use correct EUR-Lex expert syntax.
You can proceed with expert_search.
"""
    else:
        validation_result = f"""
Query Validation Results for: {query}

Issues Found:
{chr(10).join(issues)}

Suggestions:
{chr(10).join(suggestions)}

Example correct queries:
- TE ~ cheese NEAR10 transport
- TI ~ "data protection" AND DD >= 2020-01-01
- DN = 32016R0679
"""

    return [{"type": "text", "text": validation_result}]


@mcp.resource("resource://eurlex/query-examples")
async def get_query_examples() -> str:
    """Collection of working EUR-Lex query examples for common scenarios."""
    return """EUR-LEX EXPERT SEARCH - WORKING EXAMPLES
========================================

BASIC TEXT SEARCH:
  TE ~ GDPR
  TE ~ competition
  TE ~ "climate change"

TITLE SEARCH:
  TI ~ "General Data Protection"
  TI ~ artificial NEAR5 intelligence

FIND SPECIFIC DOCUMENT:
  DN = 32016R0679              (GDPR)
  DN = 32022R0868              (Data Governance Act)

SEARCH WITH DATE FILTER:
  TI ~ climate AND DD >= 2023-01-01
  TE ~ AI AND DD >= 2020-01-01 AND DD <= 2023-12-31

PROXIMITY SEARCH:
  TE ~ cheese NEAR10 transport
  TE ~ digital NEAR5 market NEAR10 regulation

DOCUMENT TYPE FILTER:
  CT = Regulation AND TE ~ competition
  CT = Directive AND TI ~ environment

AUTHOR FILTER:
  AU = Commission AND TI ~ climate
  AU = Parliament AND CT = Regulation AND DD >= 2023

CURRENTLY IN FORCE:
  TE ~ data NEAR5 protection AND FM = INFORCE
  CT = Regulation AND FM = INFORCE AND DD >= 2020

COMPLEX BOOLEAN:
  (TI ~ AI OR TI ~ "artificial intelligence") AND CT = Regulation AND DD >= 2020
  (TE ~ privacy OR TE ~ "data protection") AND AU = Commission AND FM = INFORCE

WILDCARD SEARCH:
  TI ~ environment*
  TE ~ regul?tion

Each of these examples uses proper EUR-Lex field syntax and will work correctly.
"""



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