# EUR-LEX MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

This server acts as a modern, AI-friendly bridge to the official EUR-Lex SOAP webservice. It exposes the complex EUR-Lex API as a simple and powerful **Model Context Protocol (MCP)** server, making the entire body of European Union law accessible to modern AI agents, developer tools, and applications.

By wrapping the legacy SOAP API, this server unlocks use cases in automated legal research, compliance monitoring, policy analysis, and more, without requiring AI agents to understand the complexities of WSDL or XML.

## ✨ Features

*   **Dual-Mode Transport**: Runs over **STDIO** for secure, local integration with desktop applications (like Claude Desktop, Cursor) and over **HTTP** for remote access by AI agents and web services.
*   **Expert Search Tool**: Provides an `expert_search` tool that allows for powerful, flexible querying using the native [EUR-Lex expert search syntax](https://eur-lex.europa.eu/content/help/search/expert-search.html).
*   **Direct Document Access**: Includes a `get_document_by_celex` resource template for fetching specific legal documents directly via a clean URI (`resource://eurlex/document/{celex_number}`).
*   **Clean, Structured Output**: Automatically parses the verbose SOAP/XML responses into a clean, easy-to-use JSON format, perfect for LLM consumption.
*   **Easy Installation**: Can be easily installed into compatible clients using the `fastmcp` command-line tool.

## 🔌 Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.11+** and the `uv` package manager. If you don't have `uv`, install it with:
    ```bash
    pip install uv
    ```
2.  **EUR-LEX Webservice Credentials**. You must register for a webservice account to get a username and password. You can do this by following the "Webservice registration" link on the EUR-Lex website after logging in.

## 🚀 Setup & Installation

Follow these steps to get the server up and running on your local machine.

### 1. Clone the Repository

Clone this project to your local machine.

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Set Up the Environment

Create and activate a virtual environment, then install the required dependencies using `uv`.

```bash
# Create and activate the virtual environment
uv venv

# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate

# Install dependencies
uv pip install -r pyproject.toml
```

### 3. Configure Credentials

The server requires your EUR-LEX webservice username and password. The most secure way to provide them is through a `.env` file.

1.  Create a file named `.env` in the root of the project directory.
2.  **Never commit this file to Git!** It is already included in the `.gitignore`.
3.  Add your credentials to the `.env` file like this:

    ```env
    # .env
    EURLEX_USERNAME="your-eurlex-username"
    EURLEX_PASSWORD="your-webservice-password"
    ```

## 🔧 Usage

This server can be run in two modes, depending on your use case.

### Mode 1: For Local Desktop Clients (Claude, Cursor, etc.)

This is the recommended mode for integrating with local applications. The server will run over **STDIO**.

You do not need to run the server manually. Instead, you "install" it into your client application.

#### Using `fastmcp install` (for Claude Desktop)

The easiest way to install the server is with the `fastmcp` CLI:

```bash
# Ensure you have fastmcp's CLI dependencies
uv pip install "fastmcp[cli]"

# Run the install command from the project root
fastmcp install server.py --name "EUR-LEX Search"
```

This command will automatically detect your `server.py`, read its dependencies, and configure Claude Desktop to run it. Simply **restart Claude Desktop** and the "EUR-LEX Search" tool will be available.

#### Manual Configuration (for other apps)

For other applications, you will need to manually configure them to run the `server.py` script. Find the MCP server configuration section in your application's settings and add an entry that executes the script.

<details>
<summary>Example manual JSON configuration</summary>

```json
{
  "mcpServers": {
    "eurlex-search": {
      "command": "/absolute/path/to/your/.venv/bin/python",
      "args": [
        "/absolute/path/to/your/project/server.py"
      ]
    }
  }
}
```
**Note:** You must use absolute paths for the command and script arguments.
</details>

### Mode 2: As a Remote HTTP Service

This mode is for AI agents or other web services that need to call the server over the network.

Run the following command from your terminal:

```bash
python server.py --transport http
```

The server will start and be accessible at `http://127.0.0.1:8000`.

You can customize the host and port:

```bash
python server.py --transport http --host 0.0.0.0 --port 9001
```

## 📚 API Reference

The EUR-LEX MCP Server exposes the following capabilities:

### Tool: `expert_search`

Performs a flexible search across the EUR-Lex database using the official expert search syntax.

*   **Parameters**:
    *   `query` (str): The search query, following EUR-Lex syntax. **Required**.
    *   `language` (str): The two-letter language code for the search. Default: `"en"`.
    *   `page` (int): The page number of the results. Default: `1`.
    *   `page_size` (int): The number of results per page. Default: `10`.
*   **Returns**: A list of document objects.

**Example Return Value:**
```json
[
  {
    "celex": "32016R0679",
    "title": "Regulation (EU) 2016/679 of the European Parliament and of the Council...",
    "url": "http://publications.europa.eu/resource/celex/32016R0679.EN.html"
  }
]
```

### Resource: `get_document_by_celex`

Fetches a single, specific document directly by its CELEX number.

*   **URI Template**: `resource://eurlex/document/{celex_number}`
*   **Parameters**:
    *   `celex_number` (str): The unique CELEX identifier of the document.
*   **Returns**: A single document object.

**Example URI:**
`resource://eurlex/document/32016R0679`

**Example Return Value:**
```json
{
  "celex": "32016R0679",
  "title": "Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation) (Text with EEA relevance)",
  "url": "http://publications.europa.eu/resource/celex/32016R0679.EN.html"
}
```

## 🧑‍💻 Development & Contribution

Contributions are welcome! If you'd like to improve the server, please follow these steps:

1.  Fork the repository.
2.  Clone the fork to your IDE.
2.  Create a new branch for your feature (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.