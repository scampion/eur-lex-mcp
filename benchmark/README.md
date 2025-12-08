# EU Document Classifier

A Python tool for classifying EU legal documents and extracting CELEX identifiers from RDF metadata.

## Features

- **Document Classification**: Identifies the legal document type (Decision, Regulation, Directive, Recommendation, etc.)
- **CELEX Extraction**: Extracts and parses CELEX identifiers from any document URI
- **URI Navigation**: Handles complex owl:sameAs relationships across different URI schemes (CELEX, Official Journal, Cellar, URIserv)
- **Comprehensive Metadata**: Retrieves document type, work URI, and CELEX ID in one call

## Installation

No additional dependencies required beyond Python 3.6+ standard library:
- `xml.etree.ElementTree` (standard library)

## Usage

### Basic Example

```python
from classify_document import EUDocumentClassifier

# Initialize with your RDF file
classifier = EUDocumentClassifier('tree_non_inferred.rdf')

# Classify a document
uri = "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1"
doc_type = classifier.classify_document(uri)
print(f"Document type: {doc_type}")  # Output: Decision

# Extract CELEX ID
celex_id = classifier.get_celex_id(uri)
print(f"CELEX ID: {celex_id}")  # Output: 32004D0541

# Get all information at once
info = classifier.get_document_info(uri)
print(info)
# Output: {
#   'input_uri': '...',
#   'document_type': 'Decision',
#   'work_uri': 'http://publications.europa.eu/resource/oj/JOL_2004_240_R_0006_01',
#   'celex_id': '32004D0541'
# }
```

### Listing Documents

```python
# Count documents at different levels
counts = classifier.count_documents()
print(f"Total URIs: {counts['total']}")
print(f"Unique documents: {counts['works']}")
print(f"Language versions: {counts['expressions']}")
print(f"Format versions: {counts['manifestations']}")
print(f"Individual files: {counts['items']}")

# Count by document type
type_counts = classifier.count_documents(by_type=True)
for doc_type, count in type_counts.items():
    print(f"{doc_type}: {count}")

# List all documents (work level)
works = classifier.list_all_documents(level='work', include_type=True)
for uri, doc_type in works:
    print(f"{doc_type}: {uri}")

# Filter by document type
decisions = classifier.list_all_documents(
    level='work',
    include_type=False,
    document_type_filter='Decision'
)
print(f"Found {len(decisions)} Decision(s)")

# List expressions (language versions)
expressions = classifier.list_all_documents(level='expression', include_type=False)
print(f"Found {len(expressions)} language versions")

# Export to CSV
works = classifier.list_all_documents(level='work', include_type=True)
print("Document Type,CELEX ID,Work URI")
for uri, doc_type in works:
    celex_id = classifier.get_celex_id(uri)
    print(f"{doc_type},{celex_id},{uri}")
```

### CELEX ID Components

The CELEX identifier follows the format: **XYYYYTNNNN**

- **X**: Sector (1 digit)
  - 1 = Treaties
  - 2 = International agreements
  - 3 = Legislative acts
  - 4 = Complementary legislation
  - 5 = Preparatory acts
  - 6 = Case-law
  - 7 = National implementing measures
  - 8 = References to national case-law
  - 9 = Parliamentary questions

- **YYYY**: Year (4 digits)

- **T**: Document type (1 letter)
  - R = Regulation
  - L = Directive
  - D = Decision
  - C = Communication
  - H = Recommendation
  - X = Consolidated text

- **NNNN**: Sequential number (4+ digits)

Example: **32004D0541**
- Sector: 3 (Legislative acts)
- Year: 2004
- Type: D (Decision)
- Number: 0541

## Document Types

The classifier can identify the following document types:
- **Decision**: EU decisions binding on specific parties
- **Regulation**: Directly applicable EU law in all member states
- **Directive**: EU law requiring member state implementation
- **Recommendation**: Non-binding EU guidance
- **Caselaw**: Court decisions and judgments
- **Intagr**: International agreements
- **Proposal**: Legislative proposals

## How It Works

The classifier navigates the EU document hierarchy:

1. **Item** → Individual file (e.g., DOC_1, PDF, HTML)
2. **Manifestation** → Specific format of a document
3. **Expression** → Language-specific version
4. **Work** → The abstract document containing the document type

The tool:
1. Parses the RDF file and builds relationship indexes
2. Creates bidirectional owl:sameAs mappings
3. Traces from any URI level up to the work level
4. Extracts document type and CELEX identifier

## URI Formats Supported

The classifier works with multiple URI schemes:

- **Cellar URIs**: `http://publications.europa.eu/resource/cellar/[UUID].[VERSION].[LANG]/DOC_X`
- **CELEX URIs**: `http://publications.europa.eu/resource/celex/[CELEX_ID]`
- **Official Journal URIs**: `http://publications.europa.eu/resource/oj/JOL_[...]`
- **URIserv URIs**: `http://publications.europa.eu/resource/uriserv/OJ.[...]`

## Example Output

```
Classifying document:
URI: http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1

  → Found manifestation: ...0005.03
  → Following owl:sameAs to: .../celex/32004D0541.ENG.xhtml
  → Found expression: .../celex/32004D0541.ENG
  → Following owl:sameAs for expression to: .../JOL_2004_240_R_0006_01.ENG
  → Found work: .../JOL_2004_240_R_0006_01
  ✓ Document type: Decision

CELEX ID: 32004D0541
  Sector: 3 (Legislative acts)
  Year: 2004
  Type: D (Decision)
  Number: 0541
```

## Running the Examples

```bash
# Run the main script with built-in examples
python classify_document.py

# Run the usage examples
python usage_examples.py
```

## API Reference

### `EUDocumentClassifier(rdf_file_path: str)`

Initialize the classifier with an RDF file path.

### `classify_document(resource_uri: str) -> Optional[str]`

Classify a document and return its type (Decision, Regulation, etc.).

**Parameters:**
- `resource_uri`: The URI of the resource at any level (item/manifestation/expression/work)

**Returns:**
- Document type string or None if not found

### `get_celex_id(resource_uri: str) -> Optional[str]`

Extract the CELEX identifier from any resource URI.

**Parameters:**
- `resource_uri`: The URI of the resource

**Returns:**
- CELEX identifier string (e.g., "32004D0541") or None if not found

### `get_document_info(resource_uri: str) -> Dict`

Get comprehensive information about a document.

**Parameters:**
- `resource_uri`: The URI of the resource

**Returns:**
- Dictionary with keys: `input_uri`, `document_type`, `work_uri`, `celex_id`

### `list_all_documents(level: str = 'work', include_type: bool = True, document_type_filter: Optional[str] = None) -> list`

List all document URIs in the RDF file.

**Parameters:**
- `level`: Which level to return URIs for:
  - `'work'`: Work-level URIs (default, one per unique document)
  - `'expression'`: Expression-level URIs (one per language version)
  - `'manifestation'`: Manifestation-level URIs (one per format)
  - `'item'`: Item-level URIs (individual files)
  - `'all'`: All levels combined
- `include_type`: If True, return tuples of `(uri, document_type)`, otherwise just URIs
- `document_type_filter`: Filter by document type (e.g., 'Decision', 'Regulation')

**Returns:**
- List of URIs or list of `(uri, type)` tuples

**Example:**
```python
# Get all work URIs with their types
works = classifier.list_all_documents(level='work', include_type=True)
# Returns: [('http://...', 'Decision'), ...]

# Get only Decisions
decisions = classifier.list_all_documents(
    level='work', 
    include_type=False, 
    document_type_filter='Decision'
)
# Returns: ['http://...', 'http://...']

# Get all expressions (language versions)
expressions = classifier.list_all_documents(level='expression', include_type=False)
```

### `count_documents(by_type: bool = False) -> Dict`

Count documents in the RDF file.

**Parameters:**
- `by_type`: If True, return counts grouped by document type

**Returns:**
- Dictionary with counts

**Example:**
```python
# Count by level
counts = classifier.count_documents()
# Returns: {'total': 181, 'works': 1, 'expressions': 20, 
#           'manifestations': 80, 'items': 80}

# Count by document type
type_counts = classifier.count_documents(by_type=True)
# Returns: {'Decision': 5, 'Regulation': 3, 'Directive': 2}
```

### `get_document_languages(work_uri: str) -> list`

Get all available languages for a document.

**Parameters:**
- `work_uri`: The work-level URI

**Returns:**
- List of language codes (e.g., `['ENG', 'FRA', 'DEU']`)

**Example:**
```python
languages = classifier.get_document_languages(work_uri)
# Returns: ['ENG', 'FRA', 'DEU', 'ITA', ...]
```

## License

This tool is provided as-is for working with EU Publications Office RDF metadata.
