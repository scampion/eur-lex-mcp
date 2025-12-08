# EU Document Classifier - Deliverables Summary

## Overview
Complete Python toolkit for classifying EU legal documents and extracting CELEX identifiers from RDF metadata files.

## Core Functionality

### Main Features
1. **Document Classification** - Identifies document types (Decision, Regulation, Directive, etc.)
2. **CELEX Extraction** - Extracts and parses CELEX identifiers from any URI format
3. **Document Listing** - Lists all documents in RDF file with filtering options
4. **Document Counting** - Provides statistics on documents at all hierarchy levels
5. **CSV Export** - Exports document lists to CSV for further analysis

## Files Delivered

### 📄 Core Library
- **`classify_document.py`** (31 KB)
  - Main classifier class with all functionality
  - Handles owl:sameAs relationships across URI schemes
  - Navigates document hierarchy (Item → Manifestation → Expression → Work)
  - Includes comprehensive examples in main() function

### 📚 Documentation
- **`README.md`** (8.5 KB)
  - Complete documentation with installation and usage
  - API reference for all functions
  - Examples and code snippets
  - CELEX format explanation

- **`QUICK_REFERENCE.md`** (2.9 KB)
  - Cheat sheet for common tasks
  - Quick lookup for function signatures
  - CELEX parsing table
  - Batch processing examples

### 💡 Example Scripts
- **`usage_examples.py`** (3.8 KB)
  - Basic usage examples
  - Document classification
  - CELEX extraction
  - Batch processing
  - CELEX component parsing

- **`list_documents_examples.py`** (6.0 KB)
  - Document listing at all levels
  - Filtering by document type
  - Counting statistics
  - Language version enumeration
  - Export to CSV format

- **`export_to_csv.py`** (6.7 KB)
  - Export all documents to CSV
  - Multiple export formats:
    * All works with details
    * All expressions (language versions)
    * Summary statistics
    * Detailed inventory

### 📊 Generated CSV Files
- **`all_works.csv`** (137 B)
  - All unique documents with CELEX IDs
  - Columns: Document Type, CELEX ID, Work URI, Language Count

- **`all_expressions.csv`** (3.0 KB)
  - All language versions
  - Columns: Document Type, CELEX ID, Language, Expression URI, Work URI

- **`summary_statistics.csv`** (286 B)
  - Statistical summary of RDF file
  - URIs by level, documents by type, averages

- **`detailed_inventory.csv`** (17 KB)
  - Complete inventory of all URIs
  - Columns: Level, Document Type, CELEX ID, Language, Format, URI, Related Work URI

## API Functions

### Document Information
```python
classify_document(uri)              # Get document type
get_celex_id(uri)                   # Extract CELEX ID
get_document_info(uri)              # Get all info at once
```

### Document Listing
```python
list_all_documents(level, include_type, document_type_filter)
count_documents(by_type)
get_document_languages(work_uri)
```

## Test Results

### Sample Document Analysis
**Input URI:** 
```
http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1
```

**Results:**
- **Document Type:** Decision
- **CELEX ID:** 32004D0541
- **Work URI:** http://publications.europa.eu/resource/oj/JOL_2004_240_R_0006_01
- **Available Languages:** 20 (ENG, FRA, DEU, ITA, SPA, POL, etc.)

### CELEX Breakdown
**32004D0541**
- Sector: 3 (Legislative acts)
- Year: 2004
- Type: D (Decision)
- Number: 0541

### Document Statistics
From the test RDF file:
- Total URIs: 181
- Unique documents (works): 1
- Language versions (expressions): 20
- Format versions (manifestations): 80
- Individual files (items): 80

## Supported URI Formats

✅ **Cellar URIs**
```
http://publications.europa.eu/resource/cellar/[UUID].[VERSION].[LANG]/DOC_X
```

✅ **CELEX URIs**
```
http://publications.europa.eu/resource/celex/[CELEX_ID]
```

✅ **Official Journal URIs**
```
http://publications.europa.eu/resource/oj/JOL_[...]
```

✅ **URIserv URIs**
```
http://publications.europa.eu/resource/uriserv/OJ.[...]
```

## Document Types Supported
- Decision
- Regulation
- Directive
- Recommendation
- Caselaw
- Intagr (International Agreements)
- Proposal

## Usage Examples

### Quick Start
```python
from classify_document import EUDocumentClassifier

classifier = EUDocumentClassifier('your_file.rdf')
doc_type = classifier.classify_document(uri)
celex_id = classifier.get_celex_id(uri)
```

### List All Documents
```python
# Get all unique documents
works = classifier.list_all_documents(level='work', include_type=True)

# Filter by type
decisions = classifier.list_all_documents(
    level='work',
    document_type_filter='Decision'
)

# Count statistics
counts = classifier.count_documents()
type_counts = classifier.count_documents(by_type=True)
```

### Export to CSV
```python
# Run the export script
python export_to_csv.py

# Or programmatically
works = classifier.list_all_documents(level='work', include_type=True)
with open('output.csv', 'w') as f:
    for uri, doc_type in works:
        celex = classifier.get_celex_id(uri)
        f.write(f"{doc_type},{celex},{uri}\n")
```

## Technical Details

### Document Hierarchy Navigation
The classifier navigates through the FRBR-based hierarchy:
1. **Item** - Individual files (PDF, HTML, XML, etc.)
2. **Manifestation** - Format-specific versions
3. **Expression** - Language-specific versions
4. **Work** - Abstract document (contains document type)

### owl:sameAs Handling
- Builds bidirectional owl:sameAs mappings
- Performs 2-hop search through equivalence network
- Prioritizes Official Journal URIs over CELEX URIs
- Handles missing direct relationships gracefully

### Performance
- Parses and indexes RDF file at initialization
- Fast lookups using dictionary-based indexes
- Handles files with thousands of documents efficiently

## Requirements
- Python 3.6+
- No external dependencies (uses standard library only)
- Works with EU Publications Office RDF metadata format

## Files Location
All files are available in: `/mnt/user-data/outputs/`

## Next Steps
1. Load your own RDF file
2. Run the example scripts to explore functionality
3. Use the API to integrate into your workflow
4. Export data to CSV for analysis in other tools

---

**Created:** December 8, 2024
**Version:** 1.0
**License:** Provided as-is for working with EU Publications Office RDF metadata
