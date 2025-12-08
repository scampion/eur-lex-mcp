# Quick Reference - EU Document Classifier

## Installation & Setup
```python
from classify_document import EUDocumentClassifier
classifier = EUDocumentClassifier('your_rdf_file.rdf')
```

## Common Tasks

### 1. Get Document Type
```python
doc_type = classifier.classify_document(uri)
# Returns: 'Decision', 'Regulation', 'Directive', etc.
```

### 2. Get CELEX ID
```python
celex_id = classifier.get_celex_id(uri)
# Returns: '32004D0541'
```

### 3. Get All Info
```python
info = classifier.get_document_info(uri)
# Returns: {'input_uri': '...', 'document_type': 'Decision', 
#           'work_uri': '...', 'celex_id': '32004D0541'}
```

### 4. List All Documents
```python
# List works (unique documents)
works = classifier.list_all_documents(level='work', include_type=True)
# Returns: [('http://...', 'Decision'), ...]

# List expressions (language versions)
expressions = classifier.list_all_documents(level='expression')
# Returns: [('http://.../JOL_..._ENG', 'Decision'), ...]

# Filter by type
decisions = classifier.list_all_documents(
    level='work', 
    include_type=False,
    document_type_filter='Decision'
)
```

### 5. Count Documents
```python
# Count by level
counts = classifier.count_documents()
# Returns: {'total': 181, 'works': 1, 'expressions': 20, ...}

# Count by type
type_counts = classifier.count_documents(by_type=True)
# Returns: {'Decision': 5, 'Regulation': 3, ...}
```

## CELEX Format: XYYYYTNNNN

| Component | Example | Meaning |
|-----------|---------|---------|
| X | 3 | Sector: Legislative acts |
| YYYY | 2004 | Year |
| T | D | Type: Decision |
| NNNN | 0541 | Sequential number |

## Document Types
- **Decision** - Binding on specific parties
- **Regulation** - Directly applicable in all member states
- **Directive** - Requires member state implementation
- **Recommendation** - Non-binding guidance
- **Caselaw** - Court decisions
- **Intagr** - International agreements
- **Proposal** - Legislative proposals

## Supported URI Formats
✓ Cellar: `http://publications.europa.eu/resource/cellar/[UUID].[VER].[LANG]/DOC_X`
✓ CELEX: `http://publications.europa.eu/resource/celex/[CELEX_ID]`
✓ Official Journal: `http://publications.europa.eu/resource/oj/JOL_[...]`
✓ URIserv: `http://publications.europa.eu/resource/uriserv/OJ.[...]`

## Example Usage
```python
# Your document
uri = "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1"

# One-liner to get everything
info = classifier.get_document_info(uri)
print(f"{info['celex_id']}: {info['document_type']}")
# Output: 32004D0541: Decision
```

## Batch Processing
```python
documents = [uri1, uri2, uri3, ...]
results = [classifier.get_document_info(uri) for uri in documents]
```

## Parse CELEX Components
```python
celex = '32004D0541'
sector = celex[0]      # '3' = Legislative acts
year = celex[1:5]      # '2004'
doc_type = celex[5]    # 'D' = Decision
number = celex[6:]     # '0541'
```
