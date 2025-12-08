"""
Usage Examples for EU Document Classifier
==========================================

This file demonstrates how to use the EUDocumentClassifier to:
1. Classify documents by type (Decision, Regulation, Directive, etc.)
2. Extract CELEX identifiers from any URI
3. Get comprehensive document information
"""

from classify_document import EUDocumentClassifier

# Initialize the classifier with your RDF file
classifier = EUDocumentClassifier('/mnt/user-data/uploads/tree_non_inferred.rdf')

# Example 1: Classify a document by URI
# ======================================
doc_uri = "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1"

print("Example 1: Classify Document")
print("=" * 60)
doc_type = classifier.classify_document(doc_uri)
print(f"Document type: {doc_type}")
print()

# Example 2: Extract CELEX ID from any URI
# =========================================
print("Example 2: Extract CELEX ID")
print("=" * 60)

# Works with different URI types:
uris = [
    "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1",
    "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03",
    "http://publications.europa.eu/resource/celex/32004D0541.ENG",
    "http://publications.europa.eu/resource/celex/32004D0541"
]

for uri in uris:
    celex_id = classifier.get_celex_id(uri)
    print(f"URI: ...{uri[-50:]}")
    print(f"CELEX ID: {celex_id}")
    print()

# Example 3: Get comprehensive document information
# ==================================================
print("Example 3: Get Detailed Document Info")
print("=" * 60)
info = classifier.get_document_info(doc_uri)

print(f"Input URI: {info['input_uri']}")
print(f"Document Type: {info['document_type']}")
print(f"Work URI: {info['work_uri']}")
print(f"CELEX ID: {info['celex_id']}")
print()

# Example 4: Parse CELEX ID components
# =====================================
print("Example 4: Parse CELEX Components")
print("=" * 60)
celex_id = classifier.get_celex_id(doc_uri)

if celex_id and len(celex_id) >= 8:
    sector = celex_id[0]
    year = celex_id[1:5]
    doc_type_letter = celex_id[5]
    sequential = celex_id[6:]
    
    sector_names = {
        '1': 'Treaties',
        '2': 'International agreements',
        '3': 'Legislative acts',
        '4': 'Complementary legislation',
        '5': 'Preparatory acts',
        '6': 'Case-law',
        '7': 'National implementing measures',
        '8': 'References to national case-law',
        '9': 'Parliamentary questions'
    }
    
    type_names = {
        'R': 'Regulation',
        'L': 'Directive',
        'D': 'Decision',
        'C': 'Communication',
        'H': 'Recommendation',
        'X': 'Consolidated text'
    }
    
    print(f"CELEX ID: {celex_id}")
    print(f"  Sector: {sector} - {sector_names.get(sector, 'Unknown')}")
    print(f"  Year: {year}")
    print(f"  Type: {doc_type_letter} - {type_names.get(doc_type_letter, 'Unknown')}")
    print(f"  Sequential Number: {sequential}")
    print()

# Example 5: Batch processing multiple documents
# ===============================================
print("Example 5: Batch Processing")
print("=" * 60)

# List of document URIs to process
document_uris = [
    "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1",
    # Add more URIs here
]

results = []
for uri in document_uris:
    info = classifier.get_document_info(uri)
    results.append({
        'uri': uri,
        'type': info['document_type'],
        'celex': info['celex_id']
    })

# Display results
for result in results:
    print(f"URI: ...{result['uri'][-50:]}")
    print(f"  Type: {result['type']}")
    print(f"  CELEX: {result['celex']}")
    print()

print("Done!")
