"""
Document Listing Examples
=========================

This script demonstrates how to list and filter documents from the RDF file.
"""

from classify_document import EUDocumentClassifier

# Initialize classifier
print("Initializing classifier...")
classifier = EUDocumentClassifier('/mnt/user-data/uploads/tree_non_inferred.rdf')
print("✓ Ready\n")

# Example 1: Count documents at different levels
# ===============================================
print("="*60)
print("Example 1: Document Counts by Level")
print("="*60)

counts = classifier.count_documents()
print(f"Total URIs in RDF file: {counts['total']}")
print(f"\nBreakdown:")
print(f"  • Works (unique documents): {counts['works']}")
print(f"  • Expressions (language versions): {counts['expressions']}")
print(f"  • Manifestations (format versions): {counts['manifestations']}")
print(f"  • Items (individual files): {counts['items']}")
print()

# Example 2: Count documents by type
# ===================================
print("="*60)
print("Example 2: Document Counts by Type")
print("="*60)

type_counts = classifier.count_documents(by_type=True)
print("Document types in this RDF file:")
for doc_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  • {doc_type}: {count} document(s)")
print()

# Example 3: List all work URIs
# ==============================
print("="*60)
print("Example 3: List Work URIs (Unique Documents)")
print("="*60)

works = classifier.list_all_documents(level='work', include_type=True)
print(f"Found {len(works)} work(s)\n")

for i, (uri, doc_type) in enumerate(works, 1):
    # Extract a short identifier from the URI
    short_id = uri.split('/')[-1]
    print(f"{i}. [{doc_type}] {short_id}")
    print(f"   Full URI: {uri}")
print()

# Example 4: List expressions (language versions)
# ================================================
print("="*60)
print("Example 4: List Expression URIs (Language Versions)")
print("="*60)

expressions = classifier.list_all_documents(level='expression', include_type=False)
print(f"Found {len(expressions)} expression(s) (different language versions)\n")

# Show first 5
for i, uri in enumerate(expressions[:5], 1):
    short_id = uri.split('/')[-1]
    print(f"{i}. {short_id}")

if len(expressions) > 5:
    print(f"... and {len(expressions) - 5} more")
print()

# Example 5: List manifestations (format versions)
# =================================================
print("="*60)
print("Example 5: List Manifestation URIs (Format Versions)")
print("="*60)

manifestations = classifier.list_all_documents(level='manifestation', include_type=False)
print(f"Found {len(manifestations)} manifestation(s) (different formats)\n")

# Show first 5
for i, uri in enumerate(manifestations[:5], 1):
    short_id = uri.split('/')[-1]
    print(f"{i}. {short_id}")

if len(manifestations) > 5:
    print(f"... and {len(manifestations) - 5} more")
print()

# Example 6: List items (individual files)
# =========================================
print("="*60)
print("Example 6: List Item URIs (Individual Files)")
print("="*60)

items = classifier.list_all_documents(level='item', include_type=False)
print(f"Found {len(items)} item(s) (individual files)\n")

# Show first 5
for i, uri in enumerate(items[:5], 1):
    # Items often have /DOC_1, /DOC_2, etc.
    parts = uri.split('/')
    short_id = '/'.join(parts[-2:]) if len(parts) >= 2 else uri
    print(f"{i}. {short_id}")

if len(items) > 5:
    print(f"... and {len(items) - 5} more")
print()

# Example 7: Filter by document type
# ===================================
print("="*60)
print("Example 7: Filter Documents by Type")
print("="*60)

# Get only Decisions
decisions = classifier.list_all_documents(
    level='work', 
    include_type=True, 
    document_type_filter='Decision'
)

print(f"Documents of type 'Decision': {len(decisions)}")
for uri, doc_type in decisions:
    celex_id = classifier.get_celex_id(uri)
    print(f"  • CELEX: {celex_id}")
    print(f"    URI: {uri}")
print()

# Example 8: Export to CSV format
# ================================
print("="*60)
print("Example 8: Export Document List (CSV Format)")
print("="*60)

works = classifier.list_all_documents(level='work', include_type=True)
print("Document Type,CELEX ID,Work URI")
print("-" * 60)

for uri, doc_type in works:
    celex_id = classifier.get_celex_id(uri)
    print(f"{doc_type},{celex_id},{uri}")
print()

# Example 9: Get all URIs for a specific document
# ================================================
print("="*60)
print("Example 9: Get All URIs for a Specific Document")
print("="*60)

work_uri = works[0][0] if works else None
if work_uri:
    print(f"Document: {work_uri}\n")
    
    # Find all expressions for this work
    print("Language versions (expressions):")
    expr_count = 0
    for expr_uri, w_uri in classifier.expression_to_work.items():
        if w_uri == work_uri:
            expr_count += 1
            lang_code = expr_uri.split('.')[-1] if '.' in expr_uri else expr_uri.split('/')[-1]
            print(f"  {expr_count}. {lang_code}: {expr_uri}")
    
    print(f"\nTotal expressions: {expr_count}")
print()

# Example 10: Summary statistics
# ===============================
print("="*60)
print("Example 10: Summary Statistics")
print("="*60)

counts = classifier.count_documents()
type_counts = classifier.count_documents(by_type=True)

print(f"RDF File Statistics:")
print(f"  • Total URIs: {counts['total']}")
print(f"  • Unique documents (works): {counts['works']}")
print(f"  • Languages per document (avg): {counts['expressions'] / max(counts['works'], 1):.1f}")
print(f"  • Formats per language (avg): {counts['manifestations'] / max(counts['expressions'], 1):.1f}")
print(f"  • Files per format (avg): {counts['items'] / max(counts['manifestations'], 1):.1f}")
print()

print("Document Types:")
for doc_type, count in sorted(type_counts.items()):
    percentage = (count / counts['works'] * 100) if counts['works'] > 0 else 0
    print(f"  • {doc_type}: {count} ({percentage:.1f}%)")

print("\n" + "="*60)
print("Complete!")
print("="*60)
