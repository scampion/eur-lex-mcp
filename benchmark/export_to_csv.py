"""
Export Document Lists to CSV
=============================

This script demonstrates how to export document lists to CSV files
for further analysis or import into other systems.
"""

import csv
from classify_document import EUDocumentClassifier

# Initialize classifier
print("Initializing classifier...")
classifier = EUDocumentClassifier('/mnt/user-data/uploads/tree_non_inferred.rdf')
print("✓ Ready\n")

# Export 1: All works with full details
# ======================================
print("Exporting all works to CSV...")
output_file = '/mnt/user-data/outputs/all_works.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write header
    writer.writerow(['Document Type', 'CELEX ID', 'Work URI', 'Language Count'])
    
    # Get all works
    works = classifier.list_all_documents(level='work', include_type=True)
    
    for work_uri, doc_type in works:
        celex_id = classifier.get_celex_id(work_uri)
        
        # Count languages for this work
        lang_count = sum(1 for expr_uri, w_uri in classifier.expression_to_work.items() 
                        if w_uri == work_uri)
        
        writer.writerow([doc_type, celex_id, work_uri, lang_count])

print(f"✓ Exported {len(works)} work(s) to {output_file}\n")

# Export 2: All expressions (language versions)
# ==============================================
print("Exporting all expressions to CSV...")
output_file = '/mnt/user-data/outputs/all_expressions.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write header
    writer.writerow(['Document Type', 'CELEX ID', 'Language', 'Expression URI', 'Work URI'])
    
    # Get all expressions
    expressions = classifier.list_all_documents(level='expression', include_type=True)
    
    for expr_uri, doc_type in expressions:
        # Extract language code from URI
        lang_code = 'Unknown'
        if '.' in expr_uri:
            parts = expr_uri.split('.')
            if len(parts) > 1 and len(parts[-1]) <= 3:
                lang_code = parts[-1].upper()
        
        # Get work URI
        work_uri = classifier.expression_to_work.get(expr_uri, 'Unknown')
        
        # Get CELEX from work
        celex_id = classifier.get_celex_id(expr_uri)
        
        writer.writerow([doc_type, celex_id, lang_code, expr_uri, work_uri])

print(f"✓ Exported {len(expressions)} expression(s) to {output_file}\n")

# Export 3: Document counts summary
# ==================================
print("Exporting summary statistics to CSV...")
output_file = '/mnt/user-data/outputs/summary_statistics.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write counts by level
    writer.writerow(['Metric', 'Count'])
    writer.writerow([''])
    writer.writerow(['URIs by Level', ''])
    
    counts = classifier.count_documents()
    writer.writerow(['Total URIs', counts['total']])
    writer.writerow(['Works (unique documents)', counts['works']])
    writer.writerow(['Expressions (language versions)', counts['expressions']])
    writer.writerow(['Manifestations (format versions)', counts['manifestations']])
    writer.writerow(['Items (individual files)', counts['items']])
    
    # Write counts by type
    writer.writerow([''])
    writer.writerow(['Documents by Type', ''])
    
    type_counts = classifier.count_documents(by_type=True)
    for doc_type, count in sorted(type_counts.items()):
        writer.writerow([doc_type, count])
    
    # Write averages
    writer.writerow([''])
    writer.writerow(['Averages', ''])
    if counts['works'] > 0:
        writer.writerow(['Languages per document', f"{counts['expressions'] / counts['works']:.1f}"])
    if counts['expressions'] > 0:
        writer.writerow(['Formats per language', f"{counts['manifestations'] / counts['expressions']:.1f}"])

print(f"✓ Exported summary statistics to {output_file}\n")

# Export 4: Detailed document inventory
# ======================================
print("Exporting detailed inventory to CSV...")
output_file = '/mnt/user-data/outputs/detailed_inventory.csv'

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # Write header
    writer.writerow([
        'Level', 'Document Type', 'CELEX ID', 'Language', 
        'Format', 'URI', 'Related Work URI'
    ])
    
    # Export works
    works = classifier.list_all_documents(level='work', include_type=True)
    for work_uri, doc_type in works:
        celex_id = classifier.get_celex_id(work_uri)
        writer.writerow(['Work', doc_type, celex_id, 'N/A', 'N/A', work_uri, work_uri])
    
    # Export expressions
    expressions = classifier.list_all_documents(level='expression', include_type=True)
    for expr_uri, doc_type in expressions:
        celex_id = classifier.get_celex_id(expr_uri)
        work_uri = classifier.expression_to_work.get(expr_uri, 'Unknown')
        
        # Extract language
        lang_code = 'Unknown'
        if '.' in expr_uri:
            parts = expr_uri.split('.')
            if len(parts) > 1 and len(parts[-1]) <= 3:
                lang_code = parts[-1].upper()
        
        writer.writerow(['Expression', doc_type, celex_id, lang_code, 'N/A', expr_uri, work_uri])
    
    # Export manifestations
    manifestations = classifier.list_all_documents(level='manifestation', include_type=True)
    for manif_uri, doc_type in manifestations:
        celex_id = classifier.get_celex_id(manif_uri)
        
        # Get expression and work
        expr_uri = classifier.manifestation_to_expression.get(manif_uri, 'Unknown')
        work_uri = classifier.expression_to_work.get(expr_uri, 'Unknown') if expr_uri != 'Unknown' else 'Unknown'
        
        # Extract language and format
        lang_code = 'Unknown'
        format_type = 'Unknown'
        if '.' in manif_uri:
            parts = manif_uri.split('.')
            if len(parts) >= 2:
                if len(parts[-2]) <= 3:
                    lang_code = parts[-2].upper()
                format_type = parts[-1].upper()
        
        writer.writerow(['Manifestation', doc_type, celex_id, lang_code, format_type, manif_uri, work_uri])

total_entries = len(works) + len(expressions) + len(manifestations)
print(f"✓ Exported {total_entries} entries to {output_file}\n")

# Summary
print("="*60)
print("Export Complete!")
print("="*60)
print("\nGenerated files:")
print("  1. all_works.csv - All unique documents")
print("  2. all_expressions.csv - All language versions")
print("  3. summary_statistics.csv - Statistical summary")
print("  4. detailed_inventory.csv - Complete inventory")
print("\nAll files are in: /mnt/user-data/outputs/")
