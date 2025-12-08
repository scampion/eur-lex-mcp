#!/usr/bin/env python3
"""
Script to find and classify EU legal documents from RDF metadata.
Finds the document type (Decision, Regulation, Directive, etc.) for a given resource URI.
"""

import xml.etree.ElementTree as ET
from typing import Optional, Dict, Set


class EUDocumentClassifier:
    """Classifier for EU legal documents based on RDF metadata."""
    
    # Namespace definitions
    NAMESPACES = {
        'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
        'cdm': 'http://publications.europa.eu/ontology/cdm#',
        'owl': 'http://www.w3.org/2002/07/owl#'
    }
    
    # Document types we're looking for
    DOCUMENT_TYPES = {
        'decision': 'Decision',
        'regulation': 'Regulation',
        'directive': 'Directive',
        'recommendation': 'Recommendation',
        'caselaw': 'Caselaw',
        'intagr': 'Intagr',
        'proposal': 'Proposal'
    }
    
    def __init__(self, rdf_file_path: str):
        """Initialize the classifier with an RDF file."""
        self.rdf_file_path = rdf_file_path
        self.tree = ET.parse(rdf_file_path)
        self.root = self.tree.getroot()
        
        # Build index for faster lookups
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes to map items to manifestations to expressions to works."""
        self.item_to_manifestation = {}
        self.manifestation_to_expression = {}
        self.expression_to_work = {}
        self.work_types = {}
        self.same_as_mapping = {}  # Track owl:sameAs relationships (bidirectional)
        
        # Parse all RDF descriptions
        for desc in self.root.findall('rdf:Description', self.NAMESPACES):
            about = desc.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            
            if not about:
                continue
            
            # Track owl:sameAs relationships (bidirectional)
            same_as_elems = desc.findall('owl:sameAs', self.NAMESPACES)
            if same_as_elems:
                if about not in self.same_as_mapping:
                    self.same_as_mapping[about] = []
                for same_as_elem in same_as_elems:
                    same_as_uri = same_as_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                    if same_as_uri:
                        # Add forward mapping
                        self.same_as_mapping[about].append(same_as_uri)
                        # Add reverse mapping
                        if same_as_uri not in self.same_as_mapping:
                            self.same_as_mapping[same_as_uri] = []
                        if about not in self.same_as_mapping[same_as_uri]:
                            self.same_as_mapping[same_as_uri].append(about)
            
            # Check if this is an item (has item_belongs_to_manifestation)
            manifestation_elem = desc.find('cdm:item_belongs_to_manifestation', self.NAMESPACES)
            if manifestation_elem is not None:
                manifestation_uri = manifestation_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if manifestation_uri:
                    self.item_to_manifestation[about] = manifestation_uri
            
            # Check if this is a manifestation (has manifestation_manifests_expression)
            expression_elem = desc.find('cdm:manifestation_manifests_expression', self.NAMESPACES)
            if expression_elem is not None:
                expression_uri = expression_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if expression_uri:
                    self.manifestation_to_expression[about] = expression_uri
            
            # Check if this is an expression (has expression_belongs_to_work)
            work_elem = desc.find('cdm:expression_belongs_to_work', self.NAMESPACES)
            if work_elem is not None:
                work_uri = work_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                if work_uri:
                    self.expression_to_work[about] = work_uri
            
            # Check if this is a work with document types
            type_elems = desc.findall('rdf:type', self.NAMESPACES)
            for type_elem in type_elems:
                type_uri = type_elem.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
                
                # Check if it's one of our document types
                for doc_type_key, doc_type_name in self.DOCUMENT_TYPES.items():
                    if type_uri.endswith(f'#{doc_type_key}'):
                        if about not in self.work_types:
                            self.work_types[about] = set()
                        self.work_types[about].add(doc_type_name)
    
    def classify_document(self, resource_uri: str) -> Optional[str]:
        """
        Classify a document by finding its work-level type.
        
        Args:
            resource_uri: The URI of the resource (item/manifestation/expression/work)
        
        Returns:
            The document type (e.g., 'Decision', 'Regulation') or None if not found
        """
        current_uri = resource_uri
        visited = set()  # Prevent infinite loops
        
        # Trace from item -> manifestation -> expression -> work
        
        # Step 1: If it's an item, get the manifestation
        if current_uri in self.item_to_manifestation:
            current_uri = self.item_to_manifestation[current_uri]
            print(f"  → Found manifestation: {current_uri}")
        
        # Step 2: If it's a manifestation, get the expression
        # First try direct link, then try ALL owl:sameAs equivalents
        if current_uri not in self.manifestation_to_expression:
            # Try to find an equivalent manifestation via owl:sameAs
            if current_uri in self.same_as_mapping:
                # Try preferring JOL (Official Journal) URIs over CELEX
                jol_uris = [u for u in self.same_as_mapping[current_uri] if 'JOL' in u or '/oj/' in u]
                celex_uris = [u for u in self.same_as_mapping[current_uri] if 'celex' in u]
                other_uris = [u for u in self.same_as_mapping[current_uri] if u not in jol_uris and u not in celex_uris]
                
                # Try in order: JOL, CELEX, then others
                for same_as_uri in jol_uris + celex_uris + other_uris:
                    if same_as_uri in self.manifestation_to_expression:
                        print(f"  → Following owl:sameAs to: {same_as_uri}")
                        current_uri = same_as_uri
                        break
        
        if current_uri in self.manifestation_to_expression:
            current_uri = self.manifestation_to_expression[current_uri]
            print(f"  → Found expression: {current_uri}")
        
        # Step 3: If it's an expression, get the work
        # First try direct link, then search through owl:sameAs network (up to 2 hops)
        work_uri = None
        if current_uri in self.expression_to_work:
            work_uri = self.expression_to_work[current_uri]
        else:
            # Try to find an equivalent expression via owl:sameAs (with 2-hop search)
            candidates_to_check = set()
            
            # Add direct sameAs URIs
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    candidates_to_check.add(same_as_uri)
                    # Also add second-hop sameAs URIs
                    if same_as_uri in self.same_as_mapping:
                        for second_hop_uri in self.same_as_mapping[same_as_uri]:
                            candidates_to_check.add(second_hop_uri)
            
            # Now check all candidates, prioritizing JOL over CELEX
            jol_uris = [u for u in candidates_to_check if ('JOL' in u or '/oj/' in u) and u in self.expression_to_work]
            celex_uris = [u for u in candidates_to_check if 'celex' in u and u in self.expression_to_work]
            other_uris = [u for u in candidates_to_check if u not in jol_uris and u not in celex_uris and u in self.expression_to_work]
            
            for same_as_uri in jol_uris + celex_uris + other_uris:
                print(f"  → Following owl:sameAs for expression to: {same_as_uri}")
                work_uri = self.expression_to_work[same_as_uri]
                break
        
        if work_uri:
            current_uri = work_uri
            print(f"  → Found work: {current_uri}")
        
        # Step 4: Check if we have the type for this work
        if current_uri in self.work_types:
            types = self.work_types[current_uri]
            if types:
                # Return the first document type found (usually there's only one)
                doc_type = next(iter(types))
                print(f"  ✓ Document type: {doc_type}")
                return doc_type
        
        print(f"  ✗ No document type found")
        return None
    
    def get_document_info(self, resource_uri: str) -> Dict:
        """
        Get detailed information about a document.
        
        Args:
            resource_uri: The URI of the resource
        
        Returns:
            Dictionary with document information
        """
        info = {
            'input_uri': resource_uri,
            'document_type': None,
            'work_uri': None,
            'celex_id': None
        }
        
        current_uri = resource_uri
        
        # Trace to work
        if current_uri in self.item_to_manifestation:
            current_uri = self.item_to_manifestation[current_uri]
        
        # Try to find expression via owl:sameAs if direct link doesn't exist
        if current_uri not in self.manifestation_to_expression:
            if current_uri in self.same_as_mapping:
                jol_uris = [u for u in self.same_as_mapping[current_uri] if 'JOL' in u or '/oj/' in u]
                celex_uris = [u for u in self.same_as_mapping[current_uri] if 'celex' in u]
                other_uris = [u for u in self.same_as_mapping[current_uri] if u not in jol_uris and u not in celex_uris]
                
                for same_as_uri in jol_uris + celex_uris + other_uris:
                    if same_as_uri in self.manifestation_to_expression:
                        current_uri = same_as_uri
                        break
        
        if current_uri in self.manifestation_to_expression:
            current_uri = self.manifestation_to_expression[current_uri]
        
        # Try to find work via owl:sameAs if direct link doesn't exist (with 2-hop search)
        work_uri = None
        if current_uri in self.expression_to_work:
            work_uri = self.expression_to_work[current_uri]
        else:
            candidates_to_check = set()
            
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    candidates_to_check.add(same_as_uri)
                    if same_as_uri in self.same_as_mapping:
                        for second_hop_uri in self.same_as_mapping[same_as_uri]:
                            candidates_to_check.add(second_hop_uri)
            
            jol_uris = [u for u in candidates_to_check if ('JOL' in u or '/oj/' in u) and u in self.expression_to_work]
            celex_uris = [u for u in candidates_to_check if 'celex' in u and u in self.expression_to_work]
            other_uris = [u for u in candidates_to_check if u not in jol_uris and u not in celex_uris and u in self.expression_to_work]
            
            for same_as_uri in jol_uris + celex_uris + other_uris:
                work_uri = self.expression_to_work[same_as_uri]
                break
        
        if work_uri:
            current_uri = work_uri
        
        info['work_uri'] = current_uri
        
        # Get document type
        if current_uri in self.work_types:
            types = self.work_types[current_uri]
            if types:
                info['document_type'] = next(iter(types))
        
        # Get CELEX ID using the dedicated function
        info['celex_id'] = self.get_celex_id(resource_uri)
        
        return info
    
    def _extract_celex_from_work(self, work_uri: str) -> Optional[str]:
        """Extract CELEX number from work-level metadata."""
        # Find the work description
        for desc in self.root.findall('rdf:Description', self.NAMESPACES):
            about = desc.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if about == work_uri:
                # Look for CELEX-related properties or sameAs links
                for child in desc:
                    if 'celex' in child.tag.lower():
                        return child.text
                    
                    # Check owl:sameAs links for CELEX references
                    if child.tag.endswith('sameAs'):
                        resource = child.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource', '')
                        if 'celex' in resource:
                            # Extract CELEX number from URI
                            parts = resource.split('/')
                            for part in parts:
                                if part.startswith('3') and len(part) > 5:  # CELEX format
                                    return part.split('.')[0]
        
        return None
    
    def list_all_documents(self, level: str = 'work', include_type: bool = True, 
                          document_type_filter: Optional[str] = None) -> list:
        """
        List all document URIs in the RDF file.
        
        Args:
            level: Which level to return URIs for:
                   - 'work': Work-level URIs (default, one per document)
                   - 'expression': Expression-level URIs (one per language)
                   - 'manifestation': Manifestation-level URIs (one per format)
                   - 'item': Item-level URIs (individual files)
                   - 'all': All levels combined
            include_type: If True, return tuples of (uri, document_type), 
                         otherwise just URIs
            document_type_filter: Filter by document type (e.g., 'Decision', 'Regulation')
        
        Returns:
            List of URIs or list of (uri, type) tuples
        """
        results = []
        
        if level == 'work' or level == 'all':
            # Get all work URIs (those with document types)
            for work_uri, types in self.work_types.items():
                doc_type = next(iter(types)) if types else None
                
                # Apply filter if specified
                if document_type_filter and doc_type != document_type_filter:
                    continue
                
                if include_type:
                    results.append((work_uri, doc_type))
                else:
                    results.append(work_uri)
        
        if level == 'expression' or level == 'all':
            # Get all expression URIs
            for expr_uri in self.expression_to_work.keys():
                # Get the work to determine document type
                work_uri = self.expression_to_work[expr_uri]
                doc_type = None
                if work_uri in self.work_types:
                    types = self.work_types[work_uri]
                    doc_type = next(iter(types)) if types else None
                
                # Apply filter if specified
                if document_type_filter and doc_type != document_type_filter:
                    continue
                
                if level != 'all':  # Avoid duplicates when level='all'
                    if include_type:
                        results.append((expr_uri, doc_type))
                    else:
                        results.append(expr_uri)
        
        if level == 'manifestation' or level == 'all':
            # Get all manifestation URIs
            for manif_uri in self.manifestation_to_expression.keys():
                # Trace to work to get document type
                expr_uri = self.manifestation_to_expression[manif_uri]
                doc_type = None
                if expr_uri in self.expression_to_work:
                    work_uri = self.expression_to_work[expr_uri]
                    if work_uri in self.work_types:
                        types = self.work_types[work_uri]
                        doc_type = next(iter(types)) if types else None
                
                # Apply filter if specified
                if document_type_filter and doc_type != document_type_filter:
                    continue
                
                if level != 'all':  # Avoid duplicates when level='all'
                    if include_type:
                        results.append((manif_uri, doc_type))
                    else:
                        results.append(manif_uri)
        
        if level == 'item' or level == 'all':
            # Get all item URIs
            for item_uri in self.item_to_manifestation.keys():
                # Trace to work to get document type
                manif_uri = self.item_to_manifestation[item_uri]
                doc_type = None
                if manif_uri in self.manifestation_to_expression:
                    expr_uri = self.manifestation_to_expression[manif_uri]
                    if expr_uri in self.expression_to_work:
                        work_uri = self.expression_to_work[expr_uri]
                        if work_uri in self.work_types:
                            types = self.work_types[work_uri]
                            doc_type = next(iter(types)) if types else None
                
                # Apply filter if specified
                if document_type_filter and doc_type != document_type_filter:
                    continue
                
                if level != 'all':  # Avoid duplicates when level='all'
                    if include_type:
                        results.append((item_uri, doc_type))
                    else:
                        results.append(item_uri)
        
        return results
    
    def count_documents(self, by_type: bool = False) -> Dict:
        """
        Count documents in the RDF file.
        
        Args:
            by_type: If True, return counts grouped by document type
        
        Returns:
            Dictionary with counts, e.g.:
            - {'total': 100, 'works': 1, 'expressions': 20, 'manifestations': 40, 'items': 80}
            - {'Decision': 5, 'Regulation': 3, 'Directive': 2} (if by_type=True)
        """
        if by_type:
            counts = {}
            for work_uri, types in self.work_types.items():
                doc_type = next(iter(types)) if types else 'Unknown'
                counts[doc_type] = counts.get(doc_type, 0) + 1
            return counts
        else:
            return {
                'works': len(self.work_types),
                'expressions': len(self.expression_to_work),
                'manifestations': len(self.manifestation_to_expression),
                'items': len(self.item_to_manifestation),
                'total': (len(self.work_types) + len(self.expression_to_work) + 
                         len(self.manifestation_to_expression) + len(self.item_to_manifestation))
            }
    
    def get_document_languages(self, work_uri: str) -> list:
        """
        Get all available languages for a document.
        
        Args:
            work_uri: The work-level URI
        
        Returns:
            List of language codes (e.g., ['ENG', 'FRA', 'DEU'])
        """
        languages = []
        
        # Find all expressions for this work
        for expr_uri, w_uri in self.expression_to_work.items():
            if w_uri == work_uri:
                # Try to extract language from expression URI
                # Common patterns: .ENG, .FRA, _ENG, _FRA
                for part in expr_uri.split('/'):
                    if '.' in part:
                        parts = part.split('.')
                        for p in parts:
                            if len(p) == 3 and p.isupper():
                                languages.append(p)
                    # Also check the last part
                    if part.isupper() and 2 <= len(part) <= 3:
                        languages.append(part)
        
        return list(set(languages))  # Remove duplicates
    
    def get_celex_id(self, resource_uri: str) -> Optional[str]:
        """
        Extract CELEX identifier from any resource URI.
        
        The CELEX identifier is a unique identifier for EU legal documents.
        Format: XYYYYTNNNN (e.g., 32004D0541)
        - X: Sector (1=Treaties, 2=International, 3=Legislation, etc.)
        - YYYY: Year
        - T: Document type (R=Regulation, L=Directive, D=Decision, etc.)
        - NNNN: Sequential number
        
        Args:
            resource_uri: The URI of the resource (item/manifestation/expression/work)
        
        Returns:
            CELEX identifier string or None if not found
        """
        def extract_celex_from_uri(uri: str) -> Optional[str]:
            """Helper to extract CELEX from a single URI."""
            if 'celex' not in uri.lower():
                return None
            
            parts = uri.split('/')
            for part in parts:
                # CELEX IDs start with a sector number (1-9) followed by year (4 digits)
                if part and part[0].isdigit() and len(part) >= 8:
                    # Remove language codes and extensions (e.g., .ENG, .fmx4, etc.)
                    # CELEX format is: sector(1) + year(4) + type(1) + number(4+)
                    celex_candidate = part.split('.')[0]
                    
                    # Validate CELEX format: starts with digit, has year, has letter for type
                    if (len(celex_candidate) >= 8 and 
                        celex_candidate[0].isdigit() and 
                        celex_candidate[1:5].isdigit()):
                        # Check if there's a letter after the year (document type)
                        if len(celex_candidate) > 5 and celex_candidate[5].isalpha():
                            return celex_candidate
                        # Sometimes the type letter might be at position 4
                        elif len(celex_candidate) > 4 and celex_candidate[4].isalpha():
                            return celex_candidate
            return None
        
        # Method 1: Check if the URI itself contains CELEX
        celex_id = extract_celex_from_uri(resource_uri)
        if celex_id:
            return celex_id
        
        # Method 2: Check current URI's sameAs mappings
        if resource_uri in self.same_as_mapping:
            for same_as_uri in self.same_as_mapping[resource_uri]:
                celex_id = extract_celex_from_uri(same_as_uri)
                if celex_id:
                    return celex_id
        
        # Method 3: Traverse to different levels and check sameAs
        current_uri = resource_uri
        
        # Trace through item -> manifestation -> expression
        if current_uri in self.item_to_manifestation:
            current_uri = self.item_to_manifestation[current_uri]
            # Check manifestation's sameAs
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    celex_id = extract_celex_from_uri(same_as_uri)
                    if celex_id:
                        return celex_id
        
        # Find manifestation -> expression
        if current_uri not in self.manifestation_to_expression:
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    if same_as_uri in self.manifestation_to_expression:
                        current_uri = same_as_uri
                        break
        
        if current_uri in self.manifestation_to_expression:
            current_uri = self.manifestation_to_expression[current_uri]
            # Check expression's sameAs
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    celex_id = extract_celex_from_uri(same_as_uri)
                    if celex_id:
                        return celex_id
        
        # Find expression -> work with 2-hop search
        work_uri = None
        if current_uri in self.expression_to_work:
            work_uri = self.expression_to_work[current_uri]
        else:
            # Two-hop search through sameAs
            candidates_to_check = set()
            if current_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[current_uri]:
                    candidates_to_check.add(same_as_uri)
                    # Check this candidate's sameAs too
                    celex_id = extract_celex_from_uri(same_as_uri)
                    if celex_id:
                        return celex_id
                    
                    if same_as_uri in self.same_as_mapping:
                        for second_hop_uri in self.same_as_mapping[same_as_uri]:
                            candidates_to_check.add(second_hop_uri)
                            celex_id = extract_celex_from_uri(second_hop_uri)
                            if celex_id:
                                return celex_id
            
            for candidate in candidates_to_check:
                if candidate in self.expression_to_work:
                    work_uri = self.expression_to_work[candidate]
                    break
        
        # Check work URI and its sameAs
        if work_uri:
            celex_id = extract_celex_from_uri(work_uri)
            if celex_id:
                return celex_id
            
            if work_uri in self.same_as_mapping:
                for same_as_uri in self.same_as_mapping[work_uri]:
                    celex_id = extract_celex_from_uri(same_as_uri)
                    if celex_id:
                        return celex_id
        
        return None


def main():
    """Main function demonstrating the classifier usage."""
    
    # Initialize classifier
    print("Loading RDF file...")
    classifier = EUDocumentClassifier('/mnt/user-data/uploads/tree_non_inferred.rdf')
    print(f"✓ Loaded and indexed RDF file\n")
    
    # NEW: Show document counts
    print("="*60)
    print("Document Counts:")
    print("="*60)
    counts = classifier.count_documents()
    print(f"Total URIs: {counts['total']}")
    print(f"  Works: {counts['works']}")
    print(f"  Expressions: {counts['expressions']}")
    print(f"  Manifestations: {counts['manifestations']}")
    print(f"  Items: {counts['items']}")
    
    # NEW: Show counts by document type
    print(f"\nDocuments by Type:")
    type_counts = classifier.count_documents(by_type=True)
    for doc_type, count in sorted(type_counts.items()):
        print(f"  {doc_type}: {count}")
    print()
    
    # NEW: List some work URIs
    print("="*60)
    print("Sample Work URIs:")
    print("="*60)
    works = classifier.list_all_documents(level='work', include_type=True)
    for i, (uri, doc_type) in enumerate(works[:5], 1):
        print(f"{i}. {doc_type}: {uri}")
    print(f"... and {len(works) - 5} more\n")
    
    # Test with the requested document
    test_uri = "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03/DOC_1"
    
    print(f"="*60)
    print(f"Classifying document:")
    print(f"="*60)
    print(f"URI: {test_uri}\n")
    
    # Classify the document
    doc_type = classifier.classify_document(test_uri)
    
    if doc_type:
        print(f"\n{'='*60}")
        print(f"RESULT: {doc_type}")
        print(f"{'='*60}")
    else:
        print("\nNo document type found for this URI")
    
    # Get CELEX ID
    print(f"\n{'='*60}")
    print("Extracting CELEX ID:")
    print(f"{'='*60}")
    celex_id = classifier.get_celex_id(test_uri)
    if celex_id:
        print(f"CELEX ID: {celex_id}")
        # Parse CELEX components
        if len(celex_id) >= 8:
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
            
            print(f"  Sector: {sector} ({sector_names.get(sector, 'Unknown')})")
            print(f"  Year: {year}")
            print(f"  Type: {doc_type_letter} ({type_names.get(doc_type_letter, 'Unknown')})")
            print(f"  Number: {sequential}")
    else:
        print("CELEX ID not found")
    
    # Get detailed info
    print("\n" + "="*60)
    print("Detailed Information:")
    print("="*60)
    info = classifier.get_document_info(test_uri)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    
    # Additional examples
    print("\n" + "="*60)
    print("Testing with other document variations:")
    print("="*60)
    
    # Test with manifestation URI (without /DOC_1)
    manifestation_uri = "http://publications.europa.eu/resource/cellar/96d49a9c-1316-42da-8c99-ce64d3706754.0005.03"
    print(f"\nTesting manifestation URI: ...0005.03")
    doc_type = classifier.classify_document(manifestation_uri)
    if doc_type:
        print(f"Result: {doc_type}")
    
    celex_id = classifier.get_celex_id(manifestation_uri)
    if celex_id:
        print(f"CELEX ID: {celex_id}")
    
    # Test with a CELEX URI directly
    print(f"\n{'='*60}")
    print("Testing with CELEX URI:")
    print(f"{'='*60}")
    celex_uri = "http://publications.europa.eu/resource/celex/32004D0541"
    print(f"URI: {celex_uri}")
    celex_id = classifier.get_celex_id(celex_uri)
    if celex_id:
        print(f"CELEX ID extracted: {celex_id}")
    
    # NEW: List documents by type
    print(f"\n{'='*60}")
    print("Filtering Documents by Type:")
    print(f"{'='*60}")
    decisions = classifier.list_all_documents(level='work', include_type=False, 
                                             document_type_filter='Decision')
    print(f"Found {len(decisions)} Decision(s)")
    if decisions:
        print(f"First Decision: {decisions[0]}")


if __name__ == "__main__":
    main()
