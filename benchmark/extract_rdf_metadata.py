#!/usr/bin/env python3
"""
RDF Metadata Extractor for EU Publications Office CDM (Common Data Model) files.

This script parses RDF/XML files from EU Publications and extracts metadata
into a structured JSON format.
"""

import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any
import re
import sys


# Namespace mappings commonly used in EU Publications RDF
NAMESPACES = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'owl': 'http://www.w3.org/2002/07/owl#',
    'skos': 'http://www.w3.org/2004/02/skos/core#',
    'cdm': 'http://publications.europa.eu/ontology/cdm#',
    'ann': 'http://publications.europa.eu/ontology/annotation#',
    'cmr': 'http://publications.europa.eu/ontology/cdm/cmr#',
    'xml': 'http://www.w3.org/XML/1998/namespace'
}


def extract_local_name(uri: str) -> str:
    """Extract the local name from a URI (after # or last /)."""
    if '#' in uri:
        return uri.split('#')[-1]
    return uri.rsplit('/', 1)[-1]


def parse_namespace_prefixes(root: ET.Element) -> dict:
    """Extract namespace prefix mappings from the root element."""
    ns_map = {}
    for attr, value in root.attrib.items():
        if attr.startswith('{'):
            continue
        if attr.startswith('xmlns:'):
            prefix = attr.split(':')[1]
            ns_map[prefix] = value
        elif attr == 'xmlns':
            ns_map[''] = value
    return ns_map


def get_resource_type(description: ET.Element) -> str | None:
    """Determine the RDF type of a resource."""
    for child in description:
        tag = child.tag
        if tag == f"{{{NAMESPACES['rdf']}}}type":
            resource = child.get(f"{{{NAMESPACES['rdf']}}}resource")
            if resource:
                return extract_local_name(resource)
    return None


def extract_description_metadata(description: ET.Element, ns_prefixes: dict) -> dict:
    """Extract all metadata from a single rdf:Description element."""
    metadata = {}
    
    # Get the subject URI
    about = description.get(f"{{{NAMESPACES['rdf']}}}about")
    node_id = description.get(f"{{{NAMESPACES['rdf']}}}nodeID")
    
    if about:
        metadata['uri'] = about
        metadata['identifier'] = extract_local_name(about)
    elif node_id:
        metadata['blank_node_id'] = node_id
    
    # Get the resource type
    rdf_type = get_resource_type(description)
    if rdf_type:
        metadata['type'] = rdf_type
    
    # Extract all properties
    properties = defaultdict(list)
    
    for child in description:
        tag = child.tag
        
        # Skip rdf:type as we've already processed it
        if tag == f"{{{NAMESPACES['rdf']}}}type":
            continue
        
        # Extract namespace and local name
        if tag.startswith('{'):
            ns_uri, local_name = tag[1:].split('}')
            # Find prefix for this namespace
            prefix = None
            for p, uri in NAMESPACES.items():
                if uri == ns_uri:
                    prefix = p
                    break
            if not prefix:
                for p, uri in ns_prefixes.items():
                    if uri == ns_uri:
                        prefix = p
                        break
            
            prop_name = f"{prefix}:{local_name}" if prefix else local_name
        else:
            prop_name = tag
        
        # Extract value (either resource reference or literal)
        resource = child.get(f"{{{NAMESPACES['rdf']}}}resource")
        datatype = child.get(f"{{{NAMESPACES['rdf']}}}datatype")
        lang = child.get(f"{{{NAMESPACES['xml']}}}lang")
        
        if resource:
            value = {
                'type': 'resource',
                'uri': resource,
                'local_name': extract_local_name(resource)
            }
        elif child.text:
            value = {
                'type': 'literal',
                'value': child.text.strip() if child.text else ''
            }
            if datatype:
                value['datatype'] = extract_local_name(datatype)
            if lang:
                value['language'] = lang
        else:
            continue
        
        properties[prop_name].append(value)
    
    # Convert single-item lists to single values for cleaner JSON
    for prop, values in properties.items():
        if len(values) == 1:
            metadata[prop] = values[0]
        else:
            metadata[prop] = values
    
    return metadata


def categorize_resources(all_metadata: list[dict]) -> dict:
    """Organize resources by their RDF type."""
    categorized = {
        'works': [],
        'expressions': [],
        'manifestations': [],
        'items': [],
        'languages': [],
        'axioms': [],
        'other': []
    }
    
    type_mapping = {
        'work': 'works',
        'expression': 'expressions',
        'manifestation': 'manifestations',
        'item': 'items',
        'language': 'languages',
        'Axiom': 'axioms'
    }
    
    for resource in all_metadata:
        rdf_type = resource.get('type', '')
        category = type_mapping.get(rdf_type, 'other')
        categorized[category].append(resource)
    
    # Remove empty categories
    return {k: v for k, v in categorized.items() if v}


def extract_document_summary(categorized: dict) -> dict:
    """Extract high-level document summary from categorized resources."""
    summary = {
        'total_resources': sum(len(v) for v in categorized.values()),
        'resource_counts': {k: len(v) for k, v in categorized.items()}
    }
    
    # Extract work-level metadata if available
    if categorized.get('works'):
        work = categorized['works'][0]  # Usually one main work
        summary['work'] = {
            'uri': work.get('uri'),
            'identifier': work.get('identifier')
        }
        
        # Extract titles if available
        for key, value in work.items():
            if 'title' in key.lower():
                if isinstance(value, dict):
                    summary['work']['title'] = value.get('value')
                    if value.get('language'):
                        summary['work']['title_language'] = value.get('language')
                elif isinstance(value, list):
                    summary['work']['titles'] = [
                        {'value': v.get('value'), 'language': v.get('language')}
                        for v in value if v.get('value')
                    ]
    
    # Extract available languages from expressions
    if categorized.get('expressions'):
        languages = set()
        for expr in categorized['expressions']:
            for key, value in expr.items():
                if 'language' in key.lower() and isinstance(value, dict):
                    if value.get('type') == 'resource':
                        languages.add(value.get('local_name'))
        summary['available_languages'] = sorted(list(languages))
    
    # Extract manifestation types
    if categorized.get('manifestations'):
        manifest_types = set()
        for manifest in categorized['manifestations']:
            mtype = manifest.get('cdm:manifestation_type')
            if mtype and isinstance(mtype, dict):
                manifest_types.add(mtype.get('value'))
        summary['manifestation_types'] = sorted(list(manifest_types))
    
    return summary


def parse_rdf_file(filepath: str) -> dict:
    """
    Parse an RDF/XML file and extract all metadata into JSON format.
    
    Args:
        filepath: Path to the RDF/XML file
        
    Returns:
        Dictionary containing extracted metadata
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    # Extract namespace prefixes from the document
    ns_prefixes = parse_namespace_prefixes(root)
    
    # Update our namespace mappings with any document-specific ones
    for prefix, uri in ns_prefixes.items():
        if prefix and prefix not in NAMESPACES:
            NAMESPACES[prefix] = uri
    
    all_metadata = []
    
    # Process each rdf:Description element
    for description in root.findall(f"{{{NAMESPACES['rdf']}}}Description"):
        metadata = extract_description_metadata(description, ns_prefixes)
        if metadata:
            all_metadata.append(metadata)
    
    # Categorize resources by type
    categorized = categorize_resources(all_metadata)
    
    # Build the final output
    result = {
        'metadata': {
            'source_file': filepath,
            'format': 'RDF/XML',
            'ontology': 'EU Publications CDM (Common Data Model)',
            'namespaces': NAMESPACES
        },
        'summary': extract_document_summary(categorized),
        'resources': categorized
    }
    
    return result


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract metadata from EU Publications RDF files to JSON'
    )
    parser.add_argument('input_file', help='Path to the RDF/XML file')
    parser.add_argument(
        '-o', '--output',
        help='Output JSON file (default: stdout)'
    )
    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation level (default: 2)'
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Output only the summary section'
    )
    
    args = parser.parse_args()
    
    try:
        result = parse_rdf_file(args.input_file)
        
        if args.summary_only:
            output = result['summary']
        else:
            output = result
        
        json_output = json.dumps(output, indent=args.indent, ensure_ascii=False)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(json_output)
            print(f"Output written to {args.output}")
        else:
            print(json_output)
            
    except FileNotFoundError:
        print(f"Error: File not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

