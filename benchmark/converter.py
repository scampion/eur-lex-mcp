"""
EUR-Lex HTML to JSONL Converter.

Converts EUR-Lex HTML documents and their RDF metadata into a JSONL format
suitable for indexing and evaluation.

Supports multiprocessing for faster conversion.
"""
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

from extract_rdf_metadata import parse_rdf_file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directory constants
HTML_DIR = Path("LEG_EN_HTML_20251130_01_00")
METADATA_DIR = Path("LEG_MTD_20251130_01_00")
OUTPUT_FILE = Path("eurlex.jsonl")


def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content.

    Args:
        html_content: Raw HTML string

    Returns:
        Cleaned text with normalized whitespace
    """
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n")

    # Normalize whitespace: remove empty lines and strip each line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def load_html_file(filepath: Path) -> str:
    """Load and return HTML content from a file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_metadata(metadata_dir: Path, subdir: str) -> Optional[Dict]:
    """
    Load RDF metadata for a document.

    Args:
        metadata_dir: Base metadata directory
        subdir: Subdirectory (document ID)

    Returns:
        Parsed metadata dict or None if not found
    """
    metadata_path = metadata_dir / subdir / "tree_non_inferred.rdf"

    if not metadata_path.exists():
        logger.warning(f"Metadata file not found: {metadata_path}")
        return None

    logger.debug(f"Loading metadata from {metadata_path}")
    return parse_rdf_file(str(metadata_path))


def find_matching_resource(
    metadata: Dict,
    local_name: str
) -> Optional[Dict[str, Any]]:
    """
    Find the metadata resource matching the given local name.

    Args:
        metadata: Parsed RDF metadata
        local_name: Local name to match (e.g., "subdir.filename")

    Returns:
        Matching resource item or None
    """
    resources = metadata.get("resources", {}).get("items", [])

    for item in resources:
        if is_matching_resource(item, local_name):
            return item

    return None


def is_matching_resource(item: Dict, local_name: str) -> bool:
    """
    Check if a resource item matches the given local name via owl:sameAs.

    Args:
        item: Resource item from metadata
        local_name: Local name to match

    Returns:
        True if the item matches
    """
    same_as = item.get("owl:sameAs")

    if same_as is None:
        return False

    # Handle both list and single dict cases
    same_as_list = same_as if isinstance(same_as, list) else [same_as]

    for entry in same_as_list:
        if isinstance(entry, dict):
            entry_local_name = entry.get("local_name", "")
            if entry_local_name.endswith(local_name):
                return True

    return False


def build_local_name(subdir: str, filename: str) -> str:
    """
    Build the local name used to match against RDF metadata.

    Args:
        subdir: Document subdirectory (e.g., "html" or "xhtml")
        filename: HTML filename

    Returns:
        Local name in format "subdir.filename"
    """
    return f"{subdir}.{filename}"


def process_html_file(
    filepath: Path,
    metadata_dir: Path
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Process a single HTML file and its metadata.

    Args:
        filepath: Path to the HTML file
        metadata_dir: Base metadata directory

    Returns:
        Tuple of (record dict, error message). One will be None.
    """
    try:
        # Extract path components
        # Structure: HTML_DIR / doc_id / (html|xhtml) / filename.html
        parts = filepath.parts
        filename = filepath.name
        subdir_type = parts[-2]  # "html" or "xhtml"
        doc_id = parts[-3]       # UUID document identifier

        # Load and extract text from HTML
        html_content = load_html_file(filepath)
        text_content = extract_text_from_html(html_content)

        # Load metadata
        metadata = load_metadata(metadata_dir, doc_id)
        if metadata is None:
            return None, f"Metadata not found for {filename}"

        # Find matching resource in metadata
        local_name = build_local_name(subdir_type, filename)
        resource = find_matching_resource(metadata, local_name)

        if resource is None:
            return None, f"No matching resource for {filename}"

        # Build output record
        uri = resource.get("uri", "N/A")

        record = {
            "file_id": doc_id,
            "filename": filename,
            "text": text_content,
            "uri": uri,
        }
        return record, None

    except Exception as e:
        return None, f"Error processing {filepath}: {e}"


def process_html_file_wrapper(
    args: Tuple[Path, Path]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Wrapper for process_html_file to work with multiprocessing Pool.map.

    Args:
        args: Tuple of (filepath, metadata_dir)

    Returns:
        Result from process_html_file
    """
    filepath, metadata_dir = args
    return process_html_file(filepath, metadata_dir)


def find_html_files(html_dir: Path) -> List[Path]:
    """
    Find all HTML files in the directory tree.

    Args:
        html_dir: Base HTML directory

    Returns:
        List of paths to HTML files
    """
    return list(html_dir.rglob("*.html"))


def convert_eurlex_to_jsonl(
    html_dir: Path = HTML_DIR,
    metadata_dir: Path = METADATA_DIR,
    output_file: Path = OUTPUT_FILE,
    num_workers: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Convert EUR-Lex HTML documents to JSONL format.

    Uses multiprocessing for faster conversion.

    Args:
        html_dir: Directory containing HTML documents
        metadata_dir: Directory containing RDF metadata
        output_file: Output JSONL file path
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        Tuple of (successful count, error count)
    """
    html_files = find_html_files(html_dir)
    total_files = len(html_files)
    logger.info(f"Found {total_files} HTML files to process")

    if num_workers is None:
        num_workers = cpu_count()
    logger.info(f"Using {num_workers} worker processes")

    # Prepare arguments for workers
    work_items = [(filepath, metadata_dir) for filepath in html_files]

    processed_count = 0
    error_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Use ProcessPoolExecutor (more robust than multiprocessing.Pool)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_html_file_wrapper, item): item
                for item in work_items
            }

            # Process results with progress bar as they complete
            for future in tqdm(
                as_completed(futures),
                total=total_files,
                desc="Converting documents",
                unit="doc"
            ):
                try:
                    record, error = future.result()
                    if record is not None:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        processed_count += 1
                    elif error:
                        logger.debug(error)
                        error_count += 1
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    error_count += 1

    logger.info(f"Successfully processed {processed_count} documents")
    logger.info(f"Errors: {error_count}")

    return processed_count, error_count


def convert_eurlex_to_jsonl_sequential(
    html_dir: Path = HTML_DIR,
    metadata_dir: Path = METADATA_DIR,
    output_file: Path = OUTPUT_FILE,
) -> Tuple[int, int]:
    """
    Convert EUR-Lex HTML documents to JSONL format (single-threaded version).

    Useful for debugging or when multiprocessing is not desired.

    Args:
        html_dir: Directory containing HTML documents
        metadata_dir: Directory containing RDF metadata
        output_file: Output JSONL file path

    Returns:
        Tuple of (successful count, error count)
    """
    html_files = find_html_files(html_dir)
    logger.info(f"Found {len(html_files)} HTML files to process")

    processed_count = 0
    error_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for filepath in tqdm(html_files, desc="Converting documents", unit="doc"):
            record, error = process_html_file(filepath, metadata_dir)

            if record is not None:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed_count += 1
            elif error:
                logger.debug(error)
                error_count += 1

    logger.info(f"Successfully processed {processed_count} documents")
    logger.info(f"Errors: {error_count}")

    return processed_count, error_count


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert EUR-Lex HTML documents to JSONL format"
    )
    parser.add_argument(
        "--html-dir",
        type=Path,
        default=HTML_DIR,
        help=f"Directory containing HTML documents (default: {HTML_DIR})"
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=METADATA_DIR,
        help=f"Directory containing RDF metadata (default: {METADATA_DIR})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output JSONL file path (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: CPU count = {cpu_count()})"
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use single-threaded processing (for debugging)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (show warnings)"
    )

    args = parser.parse_args()

    # Set log level based on verbosity
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting EUR-Lex HTML to JSONL conversion")
    logger.info(f"HTML directory: {args.html_dir}")
    logger.info(f"Metadata directory: {args.metadata_dir}")
    logger.info(f"Output file: {args.output}")

    if args.sequential:
        logger.info("Using sequential (single-threaded) processing")
        success, errors = convert_eurlex_to_jsonl_sequential(
            html_dir=args.html_dir,
            metadata_dir=args.metadata_dir,
            output_file=args.output,
        )
    else:
        success, errors = convert_eurlex_to_jsonl(
            html_dir=args.html_dir,
            metadata_dir=args.metadata_dir,
            output_file=args.output,
            num_workers=args.workers,
        )

    logger.info(f"Conversion complete. Success: {success}, Errors: {errors}")


if __name__ == "__main__":
    main()
