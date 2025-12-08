"""
EUR-Lex Metadata Extractor.

Extracts document metadata (URI, CELEX ID, document type) from RDF files
and outputs a JSONL file for indexing and analysis.
"""
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from tqdm import tqdm

from classify_document import EUDocumentClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Directory constants
METADATA_DIR = Path("LEG_MTD_20251130_01_00")
OUTPUT_FILE = Path("eurlex_metadata.jsonl")


def find_rdf_files(metadata_dir: Path) -> List[Path]:
    """
    Find all RDF files in the metadata directory.

    Args:
        metadata_dir: Base metadata directory

    Returns:
        List of paths to RDF files
    """
    return list(metadata_dir.rglob("tree_non_inferred.rdf"))


def extract_metadata_from_rdf(rdf_path: Path) -> List[Dict[str, Any]]:
    """
    Extract document metadata from a single RDF file.

    Args:
        rdf_path: Path to the RDF file

    Returns:
        List of metadata records for documents in the RDF file
    """
    records = []

    try:
        classifier = EUDocumentClassifier(str(rdf_path))

        # Get all works (unique documents) with their types
        works = classifier.list_all_documents(level='work', include_type=True)

        for uri, doc_type in works:
            celex_id = classifier.get_celex_id(uri)

            record = {
                "uri": uri,
                "celex_id": celex_id,
                "document_type": doc_type,
                "rdf_file": str(rdf_path.parent.name),  # UUID folder name
            }

            # Extract additional metadata if available
            try:
                info = classifier.get_document_info(uri)
                if info.get("work_uri"):
                    record["work_uri"] = info["work_uri"]
            except Exception:
                pass  # Additional info is optional

            records.append(record)

    except Exception as e:
        logger.debug(f"Error processing {rdf_path}: {e}")

    return records


def process_rdf_file(rdf_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Process a single RDF file and return metadata records.

    Args:
        rdf_path: Path to the RDF file

    Returns:
        Tuple of (list of records, error message or None)
    """
    try:
        records = extract_metadata_from_rdf(rdf_path)
        return records, None
    except Exception as e:
        return [], f"Error processing {rdf_path}: {e}"


def process_rdf_file_wrapper(rdf_path: Path) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Wrapper for multiprocessing compatibility."""
    return process_rdf_file(rdf_path)


def extract_all_metadata(
    metadata_dir: Path = METADATA_DIR,
    output_file: Path = OUTPUT_FILE,
    num_workers: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Extract metadata from all RDF files in the directory.

    Args:
        metadata_dir: Directory containing RDF files
        output_file: Output JSONL file path
        num_workers: Number of worker processes (default: CPU count)

    Returns:
        Tuple of (total records, files processed, errors)
    """
    rdf_files = find_rdf_files(metadata_dir)
    total_files = len(rdf_files)
    logger.info(f"Found {total_files} RDF files to process")

    if num_workers is None:
        num_workers = cpu_count()
    logger.info(f"Using {num_workers} worker processes")

    total_records = 0
    files_processed = 0
    error_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_rdf_file_wrapper, rdf_path): rdf_path
                for rdf_path in rdf_files
            }

            for future in tqdm(
                as_completed(futures),
                total=total_files,
                desc="Extracting metadata",
                unit="file"
            ):
                try:
                    records, error = future.result()

                    if error:
                        logger.debug(error)
                        error_count += 1
                    else:
                        files_processed += 1

                    for record in records:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_records += 1

                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    error_count += 1

    logger.info(f"Extracted {total_records} records from {files_processed} files")
    logger.info(f"Errors: {error_count}")

    return total_records, files_processed, error_count


def extract_all_metadata_sequential(
    metadata_dir: Path = METADATA_DIR,
    output_file: Path = OUTPUT_FILE,
) -> Tuple[int, int, int]:
    """
    Extract metadata sequentially (for debugging).

    Args:
        metadata_dir: Directory containing RDF files
        output_file: Output JSONL file path

    Returns:
        Tuple of (total records, files processed, errors)
    """
    rdf_files = find_rdf_files(metadata_dir)
    logger.info(f"Found {len(rdf_files)} RDF files to process")

    total_records = 0
    files_processed = 0
    error_count = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        for rdf_path in tqdm(rdf_files, desc="Extracting metadata", unit="file"):
            records, error = process_rdf_file(rdf_path)

            if error:
                logger.debug(error)
                error_count += 1
            else:
                files_processed += 1

            for record in records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += 1

    logger.info(f"Extracted {total_records} records from {files_processed} files")
    logger.info(f"Errors: {error_count}")

    return total_records, files_processed, error_count


def analyze_output(output_file: Path = OUTPUT_FILE) -> Dict[str, Any]:
    """
    Analyze the extracted metadata file.

    Args:
        output_file: Path to the JSONL file

    Returns:
        Statistics about the extracted metadata
    """
    type_counts = {}
    total = 0
    with_celex = 0

    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total += 1

            doc_type = record.get("document_type", "Unknown")
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

            if record.get("celex_id"):
                with_celex += 1

    return {
        "total_records": total,
        "with_celex_id": with_celex,
        "without_celex_id": total - with_celex,
        "document_types": type_counts,
    }


def main():
    """Main entry point with CLI argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract metadata from EUR-Lex RDF files to JSONL format"
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
        "--analyze",
        action="store_true",
        help="Analyze existing output file instead of extracting"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Analyze mode
    if args.analyze:
        if not args.output.exists():
            logger.error(f"Output file not found: {args.output}")
            return

        logger.info(f"Analyzing {args.output}")
        stats = analyze_output(args.output)

        print("\n" + "=" * 60)
        print("Metadata Statistics")
        print("=" * 60)
        print(f"Total records: {stats['total_records']}")
        print(f"With CELEX ID: {stats['with_celex_id']}")
        print(f"Without CELEX ID: {stats['without_celex_id']}")
        print("\nDocument Types:")
        for doc_type, count in sorted(
            stats['document_types'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percentage = count / stats['total_records'] * 100
            print(f"  {doc_type}: {count} ({percentage:.1f}%)")
        return

    # Extraction mode
    logger.info("Starting EUR-Lex metadata extraction")
    logger.info(f"Metadata directory: {args.metadata_dir}")
    logger.info(f"Output file: {args.output}")

    if args.sequential:
        logger.info("Using sequential (single-threaded) processing")
        records, files, errors = extract_all_metadata_sequential(
            metadata_dir=args.metadata_dir,
            output_file=args.output,
        )
    else:
        records, files, errors = extract_all_metadata(
            metadata_dir=args.metadata_dir,
            output_file=args.output,
            num_workers=args.workers,
        )

    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Total records: {records}")
    print(f"Files processed: {files}")
    print(f"Errors: {errors}")
    print(f"Output: {args.output}")

    # Quick analysis
    if args.output.exists():
        print("\nRunning quick analysis...")
        stats = analyze_output(args.output)
        print(f"\nDocument types found: {len(stats['document_types'])}")
        top_types = sorted(
            stats['document_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        for doc_type, count in top_types:
            print(f"  {doc_type}: {count}")


if __name__ == "__main__":
    main()
