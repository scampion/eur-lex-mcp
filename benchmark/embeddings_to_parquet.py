"""
EUR-Lex BGE-M3 Embeddings to Parquet.

Computes BGE-M3 embeddings for EUR-Lex documents and stores them in Parquet format.
Supports dense, sparse, and ColBERT vectors.

Usage:
  # Generate embeddings for all documents
  python embeddings_to_parquet.py --input eurlex.jsonl --output eurlex_embeddings.parquet

  # Limit documents (for testing)
  python embeddings_to_parquet.py --input eurlex.jsonl --output test.parquet --limit 100

  # Use specific device
  python embeddings_to_parquet.py --input eurlex.jsonl --output eurlex.parquet --device cuda

  # Use Ollama (dense-only, faster on Apple Silicon)
  python embeddings_to_parquet.py --input eurlex.jsonl --output eurlex.parquet --ollama

Dependencies:
  pip install FlagEmbedding pyarrow pandas tqdm
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from FlagEmbedding import BGEM3FlagModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT_FILE = Path("eurlex.jsonl")
DEFAULT_OUTPUT_FILE = Path("eurlex_embeddings.parquet")
EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = 32


@dataclass
class Document:
    """Represents a document to be embedded."""
    id: str
    text: str
    uri: str
    filename: Optional[str] = None
    file_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


def get_default_device() -> str:
    """Detect the best available device for the current platform."""
    import platform
    import torch

    if platform.system() == "Darwin" and platform.processor() == "arm":
        if torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BGE_M3_Embedder:
    """BGE-M3 embedding model wrapper with Apple Silicon support."""

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        if device == "auto":
            device = get_default_device()
            logger.info(f"Auto-detected device: {device}")

        self.device = device
        self.use_fp16 = use_fp16 and device != "mps"
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model on {self.device}...")

            self._model = BGEM3FlagModel(
                "BAAI/bge-m3",
                use_fp16=self.use_fp16,
                device=self.device
            )
            logger.info("BGE-M3 model loaded")
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 12,
        max_length: int = 1024
    ) -> Dict[str, Any]:
        """Encode texts into dense, sparse, and ColBERT vectors."""
        result = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        # Normalize key names (FlagEmbedding uses 'dense_vecs', not 'dense')
        return {
            "dense": result.get("dense_vecs", result.get("dense")),
            "lexical_weights": result.get("lexical_weights"),
            "colbert_vecs": result.get("colbert_vecs"),
        }


class OllamaEmbedder:
    """Ollama-based embedder for BGE-M3 (dense vectors only)."""

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> Dict[str, Any]:
        """Encode texts into dense vectors only (Ollama limitation)."""
        import requests

        dense_vectors = []

        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:max_length]},
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            dense_vectors.append(embedding)

        dense_array = np.array(dense_vectors)

        return {
            "dense": dense_array,
            "lexical_weights": [{} for _ in texts],
            "colbert_vecs": [dense_array[i:i+1] for i in range(len(texts))],
        }


def load_documents(input_file: Path, limit: Optional[int] = None) -> List[Document]:
    """Load documents from JSONL file."""
    documents = []

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line)
            doc_id = data.get("file_id", str(i))

            documents.append(
                Document(
                    id=doc_id,
                    text=data.get("text", ""),
                    uri=data.get("uri", ""),
                    filename=data.get("filename"),
                    file_id=data.get("file_id"),
                    metadata={
                        k: v for k, v in data.items()
                        if k not in ("text", "uri", "filename", "file_id")
                    },
                )
            )

    logger.info(f"Loaded {len(documents)} documents from {input_file}")
    return documents


def sparse_to_json(lexical_weights: Dict) -> str:
    """Convert sparse vector to JSON string for storage."""
    # Filter and convert to serializable format
    filtered = {}
    for key, value in lexical_weights.items():
        weight = float(value)
        if weight > 0:
            filtered[str(key)] = weight
    return json.dumps(filtered)


def get_checkpoint_path(output_file: Path) -> Path:
    """Get the checkpoint file path for a given output file."""
    return output_file.parent / f".{output_file.stem}_checkpoint.parquet"


def load_checkpoint(checkpoint_path: Path) -> tuple[set, List[Dict]]:
    """
    Load existing checkpoint if available.

    Returns:
        Tuple of (processed_ids set, existing records list)
    """
    if not checkpoint_path.exists():
        return set(), []

    try:
        df = pd.read_parquet(checkpoint_path)
        processed_ids = set(df["id"].tolist())
        records = df.to_dict("records")
        logger.info(f"Loaded checkpoint with {len(records)} processed documents")
        return processed_ids, records
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return set(), []


def save_checkpoint(records: List[Dict], checkpoint_path: Path):
    """Save current progress to checkpoint file."""
    if not records:
        return

    try:
        df = pd.DataFrame(records)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, checkpoint_path, compression="snappy")
        logger.debug(f"Checkpoint saved: {len(records)} records")
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def compute_and_save_embeddings(
    documents: List[Document],
    embedder: BGE_M3_Embedder,
    output_file: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_colbert: bool = True,
    checkpoint_interval: int = 250,
    resume: bool = True,
):
    """
    Compute embeddings and save to Parquet file with checkpoint support.

    Args:
        documents: List of documents to embed
        embedder: BGE-M3 embedder instance
        output_file: Output Parquet file path
        batch_size: Batch size for embedding
        include_colbert: Whether to include ColBERT vectors (increases file size)
        checkpoint_interval: Save checkpoint every N batches
        resume: Whether to resume from checkpoint if available
    """
    total = len(documents)
    logger.info(f"Computing embeddings for {total} documents")

    checkpoint_path = get_checkpoint_path(output_file)

    # Load checkpoint if resuming
    processed_ids: set = set()
    records: List[Dict] = []

    if resume:
        processed_ids, records = load_checkpoint(checkpoint_path)
        if processed_ids:
            logger.info(f"Resuming from checkpoint: {len(processed_ids)} already processed")

    # Filter out already processed documents
    if processed_ids:
        documents = [doc for doc in documents if doc.id not in processed_ids]
        logger.info(f"Remaining documents to process: {len(documents)}")

    if not documents:
        logger.info("All documents already processed")
    else:
        batches_since_checkpoint = 0

        for i in tqdm(range(0, len(documents), batch_size), desc="⚡️ Embedding", unit="batch", colour="cyan"):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.text for doc in batch_docs]

            try:
                logger.info(f"Computing embeddings for {len(texts)} documents")
                embeddings = embedder.encode(texts, batch_size=len(texts))
                logger.info(f"Saving embeddings for {len(embeddings)} documents")

                for j, doc in enumerate(batch_docs):
                    record = {
                        "id": doc.id,
                        "uri": doc.uri,
                        "text": doc.text,  # Truncate for storage
                        "text_length": len(doc.text),
                        "dense_embedding": embeddings["dense"][j].tolist(),
                        "sparse_embedding": sparse_to_json(embeddings["lexical_weights"][j]),
                    }

                    if doc.filename:
                        record["filename"] = doc.filename
                    if doc.file_id:
                        record["file_id"] = doc.file_id

                    # ColBERT vectors can be large, make optional
                    if include_colbert:
                        colbert_vecs = embeddings["colbert_vecs"][j]  # Shape: (N_tokens, 1024)

                        logger.debug(f"ColBERT shape for doc {j}: {colbert_vecs.shape}")

                        record["colbert_embedding"] = colbert_vecs.tolist()
                        record["num_tokens"] = colbert_vecs.shape[0]

                    # Add metadata fields
                    if doc.metadata:
                        for k, v in doc.metadata.items():
                            if isinstance(v, (str, int, float, bool)):
                                record[f"meta_{k}"] = v

                    records.append(record)

                batches_since_checkpoint += 1

                # Save checkpoint periodically
                if batches_since_checkpoint >= checkpoint_interval:
                    logger.info(f"Saving checkpoint for {batches_since_checkpoint} batches")
                    save_checkpoint(records, checkpoint_path)
                    batches_since_checkpoint = 0

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size}: {e}")
                # Save checkpoint on error
                save_checkpoint(records, checkpoint_path)
                continue

    logger.info(f"Processed {len(records)} documents total")

    # Create DataFrame and save to Parquet
    df = pd.DataFrame(records)

    # Convert dense embeddings to proper array format for efficient storage
    logger.info(f"Saving to {output_file}")

    # Use PyArrow for better compression and nested array support
    table = pa.Table.from_pandas(df)
    pq.write_table(
        table,
        output_file,
        compression="snappy",
        row_group_size=10000,
    )

    # Report file size
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(records)} embeddings to {output_file} ({file_size_mb:.2f} MB)")

    # Clean up checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        logger.info("Checkpoint file removed after successful completion")

    return len(records)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute BGE-M3 embeddings and save to Parquet"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output Parquet file (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (default: all)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cuda", "cpu", "mps", "auto"],
        help="Device for embeddings (default: auto-detect)"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama for embeddings (faster on M1, but dense-only)"
    )
    parser.add_argument(
        "--no-colbert",
        action="store_true",
        help="Skip ColBERT vectors (reduces file size significantly)"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=250,
        help="Save checkpoint every N batches (default: 10)"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignore any existing checkpoint"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check input file
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Initialize embedder
    if args.ollama:
        logger.info("Using Ollama embedder (dense-only)")
        embedder = OllamaEmbedder()
    else:
        embedder = BGE_M3_Embedder(device=args.device)

    # Load documents
    documents = load_documents(args.input, limit=args.limit)

    if not documents:
        logger.error("No documents to process")
        return

    # Compute and save embeddings
    count = compute_and_save_embeddings(
        documents=documents,
        embedder=embedder,
        output_file=args.output,
        batch_size=args.batch_size,
        include_colbert=not args.no_colbert,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print("Embedding Complete")
    print("=" * 60)
    print(f"Documents processed: {count}")
    print(f"Output file: {args.output}")
    print(f"File size: {args.output.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
