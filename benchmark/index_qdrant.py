"""
EUR-Lex Qdrant Indexer with BGE-M3 and ColBERT.

Indexes EUR-Lex documents into Qdrant using BGE-M3 embeddings with support for:
- Dense vectors (1024 dimensions)
- Sparse vectors (lexical matching)
- ColBERT multi-vectors (token-level late interaction)

Based on: https://github.com/yuniko-software/bge-m3-qdrant-sample


  Features:
  | Component      | Description                                     |
  |----------------|-------------------------------------------------|
  | Dense vectors  | 1024-dim semantic embeddings                    |
  | Sparse vectors | Lexical matching (BM25-like)                    |
  | ColBERT        | Token-level late interaction for reranking      |
  | Hybrid search  | Prefetch with dense+sparse, rerank with ColBERT |

  Usage:
  # Index all documents
  python index_qdrant.py --input eurlex.jsonl

  # Index with specific options
  python index_qdrant.py \
      --input eurlex.jsonl \
      --collection eurlex \
      --host localhost \
      --port 6333 \
      --batch-size 32 \
      --device cuda \
      --recreate

  # Limit documents (for testing)
  python index_qdrant.py --input eurlex.jsonl --limit 1000

  # Search mode
  python index_qdrant.py --search "GDPR data protection regulation"

  # Check collection info
  python index_qdrant.py --info

  Search architecture:
  Query → BGE-M3 → [Dense, Sparse, ColBERT vectors]
                          ↓
                ┌─────────┴─────────┐
                ↓                   ↓
           Sparse prefetch    Dense prefetch
                └─────────┬─────────┘
                          ↓
                ColBERT MaxSim reranking
                          ↓
                    Final results

  Dependencies:
  pip install FlagEmbedding qdrant-client tqdm

  Requirements:
  - Qdrant server running (docker run -p 6333:6333 qdrant/qdrant)
  - GPU recommended for BGE-M3 embeddings

"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_INPUT_FILE = Path("eurlex.jsonl")
DEFAULT_COLLECTION_NAME = "eurlex"
DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
EMBEDDING_DIM = 1024
DEFAULT_BATCH_SIZE = 32


@dataclass
class Document:
    """Represents a document to be indexed."""
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
        # Apple Silicon Mac
        if torch.backends.mps.is_available():
            return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class BGE_M3_Embedder:
    """BGE-M3 embedding model wrapper with Apple Silicon support."""

    def __init__(self, device: str = "auto", use_fp16: bool = True):
        """
        Initialize the BGE-M3 model.

        Args:
            device: Device to use ('cuda', 'cpu', 'mps', or 'auto')
            use_fp16: Use half precision for faster inference
        """
        if device == "auto":
            device = get_default_device()
            logger.info(f"Auto-detected device: {device}")

        self.device = device
        # FP16 not fully supported on MPS, use FP32
        self.use_fp16 = use_fp16 and device != "mps"
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading BGE-M3 model on {self.device}...")
            from FlagEmbedding import BGEM3FlagModel

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
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> Dict[str, Any]:
        """
        Encode texts into dense, sparse, and ColBERT vectors.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            max_length: Maximum sequence length

        Returns:
            Dict with 'dense', 'lexical_weights', and 'colbert_vecs'
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )

    def encode_single(self, text: str) -> Dict[str, Any]:
        """Encode a single text."""
        result = self.encode([text], batch_size=1)
        return {
            "dense": result["dense"][0],
            "lexical_weights": result["lexical_weights"][0],
            "colbert_vecs": result["colbert_vecs"][0],
        }


class OllamaEmbedder:
    """
    Ollama-based embedder for BGE-M3 (dense vectors only).

    Faster on Apple Silicon but doesn't support ColBERT/sparse vectors.
    Use this when you only need dense embeddings.

    Requires: ollama pull bge-m3
    """

    def __init__(self, model: str = "bge-m3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 8192,
    ) -> Dict[str, Any]:
        """
        Encode texts into dense vectors only (Ollama limitation).

        Note: ColBERT and sparse vectors are not available via Ollama.
        Returns dummy values for compatibility.
        """
        import requests
        import numpy as np

        dense_vectors = []

        for text in texts:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text[:max_length]},
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            dense_vectors.append(embedding)

        # Convert to numpy array
        dense_array = np.array(dense_vectors)

        # Ollama doesn't support sparse/ColBERT, return empty placeholders
        return {
            "dense": dense_array,
            "lexical_weights": [{} for _ in texts],  # Empty sparse
            "colbert_vecs": [dense_array[i:i+1] for i in range(len(texts))],  # Use dense as fallback
        }

    def encode_single(self, text: str) -> Dict[str, Any]:
        """Encode a single text."""
        result = self.encode([text], batch_size=1)
        return {
            "dense": result["dense"][0],
            "lexical_weights": result["lexical_weights"][0],
            "colbert_vecs": result["colbert_vecs"][0],
        }


class QdrantIndexer:
    """Qdrant indexer for EUR-Lex documents."""

    def __init__(
        self,
        host: str = DEFAULT_QDRANT_HOST,
        port: int = DEFAULT_QDRANT_PORT,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        """
        Initialize the Qdrant indexer.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self._client = None

    @property
    def client(self):
        """Lazy load Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
        return self._client

    def create_collection(self, recreate: bool = False):
        """
        Create the Qdrant collection with BGE-M3 vector configuration.

        Args:
            recreate: If True, delete existing collection first
        """
        from qdrant_client import models

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            if recreate:
                logger.info(f"Deleting existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                return

        logger.info(f"Creating collection: {self.collection_name}")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                # Dense vectors for semantic search
                "dense": models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
                # ColBERT multi-vectors for late interaction
                "colbert": models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            },
            sparse_vectors_config={
                # Sparse vectors for lexical matching
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=True)
                ),
            },
        )

        logger.info(f"Collection {self.collection_name} created")

    def _convert_sparse_vector(self, lexical_weights: Dict) -> "models.SparseVector":
        """
        Convert BGE-M3 lexical weights to Qdrant sparse vector format.

        Args:
            lexical_weights: Dict of token_id -> weight

        Returns:
            Qdrant SparseVector
        """
        from qdrant_client import models

        indices = []
        values = []

        for key, value in lexical_weights.items():
            weight = float(value)
            if weight > 0:
                # Handle string or int keys
                if isinstance(key, str) and key.isdigit():
                    indices.append(int(key))
                elif isinstance(key, int):
                    indices.append(key)
                else:
                    continue
                values.append(weight)

        return models.SparseVector(indices=indices, values=values)

    def index_documents(
        self,
        documents: List[Document],
        embedder: BGE_M3_Embedder,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Index documents into Qdrant.

        Args:
            documents: List of documents to index
            embedder: BGE-M3 embedder instance
            batch_size: Batch size for embedding and indexing
        """
        from qdrant_client import models

        total = len(documents)
        logger.info(f"Indexing {total} documents with batch size {batch_size}")

        indexed_count = 0
        error_count = 0

        # Process in batches
        for i in tqdm(range(0, total, batch_size), desc="Indexing", unit="batch"):
            batch_docs = documents[i:i + batch_size]
            texts = [doc.text for doc in batch_docs]

            try:
                # Generate embeddings for batch
                embeddings = embedder.encode(texts, batch_size=len(texts))

                # Create points
                points = []
                for j, doc in enumerate(batch_docs):
                    # Extract embeddings for this document
                    dense_vec = embeddings["dense"][j].tolist()
                    colbert_vecs = embeddings["colbert_vecs"][j].tolist()
                    sparse_vec = self._convert_sparse_vector(
                        embeddings["lexical_weights"][j]
                    )

                    # Build payload
                    payload = {
                        "uri": doc.uri,
                        "text": doc.text[:1000],  # Truncate for storage
                        "text_length": len(doc.text),
                    }
                    if doc.filename:
                        payload["filename"] = doc.filename
                    if doc.file_id:
                        payload["file_id"] = doc.file_id
                    if doc.metadata:
                        payload.update(doc.metadata)

                    points.append(
                        models.PointStruct(
                            id=indexed_count + j,
                            payload=payload,
                            vector={
                                "dense": dense_vec,
                                "colbert": colbert_vecs,
                                "sparse": sparse_vec,
                            },
                        )
                    )

                # Upsert batch
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

                indexed_count += len(batch_docs)

            except Exception as e:
                logger.error(f"Error indexing batch {i // batch_size}: {e}")
                error_count += len(batch_docs)

        logger.info(f"Indexed {indexed_count} documents, {error_count} errors")
        return indexed_count, error_count

    def search(
        self,
        query: str,
        embedder: BGE_M3_Embedder,
        limit: int = 10,
        use_hybrid: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using hybrid retrieval with ColBERT reranking.

        Args:
            query: Search query
            embedder: BGE-M3 embedder instance
            limit: Number of results to return
            use_hybrid: Use hybrid search with prefetch

        Returns:
            List of search results with scores
        """
        from qdrant_client import models

        # Encode query
        query_emb = embedder.encode_single(query)
        dense_vec = query_emb["dense"].tolist()
        colbert_vecs = query_emb["colbert_vecs"].tolist()
        sparse_vec = self._convert_sparse_vector(query_emb["lexical_weights"])

        if use_hybrid:
            # Hybrid search: prefetch with dense and sparse, rerank with ColBERT
            prefetch_limit = limit * 3  # Fetch more candidates for reranking

            prefetch = [
                models.Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=prefetch_limit,
                ),
            ]

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=colbert_vecs,
                using="colbert",
                with_payload=True,
                limit=limit,
            )
        else:
            # Simple dense search
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vec,
                using="dense",
                with_payload=True,
                limit=limit,
            )

        return [
            {
                "id": point.id,
                "score": point.score,
                "payload": point.payload,
            }
            for point in results.points
        ]

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status,
        }


def load_documents(input_file: Path, limit: Optional[int] = None) -> List[Document]:
    """
    Load documents from JSONL file.

    Args:
        input_file: Path to the JSONL file
        limit: Maximum number of documents to load (None for all)

    Returns:
        List of Document objects
    """
    documents = []

    with open(input_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            data = json.loads(line)

            # Create unique ID from available fields
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


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Index EUR-Lex documents into Qdrant with BGE-M3"
    )

    # Input/output arguments
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Input JSONL file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help=f"Qdrant collection name (default: {DEFAULT_COLLECTION_NAME})"
    )

    # Qdrant connection
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_QDRANT_HOST,
        help=f"Qdrant host (default: {DEFAULT_QDRANT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_QDRANT_PORT,
        help=f"Qdrant port (default: {DEFAULT_QDRANT_PORT})"
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for embedding/indexing (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to index (default: all)"
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
        help="Use Ollama for embeddings (faster on M1, but dense-only, no ColBERT)"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection if it exists"
    )

    # Modes
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search mode: provide a query instead of indexing"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show collection info and exit"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize components
    indexer = QdrantIndexer(
        host=args.host,
        port=args.port,
        collection_name=args.collection,
    )

    # Info mode
    if args.info:
        try:
            info = indexer.get_collection_info()
            print("\nCollection Info:")
            print(f"  Name: {info['name']}")
            print(f"  Points: {info['points_count']}")
            print(f"  Vectors: {info['vectors_count']}")
            print(f"  Status: {info['status']}")
        except Exception as e:
            print(f"Error getting collection info: {e}")
        return

    # Initialize embedder
    if args.ollama:
        logger.info("Using Ollama embedder (dense-only, no ColBERT/sparse)")
        embedder = OllamaEmbedder()
    else:
        embedder = BGE_M3_Embedder(device=args.device)

    # Search mode
    if args.search:
        logger.info(f"Searching for: {args.search}")
        results = indexer.search(
            query=args.search,
            embedder=embedder,
            limit=10,
            use_hybrid=True,
        )

        print(f"\nSearch Results for: '{args.search}'")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   URI: {result['payload'].get('uri', 'N/A')}")
            print(f"   Text: {result['payload'].get('text', '')[:200]}...")
        return

    # Index mode
    logger.info("Starting EUR-Lex indexing to Qdrant")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Qdrant: {args.host}:{args.port}")

    # Check input file exists
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Create collection
    indexer.create_collection(recreate=args.recreate)

    # Load documents
    documents = load_documents(args.input, limit=args.limit)

    if not documents:
        logger.error("No documents to index")
        return

    # Index documents
    indexed, errors = indexer.index_documents(
        documents=documents,
        embedder=embedder,
        batch_size=args.batch_size,
    )

    # Summary
    print("\n" + "=" * 60)
    print("Indexing Complete")
    print("=" * 60)
    print(f"Documents indexed: {indexed}")
    print(f"Errors: {errors}")

    # Show collection info
    info = indexer.get_collection_info()
    print(f"\nCollection '{info['name']}':")
    print(f"  Total points: {info['points_count']}")
    print(f"  Status: {info['status']}")


if __name__ == "__main__":
    main()
