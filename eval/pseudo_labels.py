"""
Phase 3: Hard Negative Mining and Pseudo-Label Generation.

This module implements the GPL (Generative Pseudo Labeling) methodology:
1. Retrieve candidate documents via MCP server
2. Generate pseudo-relevance labels with cross-encoder
3. Create BEIR-compatible evaluation dataset
"""
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from config import pseudo_label_config, OUTPUT_DIR
from corpus import Document
from query_generation import GeneratedQuery

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieved document with score."""

    celex: str
    text: str
    title: str
    retrieval_score: float
    rank: int


@dataclass
class PseudoLabel:
    """Represents a query-document relevance label."""

    query_id: str
    doc_id: str
    relevance_score: float  # Continuous 0.0-1.0
    relevance_grade: int  # Discrete 0-3
    is_source_doc: bool  # Whether this is the source document for the query


class CrossEncoderScorer:
    """Scores query-document pairs using a cross-encoder model."""

    def __init__(self, config=pseudo_label_config):
        self.config = config
        self._model = None

    @property
    def model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.config.cross_encoder_model)
            logger.info(f"Loaded cross-encoder: {self.config.cross_encoder_model}")
        return self._model

    def score_pairs(
        self,
        query: str,
        documents: List[Tuple[str, str]],  # List of (doc_id, doc_text)
    ) -> List[Tuple[str, float]]:
        """
        Score query-document pairs using cross-encoder.

        Args:
            query: Query text
            documents: List of (doc_id, doc_text) tuples

        Returns:
            List of (doc_id, score) tuples
        """
        if not documents:
            return []

        pairs = [(query, doc_text) for _, doc_text in documents]
        scores = self.model.predict(pairs, batch_size=self.config.cross_encoder_batch_size)

        return [(doc_id, float(score)) for (doc_id, _), score in zip(documents, scores)]

    def score_to_grade(self, score: float) -> int:
        """Convert continuous score to graded relevance (0-3)."""
        thresholds = self.config.relevance_thresholds
        for grade in [3, 2, 1]:
            if score >= thresholds[grade]:
                return grade
        return 0


class PseudoLabelGenerator:
    """Generates pseudo-labels for query-document pairs."""

    def __init__(self, config=pseudo_label_config):
        self.config = config
        self.scorer = CrossEncoderScorer(config)
        self._mcp_client = None

    @property
    def mcp_client(self):
        """Lazy load MCP client."""
        if self._mcp_client is None:
            from mcp_client import create_client

            # create QueryConversionLLMConfig with base_url pointing to local MCP server
            self._mcp_client = create_client("http", base_url="http://0.0.0.0:9001/")


        return self._mcp_client

    async def retrieve_candidates(
        self, query: str, k: int = 100
    ) -> List[RetrievalResult]:
        """Retrieve candidate documents from MCP server."""
        try:
            eurlex_query = await self.mcp_client.natural_language_to_query(query, max_iterations=5)

            # Then search
            results = await self.mcp_client.search(eurlex_query, use_llm=False, limit=k)
            #results = await self.mcp_client.search(query=query, limit=k)
            return [
                RetrievalResult(
                    celex=doc["celex"],
                    text=doc.get("text", ""),
                    title=doc.get("title", ""),
                    retrieval_score=doc.get("score", 0.0),
                    rank=i,
                )
                for i, doc in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"MCP retrieval failed for query: {e}")
            return []

    def generate_labels_for_query(
        self,
        query: GeneratedQuery,
        candidates: List[RetrievalResult],
    ) -> List[PseudoLabel]:
        """Generate pseudo-labels for a query and its candidate documents."""
        if not candidates:
            return []

        # Prepare document pairs for scoring
        doc_pairs = [(c.celex, c.text[:512]) for c in candidates]

        # Score with cross-encoder
        scores = self.scorer.score_pairs(query.text, doc_pairs)

        # Create pseudo-labels
        labels = []
        for celex, score in scores:
            grade = self.scorer.score_to_grade(score)
            labels.append(
                PseudoLabel(
                    query_id=query.query_id,
                    doc_id=celex,
                    relevance_score=score,
                    relevance_grade=grade,
                    is_source_doc=(celex == query.source_celex),
                )
            )

        return labels

    async def generate_all_labels(
        self,
        queries: List[GeneratedQuery],
        show_progress: bool = True,
    ) -> Dict[str, List[PseudoLabel]]:
        """Generate pseudo-labels for all queries."""
        all_labels = {}

        iterator = tqdm(queries, desc="Generating pseudo-labels") if show_progress else queries

        for query in iterator:
            # Retrieve candidates
            candidates = await self.retrieve_candidates(
                query.text, k=self.config.retrieval_top_k
            )

            # Generate labels
            labels = self.generate_labels_for_query(query, candidates)
            all_labels[query.query_id] = labels

        return all_labels


class BEIRDatasetBuilder:
    """Builds BEIR-compatible evaluation dataset."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR / "beir_dataset"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_dataset(
        self,
        corpus: Dict[str, Document],
        queries: List[GeneratedQuery],
        labels: Dict[str, List[PseudoLabel]],
        use_graded: bool = True,
    ) -> Path:
        """
        Build BEIR-compatible dataset structure.

        Structure:
            beir_dataset/
            ├── corpus.jsonl
            ├── queries.jsonl
            └── qrels/
                └── test.tsv
        """
        # Write corpus.jsonl
        corpus_path = self.output_dir / "corpus.jsonl"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for celex, doc in corpus.items():
                record = {
                    "_id": celex,
                    "title": doc.title,
                    "text": doc.text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(corpus)} documents to {corpus_path}")

        # Write queries.jsonl
        queries_path = self.output_dir / "queries.jsonl"
        with open(queries_path, "w", encoding="utf-8") as f:
            for query in queries:
                record = {
                    "_id": query.query_id,
                    "text": query.text,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(queries)} queries to {queries_path}")

        # Write qrels
        qrels_dir = self.output_dir / "qrels"
        qrels_dir.mkdir(exist_ok=True)
        qrels_path = qrels_dir / "test.tsv"

        with open(qrels_path, "w", encoding="utf-8") as f:
            f.write("query-id\tcorpus-id\tscore\n")
            for query_id, query_labels in labels.items():
                for label in query_labels:
                    score = label.relevance_grade if use_graded else (
                        1 if label.relevance_score >= 0.5 else 0
                    )
                    if score > 0:  # Only write positive labels
                        f.write(f"{query_id}\t{label.doc_id}\t{score}\n")

        logger.info(f"Wrote qrels to {qrels_path}")

        return self.output_dir

    def load_qrels(self) -> Dict[str, Dict[str, int]]:
        """Load qrels from TSV file."""
        qrels_path = self.output_dir / "qrels" / "test.tsv"
        qrels = {}

        with open(qrels_path, "r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    query_id, doc_id, score = parts
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][doc_id] = int(score)

        return qrels


class HardNegativeMiner:
    """Mines hard negatives for contrastive learning."""

    def __init__(self, config=pseudo_label_config):
        self.config = config
        self.scorer = CrossEncoderScorer(config)

    def mine_hard_negatives(
        self,
        query: GeneratedQuery,
        candidates: List[RetrievalResult],
        n_hard_negatives: int = 5,
    ) -> List[str]:
        """
        Find hard negatives: documents that rank high but aren't relevant.

        Args:
            query: The query
            candidates: Retrieved candidate documents
            n_hard_negatives: Number of hard negatives to return

        Returns:
            List of CELEX IDs for hard negative documents
        """
        # Score all candidates
        doc_pairs = [(c.celex, c.text[:512]) for c in candidates]
        scores = self.scorer.score_pairs(query.text, doc_pairs)

        # Find hard negatives: low cross-encoder score but high retrieval rank
        hard_negatives = []
        for celex, score in scores:
            if score < 0.3 and celex != query.source_celex:
                # Find the candidate to get its rank
                for c in candidates:
                    if c.celex == celex:
                        hard_negatives.append((celex, c.rank, score))
                        break

        # Sort by retrieval rank (lower is "harder")
        hard_negatives.sort(key=lambda x: x[1])

        return [celex for celex, _, _ in hard_negatives[:n_hard_negatives]]


class PseudoLabelPipeline:
    """Main pipeline for pseudo-label generation."""

    def __init__(self, config=pseudo_label_config):
        self.config = config
        self.generator = PseudoLabelGenerator(config)
        self.dataset_builder = BEIRDatasetBuilder()
        self.hard_negative_miner = HardNegativeMiner(config)

    async def run(
        self,
        corpus: Dict[str, Document],
        queries: List[GeneratedQuery],
    ) -> Path:
        """Run the full pseudo-labeling pipeline."""
        logger.info("Starting pseudo-label generation pipeline...")

        # Generate labels
        labels = await self.generator.generate_all_labels(queries)

        # Build BEIR dataset
        dataset_path = self.dataset_builder.build_dataset(corpus, queries, labels)

        logger.info(f"Pipeline complete. Dataset saved to {dataset_path}")
        return dataset_path

    def save_labels(
        self,
        labels: Dict[str, List[PseudoLabel]],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save raw pseudo-labels to JSON."""
        output_path = output_path or OUTPUT_DIR / "pseudo_labels.json"

        serializable = {}
        for query_id, query_labels in labels.items():
            serializable[query_id] = [asdict(label) for label in query_labels]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved pseudo-labels to {output_path}")
        return output_path


async def main():
    """Main entry point for pseudo-label generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from corpus import CorpusManager
    from query_generation import QueryGenerationPipeline

    # Load corpus and queries
    logger.info("Loading corpus and queries...")
    corpus_manager = CorpusManager()
    corpus = corpus_manager.load_corpus()

    query_pipeline = QueryGenerationPipeline()
    queries = query_pipeline.load_queries()

    # Run pseudo-labeling
    pipeline = PseudoLabelPipeline()
    dataset_path = await pipeline.run(corpus, queries)

    print(f"\nPseudo-labeling complete!")
    print(f"Dataset saved to: {dataset_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
