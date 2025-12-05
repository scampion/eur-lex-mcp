"""
Phase 1: Corpus Acquisition and Stratified Document Sampling.

This module handles downloading EUR-Lex documents from the HuggingFace dataset
and performing stratified sampling across document types.
"""
import json
import lzma
import logging
import random
from pathlib import Path
from typing import Dict, Generator, List, Optional
from dataclasses import dataclass, asdict

import requests
from tqdm import tqdm

from config import corpus_config, DATA_DIR, OUTPUT_DIR

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a EUR-Lex document."""

    celex: str
    language: str
    date: str
    title: str
    text: str
    doc_type: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict, doc_type: str) -> "Document":
        return cls(
            celex=data.get("celex", ""),
            language=data["language"],
            date=data["date"],
            title=data.get("title", ""),
            text=data["text"],
            doc_type=doc_type,
        )


class CorpusDownloader:
    """Downloads EUR-Lex resources from HuggingFace."""

    RESOURCES = [
        "caselaw",
        "decision",
        "directive",
        "intagr",
        "proposal",
        "recommendation",
        "regulation",
    ]

    def __init__(self, config=corpus_config):
        self.config = config
        self.data_dir = DATA_DIR
        self.data_dir.mkdir(exist_ok=True)

    def download_resource(self, resource: str, language: str = "en") -> Path:
        """Download a single resource file."""
        url = f"{self.config.hf_dataset_base_url}/{language}/{resource}.jsonl.xz"
        output_path = self.data_dir / f"{resource}_{language}.jsonl.xz"

        if output_path.exists():
            logger.info(f"Skipping {resource} in {language}, file already exists.")
            return output_path

        logger.info(f"Downloading {resource} in {language}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        logger.info(f"Downloaded {resource} in {language}")
        return output_path

    def download_all(self, language: str = "en") -> Dict[str, Path]:
        """Download all resources."""
        paths = {}
        for resource in self.RESOURCES:
            paths[resource] = self.download_resource(resource, language)
        return paths

    def iter_documents(
        self, resource: str, language: str = "en"
    ) -> Generator[Document, None, None]:
        """Iterate over documents in a resource file."""
        file_path = self.data_dir / f"{resource}_{language}.jsonl.xz"

        if not file_path.exists():
            self.download_resource(resource, language)

        with lzma.open(file_path, "rt", encoding="utf-8") as f:
            # parse each line with tqdm
            for line in tqdm(f, desc=f"Reading {resource}"):
                data = json.loads(line)
                try:
                    yield Document.from_dict(data, doc_type=resource)
                except KeyError as e:
                    logger.debug(f"Missing key in document: {e} - skipping document.")
                    continue

    def count_documents(self, resource: str, language: str = "en") -> int:
        """Count documents in a resource file."""
        count = 0
        for _ in self.iter_documents(resource, language):
            count += 1
        return count


class StratifiedSampler:
    """Performs stratified sampling across document types."""

    def __init__(self, downloader: CorpusDownloader, config=corpus_config):
        self.downloader = downloader
        self.config = config
        self.random_state = random.Random(42)

    def sample_documents(
        self,
        targets: Optional[Dict[str, int]] = None,
        language: str = "en",
    ) -> Dict[str, List[Document]]:
        """
        Sample documents with stratification by document type.

        Args:
            targets: Dict mapping doc_type to target sample size.
                    If None, uses config defaults.
            language: Language code.

        Returns:
            Dict mapping doc_type to list of sampled documents.
        """
        targets = targets or self.config.doc_type_targets
        sampled = {}

        for doc_type, target_count in targets.items():
            logger.info(f"Sampling {target_count} documents of type '{doc_type}'...")

            # Collect all documents of this type
            all_docs = list(self.downloader.iter_documents(doc_type, language))
            total_available = len(all_docs)

            if total_available < self.config.min_docs_per_type:
                logger.warning(
                    f"Only {total_available} docs available for {doc_type}, "
                    f"below minimum {self.config.min_docs_per_type}"
                )

            # Sample with replacement if needed
            if total_available >= target_count:
                sampled[doc_type] = self.random_state.sample(all_docs, target_count)
            else:
                logger.warning(
                    f"Only {total_available} docs for {doc_type}, "
                    f"using all available (target was {target_count})"
                )
                sampled[doc_type] = all_docs

            logger.info(f"Sampled {len(sampled[doc_type])} documents for {doc_type}")

        return sampled

    def get_statistics(self, sampled: Dict[str, List[Document]]) -> Dict:
        """Get statistics about the sampled corpus."""
        stats = {
            "total_documents": sum(len(docs) for docs in sampled.values()),
            "by_type": {},
        }

        for doc_type, docs in sampled.items():
            text_lengths = [len(d.text) for d in docs]
            stats["by_type"][doc_type] = {
                "count": len(docs),
                "avg_text_length": sum(text_lengths) / len(text_lengths) if docs else 0,
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
            }

        return stats


class CorpusManager:
    """High-level interface for corpus management."""

    def __init__(self, config=corpus_config):
        self.config = config
        self.downloader = CorpusDownloader(config)
        self.sampler = StratifiedSampler(self.downloader, config)

    def prepare_corpus(
        self,
        targets: Optional[Dict[str, int]] = None,
        save: bool = True,
    ) -> Dict[str, List[Document]]:
        """
        Download, sample, and optionally save the evaluation corpus.

        Args:
            targets: Custom sampling targets per document type.
            save: Whether to save the corpus to disk.

        Returns:
            Dict of sampled documents by type.
        """
        # Download all resources
        logger.info("Downloading EUR-Lex resources...")
        self.downloader.download_all(self.config.language)

        # Perform stratified sampling
        logger.info("Performing stratified sampling...")
        sampled = self.sampler.sample_documents(targets, self.config.language)

        # Get and log statistics
        stats = self.sampler.get_statistics(sampled)
        logger.info(f"Corpus statistics: {json.dumps(stats, indent=2)}")

        if save:
            self.save_corpus(sampled, stats)

        return sampled

    def save_corpus(
        self,
        corpus: Dict[str, List[Document]],
        stats: Optional[Dict] = None,
    ) -> Path:
        """Save corpus to JSONL files in BEIR-compatible format."""
        output_dir = OUTPUT_DIR / "corpus"
        output_dir.mkdir(exist_ok=True)

        # Save corpus.jsonl (BEIR format)
        corpus_path = output_dir / "corpus.jsonl"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for doc_type, docs in corpus.items():
                for doc in docs:
                    record = {
                        "_id": doc.celex,
                        "title": doc.title,
                        "text": doc.text,
                        "metadata": {
                            "doc_type": doc.doc_type,
                            "date": doc.date,
                            "language": doc.language,
                        },
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved corpus to {corpus_path}")

        # Save statistics
        if stats:
            stats_path = output_dir / "corpus_stats.json"
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved statistics to {stats_path}")

        return corpus_path

    def load_corpus(self) -> Dict[str, Document]:
        """Load corpus from saved JSONL file."""
        corpus_path = OUTPUT_DIR / "corpus" / "corpus.jsonl"
        corpus = {}

        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc = Document(
                    celex=record["_id"],
                    language=record["metadata"]["language"],
                    date=record["metadata"]["date"],
                    title=record["title"],
                    text=record["text"],
                    doc_type=record["metadata"]["doc_type"],
                )
                corpus[doc.celex] = doc

        return corpus


def main():
    """Main entry point for corpus preparation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    manager = CorpusManager()
    corpus = manager.prepare_corpus()

    print(f"\nCorpus preparation complete!")
    print(f"Total documents: {sum(len(docs) for docs in corpus.values())}")
    for doc_type, docs in corpus.items():
        print(f"  {doc_type}: {len(docs)} documents")


if __name__ == "__main__":
    main()
