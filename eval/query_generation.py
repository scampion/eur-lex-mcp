"""
Phase 2: Synthetic Query Generation Pipeline.

This module generates diverse, realistic legal queries from sampled documents
using three complementary approaches:
- Approach A: Aspect-guided generation with local LLM
- Approach B: DocT5Query for keyword-style queries
- Approach C: Query type diversification
"""
import hashlib
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from openai import OpenAI

import requests
from tqdm import tqdm

from config import query_gen_config, OUTPUT_DIR, CACHE_DIR
from corpus import Document


class QueryCache:
    """Cache for generated queries, keyed by model configuration and document."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir / "query_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, List[Dict]] = {}
        self._index_file = self.cache_dir / "cache_index.json"
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2)

    def _make_cache_key(self, base_url: str, model: str, celex: str, method: str) -> str:
        """Generate cache key from configuration and document ID."""
        key_string = f"{base_url}|{model}|{celex}|{method}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:32]

    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for a key."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self, base_url: str, model: str, celex: str, method: str = "llm"
    ) -> Optional[List[Dict]]:
        """
        Get cached queries for a document.

        Args:
            base_url: API base URL
            model: Model name
            celex: Document CELEX number
            method: Generation method ('llm' or 't5')

        Returns:
            List of query dicts if cached, None otherwise
        """
        cache_key = self._make_cache_key(base_url, model, celex, method)

        # Check memory cache first
        if cache_key in self._memory_cache:
            return self._memory_cache[cache_key]

        # Check disk cache
        cache_file = self._get_cache_file(cache_key)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    queries = pickle.load(f)
                self._memory_cache[cache_key] = queries
                return queries
            except (pickle.PickleError, IOError) as e:
                logging.warning(f"Failed to load cache for {celex}: {e}")
                return None

        return None

    def set(
        self,
        base_url: str,
        model: str,
        celex: str,
        queries: List[Dict],
        method: str = "llm",
    ):
        """
        Cache queries for a document.

        Args:
            base_url: API base URL
            model: Model name
            celex: Document CELEX number
            queries: List of query dicts to cache
            method: Generation method ('llm' or 't5')
        """
        cache_key = self._make_cache_key(base_url, model, celex, method)

        # Save to memory cache
        self._memory_cache[cache_key] = queries

        # Save to disk
        cache_file = self._get_cache_file(cache_key)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(queries, f)

            # Update index
            self._index[cache_key] = {
                "base_url": base_url,
                "model": model,
                "celex": celex,
                "method": method,
                "query_count": len(queries),
            }
            self._save_index()
        except (pickle.PickleError, IOError) as e:
            logging.warning(f"Failed to save cache for {celex}: {e}")

    def has(self, base_url: str, model: str, celex: str, method: str = "llm") -> bool:
        """Check if queries are cached for a document."""
        cache_key = self._make_cache_key(base_url, model, celex, method)
        if cache_key in self._memory_cache:
            return True
        return self._get_cache_file(cache_key).exists()

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "total_cached": len(self._index),
            "memory_cached": len(self._memory_cache),
            "cache_dir": str(self.cache_dir),
        }

    def clear(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Clear cache entries.

        Args:
            base_url: If provided, only clear entries for this base_url
            model: If provided, only clear entries for this model
        """
        keys_to_remove = []
        for key, info in self._index.items():
            if base_url and info.get("base_url") != base_url:
                continue
            if model and info.get("model") != model:
                continue
            keys_to_remove.append(key)

        for key in keys_to_remove:
            cache_file = self._get_cache_file(key)
            if cache_file.exists():
                cache_file.unlink()
            if key in self._memory_cache:
                del self._memory_cache[key]
            if key in self._index:
                del self._index[key]

        self._save_index()
        logging.info(f"Cleared {len(keys_to_remove)} cache entries")

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of generated queries."""

    KEYWORD = "keyword"
    PHRASE = "phrase"
    NATURAL_LANGUAGE = "natural_language"
    BOOLEAN = "boolean"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    INTERPRETIVE = "interpretive"


@dataclass
class GeneratedQuery:
    """Represents a generated query with metadata."""

    query_id: str
    text: str
    source_celex: str
    query_type: str
    generation_method: str  # "llm", "doc2query", "manual"
    aspect: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# Prompt template for aspect-guided generation
ASPECT_GENERATION_PROMPT = """You are a legal research assistant. Analyze this EUR-Lex document and:
1. Identify 3-5 distinct legal aspects, concepts, or provisions covered
2. For each aspect, generate a realistic search query a legal professional might use

Document Type: {doc_type}
CELEX Number: {celex}
Title: {title}
Content (first 2000 chars): {content}

For each aspect, generate varied query types:
- Factual query: seeking specific provisions or requirements
- Procedural query: how to comply or implement
- Interpretive query: meaning or scope of legal concepts

Output your response as JSON with the following structure:
{{
    "aspects": [
        {{
            "name": "aspect description",
            "factual": "factual query text",
            "procedural": "procedural query text",
            "interpretive": "interpretive query text"
        }}
    ]
}}

Only output valid JSON, no other text."""


class LLMQueryGenerator:
    """Approach A: Aspect-guided query generation with local LLM."""

    def __init__(self, config=query_gen_config):
        self.config = config
        self.base_url = config.llm_base_url
        self.model = config.llm_model
        self.client = OpenAI(base_url=self.base_url, api_key=os.environ['OLLAMA_TOKEN'])

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for text generation."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "top_p": 0.9,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OLLAMA_TOKEN']}"
        }
        try:
            response = self.client.chat.completions.create(
               model=self.model,
               messages=[{"role": "user", "content": prompt}],
               temperature=0.7,
               top_p=0.9,
            )
            #response = requests.post(url, payload, headers=headers, timeout=120)
            #print(response.json())
            return response.choices[0].message.content # response.json()["response"]
        except requests.RequestException as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except OpenAI.exceptions.OpenAIException as e:
            logger.error(f"Ollama OpenAI error: {e}")
            raise

    def generate_queries(self, document: Document) -> List[GeneratedQuery]:
        """Generate queries for a document using aspect-guided approach."""
        prompt = ASPECT_GENERATION_PROMPT.format(
            doc_type=document.doc_type,
            celex=document.celex,
            title=document.title,
            content=document.text[: self.config.max_content_chars],
        )

        try:
            response = self._call_ollama(prompt)
            queries = self._parse_llm_response(response, document)
            return queries
        except Exception as e:
            logger.error(f"Failed to generate queries for {document.celex}: {e}")
            return []

    def _parse_llm_response(
        self, response: str, document: Document
    ) -> List[GeneratedQuery]:
        """Parse LLM JSON response into GeneratedQuery objects."""
        queries = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning(f"No JSON found in LLM response for {document.celex}")
            return queries

        try:
            data = json.loads(json_match.group())
            aspects = data.get("aspects", [])

            for i, aspect in enumerate(aspects):
                aspect_name = aspect.get("name", f"aspect_{i}")

                for query_type in ["factual", "procedural", "interpretive"]:
                    if query_type in aspect and aspect[query_type]:
                        query_id = f"{document.celex}_llm_{i}_{query_type}"
                        queries.append(
                            GeneratedQuery(
                                query_id=query_id,
                                text=aspect[query_type],
                                source_celex=document.celex,
                                query_type=query_type,
                                generation_method="llm",
                                aspect=aspect_name,
                            )
                        )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON for {document.celex}: {e}")

        return queries


class DocT5QueryGenerator:
    """Approach B: DocT5Query for keyword-style queries."""

    def __init__(self, config=query_gen_config):
        self.config = config
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        """Lazy load the T5 model."""
        if self._model is None:
            from transformers import T5ForConditionalGeneration

            self._model = T5ForConditionalGeneration.from_pretrained(
                self.config.doc2query_model
            )
        return self._model

    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            from transformers import T5Tokenizer

            self._tokenizer = T5Tokenizer.from_pretrained(
                self.config.doc2query_model
            )
        return self._tokenizer

    def generate_queries(self, document: Document) -> List[GeneratedQuery]:
        """Generate keyword-style queries using DocT5Query."""
        queries = []

        # Truncate text for T5 input
        text = document.text[:512]

        try:
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.config.doc2query_max_length,
                do_sample=True,
                top_p=self.config.doc2query_top_p,
                temperature=self.config.doc2query_temperature,
                num_return_sequences=self.config.doc2query_num_queries,
            )

            for i, output in enumerate(outputs):
                query_text = self.tokenizer.decode(output, skip_special_tokens=True)
                query_id = f"{document.celex}_t5_{i}"

                queries.append(
                    GeneratedQuery(
                        query_id=query_id,
                        text=query_text,
                        source_celex=document.celex,
                        query_type=QueryType.KEYWORD.value,
                        generation_method="doc2query",
                    )
                )

        except Exception as e:
            logger.error(f"DocT5Query generation failed for {document.celex}: {e}")

        return queries


class QueryDiversifier:
    """Approach C: Ensure query type diversity."""

    def __init__(self, config=query_gen_config):
        self.config = config

    def diversify_queries(
        self, queries: List[GeneratedQuery]
    ) -> List[GeneratedQuery]:
        """
        Ensure coverage across query complexity levels.
        Tags and filters queries to match target distribution.
        """
        # Classify queries by type
        type_counts = {qt.value: 0 for qt in QueryType}

        for query in queries:
            detected_type = self._classify_query(query.text)
            query.query_type = detected_type
            type_counts[detected_type] = type_counts.get(detected_type, 0) + 1

        logger.info(f"Query type distribution: {type_counts}")
        return queries

    def _classify_query(self, query_text: str) -> str:
        """Classify a query into a type based on patterns."""
        query_lower = query_text.lower()

        # Boolean patterns
        if any(op in query_lower for op in [" and ", " or ", " not "]):
            return QueryType.BOOLEAN.value

        # Natural language (question words)
        if any(
            query_lower.startswith(w)
            for w in ["what", "how", "when", "where", "why", "which", "who", "is", "are", "can", "does"]
        ):
            return QueryType.NATURAL_LANGUAGE.value

        # Phrase (quoted or multi-word descriptive)
        if '"' in query_text or len(query_text.split()) >= 4:
            return QueryType.PHRASE.value

        # Default to keyword
        return QueryType.KEYWORD.value


class QueryFilterPipeline:
    """Three-stage filtering pipeline for quality control."""

    def __init__(self, config=query_gen_config):
        self.config = config
        self._cross_encoder = None

    @property
    def cross_encoder(self):
        """Lazy load cross-encoder for scoring."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
        return self._cross_encoder

    def filter_queries(
        self,
        queries: List[GeneratedQuery],
        documents: Dict[str, Document],
    ) -> List[GeneratedQuery]:
        """Apply three-stage filtering pipeline."""
        logger.info(f"Starting with {len(queries)} queries")

        # Stage 1: Length filtering
        filtered = self._filter_by_length(queries)
        logger.info(f"After length filtering: {len(filtered)} queries")

        # Stage 2: Cross-encoder scoring
        filtered = self._filter_by_relevance(filtered, documents)
        logger.info(f"After relevance filtering: {len(filtered)} queries")

        return filtered

    def _filter_by_length(
        self, queries: List[GeneratedQuery]
    ) -> List[GeneratedQuery]:
        """Stage 1: Keep queries with 3-50 tokens."""
        filtered = []
        for query in queries:
            token_count = len(query.text.split())
            if self.config.min_query_tokens <= token_count <= self.config.max_query_tokens:
                filtered.append(query)
        return filtered

    def _filter_by_relevance(
        self,
        queries: List[GeneratedQuery],
        documents: Dict[str, Document],
    ) -> List[GeneratedQuery]:
        """Stage 2: Keep top 70% by cross-encoder relevance score."""
        scored_queries = []

        for query in tqdm(queries, desc="Scoring queries"):
            if query.source_celex not in documents:
                continue

            doc = documents[query.source_celex]
            score = self.cross_encoder.predict([(query.text, doc.text[:512])])[0]
            query.confidence = float(score)
            scored_queries.append(query)

        # Sort by score and keep top percentage
        scored_queries.sort(key=lambda q: q.confidence or 0, reverse=True)
        cutoff = int(len(scored_queries) * self.config.cross_encoder_top_percent)
        return scored_queries[:cutoff]


class QueryGenerationPipeline:
    """Main pipeline orchestrating all query generation approaches."""

    def __init__(self, config=query_gen_config, use_cache: bool = True):
        self.config = config
        self.llm_generator = LLMQueryGenerator(config)
        self.t5_generator = DocT5QueryGenerator(config)
        self.diversifier = QueryDiversifier(config)
        self.filter_pipeline = QueryFilterPipeline(config)
        self.use_cache = use_cache
        self.cache = QueryCache() if use_cache else None

    def _queries_to_dicts(self, queries: List[GeneratedQuery]) -> List[Dict]:
        """Convert GeneratedQuery objects to dicts for caching."""
        return [q.to_dict() for q in queries]

    def _dicts_to_queries(self, dicts: List[Dict]) -> List[GeneratedQuery]:
        """Convert cached dicts back to GeneratedQuery objects."""
        return [
            GeneratedQuery(
                query_id=d["query_id"],
                text=d["text"],
                source_celex=d["source_celex"],
                query_type=d["query_type"],
                generation_method=d["generation_method"],
                aspect=d.get("aspect"),
                confidence=d.get("confidence"),
            )
            for d in dicts
        ]

    def generate_all_queries(
        self,
        documents: Dict[str, Document],
        use_llm: bool = True,
        use_t5: bool = True,
    ) -> List[GeneratedQuery]:
        """Generate queries using all approaches with caching support."""
        all_queries = []
        cache_hits = 0
        cache_misses = 0

        # Get cache keys from config
        base_url = self.config.llm_base_url
        llm_model = self.config.llm_model
        t5_model = self.config.doc2query_model

        for celex, doc in tqdm(documents.items(), desc="Generating queries"):
            doc_queries = []

            # Approach A: LLM-based generation
            if use_llm:
                # Check cache first
                cached = None
                if self.cache:
                    cached = self.cache.get(base_url, llm_model, celex, method="llm")

                if cached is not None:
                    # Cache hit - restore from cache
                    llm_queries = self._dicts_to_queries(cached)
                    doc_queries.extend(llm_queries)
                    cache_hits += 1
                else:
                    # Cache miss - generate and cache
                    try:
                        llm_queries = self.llm_generator.generate_queries(doc)
                        doc_queries.extend(llm_queries)
                        cache_misses += 1

                        # Save to cache
                        if self.cache and llm_queries:
                            self.cache.set(
                                base_url,
                                llm_model,
                                celex,
                                self._queries_to_dicts(llm_queries),
                                method="llm",
                            )
                    except Exception as e:
                        logger.warning(f"LLM generation failed for {celex}: {e}")

            # Approach B: DocT5Query
            if use_t5:
                # Check cache first
                cached = None
                if self.cache:
                    cached = self.cache.get(base_url, t5_model, celex, method="t5")

                if cached is not None:
                    # Cache hit
                    t5_queries = self._dicts_to_queries(cached)
                    doc_queries.extend(t5_queries)
                    cache_hits += 1
                else:
                    # Cache miss
                    try:
                        t5_queries = self.t5_generator.generate_queries(doc)
                        doc_queries.extend(t5_queries)
                        cache_misses += 1

                        # Save to cache
                        if self.cache and t5_queries:
                            self.cache.set(
                                base_url,
                                t5_model,
                                celex,
                                self._queries_to_dicts(t5_queries),
                                method="t5",
                            )
                    except Exception as e:
                        logger.warning(f"T5 generation failed for {celex}: {e}")

            all_queries.extend(doc_queries)

        # Log cache statistics
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            logger.info(
                f"Cache stats: {cache_hits} hits, {cache_misses} misses "
                f"({cache_hits / total_requests * 100:.1f}% hit rate)"
            )
        if self.cache:
            logger.info(f"Cache index: {self.cache.get_stats()}")

        # Approach C: Diversification
        all_queries = self.diversifier.diversify_queries(all_queries)

        # Quality filtering
        filtered_queries = self.filter_pipeline.filter_queries(all_queries, documents)

        return filtered_queries

    def clear_cache(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """Clear the query cache."""
        if self.cache:
            self.cache.clear(base_url=base_url, model=model)
            logger.info("Query cache cleared")

    def save_queries(
        self, queries: List[GeneratedQuery], output_path: Optional[Path] = None
    ) -> Path:
        """Save generated queries in BEIR-compatible format."""
        output_dir = OUTPUT_DIR / "queries"
        output_dir.mkdir(exist_ok=True)
        output_path = output_path or output_dir / "queries.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for query in queries:
                record = {
                    "_id": query.query_id,
                    "text": query.text,
                    "metadata": {
                        "source_celex": query.source_celex,
                        "query_type": query.query_type,
                        "generation_method": query.generation_method,
                        "aspect": query.aspect,
                        "confidence": query.confidence,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(queries)} queries to {output_path}")
        return output_path

    def load_queries(self, input_path: Optional[Path] = None) -> List[GeneratedQuery]:
        """Load queries from JSONL file."""
        input_path = input_path or OUTPUT_DIR / "queries" / "queries.jsonl"
        queries = []

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                queries.append(
                    GeneratedQuery(
                        query_id=record["_id"],
                        text=record["text"],
                        source_celex=record["metadata"]["source_celex"],
                        query_type=record["metadata"]["query_type"],
                        generation_method=record["metadata"]["generation_method"],
                        aspect=record["metadata"].get("aspect"),
                        confidence=record["metadata"].get("confidence"),
                    )
                )

        return queries


def main():
    """Main entry point for query generation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from corpus import CorpusManager

    # Load corpus
    logger.info("Loading corpus...")
    corpus_manager = CorpusManager()
    documents = corpus_manager.load_corpus()

    # Generate queries
    logger.info("Generating queries...")
    pipeline = QueryGenerationPipeline()

    # For initial development, you may want to disable LLM generation
    queries = pipeline.generate_all_queries(
        documents,
        use_llm=True,   # Requires Ollama running
        use_t5=False,    # Requires transformers
    )

    # Save queries
    pipeline.save_queries(queries)

    print(f"\nQuery generation complete!")
    print(f"Total queries: {len(queries)}")

    # Statistics by type
    type_counts = {}
    for q in queries:
        type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1
    print("By type:", type_counts)


if __name__ == "__main__":
    main()
