"""
Phase 2: Synthetic Query Generation Pipeline.

This module generates diverse, realistic legal queries from sampled documents
using three complementary approaches:
- Approach A: Aspect-guided generation with local LLM
- Approach B: DocT5Query for keyword-style queries
- Approach C: Query type diversification
"""
import json
import logging
import os
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

    def __init__(self, config=query_gen_config):
        self.config = config
        self.llm_generator = LLMQueryGenerator(config)
        self.t5_generator = DocT5QueryGenerator(config)
        self.diversifier = QueryDiversifier(config)
        self.filter_pipeline = QueryFilterPipeline(config)

    def generate_all_queries(
        self,
        documents: Dict[str, Document],
        use_llm: bool = True,
        use_t5: bool = True,
    ) -> List[GeneratedQuery]:
        """Generate queries using all approaches."""
        all_queries = []

        for celex, doc in tqdm(documents.items(), desc="Generating queries"):
            doc_queries = []

            # Approach A: LLM-based generation
            if use_llm:
                try:
                    llm_queries = self.llm_generator.generate_queries(doc)
                    doc_queries.extend(llm_queries)
                except Exception as e:
                    logger.warning(f"LLM generation failed for {celex}: {e}")

            # Approach B: DocT5Query
            if use_t5:
                try:
                    t5_queries = self.t5_generator.generate_queries(doc)
                    doc_queries.extend(t5_queries)
                except Exception as e:
                    logger.warning(f"T5 generation failed for {celex}: {e}")

            all_queries.extend(doc_queries)

        # Approach C: Diversification
        all_queries = self.diversifier.diversify_queries(all_queries)

        # Quality filtering
        filtered_queries = self.filter_pipeline.filter_queries(all_queries, documents)

        return filtered_queries

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
