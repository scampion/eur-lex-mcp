"""
Phase 4: LLM-as-Judge Evaluation Framework.

This module implements LLM-based relevance assessment for queries where
cross-encoder scoring may be insufficient, particularly for complex legal reasoning.
"""
import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from config import llm_judge_config, OUTPUT_DIR
from corpus import Document
from query_generation import GeneratedQuery

logger = logging.getLogger(__name__)


class Judgment(Enum):
    """Binary relevance judgment."""

    RELEVANT = "RELEVANT"
    NOT_RELEVANT = "NOT_RELEVANT"


class Confidence(Enum):
    """Confidence level of judgment."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class JudgmentResult:
    """Result of LLM relevance judgment."""

    query_id: str
    doc_id: str
    judgment: str
    confidence: str
    legal_issue: str
    analysis: str
    raw_response: str


# Binary relevance judgment prompt template
JUDGMENT_PROMPT = """You are an expert legal research evaluator assessing document relevance for EUR-Lex search.

QUERY: {query}
QUERY TYPE: {query_type}

DOCUMENT (CELEX {celex}):
Type: {doc_type}
Title: {title}
Content: {content}

EVALUATION CRITERIA:
A document is RELEVANT if it:
- Addresses the legal issue, provision, or topic in the query
- Contains applicable EU law (regulations, directives, case holdings)
- Would be useful to a legal professional researching this topic
- Provides legal principles, requirements, or reasoning related to the query

A document is NOT RELEVANT if it:
- Discusses unrelated legal matters
- Mentions query terms only incidentally without substantive legal connection
- Lacks legal authority or analysis related to the query

INSTRUCTIONS:
1. Identify the core legal question or research need in the query
2. Analyze how the document content relates to this need
3. Consider document type appropriateness (e.g., case law for precedent queries)
4. Provide your reasoning, then your judgment

OUTPUT (JSON only):
{{
  "legal_issue": "<core legal question identified>",
  "document_relevance_analysis": "<how document relates to query>",
  "judgment": "RELEVANT" | "NOT_RELEVANT",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}}"""


class OllamaClient:
    """Client for Ollama LLM API."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate text using Ollama."""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            },
        }

        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]


class LLMJudge:
    """LLM-based relevance judge."""

    def __init__(self, config=llm_judge_config):
        self.config = config
        self.client = OllamaClient(config.judge_base_url, config.judge_model)
        self.random = random.Random(42)

    def judge_relevance(
        self,
        query: GeneratedQuery,
        document: Document,
    ) -> JudgmentResult:
        """Judge relevance of a document to a query."""
        prompt = JUDGMENT_PROMPT.format(
            query=query.text,
            query_type=query.query_type,
            celex=document.celex,
            doc_type=document.doc_type,
            title=document.title,
            content=document.text[: self.config.max_content_chars],
        )

        try:
            response = self.client.generate(prompt)
            result = self._parse_response(query.query_id, document.celex, response)
            return result
        except Exception as e:
            logger.error(f"LLM judgment failed for {query.query_id}/{document.celex}: {e}")
            return JudgmentResult(
                query_id=query.query_id,
                doc_id=document.celex,
                judgment=Judgment.NOT_RELEVANT.value,
                confidence=Confidence.LOW.value,
                legal_issue="",
                analysis=f"Error: {str(e)}",
                raw_response="",
            )

    def _parse_response(
        self, query_id: str, doc_id: str, response: str
    ) -> JudgmentResult:
        """Parse LLM response into JudgmentResult."""
        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)

        if not json_match:
            logger.warning(f"No JSON found in response for {query_id}/{doc_id}")
            return JudgmentResult(
                query_id=query_id,
                doc_id=doc_id,
                judgment=Judgment.NOT_RELEVANT.value,
                confidence=Confidence.LOW.value,
                legal_issue="",
                analysis="Failed to parse response",
                raw_response=response,
            )

        try:
            data = json.loads(json_match.group())
            return JudgmentResult(
                query_id=query_id,
                doc_id=doc_id,
                judgment=data.get("judgment", Judgment.NOT_RELEVANT.value),
                confidence=data.get("confidence", Confidence.MEDIUM.value),
                legal_issue=data.get("legal_issue", ""),
                analysis=data.get("document_relevance_analysis", ""),
                raw_response=response,
            )
        except json.JSONDecodeError:
            logger.warning(f"JSON parse error for {query_id}/{doc_id}")
            return JudgmentResult(
                query_id=query_id,
                doc_id=doc_id,
                judgment=Judgment.NOT_RELEVANT.value,
                confidence=Confidence.LOW.value,
                legal_issue="",
                analysis="JSON parse error",
                raw_response=response,
            )

    def judge_batch(
        self,
        pairs: List[Tuple[GeneratedQuery, Document]],
        randomize_order: bool = True,
    ) -> List[JudgmentResult]:
        """Judge a batch of query-document pairs."""
        if randomize_order and self.config.randomize_order:
            pairs = pairs.copy()
            self.random.shuffle(pairs)

        results = []
        for query, document in tqdm(pairs, desc="Judging relevance"):
            result = self.judge_relevance(query, document)
            results.append(result)

        return results


class CrossValidator:
    """Cross-validates low-confidence judgments."""

    def __init__(self, primary_judge: LLMJudge, config=llm_judge_config):
        self.primary_judge = primary_judge
        self.config = config

        # Create secondary judge with different model if configured
        secondary_model = self.config.model_options.get("production", {}).get("model")
        if secondary_model and secondary_model != self.config.judge_model:
            self.secondary_judge = LLMJudge(config)
            self.secondary_judge.client.model = secondary_model
        else:
            self.secondary_judge = None

    def cross_validate(
        self,
        results: List[JudgmentResult],
        queries: Dict[str, GeneratedQuery],
        documents: Dict[str, Document],
    ) -> List[JudgmentResult]:
        """Re-evaluate low-confidence judgments with secondary model."""
        if not self.secondary_judge:
            logger.info("No secondary judge configured, skipping cross-validation")
            return results

        validated = []
        for result in results:
            if result.confidence == Confidence.LOW.value:
                # Re-evaluate with secondary model
                query = queries.get(result.query_id)
                doc = documents.get(result.doc_id)

                if query and doc:
                    secondary_result = self.secondary_judge.judge_relevance(query, doc)

                    # If judgments disagree, flag for manual review
                    if secondary_result.judgment != result.judgment:
                        result.analysis += " [NEEDS_MANUAL_REVIEW]"
                    else:
                        # Agreement increases confidence
                        result.confidence = Confidence.MEDIUM.value

            validated.append(result)

        return validated


class RAGASEvaluator:
    """Integration with RAGAS framework for comprehensive RAG evaluation."""

    def __init__(self, config=llm_judge_config):
        self.config = config

    def evaluate(
        self,
        queries: List[str],
        retrieved_contexts: List[List[str]],
        generated_answers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate using RAGAS metrics.

        Note: This requires generated answers from the MCP server.
        If the server only does retrieval, use context_precision only.
        """
        try:
            from ragas import evaluate
            from ragas.metrics import LLMContextPrecisionWithoutReference
            from ragas.llms import LangchainLLMWrapper
            from langchain_community.llms import Ollama
            from datasets import Dataset

            # Prepare dataset
            data = {
                "question": queries,
                "contexts": retrieved_contexts,
            }
            if generated_answers:
                data["answer"] = generated_answers

            dataset = Dataset.from_dict(data)

            # Setup evaluator LLM
            evaluator_llm = LangchainLLMWrapper(
                Ollama(
                    model=self.config.judge_model,
                    base_url=self.config.judge_base_url,
                )
            )

            # Select metrics based on what's available
            metrics = [LLMContextPrecisionWithoutReference(llm=evaluator_llm)]

            # Run evaluation
            results = evaluate(dataset=dataset, metrics=metrics)

            return {str(k): float(v) for k, v in results.items()}

        except ImportError as e:
            logger.warning(f"RAGAS not available: {e}")
            return {}
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return {}


class LLMJudgePipeline:
    """Main pipeline for LLM-based evaluation."""

    def __init__(self, config=llm_judge_config):
        self.config = config
        self.judge = LLMJudge(config)
        self.cross_validator = CrossValidator(self.judge, config)
        self.ragas_evaluator = RAGASEvaluator(config)

    def evaluate_pairs(
        self,
        pairs: List[Tuple[GeneratedQuery, Document]],
        queries_dict: Dict[str, GeneratedQuery],
        documents_dict: Dict[str, Document],
        cross_validate: bool = True,
    ) -> List[JudgmentResult]:
        """Run full LLM evaluation pipeline."""
        logger.info(f"Evaluating {len(pairs)} query-document pairs...")

        # Initial judgments
        results = self.judge.judge_batch(pairs)

        # Cross-validation for low-confidence cases
        if cross_validate and self.config.cross_validate_low_confidence:
            results = self.cross_validator.cross_validate(
                results, queries_dict, documents_dict
            )

        return results

    def save_results(
        self,
        results: List[JudgmentResult],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save judgment results."""
        output_path = output_path or OUTPUT_DIR / "llm_judgments.json"

        serializable = [asdict(r) for r in results]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        logger.info(f"Saved {len(results)} judgments to {output_path}")
        return output_path

    def load_results(self, input_path: Optional[Path] = None) -> List[JudgmentResult]:
        """Load judgment results from file."""
        input_path = input_path or OUTPUT_DIR / "llm_judgments.json"

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return [
            JudgmentResult(
                query_id=r["query_id"],
                doc_id=r["doc_id"],
                judgment=r["judgment"],
                confidence=r["confidence"],
                legal_issue=r["legal_issue"],
                analysis=r["analysis"],
                raw_response=r["raw_response"],
            )
            for r in data
        ]

    def compute_statistics(self, results: List[JudgmentResult]) -> Dict:
        """Compute statistics about judgments."""
        total = len(results)
        if total == 0:
            return {}

        relevant_count = sum(1 for r in results if r.judgment == Judgment.RELEVANT.value)
        confidence_counts = {}
        for r in results:
            confidence_counts[r.confidence] = confidence_counts.get(r.confidence, 0) + 1

        needs_review = sum(1 for r in results if "[NEEDS_MANUAL_REVIEW]" in r.analysis)

        return {
            "total_judgments": total,
            "relevant_count": relevant_count,
            "not_relevant_count": total - relevant_count,
            "relevance_rate": relevant_count / total,
            "confidence_distribution": confidence_counts,
            "needs_manual_review": needs_review,
        }


def main():
    """Main entry point for LLM judge evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    from corpus import CorpusManager
    from query_generation import QueryGenerationPipeline

    # Load data
    logger.info("Loading corpus and queries...")
    corpus_manager = CorpusManager()
    corpus = corpus_manager.load_corpus()

    query_pipeline = QueryGenerationPipeline()
    queries = query_pipeline.load_queries()
    queries_dict = {q.query_id: q for q in queries}

    # Create sample pairs for evaluation (in practice, use retrieved results)
    pairs = []
    for query in queries[:100]:  # Sample for demo
        if query.source_celex in corpus:
            pairs.append((query, corpus[query.source_celex]))

    # Run evaluation
    pipeline = LLMJudgePipeline()
    results = pipeline.evaluate_pairs(pairs, queries_dict, corpus)

    # Save and report
    pipeline.save_results(results)
    stats = pipeline.compute_statistics(results)

    print("\nLLM Judge Evaluation Complete!")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
