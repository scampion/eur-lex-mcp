"""
EUR-Lex MCP Server Evaluation Framework.

A comprehensive evaluation pipeline for the EUR-Lex full-text search MCP server,
implementing synthetic query generation, LLM-as-judge relevance assessment,
and automated metrics computation.

Phases:
    1. Corpus Acquisition and Stratified Sampling (corpus.py)
    2. Synthetic Query Generation (query_generation.py)
    3. Hard Negative Mining and Pseudo-Labels (pseudo_labels.py)
    4. LLM-as-Judge Evaluation (llm_judge.py)
    5. Human Annotation Calibration (annotation.py)
    6. Automated Metrics and Coverage Testing (metrics.py)

Usage:
    # Run full evaluation
    python run_evaluation.py --phase all

    # Run specific phase
    python run_evaluation.py --phase corpus
    python run_evaluation.py --phase queries
    python run_evaluation.py --phase labels
    python run_evaluation.py --phase judge
    python run_evaluation.py --phase annotation
    python run_evaluation.py --phase metrics

    # Programmatic usage
    from eval import EvaluationRunner
    runner = EvaluationRunner()
    results = await runner.run_all_phases()
"""

__version__ = "0.1.0"

from .config import (
    corpus_config,
    query_gen_config,
    pseudo_label_config,
    llm_judge_config,
    annotation_config,
    metrics_config,
)

from .corpus import CorpusManager, Document
from .query_generation import QueryGenerationPipeline, GeneratedQuery
from .pseudo_labels import PseudoLabelPipeline, PseudoLabel
from .llm_judge import LLMJudgePipeline, JudgmentResult
from .annotation import AnnotationPipeline, AnnotationSample
from .metrics import MetricsPipeline, RetrievalMetrics
from .mcp_client import MCPClient, MockMCPClient
from .run_evaluation import EvaluationRunner

__all__ = [
    # Config
    "corpus_config",
    "query_gen_config",
    "pseudo_label_config",
    "llm_judge_config",
    "annotation_config",
    "metrics_config",
    # Phase 1
    "CorpusManager",
    "Document",
    # Phase 2
    "QueryGenerationPipeline",
    "GeneratedQuery",
    # Phase 3
    "PseudoLabelPipeline",
    "PseudoLabel",
    # Phase 4
    "LLMJudgePipeline",
    "JudgmentResult",
    # Phase 5
    "AnnotationPipeline",
    "AnnotationSample",
    # Phase 6
    "MetricsPipeline",
    "RetrievalMetrics",
    # Client
    "MCPClient",
    "MockMCPClient",
    # Runner
    "EvaluationRunner",
]
