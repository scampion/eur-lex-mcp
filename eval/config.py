"""
Configuration settings for EUR-Lex MCP Server Evaluation.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"

# Ensure directories exist
for d in [DATA_DIR, OUTPUT_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)


@dataclass
class CorpusConfig:
    """Phase 1: Corpus acquisition settings."""

    # Document type targets for stratified sampling
    doc_type_targets: Dict[str, int] = field(default_factory=lambda: {
        "regulation": 1500,      # Sector 3, Descriptor R
        "directive": 1000,       # Sector 3, Descriptor L
        "decision": 1000,        # Sector 3, Descriptor D
        "caselaw": 800,          # Sector 6, CJ/TJ
        "proposal": 600,         # Sector 5, PC/DC
        "intagr": 400,           # Sector 2, International agreements
        "recommendation": 400,   # Sector 3, H/A
    })

    language: str = "en"
    min_docs_per_type: int = 100

    # HuggingFace dataset source
    hf_dataset_base_url: str = "https://huggingface.co/datasets/joelito/eurlex_resources/resolve/main/data"


@dataclass
class QueryGenerationConfig:
    """Phase 2: Synthetic query generation settings."""

    # LLM settings for aspect-guided generation
    llm_model: str = "llama3.1:70b"
    llm_model: str = "gpt-oss-120b"

    llm_base_url: str = "http://localhost:11434"  # Ollama default
    llm_base_url: str = "https://api-gpt.jrc.ec.europa.eu/v1"

    # DocT5Query settings
    doc2query_model: str = "doc2query/msmarco-t5-base-v1"
    doc2query_num_queries: int = 10
    doc2query_max_length: int = 64
    doc2query_top_p: float = 0.95
    doc2query_temperature: float = 0.8

    # Query type distribution
    query_type_distribution: Dict[str, float] = field(default_factory=lambda: {
        "keyword": 0.25,
        "phrase": 0.25,
        "natural_language": 0.30,
        "boolean": 0.20,
    })

    # Queries per document
    queries_per_doc_llm: int = 5
    queries_per_doc_t5: int = 10

    # Filtering thresholds
    min_query_tokens: int = 3
    max_query_tokens: int = 50
    cross_encoder_top_percent: float = 0.70
    consistency_top_k: int = 40

    # Content truncation for LLM
    max_content_chars: int = 2000


@dataclass
class PseudoLabelConfig:
    """Phase 3: Hard negative mining and pseudo-labeling settings."""

    # Retrieval settings
    retrieval_top_k: int = 100

    # Cross-encoder model
    cross_encoder_model: str = "BAAI/bge-reranker-large"
    cross_encoder_batch_size: int = 32

    # Relevance thresholds for graded labels (0-3 scale)
    relevance_thresholds: Dict[int, float] = field(default_factory=lambda: {
        3: 0.8,   # Highly relevant
        2: 0.5,   # Relevant
        1: 0.2,   # Marginally relevant
        0: 0.0,   # Not relevant
    })

    # Output format
    output_format: str = "beir"  # BEIR-compatible format


@dataclass
class LLMJudgeConfig:
    """Phase 4: LLM-as-judge evaluation settings."""

    # Model selection
    judge_model: str = "llama3.1:70b"
    judge_base_url: str = "http://localhost:11434"

    # Alternative models by resource
    model_options: Dict[str, Dict] = field(default_factory=lambda: {
        "development": {"model": "llama3.1:8b", "vram_gb": 6},
        "production": {"model": "qwen2.5:32b", "vram_gb": 20},
        "highest_quality": {"model": "llama3.1:70b", "vram_gb": 40},
    })

    # Evaluation settings
    max_content_chars: int = 2000
    batch_size: int = 10

    # Position bias mitigation
    randomize_order: bool = True
    cross_validate_low_confidence: bool = True
    confidence_threshold: float = 0.7


@dataclass
class AnnotationConfig:
    """Phase 5: Human annotation calibration settings."""

    # Sample selection
    total_samples: int = 100
    samples_per_doc_type: int = 15
    high_confidence_samples: int = 10

    # Inter-annotator agreement
    target_kappa: float = 0.6  # Substantial agreement
    overlap_samples: int = 50

    # Agreement thresholds
    llm_human_agreement_target: float = 0.80


@dataclass
class MetricsConfig:
    """Phase 6: Automated metrics and coverage testing settings."""

    # Core metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "ndcg@5", "ndcg@10",
        "mrr",
        "precision@5", "precision@10",
        "recall@10", "recall@100",
        "map@100",
    ])

    # Success criteria
    target_ndcg_10: float = 0.5
    target_mrr: float = 0.6
    target_coverage_pass_rate: float = 0.95

    # Coverage test minimum queries per doc type
    min_coverage_queries: Dict[str, int] = field(default_factory=lambda: {
        "regulation": 100,
        "directive": 100,
        "decision": 80,
        "caselaw_ecj": 100,
        "caselaw_gc": 60,
        "ag_opinions": 40,
        "proposal": 60,
        "intagr": 40,
        "consolidated": 40,
    })

    # MLflow settings
    mlflow_experiment_name: str = "eurlex-mcp-eval"
    mlflow_tracking_uri: str = "mlruns"


@dataclass
class MCPClientConfig:
    """MCP server client configuration."""

    server_command: str = "python"
    server_args: List[str] = field(default_factory=lambda: ["-m", "eurlex_mcp"])
    timeout_seconds: int = 30
    max_retries: int = 3


# Default configuration instances
corpus_config = CorpusConfig()
query_gen_config = QueryGenerationConfig()
pseudo_label_config = PseudoLabelConfig()
llm_judge_config = LLMJudgeConfig()
annotation_config = AnnotationConfig()
metrics_config = MetricsConfig()
mcp_client_config = MCPClientConfig()
