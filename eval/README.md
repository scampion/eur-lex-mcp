# EUR-Lex MCP Server Evaluation Framework

A comprehensive evaluation pipeline for the EUR-Lex full-text search MCP server, implementing synthetic query generation, LLM-as-judge relevance assessment, and automated metrics computation.

> **Note**: This framework follows the methodology outlined in `evaluation_workplan.md`.

## Overview

The evaluation framework consists of six phases designed to rigorously assess the EUR-Lex MCP server's retrieval quality with minimal human annotation:

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `corpus.py` | Corpus acquisition and stratified document sampling |
| 2 | `query_generation.py` | Synthetic query generation (LLM + DocT5Query) |
| 3 | `pseudo_labels.py` | Hard negative mining and pseudo-label generation |
| 4 | `llm_judge.py` | LLM-as-judge relevance evaluation |
| 5 | `annotation.py` | Human annotation calibration (50-100 samples) |
| 6 | `metrics.py` | Automated metrics computation and coverage testing |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama for LLM-based evaluation
# See https://ollama.ai for installation instructions
ollama pull llama3.1:70b
```

## Quick Start

```bash
# Run full evaluation pipeline
python run_evaluation.py --phase all

# Run with mock client (no MCP server needed)
python run_evaluation.py --phase all --use-mock

# Skip LLM operations (faster, uses only DocT5Query)
python run_evaluation.py --phase all --skip-llm
```

## Usage

### Command Line Interface

```bash
# Run specific phases
python run_evaluation.py --phase corpus      # Phase 1: Download and sample corpus
python run_evaluation.py --phase queries     # Phase 2: Generate synthetic queries
python run_evaluation.py --phase labels      # Phase 3: Generate pseudo-labels
python run_evaluation.py --phase judge       # Phase 4: LLM-as-judge evaluation
python run_evaluation.py --phase annotation  # Phase 5: Prepare annotation batch
python run_evaluation.py --phase metrics     # Phase 6: Compute metrics

# Options
python run_evaluation.py --phase all \
    --skip-llm \                    # Skip LLM-based operations
    --sample-size 200 \             # Number of samples for LLM judge
    --use-mock \                    # Use mock MCP client
    --output-dir ./my_results \     # Custom output directory
    -v                              # Verbose logging
```

### Programmatic Usage

```python
import asyncio
from eval import EvaluationRunner

async def run_evaluation():
    runner = EvaluationRunner()

    # Run full pipeline
    results = await runner.run_all_phases()

    # Or run individual phases
    runner.run_phase1_corpus()
    runner.run_phase2_queries(use_llm=True, use_t5=True)
    await runner.run_phase3_labels()
    runner.run_phase4_judge(sample_size=100)
    runner.run_phase5_annotation()
    await runner.run_phase6_metrics()

    return results

results = asyncio.run(run_evaluation())
```

## Directory Structure

```
eval/
├── __init__.py              # Package exports
├── config.py                # Configuration settings
├── corpus.py                # Phase 1: Corpus acquisition
├── query_generation.py      # Phase 2: Query generation
├── pseudo_labels.py         # Phase 3: Pseudo-labeling
├── llm_judge.py             # Phase 4: LLM evaluation
├── annotation.py            # Phase 5: Human annotation
├── metrics.py               # Phase 6: Metrics computation
├── mcp_client.py            # MCP server client
├── utils.py                 # Shared utilities
├── run_evaluation.py        # Main orchestrator
├── requirements.txt         # Dependencies
├── dataset.py               # Data download script
├── evaluation_workplan.md   # Detailed methodology
├── data/                    # Downloaded corpus data
├── output/                  # Evaluation outputs
│   ├── corpus/              # Sampled corpus (BEIR format)
│   ├── queries/             # Generated queries
│   ├── beir_dataset/        # Complete BEIR dataset
│   ├── annotation/          # Label Studio exports
│   └── runs/                # Timestamped run results
└── cache/                   # Cached computations
```

## Phase Details

### Phase 1: Corpus Acquisition

Downloads EUR-Lex documents from the [joelito/eurlex_resources](https://huggingface.co/datasets/joelito/eurlex_resources) HuggingFace dataset and performs stratified sampling.

**Default sampling targets:**
- Regulations: 1,500 documents
- Directives: 1,000 documents
- Decisions: 1,000 documents
- Case law: 800 documents
- Proposals: 600 documents
- International agreements: 400 documents
- Recommendations: 400 documents

**Output:** `output/corpus/corpus.jsonl` (BEIR format)

### Phase 2: Synthetic Query Generation

Generates diverse queries using three approaches:

1. **Aspect-guided LLM generation**: Uses Llama 3.1 via Ollama to generate factual, procedural, and interpretive queries for each document aspect
2. **DocT5Query**: Generates keyword-style queries using the `doc2query/msmarco-t5-base-v1` model
3. **Quality filtering**: Three-stage pipeline (length, cross-encoder scoring, consistency)

**Query type distribution:**
- Keyword queries: 25%
- Phrase queries: 25%
- Natural language questions: 30%
- Boolean/complex queries: 20%

**Output:** `output/queries/queries.jsonl`

### Phase 3: Hard Negative Mining

Implements the GPL (Generative Pseudo Labeling) methodology:

1. Retrieves candidate documents via MCP server (top-100)
2. Scores all (query, candidate) pairs with cross-encoder (`BAAI/bge-reranker-large`)
3. Converts continuous scores to graded relevance (0-3 scale)
4. Builds BEIR-compatible evaluation dataset

**Output:** `output/beir_dataset/` with `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`

### Phase 4: LLM-as-Judge

Deploys an LLM judge for complex relevance assessment:

- **Model**: Llama 3.1 70B via Ollama (configurable)
- **Output**: Binary relevance (RELEVANT/NOT_RELEVANT) with confidence
- **Features**: Position bias mitigation, cross-validation for low-confidence cases

**Judgment prompt includes:**
- Legal-specific evaluation criteria
- Document type appropriateness
- Authority and applicability assessment

**Output:** `output/llm_judgments.json`

### Phase 5: Human Annotation

Prepares calibration samples for human annotation:

- **Sample selection**: Stratified by document type and LLM confidence
- **Export format**: Label Studio JSON
- **Agreement metrics**: Cohen's Kappa (target κ > 0.6)
- **Calibration**: Compares human labels to pseudo-labels and LLM judgments

**Relevance scale (TREC-adapted):**
| Score | Label | Description |
|-------|-------|-------------|
| 3 | Highly Relevant | Essential for legal research, binding authority |
| 2 | Relevant | Useful legal information, related provisions |
| 1 | Marginally Relevant | Tangential, different jurisdiction |
| 0 | Not Relevant | Off-topic, incidental mention only |

**Output:** `output/annotation/` with Label Studio export

### Phase 6: Automated Metrics

Computes comprehensive retrieval metrics:

**Core metrics (via ranx):**
- nDCG@5, nDCG@10
- MRR (Mean Reciprocal Rank)
- Precision@5, Precision@10
- Recall@10, Recall@100
- MAP@100

**Coverage testing:**
- Tests all EUR-Lex document types (regulations, directives, case law, etc.)
- Minimum query coverage per type

**Edge case tests:**
- Citation format variations (e.g., "Regulation (EU) 2016/679" → GDPR)
- Special characters and boolean queries
- Multilingual queries
- Temporal queries

**Success criteria:**
- nDCG@10 > 0.5
- MRR > 0.6
- Coverage pass rate > 95%

**Output:** `output/evaluation_results.json`, HTML report, MLflow logs

## Configuration

All settings are in `config.py`. Key configurations:

```python
from eval.config import (
    corpus_config,        # Sampling targets, language
    query_gen_config,     # LLM settings, query types
    pseudo_label_config,  # Cross-encoder, thresholds
    llm_judge_config,     # Judge model, batch size
    annotation_config,    # Sample counts, agreement targets
    metrics_config,       # Metric list, success criteria
)

# Modify settings
corpus_config.doc_type_targets["regulation"] = 2000
llm_judge_config.judge_model = "qwen2.5:32b"
```

## MLflow Tracking

Evaluation runs are logged to MLflow:

```bash
# View MLflow UI
mlflow ui --backend-store-uri ./eval/mlruns

# Access at http://localhost:5000
```

Logged data includes:
- All retrieval metrics
- Coverage and edge case pass rates
- Success criteria status
- Evaluation artifacts (results JSON, reports)

## Requirements

### Core Dependencies
- Python 3.9+
- requests, tqdm, aiohttp
- transformers, torch, sentence-transformers
- ranx, scikit-learn
- mlflow

### Optional
- **Ollama**: For LLM-based query generation and judging
- **Label Studio**: For human annotation interface
- **RAGAS**: For RAG-specific evaluation metrics

## Troubleshooting

### Ollama connection issues
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required model
ollama pull llama3.1:70b
```

### Memory issues with large models
```python
# Use smaller models in config.py
query_gen_config.llm_model = "llama3.1:8b"
llm_judge_config.judge_model = "llama3.1:8b"
```

### Skip LLM operations entirely
```bash
python run_evaluation.py --phase queries --skip-llm
```

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update this README for new features
4. Run the full pipeline before submitting changes

## References

- [Evaluation Workplan](evaluation_workplan.md) - Detailed methodology
- [BEIR Benchmark](https://github.com/beir-cellar/beir) - Dataset format
- [GPL Paper](https://aclanthology.org/2022.naacl-main.168/) - Pseudo-labeling methodology
- [ranx Documentation](https://amenra.github.io/ranx/) - Metrics computation
