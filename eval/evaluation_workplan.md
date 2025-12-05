# Workplan for EUR-Lex MCP Server Evaluation

> [!WARNING]  
> Work in progress


A full-text search server providing access to the EUR-Lex legal database can be rigorously evaluated using synthetic query generation, LLM-as-judge relevance assessment, and automated metrics—achieving **minimal human annotation** while maintaining statistical validity. This workplan details a six-phase approach requiring approximately **4-6 weeks** of implementation effort, leveraging entirely open-source tooling.

## Phase 1: Corpus acquisition and stratified document sampling

Before generating evaluation queries, you need representative documents from all EUR-Lex sectors. The CELEX numbering system provides natural stratification: [Link](https://eur-lex.europa.eu/content/help/eurlex-content/celex-number.html) **Sector 3** contains regulations (R), directives (L), and decisions (D); **Sector 5** holds preparatory acts (COM proposals, working documents); **Sector 6** contains ECJ and General Court case law; **Sector 2** covers international agreements. [Link](https://eur-lex.europa.eu/content/tools/TableOfSectors/types_of_documents_in_eurlex.html)

**Implementation approach:** Use the EUR-Lex SPARQL endpoint [Link](https://eur-lex.europa.eu/content/help/data-reuse/reuse-contents-eurlex-details.html) at `publications.europa.eu/webapi/rdf/sparql` or the `eurlex` R package to bulk-query document metadata. [Link](https://www.rdocumentation.org/packages/eurlex/versions/0.3.1) [Link](https://michalovadek.github.io/eurlex/articles/eurlexpkg.html) Target **5,000-10,000 documents** with proportional stratification across document types, ensuring minimum **100 documents per major type** (regulations, directives, decisions, case law, preparatory acts). Include metadata fields: CELEX number, EuroVoc descriptors, document date, and in-force status.

python

```python
# Sample stratification approach
doc_type_targets = {
    "regulation": 1500,      # Sector 3, Descriptor R
    "directive": 1000,       # Sector 3, Descriptor L  
    "decision": 1000,        # Sector 3, Descriptor D
    "case_law_ecj": 800,     # Sector 6, CJ/TJ
    "preparatory_act": 600,  # Sector 5, PC/DC
    "international_agreement": 400,  # Sector 2
    "consolidated_text": 300,  # Sector 0
    "recommendation_opinion": 400   # Sector 3, H/A
}
```

**Estimated effort:** 3-5 days for data acquisition pipeline, including SPARQL query development and document text extraction. **Dependencies:** None—this phase runs first.

---

## Phase 2: Synthetic query generation pipeline

Generate diverse, realistic legal queries from sampled documents using three complementary approaches. This creates query-document pairs with **known pseudo-relevance** without human annotation.

### Approach A: Aspect-guided generation with local LLM

Using MiniMax M2 or Llama 3.1 70B via OpenRouter, generate **3-5 queries per document** by first identifying distinct legal aspects, then creating queries for each aspect. Research on Vietnamese legal retrieval shows this yields higher query diversity than single-query generation, with **82% passage hit rate** after filtering. [Link](https://arxiv.org/html/2412.00657)

```
Prompt template for legal query generation:

You are a legal research assistant. Analyze this EUR-Lex document and:
1. Identify 3-5 distinct legal aspects, concepts, or provisions covered
2. For each aspect, generate a realistic search query a legal professional might use

Document Type: {doc_type} (e.g., Directive, Regulation, Case Law)
CELEX Number: {celex}
Title: {title}
Content (first 2000 chars): {content}

For each aspect, generate varied query types:
- Factual query: seeking specific provisions or requirements
- Procedural query: how to comply or implement
- Interpretive query: meaning or scope of legal concepts

Format:
Aspect 1: [description]
- Factual: [query]
- Procedural: [query]
- Interpretive: [query]
```

### Approach B: DocT5Query for keyword-style queries

Use the pre-trained `doc2query/msmarco-t5-base-v1` model to generate **5-10 short keyword queries** per document. [Link](https://huggingface.co/doc2query/msmarco-t5-base-v1) These complement the natural language queries from Approach A by covering simple lookup patterns.

python

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained('doc2query/msmarco-t5-base-v1')
tokenizer = T5Tokenizer.from_pretrained('doc2query/msmarco-t5-base-v1')

# Generate diverse queries with nucleus sampling
outputs = model.generate(
    input_ids=tokenizer.encode(document_text[:512], return_tensors='pt'),
    max_length=64, do_sample=True, top_p=0.95, 
    temperature=0.8, num_return_sequences=10
)
```

### Approach C: Query type diversification

Ensure coverage across query complexity levels by explicitly generating different types:

|Query Type|Example for GDPR|Target %|
|---|---|---|
|Keyword|"GDPR data portability"|25%|
|Phrase|"right to be forgotten requirements"|25%|
|Natural language question|"What are the conditions for lawful data processing under GDPR?"|30%|
|Boolean/complex|"data breach AND notification AND 72 hours"|20%|

### Quality filtering pipeline

Apply three-stage filtering to remove low-quality synthetic queries:

1. **Length filtering:** Keep queries with 3-50 tokens (removes trivial and verbose queries)
2. **Cross-encoder scoring:** Use `BAAI/bge-reranker-base` to score (query, source_document) pairs; retain top 70% by relevance score
3. **Consistency filtering:** Train a bi-encoder on initial synthetic data, then keep only queries where the source document appears in top-40 retrieval results [Link](https://www.researchgate.net/publication/363843345_Promptagator_Few-shot_Dense_Retrieval_From_8_Examples)

**Target output:** 15,000-25,000 filtered query-document pairs across all document types and query styles.

**Estimated effort:** 5-7 days for pipeline development and execution. **Dependencies:** Phase 1 document corpus.

---

## Phase 3: Hard negative mining and pseudo-label generation

Transform query-document pairs into training/evaluation data with graded relevance using the **GPL (Generative Pseudo Labeling)** methodology. [Link](https://aclanthology.org/2022.naacl-main.168/)

### Step 1: Retrieve candidate documents via MCP server

For each synthetic query, call the EUR-Lex MCP server to retrieve top-100 documents. This simultaneously tests the server and generates candidate relevance pairs.

python

```python
async def retrieve_candidates(mcp_client, query: str, k: int = 100):
    """Call MCP server and retrieve candidates"""
    response = await mcp_client.search(query=query, limit=k)
    return [(doc['celex'], doc['text'], doc['score']) for doc in response]
```

### Step 2: Generate pseudo-relevance labels with cross-encoder

Use a cross-encoder to score all (query, candidate_document) pairs with **continuous relevance scores** (0.0-1.0), providing finer granularity than binary labels.

python

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('BAAI/bge-reranker-large')  # Best quality
# Or 'cross-encoder/ms-marco-MiniLM-L-6-v2' for faster inference

def generate_pseudo_labels(query: str, candidates: list) -> list:
    pairs = [(query, doc_text) for _, doc_text, _ in candidates]
    scores = reranker.predict(pairs)
    return [(celex, score) for (celex, _, _), score in zip(candidates, scores)]
```

### Step 3: Create evaluation dataset structure

Organize data in BEIR-compatible format for standardized evaluation:

```
eurlex_eval/
├── corpus.jsonl        # {"_id": "celex", "title": "...", "text": "..."}
├── queries.jsonl       # {"_id": "q001", "text": "query text"}
└── qrels/
    └── test.tsv        # query_id \\t doc_id \\t relevance_score
```

Convert continuous cross-encoder scores to graded relevance (0-3 scale) using thresholds calibrated against a small human-annotated sample in Phase 5.

**Estimated effort:** 4-5 days. **Dependencies:** Phases 1-2, working MCP server connection.

---

## Phase 4: LLM-as-judge evaluation framework

Deploy an LLM judge to assess retrieval relevance for queries where cross-encoder scoring may be insufficient, particularly for complex legal reasoning.

### Model selection for local deployment

For local evaluation without API costs, deploy via **Ollama** or **vLLM**:

|Model|VRAM (Q4)|Speed|Recommended For|
|---|---|---|---|
|Llama 3.1 8B-Instruct|~6GB|Fast|Development/iteration|
|Qwen2.5 32B-Instruct|~20GB|Moderate|Production evaluation|
|Llama 3.1 70B-Instruct|~40GB|Slower|Highest quality|

bash

```bash
# Install and run locally
ollama pull llama3.1:70b
# Or via OpenRouter for free-tier access to larger models
```

### Binary relevance judgment prompt

Research shows binary classification is more reliable than graded scales for LLM judges. [Link](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) Use this prompt template:

```
You are an expert legal research evaluator assessing document relevance for EUR-Lex search.

QUERY: {query}
QUERY TYPE: {query_type}  # e.g., "seeking applicable regulation", "finding case law precedent"

DOCUMENT (CELEX {celex}):
Type: {doc_type}
Title: {title}
Content: {content_truncated_to_2000_chars}

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
{
  "legal_issue": "<core legal question identified>",
  "document_relevance_analysis": "<how document relates to query>",
  "judgment": "RELEVANT" | "NOT_RELEVANT",
  "confidence": "HIGH" | "MEDIUM" | "LOW"
}
```

### Position bias mitigation

When evaluating multiple documents per query, randomize presentation order and flag low-confidence judgments for cross-validation with alternative model or human review.

### Integration with RAGAS framework

For comprehensive RAG-style evaluation if the MCP server supports answer generation:

python

```python
from ragas import evaluate
from ragas.metrics import LLMContextPrecisionWithoutReference, Faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama

evaluator_llm = LangchainLLMWrapper(Ollama(model="llama3.1:70b"))
metrics = [LLMContextPrecisionWithoutReference(llm=evaluator_llm)]

results = evaluate(dataset=eval_samples, metrics=metrics)
```

**Estimated effort:** 5-7 days for prompt engineering, calibration, and pipeline integration. **Dependencies:** Phase 3 candidate generation.

---

## Phase 5: Minimal human annotation for calibration

While the pipeline is primarily automated, **50-100 human-annotated samples** are essential for calibrating pseudo-labels and validating LLM judge reliability.

### Annotation protocol

Use **stratified sampling** to select calibration samples:

- 10-15 samples per major document type (regulations, directives, case law, etc.)
- Mix of high-confidence and low-confidence LLM judgments
- Include edge cases (partial relevance, outdated documents, cross-references)

### Annotation guidelines for legal relevance

```
RELEVANCE SCALE (adapted from TREC):
3 - HIGHLY RELEVANT: Document directly addresses the legal query; contains 
    binding or persuasive authority; would be essential for legal research
2 - RELEVANT: Contains useful legal information related to query; relevant 
    dicta, related provisions, or background context
1 - MARGINALLY RELEVANT: Touches on topic but not directly applicable; 
    different jurisdiction, superseded version, or tangential mention
0 - NOT RELEVANT: Off-topic; query terms appear without legal connection

SPECIAL CONSIDERATIONS:
- For case law queries: Distinguish holdings from dicta
- For regulatory queries: Check in-force status
- For procedural queries: Assess practical applicability
```

### Inter-annotator agreement measurement

With two annotators on 50+ overlapping samples, calculate Cohen's Kappa: [Link](https://surge-ai.medium.com/inter-annotator-agreement-an-introduction-to-cohens-kappa-statistic-dcc15ffa5ac4)

python

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(annotator1_labels, annotator2_labels)
# Target: κ > 0.6 (substantial agreement)
```

### Active learning for efficient annotation

If more annotation is needed, use **uncertainty sampling** to prioritize:

python

```python
def select_for_annotation(predictions, n_samples=50):
    # Sort by confidence (ascending) to get most uncertain
    uncertain = sorted(predictions, key=lambda x: x['confidence'])
    # Also include some high-confidence for calibration
    high_conf = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:10]
    return uncertain[:n_samples-10] + high_conf
```

### Calibration workflow

1. Compare cross-encoder pseudo-labels to human labels; adjust thresholds if needed
2. Compare LLM judge outputs to human labels; measure agreement (target >80%)
3. Identify systematic biases (e.g., LLM favoring longer documents)
4. Iterate on prompts if agreement is below threshold

**Estimated effort:** 3-5 days for annotation interface setup and ~8-16 hours of annotation time. **Dependencies:** Phases 3-4 outputs.

---

## Phase 6: Automated metrics computation and coverage testing

### Core retrieval metrics

Compute standard IR metrics using the `ranx` library (recommended for speed and completeness):

python

```python
from ranx import Qrels, Run, evaluate

# Load qrels from pseudo-labels or human annotations
qrels = Qrels(qrels_dict)  # {query_id: {doc_id: relevance_score}}
run = Run(run_dict)        # {query_id: {doc_id: retrieval_score}}

metrics = evaluate(qrels, run, [
    "ndcg@5", "ndcg@10",      # Primary ranking quality
    "mrr",                    # First relevant result
    "precision@5", "precision@10",
    "recall@10", "recall@100",
    "map@100"                 # Overall ranking
])
```

### Coverage testing matrix

Systematically verify the MCP server handles all EUR-Lex document types:

|Document Type|CELEX Sector|Query Types to Test|Min Queries|
|---|---|---|---|
|Regulations|3R|Keyword, phrase, citation|100|
|Directives|3L|Transposition, requirements|100|
|Decisions|3D|Addressee, subject matter|80|
|ECJ Judgments|6CJ|Case number, legal issue|100|
|General Court|6TJ|Trademark, competition|60|
|AG Opinions|6CC|Case reference, legal question|40|
|COM Proposals|5PC|Proposal subject, COM number|60|
|International Agreements|2A|Country, subject matter|40|
|Consolidated Texts|0|Base act + amendments|40|

### Edge case test suite

Create explicit tests for known challenges:

python

```python
edge_cases = [
    # Citation format variations
    {"query": "Regulation (EU) 2016/679", "must_return": "32016R0679"},
    {"query": "GDPR", "must_return": "32016R0679"},
    {"query": "Case C-131/12", "must_return": "62012CJ0131"},
    
    # Special characters
    {"query": "Article 101(1) TFEU", "category": "parentheses"},
    {"query": "Section 230", "category": "numeric_section"},
    
    # Boolean queries
    {"query": "data protection AND portability", "category": "boolean"},
    
    # Multilingual (if supported)
    {"query": "Datenschutz-Grundverordnung", "must_return": "32016R0679"},
    
    # Temporal queries
    {"query": "directive 2006/54 amendments", "category": "temporal"},
]
```

### Logging and monitoring infrastructure

Use MLflow for experiment tracking: ??
(How to integrate that in a global Observability stack)

python

```python
import mlflow

mlflow.set_experiment("eurlex-mcp-eval")

with mlflow.start_run(run_name="baseline-evaluation"):
    mlflow.log_params({
        "corpus_size": 5000,
        "query_count": 15000,
        "pseudo_label_method": "cross-encoder",
        "llm_judge_model": "llama3.1-70b"
    })
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("evaluation_results.json")
```

**Estimated effort:** 4-6 days for pipeline implementation and test execution. **Dependencies:** All previous phases.

---

## Tool stack summary

|Component|Recommended Tool|Installation|
|---|---|---|
|Query generation (LLM)|Llama 3.1 70B via Ollama/OpenRouter|`ollama pull llama3.1:70b`|
|Query generation (T5)|doc2query/msmarco-t5-base-v1|`pip install transformers`|
|Cross-encoder scoring|BAAI/bge-reranker-large|`pip install sentence-transformers`|
|LLM-as-judge|Qwen2.5 32B or Llama 3.1 70B|Via Ollama or vLLM|
|RAG evaluation|RAGAS|`pip install ragas`|
|IR metrics|ranx|`pip install ranx`|
|Annotation (if needed)|Label Studio|`pip install label-studio`|
|Experiment tracking|MLflow|`pip install mlflow`|

---

## Dependency graph and timeline

```
Week 1:
[Phase 1: Corpus Acquisition] ────────────────────────┐
                                                       │
Week 2:                                               ▼
[Phase 2: Synthetic Query Generation] ◄───────────────┤
                                                       │
Week 3:                                               ▼
[Phase 3: Hard Negative Mining + Pseudo-Labels] ◄─────┤
                                                       │
Week 3-4:                                             ▼
[Phase 4: LLM-as-Judge Framework] ◄───────────────────┤
                                                       │
Week 4:                                               ▼
[Phase 5: Human Calibration (parallel)] ◄─────────────┤
                                                       │
Week 5-6:                                             ▼
[Phase 6: Automated Metrics + Coverage Testing] ◄─────┘
```

---

## Expected outputs and success criteria

The evaluation pipeline should produce:

- **15,000+ query-document relevance pairs** across all EUR-Lex document types
- **Retrieval metrics** (nDCG@10, MRR, Precision@10) with confidence intervals
- **Coverage verification** confirming search works for all CELEX sectors
- **Edge case test results** identifying specific failure modes
- **LLM-judge calibration data** showing >80% agreement with human annotations
- **Reproducible evaluation scripts** for regression testing future versions

Success criteria: nDCG@10 > **0.5** indicates reasonable retrieval quality; MRR > **0.6** indicates the first result is usually relevant; coverage tests should achieve **>95% pass rate** across document types. Human annotation requirement: **<100 samples** for full calibration.

This workplan provides a rigorous, automated evaluation framework that balances statistical validity with practical resource constraints, leveraging state-of-the-art techniques in synthetic data generation and LLM-based evaluation.