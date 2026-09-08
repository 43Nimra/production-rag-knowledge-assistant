# ADR-005: Evaluation Strategy — RAGAS + Golden Dataset + CI Gate

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

The most common failure mode in portfolio RAG projects is: build the system, run a few manual queries, say "it works," ship it.

This approach has two fundamental problems:

1. **There is no baseline.** Without numbers, you cannot say whether a change improved or degraded the system.
2. **There is no regression detection.** A prompt change, a chunking tweak, or a model upgrade can silently reduce quality.

A RAG system without evaluation is not a production system — it is a demo.

This ADR defines the evaluation strategy: what we measure, how we measure it, what dataset we use, and how evaluation integrates into CI.

---

## Decision

We will implement a three-layer evaluation pipeline:

1. **Retrieval metrics** — does the system find the right chunks?
2. **Answer quality metrics** — does the system generate correct, grounded answers?
3. **CI quality gate** — does the system regress on any commit?

All three layers run against a curated golden dataset of 50 Q&A pairs.

---

## Layer 1: Retrieval Metrics

Retrieval evaluation does not require an LLM. It is fast, cheap, and can run on every commit.

### Metrics

**Recall@K** (primary metric)  
Proportion of queries where at least one relevant chunk appears in the top-K retrieved results.

```
Recall@K = (queries where relevant chunk in top-K) / (total queries)
```

We report Recall@5 and Recall@10. Recall@5 is the CI gate metric.

**MRR — Mean Reciprocal Rank**  
Average of the reciprocal rank of the first relevant result. Measures how high in the list the first correct chunk appears.

```
MRR = (1/|Q|) * Σ 1/rank_i
```

An MRR of 1.0 means the relevant chunk is always first. An MRR of 0.5 means it is on average second.

### Retrieval comparison

Three strategies are evaluated side-by-side:

| Strategy | What it tests |
|----------|--------------|
| BM25-only | Keyword matching baseline |
| Dense-only | Semantic embedding baseline |
| Hybrid RRF | Combined approach (expected best) |

**Why compare all three?**  
This is the core portfolio differentiator. The comparison demonstrates:
- We know BM25 is not obsolete; it is a legitimate baseline
- We know dense retrieval has a failure mode on exact-match queries
- We chose hybrid RRF based on measured evidence, not trend-following

The comparison table is a permanent artifact in the README.

---

## Layer 2: Answer Quality Metrics (RAGAS)

RAGAS (Retrieval Augmented Generation Assessment) is the standard open-source framework for RAG evaluation. It uses an LLM-as-judge approach to measure quality dimensions that cannot be assessed by string matching alone.

### Metrics

**Faithfulness** (primary metric — CI gate)  
Measures whether every claim in the generated answer is supported by the retrieved context. A faithfulness score of 1.0 means the answer contains no information not present in the retrieved chunks.

```
Faithfulness = (claims in answer supported by context) / (total claims in answer)
```

This is the hallucination detection metric. It is the most important metric in any RAG system.

**Answer Relevancy**  
Measures whether the answer actually addresses the question asked. Penalises answers that are grounded but off-topic.

**Context Recall**  
Measures whether the retrieved context contains the information needed to answer the question. Requires ground-truth answers from the golden dataset.

```
Context Recall = (ground truth statements covered by context) / (total ground truth statements)
```

**Context Precision**  
Measures what proportion of retrieved chunks were actually relevant. High precision means we are not polluting the context with irrelevant chunks.

### Why RAGAS over a custom eval

RAGAS is the established standard. Using it signals familiarity with the RAG evaluation ecosystem. Writing a custom evaluator would be reinventing a solved problem — and would not be better.

RAGAS uses LLM calls for faithfulness and relevancy metrics. This means eval runs have a small API cost (~$2-5 per full run with Sonnet). This is acceptable.

### Model used for evaluation

RAGAS evaluation runs use `claude-sonnet-4-6` as the judge model, not `claude-haiku-4-5`. The judge model needs sufficient reasoning capability to detect faithfulness violations. This is documented as a conscious cost decision.

---

## Layer 3: The Golden Dataset

### What it is

A manually curated JSON file: `evaluation/dataset/golden_qa.json`

50 question-answer pairs based on the actual documents indexed in the system.

### Format

```json
[
  {
    "id": "q001",
    "question": "What must be completed before starting the application server during deployment?",
    "ground_truth_answer": "Database migrations must be completed before starting the application server.",
    "relevant_document": "deployment_guide.pdf",
    "relevant_chunk_keywords": ["migrations", "application server", "deployment"],
    "difficulty": "medium",
    "category": "deployment"
  },
  ...
]
```

### Design principles for good questions

**Bad question (too easy):**
```
"What is the title of the deployment guide?"
```
Keyword retrieval trivially answers this. It tests nothing.

**Good question (requires understanding):**
```
"According to the deployment guide, what is the consequence of starting 
 the application server before running migrations?"
```
This requires:
1. Finding the right chunk (retrieval quality)
2. Understanding the causal relationship (LLM reasoning)
3. Not fabricating consequences not mentioned in the document (faithfulness)

### 50 questions — why this number

- Fewer than 30: too small for statistically meaningful metrics
- More than 100: excessive manual effort, diminishing returns for portfolio purposes
- 50: sufficient signal, roughly 3-4 hours of careful manual work

### Category distribution (recommended)

| Category | Count | Purpose |
|----------|-------|---------|
| Factual lookup | 15 | Tests basic retrieval |
| Multi-sentence reasoning | 15 | Tests context assembly |
| Exact-match terms (codes, versions) | 10 | Tests BM25 vs dense |
| Negative (answer not in docs) | 5 | Tests hallucination resistance |
| Cross-document | 5 | Tests multi-chunk synthesis |

The "negative" category is critical: 5 questions whose answers are NOT in the indexed documents. The system must respond "I don't have information about that" — not hallucinate an answer. Faithfulness on these questions is the most important quality signal.

### The dataset is a first-class artifact

`evaluation/dataset/golden_qa.json` is committed to the repository. It is reviewed as part of code review. It is never auto-generated by an LLM — every pair is manually verified.

---

## Layer 4: CI Quality Gate

### Gate thresholds

| Metric | Threshold | Consequence of failure |
|--------|-----------|----------------------|
| Recall@5 (hybrid) | ≥ 0.70 | CI fails |
| Faithfulness | ≥ 0.80 | CI fails |

**Why these thresholds:**
- Recall@5 ≥ 0.70: 70% of questions have a relevant chunk in the top 5. Below this, the context passed to the LLM is too frequently wrong.
- Faithfulness ≥ 0.80: 80% of answer claims are grounded in retrieved context. Below this, the system is hallucinating too frequently to be trustworthy.

These are baseline thresholds. After initial evaluation, they will be adjusted to match the actual baseline performance (you cannot set a threshold before you have the first measurement).

### When evaluation runs in CI

```yaml
# .github/workflows/ci.yml

on:
  push:
    branches: [main]      # Eval runs on main only
  workflow_dispatch:       # Manual trigger for development

jobs:
  eval:
    needs: [test, build]   # Only runs if tests pass and image builds
    steps:
      - run: python evaluation/run_eval.py --fail-below-thresholds
```

**Why only on `main`:** Eval uses `claude-sonnet-4-6` as judge, costing ~$2-5 per run. Running on every feature branch push would be expensive and unnecessary. The gate on `main` is what prevents regressions from being merged.

### What triggers a CI eval failure

| Change type | Eval needed? | Reason |
|-------------|-------------|--------|
| Prompt template change | Yes | Faithfulness often changes with prompt wording |
| Chunking parameter change | Yes | Affects which chunks are retrieved |
| Embedding model change | Yes | Changes the entire retrieval space |
| LLM model change | Yes | Different generation quality |
| Database migration | No | Schema change doesn't affect quality |
| Adding API endpoint | No | No effect on retrieval or generation |

---

## Evaluation Results as Repository Artifacts

`evaluation/results/` contains timestamped JSON files:

```
evaluation/results/
├── 2025-09-15_baseline.json
├── 2025-09-22_hybrid_rrf_added.json
└── 2025-10-01_prompt_v2.json
```

Each file records the full metric breakdown, the retrieval comparison table, and the model configuration at evaluation time. This history is the evidence of iterative improvement — a timeline of engineering decisions and their measured effects.

The README displays the most recent results.

---

## Consequences

**Positive:**
- Every change to retrieval or generation is immediately measurable
- The comparison table (BM25 vs dense vs hybrid) is a quantitative portfolio artifact
- Hallucination detection is automated, not manual
- The golden dataset demonstrates careful domain thinking
- CI gate prevents silent quality regression

**Negative:**
- Eval CI job costs $2-5 per run in API calls
- Golden dataset requires 3-4 hours of manual work upfront
- RAGAS evaluation adds ~2-3 minutes to CI pipeline

**Accepted trade-offs:** The manual work and API cost are investments. The golden dataset and evaluation results are the primary differentiators between this portfolio project and generic RAG tutorials.

---

## Alternatives Considered

### No formal evaluation (manual testing only)

Fast to build. Zero insight into quality. Not acceptable for a system whose purpose includes demonstrating evaluation competence.

### Custom evaluation metrics (no RAGAS)

Would demonstrate ability to implement evaluation from scratch. RAGAS is the established standard — using it signals ecosystem familiarity. Writing a custom evaluator would be reinventing a solved problem without clear benefit.

### LLM-generated golden dataset

Fast to generate. Low quality. Questions generated by an LLM from the same documents tend to be trivially answerable by retrieval. Manual curation is required to produce questions that genuinely test the system.

### Continuous evaluation on every commit

Maximises regression detection speed. Cost-prohibitive ($2-5 × number of commits per day). `main`-only gate is the right trade-off.

---

## Review Trigger

This decision should be revisited if:
- Initial evaluation shows Recall@5 or Faithfulness significantly above thresholds (raise the bar)
- Initial evaluation shows metrics below thresholds (investigate retrieval or generation before raising thresholds)
- RAGAS releases a major new version with substantially different metric methodology
