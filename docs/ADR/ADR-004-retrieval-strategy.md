# ADR-004: Hybrid Retrieval (Dense + BM25 via RRF) and Custom Chunker

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

Two core decisions in a RAG system determine retrieval quality above all others:

1. **How documents are split into chunks** (chunking strategy)
2. **How relevant chunks are found at query time** (retrieval strategy)

Poor choices here cannot be compensated by a better LLM or a more sophisticated prompt. Garbage in, garbage out.

This ADR covers both decisions together because they are tightly coupled: the chunking strategy affects the BM25 corpus, the token budget affects embedding quality, and the retrieval comparison (BM25 vs dense vs hybrid) only has meaning if chunking is consistent across all three.

---

## Decision 1: Custom Sentence-Aware Chunker

We will implement a sentence-aware chunker directly (~60 lines) rather than importing a framework chunker (LangChain, LlamaIndex).

**Configuration:**
- Target chunk size: 512 tokens (configurable via `CHUNK_SIZE` env var)
- Overlap: 64 tokens (configurable via `CHUNK_OVERLAP` env var)
- Sentence boundary detection: NLTK `sent_tokenize` or Python `re` sentence splitter

### Why not fixed-size character splitting

Fixed-size splitting cuts at arbitrary character positions. This produces:

```
# Fixed-size split — bad
chunk_1: "The deployment process requires running migrations before"
chunk_2: " starting the application server. Failure to do so will"
chunk_3: " result in undefined behaviour and potential data loss."
```

The sentence "Failure to do so will result in undefined behaviour" is split across two chunks. Neither chunk contains a complete semantic unit. The embedding of each chunk is degraded because the sentence has no syntactic or semantic closure.

Sentence-aware splitting respects natural language boundaries:

```
# Sentence-aware split — correct
chunk_1: "The deployment process requires running migrations before 
          starting the application server."
chunk_2: "Failure to do so will result in undefined behaviour and 
          potential data loss."
```

Each chunk is a complete semantic unit. Its embedding vector captures a coherent concept.

### Why not LangChain's RecursiveCharacterTextSplitter

LangChain is a legitimate production tool. For this project, importing it for the chunker specifically has two costs:

1. **Hundreds of transitive dependencies** enter the project for one component
2. **The chunking logic is hidden** — an interviewer reading the code cannot see how chunks are formed

A 60-line chunker written in this codebase is:
- Fully readable and explainable line by line
- Unit testable with complete coverage
- Zero transitive dependencies
- A stronger portfolio signal than a one-line framework call

### Why 512 tokens

512 tokens is the empirically established sweet spot for RAG chunk size in English enterprise documents:
- Large enough to contain a complete concept with context
- Small enough that each chunk is focused (not diluted by surrounding text)
- Stays well within the embedding model's effective context window

### Why 64-token overlap

Overlap ensures that a concept sitting at the boundary between two chunks is fully captured in at least one of them. 64 tokens (~48 words) is approximately one to two sentences — enough to preserve boundary context without significant duplication.

---

## Decision 2: Hybrid Retrieval with Reciprocal Rank Fusion

We will implement hybrid retrieval combining:
- **Dense retrieval** via pgvector HNSW approximate nearest neighbor search
- **Sparse retrieval** via BM25 keyword matching
- **Fusion** via Reciprocal Rank Fusion (RRF)

### Why dense retrieval alone is insufficient

Dense embeddings capture semantic similarity. This is powerful but has a known failure mode: queries containing specific technical terms, model numbers, error codes, or proper nouns can retrieve semantically related but factually wrong chunks.

Example:
```
Query: "How do I fix ORA-12541 error?"
Dense retrieval might return: chunks about Oracle connection errors generally
BM25 retrieval returns: chunks containing the exact string "ORA-12541"
```

For enterprise documentation — which is dense with product names, version numbers, error codes, and proper nouns — exact-match capability is not optional.

### Why not dense + re-ranking instead

A cross-encoder reranker is deferred (see ADR-007). Hybrid retrieval provides complementary signal fusion at the retrieval stage, before the reranker would apply. These are not mutually exclusive — hybrid retrieval + reranking is the gold standard. We start with hybrid retrieval and add reranking if eval data shows a gap.

### RRF Formula and Parameter Choice

```
RRF_score(chunk) = Σ  1 / (k + rank_i(chunk))
                  i

where:
  k     = 60   (standard constant from the original RRF paper)
  rank_i = position in ranked list from retriever i (1-indexed, lower = better)
  Σ      = sum over all retrievers (dense + BM25 = 2 retrievers)
```

**Why k=60:** The constant k dampens the influence of very highly-ranked results from a single retriever, preventing one retriever from dominating when the other has no opinion. k=60 is the value from the original Cormack et al. 2009 paper and is widely validated. It requires no tuning.

**Why not learned fusion weights:** Learned weights require a training set of relevance judgments. We have a golden dataset of 50 pairs — insufficient to reliably learn weights. RRF's parameter-free design is more robust at this dataset size.

### Retrieval pipeline configuration

- Dense candidates: top-20 from pgvector ANN
- BM25 candidates: top-20 from BM25 over full chunk corpus
- After RRF fusion: return top-K (default K=5, configurable)

Top-20 from each retriever before fusion provides enough candidates that both signals can contribute meaningfully, without the latency cost of retrieving top-100.

---

## Evaluation Comparison

A three-way retrieval comparison is a first-class artifact of this project:

```
┌─────────────────┬──────────┬──────────┬──────────────┐
│ Strategy        │ Recall@5 │ MRR      │ Notes        │
├─────────────────┼──────────┼──────────┼──────────────┤
│ BM25 only       │ TBD      │ TBD      │ Baseline     │
│ Dense only      │ TBD      │ TBD      │ Semantic     │
│ Hybrid RRF      │ TBD      │ TBD      │ Combined     │
└─────────────────┴──────────┴──────────┴──────────────┘
```

Numbers are filled in after evaluation runs. The table appears in the project README.

This comparison demonstrates measurement-driven engineering: "We didn't choose hybrid because RAG articles recommend it. We chose it because our eval showed X% improvement over dense-only on our specific document domain."

---

## BM25 Implementation

Library: `rank_bm25` (pure Python, no compiled dependencies, no external service).

The BM25 corpus is built from all chunk texts at query time with an LRU-cached index. At portfolio scale this is fast enough. At larger scale, the BM25 index would be pre-built and persisted.

```python
# Conceptual — not the implementation
corpus = [chunk.text for chunk in all_chunks]  # fetched from DB, cached
bm25 = BM25Okapi([text.split() for text in corpus])
scores = bm25.get_scores(query.split())
```

**Known limitation:** The BM25 index is rebuilt per query if not cached, and the full corpus must fit in memory. For the portfolio scale of tens of thousands of chunks, this is acceptable. At >1M chunks, a pre-built persistent index (Elasticsearch, OpenSearch) would be required.

---

## Consequences

**Positive:**
- Hybrid retrieval demonstrably outperforms single-strategy retrieval on mixed query types
- Custom chunker is fully readable, testable, and explainable in interviews
- RRF requires no training data and no parameter tuning
- The three-way comparison produces a quantitative portfolio artifact

**Negative:**
- BM25 corpus must fit in memory (acceptable at portfolio scale)
- BM25 index rebuild cost grows linearly with chunk count
- Custom chunker has a bug surface we own; mitigated by comprehensive unit tests

**Accepted trade-offs:**
- Memory-resident BM25 index: documented limitation with known migration path (pre-built index)
- Custom chunker complexity: offset by testability and interview explainability

---

## Alternatives Considered

### Fixed-size character chunking

Simpler to implement. Demonstrably worse retrieval quality due to mid-sentence cuts. Not considered seriously.

### Parent-child chunking (small chunks for retrieval, large chunks for context)

Sophisticated approach. Requires additional data model complexity (parent-child chunk relationships). Deferred as a potential Phase 2 enhancement if eval shows that retrieved chunks lack sufficient context.

### Dense retrieval only

Simpler retrieval path. Known failure mode on exact-match queries. The BM25 baseline comparison makes this failure mode measurable and documented.

### Elasticsearch / OpenSearch for BM25

Production-grade BM25 with persistent index. Requires a third Docker service. Zero portfolio benefit over `rank_bm25` at this scale. Excluded.

---

## Review Trigger

This decision should be revisited if:
- BM25 memory usage becomes a measurable problem (chunk count > 500K)
- Eval shows that hybrid RRF does not outperform dense-only on the specific document domain
- A reranker is added (see ADR-007) — at which point retrieval candidate count (currently top-20) may need adjustment
