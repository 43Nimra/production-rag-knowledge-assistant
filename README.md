# Production RAG Knowledge Assistant

A portfolio-grade Retrieval-Augmented Generation (RAG) system demonstrating end-to-end AI Engineering skills: document ingestion, hybrid retrieval, LLM-based generation, RAGAS evaluation, observability, and Docker deployment.

> **Status:** Architecture approved — Phase 1 implementation starting.

---

## What This System Does

Accepts documents (PDF, DOCX, HTML, TXT) → indexes them as searchable vector chunks → answers natural language questions with grounded responses and source citations.

```
Document upload                     Natural language query
      │                                       │
      ▼                                       ▼
   Parse                              Embed query
      │                                       │
   Chunk (sentence-aware)          Hybrid retrieval
      │                             ├─ Dense (pgvector ANN)
   Embed (OpenAI)                   ├─ Sparse (BM25)
      │                             └─ RRF fusion
   Store (PostgreSQL + pgvector)           │
                                    Build prompt + context
                                           │
                                    LLM generation (Claude)
                                           │
                                    Answer + citations
```

---

## Architecture Decisions

Every major technology decision is documented with rationale and trade-offs.

| ADR | Decision | Summary |
|-----|----------|---------|
| [ADR-001](docs/ADR/ADR-001-single-service-architecture.md) | Single service | One deployable unit with clean internal module boundaries |
| [ADR-002](docs/ADR/ADR-002-database-and-vector-search.md) | PostgreSQL + pgvector | One database for metadata and vectors; pgvector sufficient at portfolio scale |
| [ADR-003](docs/ADR/ADR-003-embedding-and-llm-provider-abstraction.md) | Provider abstraction | Embedder and LLM swappable via config; no hardcoded providers |
| [ADR-004](docs/ADR/ADR-004-retrieval-strategy.md) | Hybrid RRF retrieval | Dense + BM25 via Reciprocal Rank Fusion; custom chunker over LangChain |
| [ADR-005](docs/ADR/ADR-005-evaluation-strategy.md) | RAGAS + CI gate | 50-pair golden dataset; eval fails CI on quality regression |
| [ADR-006](docs/ADR/ADR-006-deployment-and-ci.md) | Docker Compose + GH Actions | Multi-stage Dockerfile; 4-stage CI with eval gate on main |

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API framework | FastAPI + Python 3.11 | Async-native, Pydantic validation, auto OpenAPI |
| Database | PostgreSQL 16 + pgvector | One DB for metadata + vectors; ACID + ANN search |
| Embedding | OpenAI text-embedding-3-small | Best cost/quality for English; provider-abstracted |
| LLM | Anthropic Claude (Haiku dev / Sonnet eval) | Strong citation adherence; cost-tiered |
| Sparse retrieval | BM25 via rank-bm25 | Exact-match queries where dense search underperforms |
| Retrieval fusion | Reciprocal Rank Fusion (k=60) | No learned weights; robust at small dataset sizes |
| Evaluation | RAGAS | Standard RAG eval framework; faithfulness + relevancy |
| Observability | structlog | Structured JSON logs with per-request trace_id |
| Deployment | Docker Compose | Single command; multi-stage image (~200MB) |
| CI | GitHub Actions | lint → test → build → eval gate |

**Intentionally excluded:** Kubernetes, Celery/Redis, Qdrant, LangChain, LlamaIndex, S3, microservices. See [ADRs](docs/ADR/) for reasoning.

---

## Quick Start

```bash
git clone https://github.com/your-username/rag-assistant
cd rag-assistant
cp .env.example .env          # Add your API keys
docker compose up             # Starts API + PostgreSQL
```

API available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## API Reference

### Authentication
```bash
curl -X POST http://localhost:8000/auth/token \
  -d "username=admin&password=changeme"
# Returns: {"access_token": "...", "token_type": "bearer"}
```

### Ingest a document
```bash
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@./sample_docs/deployment_guide.pdf"
# Returns: {"document_id": "uuid", "status": "INDEXED", "chunk_count": 42}
```

### Query the knowledge base
```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What must be done before starting the application server?"}'
# Returns: {"answer": "...", "sources": [...], "trace_id": "uuid", "latency_ms": 1847}
```

### Health check
```bash
curl http://localhost:8000/health
# Returns: {"status": "ok", "db": "connected"}
```

---

## Batch Ingestion

```bash
python scripts/ingest_docs.py ./sample_docs/
# Ingests all PDF, DOCX, HTML, TXT files in the directory
```

---

## Evaluation

### Run full evaluation
```bash
python evaluation/run_eval.py
```

### Latest results

> Results populated after Phase 5 implementation.

| Metric | Value | Threshold |
|--------|-------|-----------|
| Recall@5 (Hybrid RRF) | TBD | ≥ 0.70 |
| Recall@5 (Dense only) | TBD | — |
| Recall@5 (BM25 only) | TBD | — |
| MRR (Hybrid RRF) | TBD | — |
| Faithfulness | TBD | ≥ 0.80 |
| Answer Relevancy | TBD | — |
| Context Recall | TBD | — |
| Context Precision | TBD | — |

The retrieval comparison (BM25 vs Dense vs Hybrid) is the primary evaluation artifact — it demonstrates that technology choices were measured, not assumed.

---

## Project Documentation

| Document | Purpose |
|----------|---------|
| [SRS](docs/SRS.md) | Functional and non-functional requirements |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Component design, data flows, schema |
| [IMPLEMENTATION PLAN](docs/IMPLEMENTATION_PLAN.md) | Phased roadmap, task breakdown |
| [ADR directory](docs/ADR/) | All architecture decision records |

---

## Repository Structure

```
src/
├── api/           # FastAPI routes, auth, schemas, middleware
├── ingestion/     # Document parsers, sentence-aware chunker, pipeline
├── embeddings/    # EmbedderBase abstraction + OpenAI implementation
├── retrieval/     # pgvector ANN, BM25, RRF fusion, HybridRetriever
├── generation/    # Prompt builder, LLM client abstraction, citation parser
├── db/            # SQLAlchemy models, session, repository pattern
└── observability/ # structlog configuration, trace_id, metrics

evaluation/
├── dataset/       # golden_qa.json — 50 manually curated Q&A pairs
├── run_eval.py    # Main eval script with --ci-mode flag
└── results/       # Timestamped eval output JSONs

tests/
├── unit/          # Chunker, parsers, prompt builder, RRF (no DB)
└── integration/   # Ingestion pipeline, retrieval (real PostgreSQL)
```

---

## What Is Intentionally Not Built

| Excluded | Reason |
|----------|--------|
| Celery + Redis | FastAPI background tasks sufficient at this scale |
| Kubernetes | Docker Compose is the right tool; K8s would be premature complexity |
| Qdrant / Pinecone | pgvector adequate below 500K vectors; migration path preserved via abstraction |
| LangChain / LlamaIndex | Abstracts the components this project exists to demonstrate |
| Cross-encoder reranker | Deferred until eval data shows a gap |
| Streaming responses | UX feature; zero AI engineering signal |
| Multi-tenancy / ACL | Out of scope; documented extension point |
