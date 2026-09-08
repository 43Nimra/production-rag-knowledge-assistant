# Software Requirements Specification
## Production RAG Knowledge Assistant

**Version:** 1.0  
**Status:** Approved  
**Last updated:** 2025-09-08

---

## 1. Purpose

This document defines the functional and non-functional requirements for the Production RAG Knowledge Assistant — a portfolio-grade system that demonstrates end-to-end AI engineering skills including document ingestion, vector retrieval, LLM-based generation, evaluation, observability, and deployment.

The system is intentionally scoped to demonstrate depth of understanding over breadth of features. Every component exists because it signals engineering judgment, not because it adds complexity.

---

## 2. System Overview

The system is a single deployable Python service that:

1. Accepts documents (PDF, DOCX, HTML, TXT) and indexes them into a vector store
2. Answers natural language questions using Retrieval-Augmented Generation (RAG)
3. Returns grounded answers with traceable source citations
4. Measures its own quality via an offline evaluation pipeline
5. Exposes observability data on every request

```
┌─────────────────────────────────────────────────┐
│              RAG Knowledge Assistant             │
│                                                 │
│  ┌─────────────┐        ┌────────────────────┐  │
│  │  Ingestion  │        │   Query Pipeline   │  │
│  │  Pipeline   │        │                    │  │
│  │             │        │  embed → retrieve  │  │
│  │  parse →    │        │  → rerank? → gen   │  │
│  │  chunk →    │        │  → cite → respond  │  │
│  │  embed →    │        │                    │  │
│  │  store      │        └────────────────────┘  │
│  └─────────────┘                                │
│          │                        │             │
│          ▼                        ▼             │
│     ┌─────────────────────────────────┐         │
│     │   PostgreSQL 16 + pgvector      │         │
│     │   documents | chunks | vectors  │         │
│     │   query_logs | eval_results     │         │
│     └─────────────────────────────────┘         │
└─────────────────────────────────────────────────┘
```

---

## 3. Users and Use Cases

### 3.1 Users

**Primary user:** A developer or recruiter evaluating the repository as a portfolio artifact.

**Secondary user (demo context):** Any person who wants to query a set of indexed documents via REST API.

### 3.2 Use Cases

| ID | Use Case | Priority |
|----|----------|----------|
| UC-01 | Upload a document for indexing | P0 |
| UC-02 | Query the indexed knowledge base with a natural language question | P0 |
| UC-03 | Receive a grounded answer with source citations | P0 |
| UC-04 | Authenticate before accessing the API | P0 |
| UC-05 | Check system health and readiness | P0 |
| UC-06 | Run offline evaluation against a golden dataset | P1 |
| UC-07 | View structured logs for a specific request via trace ID | P1 |
| UC-08 | Ingest a batch of documents via CLI script | P1 |

---

## 4. Functional Requirements

### 4.1 Document Ingestion

| ID | Requirement |
|----|-------------|
| FR-01 | The system SHALL accept PDF, DOCX, HTML, and TXT files via `POST /documents` |
| FR-02 | The system SHALL extract text from each document using format-appropriate parsers |
| FR-03 | The system SHALL detect and gracefully handle parse failures (e.g. scanned PDFs with no text layer), logging them as `PARSE_FAILED` |
| FR-04 | The system SHALL split extracted text into chunks using sentence-aware splitting with configurable token budget (default: 512 tokens) and overlap (default: 64 tokens) |
| FR-05 | The system SHALL attach metadata to every chunk: `document_id`, `source_filename`, `chunk_index`, `page_number` (where available) |
| FR-06 | The system SHALL generate an embedding vector for each chunk using the configured embedding model |
| FR-07 | The system SHALL store document metadata, chunk text, chunk metadata, and embedding vectors in PostgreSQL |
| FR-08 | Ingestion SHALL be idempotent — re-ingesting the same document by filename SHALL update the existing record, not create a duplicate |
| FR-09 | The system SHALL expose ingestion status per document: `PENDING`, `INDEXED`, `PARSE_FAILED` |

### 4.2 Query and Retrieval

| ID | Requirement |
|----|-------------|
| FR-10 | The system SHALL accept a natural language question via `POST /query` |
| FR-11 | The system SHALL embed the query using the same embedding model used during ingestion |
| FR-12 | The system SHALL retrieve candidate chunks using hybrid retrieval: dense vector search (pgvector ANN) combined with sparse keyword search (BM25) fused via Reciprocal Rank Fusion (RRF) |
| FR-13 | The system SHALL return the top-K chunks (default K=5, configurable) after RRF fusion |
| FR-14 | The system SHALL construct a prompt that includes retrieved chunks with source labels and the user's question |
| FR-15 | The system SHALL call the configured LLM to generate a grounded answer |
| FR-16 | The system SHALL parse and validate citations from the LLM response against retrieved chunk sources |
| FR-17 | The system SHALL return the answer, list of source references, trace ID, and latency in milliseconds |
| FR-18 | The system SHALL log every query with: `trace_id`, `query_hash`, `retrieved_chunk_count`, `retrieval_latency_ms`, `llm_latency_ms`, `total_latency_ms`, `token_usage` |

### 4.3 Authentication

| ID | Requirement |
|----|-------------|
| FR-19 | The system SHALL require JWT Bearer token authentication on all `/documents` and `/query` endpoints |
| FR-20 | The system SHALL expose `POST /auth/token` to exchange credentials for a JWT |
| FR-21 | JWT tokens SHALL expire after a configurable TTL (default: 24 hours) |
| FR-22 | The `/health` endpoint SHALL be publicly accessible without authentication |

### 4.4 Observability

| ID | Requirement |
|----|-------------|
| FR-23 | Every request SHALL be assigned a unique `trace_id` (UUID) injected as a request-scoped context variable |
| FR-24 | All log output SHALL be structured JSON (via `structlog`) |
| FR-25 | Logs SHALL never contain raw query text — only `query_hash` (SHA-256 truncated) |
| FR-26 | The system SHALL expose `GET /health` returning service status and database connectivity |
| FR-27 | The system SHALL expose `GET /metrics` returning aggregate token usage and request count since startup |

### 4.5 Evaluation

| ID | Requirement |
|----|-------------|
| FR-28 | The system SHALL include an offline evaluation script `evaluation/run_eval.py` |
| FR-29 | Evaluation SHALL run against a golden dataset of ≥50 manually labeled Q&A pairs |
| FR-30 | Evaluation SHALL compute: `recall@5`, `MRR`, `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision` |
| FR-31 | Evaluation SHALL compare three retrieval strategies: BM25-only, dense-only, hybrid RRF |
| FR-32 | Evaluation results SHALL be written to `evaluation/results/` as timestamped JSON |
| FR-33 | CI SHALL fail if `faithfulness < 0.80` or `recall@5 < 0.70` on the golden dataset |

---

## 5. Non-Functional Requirements

### 5.1 Performance

| ID | Requirement | Rationale |
|----|-------------|-----------|
| NFR-01 | P50 query latency SHALL be under 4 seconds end-to-end | Acceptable for a knowledge assistant; LLM calls dominate |
| NFR-02 | P95 query latency SHALL be under 8 seconds | Degraded but still usable |
| NFR-03 | Document ingestion SHALL process at least 10 documents per minute | Single-worker baseline |
| NFR-04 | The system SHALL handle at least 10 concurrent query requests without degradation | Portfolio demo scale |

### 5.2 Reliability

| ID | Requirement |
|----|-------------|
| NFR-05 | A failed document parse SHALL not affect other documents in a batch |
| NFR-06 | A failed LLM call SHALL return HTTP 503 with a clear error message; it SHALL NOT return a hallucinated answer |
| NFR-07 | Database write failures during ingestion SHALL trigger a transaction rollback; the document SHALL be marked `PENDING` for retry |
| NFR-08 | The system SHALL start cleanly from `docker compose up` with no manual setup steps beyond copying `.env.example` |

### 5.3 Security

| ID | Requirement |
|----|-------------|
| NFR-09 | API keys and secrets SHALL only be read from environment variables; never hardcoded |
| NFR-10 | User input SHALL be validated by Pydantic before reaching any business logic |
| NFR-11 | Retrieved context and user query SHALL be passed as separate named placeholders in the prompt — never string-concatenated directly |
| NFR-12 | Raw query text SHALL NOT appear in logs — only its hash |
| NFR-13 | The system prompt SHALL explicitly instruct the LLM to answer only from provided context |

### 5.4 Maintainability

| ID | Requirement |
|----|-------------|
| NFR-14 | Embedding provider SHALL be swappable via config change with no application code change |
| NFR-15 | LLM provider SHALL be swappable via config change with no application code change |
| NFR-16 | All configuration SHALL be read from environment variables via Pydantic Settings |
| NFR-17 | Database schema changes SHALL be managed via Alembic migrations |
| NFR-18 | No circular imports between modules |

### 5.5 Testability

| ID | Requirement |
|----|-------------|
| NFR-19 | Chunker SHALL have unit tests covering: empty input, single sentence, multi-paragraph, boundary overlap |
| NFR-20 | Each parser (PDF, DOCX, HTML, TXT) SHALL have unit tests with fixture files |
| NFR-21 | Retrieval pipeline SHALL have integration tests against a real PostgreSQL + pgvector instance |
| NFR-22 | LLM and embedding calls SHALL be mockable for unit test isolation |

### 5.6 Cost Awareness

| ID | Requirement |
|----|-------------|
| NFR-23 | Development and integration tests SHALL use `claude-haiku-4-5` to minimize cost |
| NFR-24 | Evaluation runs SHALL use `claude-sonnet-4-6` for quality accuracy |
| NFR-25 | CI evaluation job SHALL run only on pushes to `main` branch to control API cost |
| NFR-26 | Embedding calls during ingestion SHALL be batched (max batch size: 100 chunks per API call) |

---

## 6. Constraints

- **Language:** Python 3.11 only
- **Database:** PostgreSQL 16 + pgvector extension. No separate vector database.
- **Deployment:** Docker Compose. No Kubernetes.
- **Framework selection:** No LangChain or LlamaIndex. Core RAG components are implemented directly.
- **Document formats:** PDF, DOCX, HTML, TXT only. No images, spreadsheets, or presentations.
- **Scale:** Designed for up to ~100K document chunks. pgvector is sufficient at this scale.

---

## 7. Explicitly Out of Scope

The following are intentionally excluded from this system. Each represents a deliberate scope decision, not an oversight.

| Feature | Reason Excluded |
|---------|-----------------|
| OCR for scanned PDFs | Significant dependency (Tesseract); detect and fail gracefully instead |
| Multi-tenancy / document ACL | Single-user system; documented extension point |
| Async task queue (Celery/Redis) | FastAPI background tasks sufficient at portfolio scale |
| Reranking (cross-encoder) | Deferred until eval data justifies the latency cost |
| Streaming LLM responses | UX feature; zero AI engineering signal |
| Admin UI / frontend | Backend/ML portfolio; Swagger UI is sufficient |
| Conversation history | Each query is independent; multi-turn adds state complexity |
| Object storage (S3/MinIO) | Local filesystem + Docker volume is sufficient |
| Kubernetes / container orchestration | Premature at this scale; Docker Compose is correct |
| Microservices | Single service is the right architecture for this problem size |

---

## 8. Glossary

| Term | Definition |
|------|------------|
| RAG | Retrieval-Augmented Generation — answering questions by retrieving relevant context before generating a response |
| Chunk | A text segment produced by splitting a document; the unit of retrieval |
| Embedding | A high-dimensional vector representation of text, capturing semantic meaning |
| ANN | Approximate Nearest Neighbor search — fast vector similarity search |
| BM25 | Best Match 25 — a classical keyword-based ranking function |
| RRF | Reciprocal Rank Fusion — a score fusion method combining rankings from multiple retrievers |
| Faithfulness | RAGAS metric: proportion of the answer that is grounded in the retrieved context |
| Trace ID | A UUID assigned per request, propagated through all log lines for that request |
| Golden dataset | A manually curated set of Q&A pairs used as ground truth for evaluation |
| RAGAS | Retrieval Augmented Generation Assessment — an open-source evaluation framework |
