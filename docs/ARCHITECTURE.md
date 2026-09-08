# Architecture Document
## Production RAG Knowledge Assistant

**Version:** 1.0  
**Status:** Approved — ready for implementation  
**Last updated:** 2025-09-08

---

## 1. Architectural Philosophy

This system is designed around one guiding principle: **portfolio signal per unit of complexity**.

Every technology and every component must be justifiable on two grounds:
1. It solves a real problem in this system
2. It enables a meaningful technical conversation in an interview

Complexity that exists only to look impressive is treated as technical debt.

The architecture is deliberately **monolith-first**: one deployable Python service, one PostgreSQL database, one Docker Compose file. The internal module boundaries are as clean as a microservices split — but without the operational overhead that would be premature at this scale.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌──────────────────────────────────────────────────────────┐
│                     Client (REST)                        │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTPS / HTTP
┌──────────────────────▼───────────────────────────────────┐
│                  FastAPI Service                          │
│                                                          │
│  ┌─────────────────────┐  ┌──────────────────────────┐   │
│  │   Ingestion Pipeline │  │     Query Pipeline       │   │
│  │                     │  │                          │   │
│  │  Parser             │  │  Embedder (same model)   │   │
│  │    └─ PDF           │  │  HybridRetriever         │   │
│  │    └─ DOCX          │  │    ├─ pgvector ANN       │   │
│  │    └─ HTML          │  │    ├─ BM25               │   │
│  │    └─ TXT           │  │    └─ RRF fusion         │   │
│  │  SentenceChunker    │  │  ContextBuilder          │   │
│  │  Embedder           │  │  LLMClient               │   │
│  │  VectorStore.write  │  │  CitationParser          │   │
│  └──────────┬──────────┘  └─────────────┬────────────┘   │
│             │                           │                 │
│  ┌──────────▼───────────────────────────▼────────────┐   │
│  │              Repository (DB layer)                │   │
│  └──────────────────────────┬─────────────────────────┘   │
│                             │                             │
│  ┌──────────────────────────▼─────────────────────────┐   │
│  │           Observability (structlog + trace_id)     │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────┐
│              PostgreSQL 16 + pgvector                    │
│                                                          │
│   documents    chunks    query_logs    eval_results      │
└──────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

| Component | Responsibility | Key constraint |
|-----------|---------------|----------------|
| `api/` | HTTP layer, auth, request validation, routing | No business logic; delegates to pipelines |
| `ingestion/` | Parse → chunk → embed → store | Idempotent; fails atomically per document |
| `retrieval/` | Dense + sparse search, RRF fusion | Same embedding model as ingestion |
| `generation/` | Prompt construction, LLM call, citation parsing | Never passes raw user input to prompt directly |
| `embeddings/` | Abstract interface + OpenAI implementation | Provider-swappable via config |
| `db/` | ORM models, session, repository pattern | No raw SQL in routes or pipelines |
| `observability/` | Structured logging, trace ID propagation | trace_id injected at request boundary |
| `evaluation/` | Offline eval harness, RAGAS, CI gate | Runs against golden dataset; separate from app |

---

## 3. Module Dependency Graph

Dependencies flow in one direction only. No circular imports.

```
api/
 └── depends on → ingestion/, retrieval/, generation/, db/, observability/

ingestion/
 └── depends on → embeddings/, db/, observability/

retrieval/
 └── depends on → embeddings/, db/, observability/

generation/
 └── depends on → observability/

embeddings/
 └── depends on → observability/

db/
 └── depends on → (external: PostgreSQL only)

observability/
 └── depends on → (nothing internal)

evaluation/
 └── depends on → retrieval/, generation/, db/ (read-only)
```

`observability/` is the only module imported by all others. It has no internal dependencies, preventing any circular import risk.

---

## 4. Data Models

### 4.1 PostgreSQL Schema

```sql
-- documents: one row per ingested file
CREATE TABLE documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename    TEXT NOT NULL UNIQUE,
    file_type   TEXT NOT NULL,           -- 'pdf' | 'docx' | 'html' | 'txt'
    status      TEXT NOT NULL,           -- 'PENDING' | 'INDEXED' | 'PARSE_FAILED'
    chunk_count INTEGER,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- chunks: one row per text segment
CREATE TABLE chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    page_number INTEGER,                 -- NULL for formats without pages
    embedding   VECTOR(1536),           -- text-embedding-3-small output dim
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks (document_id);

-- query_logs: one row per API query request
CREATE TABLE query_logs (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id             UUID NOT NULL,
    query_hash           TEXT NOT NULL,   -- SHA-256[:16] of query text
    retrieved_chunk_ids  UUID[],
    retrieval_latency_ms INTEGER,
    llm_latency_ms       INTEGER,
    total_latency_ms     INTEGER,
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 4.2 Schema Design Decisions

**`filename UNIQUE`:** Enforces ingestion idempotency at the database level, not just application level.

**`embedding VECTOR(1536)`:** Matches `text-embedding-3-small` output dimensions. If the embedding model changes, this column must be migrated — which is intentional: it makes model changes explicit and auditable via Alembic.

**`query_hash` not `query_text`:** Raw query text is never persisted. SHA-256 truncated to 16 chars enables correlation without storing PII.

**`HNSW` index on embeddings:** Chosen over IVFFlat because HNSW does not require a training step and handles small-to-medium datasets better. At >500K vectors, IVFFlat's lower memory footprint may become preferable.

---

## 5. Ingestion Pipeline

### 5.1 Data Flow

```
POST /documents  (multipart file upload)
         │
         ▼
  [Validation]
  - File type check (extension + MIME)
  - File size check (max 50MB)
  - Pydantic schema validation
         │
         ▼
  [Parser dispatch]
  - .pdf  → PdfParser (pdfplumber)
  - .docx → DocxParser (python-docx)
  - .html → HtmlParser (BeautifulSoup4 + lxml)
  - .txt  → PlainTextParser
         │
         ▼
  [Text extraction]
  - Extract raw text (+ page numbers for PDF)
  - Detect empty extraction → status = PARSE_FAILED, stop
  - Normalize whitespace
  - Remove boilerplate (headers, footers if detectable)
         │
         ▼
  [Sentence-aware chunking]
  - Split at sentence boundaries
  - Target: 512 tokens per chunk
  - Overlap: 64 tokens (sliding window)
  - Attach: document_id, chunk_index, page_number
         │
         ▼
  [Embedding (batched)]
  - Batch chunks (max 100 per API call)
  - Call OpenAI text-embedding-3-small
  - Retry on rate limit (exponential backoff, max 3 attempts)
         │
         ▼
  [Database write (transaction)]
  - Upsert document record (PENDING → INDEXED)
  - Insert chunk rows
  - Insert embedding vectors
  - Commit → success / Rollback → document stays PENDING
         │
         ▼
  Response: { document_id, status, chunk_count, duration_ms }
```

### 5.2 Failure Modes and Handling

| Failure | Detection | Handling |
|---------|-----------|----------|
| Scanned PDF (no text layer) | `len(extracted_text) < MIN_TEXT_THRESHOLD` | Mark `PARSE_FAILED`, log `{doc_id, reason: "empty_extraction"}`, return 422 |
| Password-protected DOCX | `python-docx` raises `PackageNotFoundError` | Catch, mark `PARSE_FAILED`, return 422 |
| Embedding API rate limit | HTTP 429 response | Exponential backoff: 1s, 2s, 4s. After 3 failures: mark `PENDING`, return 503 |
| Embedding API outage | HTTP 5xx or timeout | Same as rate limit |
| DB write failure | SQLAlchemy exception | Rollback transaction, document stays `PENDING`, return 500 |
| Malformed file content | Parser exception | Catch at parser boundary, mark `PARSE_FAILED`, log exception |
| Empty chunks after cleaning | `len(chunks) == 0` | Mark `PARSE_FAILED`, log warning, return 422 |

---

## 6. Query Pipeline

### 6.1 Data Flow

```
POST /query  { "question": "...", "top_k": 5 }
         │
         ▼
  [Validation]
  - JWT authentication
  - Pydantic: question length 1–2000 chars
  - top_k: 1–20
         │
         ▼
  [Query embedding]
  - Embed question via same EmbedderBase implementation
  - (Same model as ingestion — critical for vector space alignment)
         │
         ▼
  [Hybrid retrieval]
  ┌──────────────────────────────────┐
  │  Dense retrieval (pgvector)      │
  │  - ANN search: top-20 candidates │
  │  - cosine similarity             │
  └──────────────┬───────────────────┘
                 │         ┌──────────────────────────────────┐
                 │         │  Sparse retrieval (BM25)         │
                 │         │  - BM25 over chunk text corpus   │
                 │         │  - top-20 candidates             │
                 │         └──────────────┬───────────────────┘
                 │                        │
                 ▼                        ▼
  [RRF score fusion]
  - score(d) = Σ 1 / (k + rank_i(d))   where k=60
  - Produces unified ranking
  - Select top-K (default 5)
         │
         ▼
  [Context construction]
  - Format: "[Source: {filename}, chunk {i}]\n{chunk_text}"
  - Concatenate top-K chunks
  - Calculate total token count
  - Truncate if approaching context window limit (log warning)
         │
         ▼
  [Prompt assembly]
  - System prompt: instructs LLM to answer only from context
  - Context block: formatted chunks with source labels
  - User turn: validated question (never interpolated directly)
         │
         ▼
  [LLM call]
  - AnthropicLLMClient (or swappable via config)
  - Model: claude-haiku-4-5 (dev) / claude-sonnet-4-6 (configured)
  - Timeout: 30 seconds
  - On timeout → 503, do NOT retry immediately
         │
         ▼
  [Citation parsing]
  - Extract source references from response
  - Validate citations against retrieved chunk sources
  - Flag if answer references source not in retrieved set
         │
         ▼
  [Logging]
  - Write query_log row (async, does not block response)
         │
         ▼
  Response: {
    "answer": "...",
    "sources": [{ "filename": "...", "chunk_index": N, "page_number": N }],
    "trace_id": "uuid",
    "retrieval_latency_ms": N,
    "llm_latency_ms": N
  }
```

### 6.2 Failure Modes and Handling

| Failure | Handling |
|---------|----------|
| Zero chunks retrieved | Return: `{"answer": "No relevant information found.", "sources": []}` — never hallucinate |
| LLM timeout (>30s) | HTTP 503, log `{trace_id, error: "llm_timeout"}` |
| LLM returns uncited answer | Log warning with trace_id; return answer with empty sources list |
| Context exceeds token limit | Truncate to top-3 chunks, log warning `{trace_id, truncation: true}` |
| Embedding fails for query | HTTP 503, do not proceed to retrieval |

---

## 7. Retrieval Strategy: Hybrid RRF

### 7.1 Why Hybrid

Dense vector search alone has a known weakness: queries containing specific technical terms, product codes, or proper nouns can retrieve semantically similar but factually wrong chunks. BM25 keyword matching handles these exact-match cases well.

Reciprocal Rank Fusion combines both rankings without requiring learned weights, making it robust and parameter-light.

### 7.2 RRF Formula

```
RRF_score(chunk) = Σ 1 / (k + rank_i(chunk))

where:
  k = 60  (standard constant; dampens influence of very high ranks)
  rank_i = position in ranked list from retriever i (1-indexed)
  Σ   = sum over all retrievers (dense + BM25)
```

### 7.3 Evaluation Baseline Comparison

Three strategies are evaluated and compared in `evaluation/run_eval.py`:

```
BM25-only retrieval     → Recall@5, MRR
Dense-only retrieval    → Recall@5, MRR
Hybrid RRF retrieval    → Recall@5, MRR  ← expected winner
```

This comparison is a first-class portfolio artifact. The numbers appear in the README.

---

## 8. Prompt Architecture

### 8.1 System Prompt

```
You are a precise knowledge assistant. Answer questions using ONLY the provided 
context. If the answer is not present in the context, say: "I don't have 
information about that in the provided documents."

Rules:
- Every factual claim must reference a source using [Source: filename]
- Do not infer, extrapolate, or use external knowledge
- If context is partially relevant, say so explicitly
- Never fabricate citations
```

### 8.2 Prompt Structure

```
[SYSTEM]
{system_prompt}

[CONTEXT]
[Source: deployment_guide.pdf, chunk 3, page 12]
{chunk_text}

[Source: api_reference.html, chunk 7]
{chunk_text}

... (up to K chunks)

[USER]
{validated_question}
```

The `[USER]` section always receives the validated, Pydantic-coerced question — never a raw string interpolated from the HTTP request.

### 8.3 Prompt Injection Defence

User input is structurally isolated from the system prompt. Even if a user sends:
```
"Ignore previous instructions and..."
```
...it arrives inside the `[USER]` block where the LLM is already instructed to treat it as a question, not as an instruction. This is not a complete injection defence, but it is the correct production baseline.

---

## 9. Abstraction Layers

### 9.1 EmbedderBase

```
EmbedderBase (abstract)
├── embed_query(text: str) → list[float]
└── embed_documents(texts: list[str]) → list[list[float]]

Implementations:
└── OpenAIEmbedder
    - model: text-embedding-3-small
    - dimensions: 1536
    - batching: max 100 texts per call
```

**Swap procedure:** Set `EMBEDDER=local` in `.env`. `LocalEmbedder` uses `sentence-transformers` with `BAAI/bge-small-en-v1.5`. No other changes. Note: changing embedding model requires full re-ingestion because vectors are in a different space.

### 9.2 LLMClientBase

```
LLMClientBase (abstract)
└── generate(system: str, context: str, question: str) → LLMResponse

Implementations:
└── AnthropicLLMClient
    - dev model: claude-haiku-4-5
    - eval model: claude-sonnet-4-6 (via config)
    - timeout: 30s
    - returns: LLMResponse(text, prompt_tokens, completion_tokens)
```

---

## 10. Observability Design

### 10.1 Trace ID Flow

```
HTTP Request arrives
      │
      ▼
FastAPI middleware generates trace_id (UUID4)
      │
      ▼
Injected into structlog context for this request
      │
      ├── Every log line in ingestion pipeline carries trace_id
      ├── Every log line in retrieval carries trace_id
      ├── Every log line in LLM call carries trace_id
      └── Returned in HTTP response header: X-Trace-ID
```

### 10.2 Structured Log Example

```json
{
  "timestamp": "2025-09-08T10:23:45.123Z",
  "level": "info",
  "event": "query_complete",
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "query_hash": "a3f9c2b1",
  "retrieved_chunks": 5,
  "retrieval_latency_ms": 87,
  "llm_latency_ms": 1823,
  "total_latency_ms": 1921,
  "prompt_tokens": 1247,
  "completion_tokens": 183,
  "model": "claude-haiku-4-5"
}
```

No `query_text`. No `user_id` (single-user system). No raw content. Safe to ship to any log aggregator.

---

## 11. Security Model

| Concern | Mitigation |
|---------|------------|
| Unauthenticated access | JWT required on all data endpoints |
| API key leakage | All secrets via env vars; `.env` in `.gitignore`; `.env.example` with placeholders |
| Prompt injection via user input | Structural isolation: user input in `[USER]` block, never in system prompt |
| Prompt injection via document content | Documents indexed as data, not as instructions; system prompt establishes authority |
| PII in logs | Query text never logged; only truncated hash |
| Malformed input crashing service | Pydantic validation at API boundary; parser exceptions caught per-document |
| Oversized uploads | File size limit enforced before parsing (50MB max) |

**What this security model does NOT cover** (out of scope, documented):
- Document-level access control (all indexed documents visible to any authenticated user)
- User management / registration
- Audit logging of which user accessed which document
- Rate limiting per user (global rate limit only)

---

## 12. Deployment Architecture

```
┌─────────────────────────────────────────────────────┐
│                docker-compose.yml                   │
│                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  │
│  │  api (FastAPI)      │  │  postgres            │  │
│  │                     │  │                      │  │
│  │  port: 8000         │  │  port: 5432 (local)  │  │
│  │  multi-stage build  │  │  image: postgres:16  │  │
│  │  (builder + slim)   │  │  pgvector enabled    │  │
│  └──────────┬──────────┘  └──────────────────────┘  │
│             │  volume: ./sample_docs → /app/docs     │
│             │  env_file: .env                        │
└─────────────────────────────────────────────────────┘
```

**Multi-stage Dockerfile rationale:** Builder stage installs all build dependencies and compiles packages. Runtime stage copies only the installed packages and source — no build tools, no cache. Result: ~200MB image vs ~2GB single-stage.

**`docker compose up` is the only setup command** beyond copying `.env.example`.

---

## 13. What Is Intentionally Not Built

This is an architecture document. What is excluded is as important as what is included.

| Excluded | Why |
|----------|-----|
| Celery + Redis | FastAPI background tasks are sufficient; distributed queue adds an operator, a process, and a failure domain with zero benefit at this scale |
| Qdrant / Weaviate / Pinecone | pgvector is adequate below 500K vectors; abstraction layer enables migration; one database is one backup strategy |
| LangChain / LlamaIndex | Abstracts the components we specifically want to demonstrate understanding of; hides chunking, retrieval, and prompt logic from code reviewers |
| Kubernetes | Single-host deployment is correct; K8s adds YAML surface area without engineering substance at this scale |
| Reranker (cross-encoder) | Deferred until eval data shows gap; adding before measurement is premature optimization |
| Streaming responses | UX feature with no AI engineering signal |
| Microservices | The problem does not require distributed compute; module boundaries provide separation without operational overhead |
| S3 / MinIO | Local filesystem + Docker volume; object storage configuration is not a portfolio signal |
| Multi-turn conversation | Stateful conversation adds complexity; each query stands alone |
