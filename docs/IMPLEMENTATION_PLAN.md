# Implementation Plan
## Production RAG Knowledge Assistant

**Version:** 1.0  
**Status:** Approved — ready to execute  
**Last updated:** 2025-09-08

---

## Guiding Principles

1. **Every phase ends with a working, demonstrable system.** No phase ends with half-implemented features that require the next phase to be useful.
2. **Tests are written alongside implementation, not after.** A feature without tests is not done.
3. **No new technologies are introduced during implementation.** Technology decisions are closed. See the ADRs.
4. **Incremental complexity.** The simplest thing that works is implemented first. Sophistication is added only when the simpler version is working and tested.
5. **The evaluation pipeline is built in Phase 5, not last.** Eval is a core deliverable, not an afterthought.

---

## Repository Structure (Target)

This is the full target structure. Each phase incrementally builds toward it.

```
rag-assistant/
│
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app, lifespan hooks, middleware
│   │   ├── dependencies.py      # Auth verification, DB session, embedder, LLM
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py          # POST /auth/token
│   │   │   ├── documents.py     # POST /documents
│   │   │   ├── query.py         # POST /query
│   │   │   └── health.py        # GET /health, GET /metrics
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── auth.py          # TokenRequest, TokenResponse
│   │       ├── documents.py     # IngestResponse
│   │       └── query.py         # QueryRequest, QueryResponse, SourceReference
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Orchestrates parse → chunk → embed → store
│   │   ├── parsers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # ParseResult dataclass, ParserBase ABC
│   │   │   ├── pdf.py           # PdfParser (pdfplumber)
│   │   │   ├── docx.py          # DocxParser (python-docx)
│   │   │   ├── html.py          # HtmlParser (BeautifulSoup4)
│   │   │   └── txt.py           # PlainTextParser
│   │   └── chunker.py           # SentenceAwareChunker, ChunkResult dataclass
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py              # EmbedderBase ABC
│   │   └── openai.py            # OpenAIEmbedder
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py      # VectorStoreBase ABC + PgVectorStore
│   │   └── retriever.py         # HybridRetriever (dense + BM25 + RRF)
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── prompt_builder.py    # Assembles system prompt + context + question
│   │   ├── llm_client.py        # LLMClientBase ABC + AnthropicLLMClient
│   │   └── response_parser.py   # Extracts and validates citations
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy ORM: Document, Chunk, QueryLog
│   │   ├── session.py           # Engine factory, get_session dependency
│   │   └── repository.py        # DocumentRepo, ChunkRepo, QueryLogRepo
│   │
│   └── observability/
│       ├── __init__.py
│       ├── logger.py            # structlog configuration, get_logger()
│       └── metrics.py           # In-memory counters: requests, tokens, latency
│
├── evaluation/
│   ├── dataset/
│   │   └── golden_qa.json       # 50 manually curated Q&A pairs
│   ├── run_eval.py              # Main eval script (CLI: --ci-mode flag)
│   ├── retrieval_metrics.py     # Recall@K, MRR computation
│   ├── ragas_eval.py            # RAGAS faithfulness, relevancy, recall, precision
│   └── results/                 # Timestamped eval output JSONs (.gitkeep)
│
├── tests/
│   ├── conftest.py              # Shared fixtures: test DB, mock embedder, mock LLM
│   ├── unit/
│   │   ├── test_chunker.py      # Chunker boundary conditions
│   │   ├── test_parsers.py      # Parser extraction correctness
│   │   ├── test_prompt_builder.py  # Prompt assembly structure
│   │   ├── test_response_parser.py # Citation extraction
│   │   └── test_rrf.py          # RRF score fusion logic
│   └── integration/
│       ├── test_ingestion_pipeline.py  # Full parse → chunk → embed → store
│       └── test_retrieval.py    # Full query embed → retrieve → rank
│
├── migrations/
│   ├── env.py                   # Alembic environment
│   ├── script.py.mako
│   └── versions/
│       └── 001_initial_schema.py
│
├── scripts/
│   └── ingest_docs.py           # CLI: python scripts/ingest_docs.py ./sample_docs/
│
├── docker/
│   ├── Dockerfile               # Multi-stage: builder + slim runtime
│   └── docker-compose.yml       # api + postgres services
│
├── docs/
│   ├── SRS.md                   # Software Requirements Specification ✓
│   ├── ARCHITECTURE.md          # System architecture document ✓
│   ├── IMPLEMENTATION_PLAN.md   # This file ✓
│   └── ADR/
│       ├── ADR-001-single-service-architecture.md ✓
│       ├── ADR-002-database-and-vector-search.md ✓
│       ├── ADR-003-embedding-and-llm-provider-abstraction.md ✓
│       ├── ADR-004-retrieval-strategy.md ✓
│       ├── ADR-005-evaluation-strategy.md ✓
│       └── ADR-006-deployment-and-ci.md ✓
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── sample_docs/                 # Demo documents for ingestion (committed)
│   └── .gitkeep
│
├── config.py                    # Pydantic Settings — single source of config truth
├── .env.example                 # All required env vars with placeholder values
├── pyproject.toml               # Dependencies + tool config (ruff, mypy, pytest)
└── README.md                    # Setup instructions + architecture diagram + eval results
```

---

## Phase 1: Project Foundation

**Goal:** A running FastAPI service connected to PostgreSQL, with CI passing, before any AI components.

**Deliverable:** `docker compose up` starts the API. `GET /health` returns `{"status": "ok", "db": "connected"}`. GitHub Actions runs lint + type check.

### Tasks

**1.1 — Project scaffolding**
- `pyproject.toml` with all approved dependencies pinned
- `config.py` with Pydantic Settings reading from environment
- `.env.example` documenting all required variables
- `.gitignore` excluding `.env`, `__pycache__`, `evaluation/results/*.json`

**1.2 — Database setup**
- `src/db/models.py`: SQLAlchemy ORM models for `documents`, `chunks`, `query_logs`
- `src/db/session.py`: async engine factory, `get_session` FastAPI dependency
- `src/db/repository.py`: `DocumentRepository`, `ChunkRepository`, `QueryLogRepository`
- `migrations/versions/001_initial_schema.py`: Alembic migration for all tables + pgvector extension + HNSW index

**1.3 — Observability foundation**
- `src/observability/logger.py`: `structlog` configuration, JSON output, `get_logger()`
- `src/observability/metrics.py`: in-memory counters (requests, tokens, latency)
- Middleware: generate `trace_id` per request, bind to structlog context, inject into response header `X-Trace-ID`

**1.4 — API skeleton**
- `src/api/main.py`: FastAPI app with lifespan (DB connection check on startup), CORS, logging middleware
- `src/api/routes/health.py`: `GET /health`, `GET /metrics`
- `src/api/routes/auth.py`: `POST /auth/token` (JWT issuance)
- `src/api/dependencies.py`: JWT verification dependency

**1.5 — Docker**
- `docker/Dockerfile`: multi-stage builder + slim runtime, non-root user
- `docker/docker-compose.yml`: `api` + `postgres` with health check and dependency ordering

**1.6 — CI**
- `.github/workflows/ci.yml`: quality job (ruff + mypy), test job (pytest with postgres service container)

### Tests for Phase 1
- `tests/unit/`: none yet (no business logic)
- `tests/integration/test_health.py`: health endpoint returns 200 + db connected

### Phase 1 done when
- [ ] `docker compose up` starts without errors
- [ ] `GET /health` returns `{"status": "ok", "db": "connected"}`
- [ ] `POST /auth/token` returns a valid JWT
- [ ] GitHub Actions quality job passes (ruff + mypy)
- [ ] `alembic upgrade head` runs cleanly and creates all tables

---

## Phase 2: Ingestion Pipeline

**Goal:** A document can be uploaded and indexed end-to-end. Chunks and vectors appear in the database.

**Deliverable:** `POST /documents` with a PDF returns `{"status": "INDEXED", "chunk_count": N}`. The `scripts/ingest_docs.py` CLI can batch-ingest a folder.

### Tasks

**2.1 — Parsers**
- `src/ingestion/parsers/base.py`: `ParseResult(text: str, page_numbers: dict[int, str] | None)`, `ParserBase` ABC
- `src/ingestion/parsers/pdf.py`: `PdfParser` using `pdfplumber` — extracts text with page numbers, detects empty extraction
- `src/ingestion/parsers/docx.py`: `DocxParser` using `python-docx` — extracts paragraph text
- `src/ingestion/parsers/html.py`: `HtmlParser` using `BeautifulSoup4` + `lxml` — strips tags, extracts body text
- `src/ingestion/parsers/txt.py`: `PlainTextParser` — reads file, normalises whitespace
- Parser dispatch: `get_parser(filename: str) -> ParserBase` based on file extension

**2.2 — Chunker**
- `src/ingestion/chunker.py`: `SentenceAwareChunker`
  - Splits text at sentence boundaries (NLTK `sent_tokenize`)
  - Accumulates sentences until token budget (`tiktoken` for token counting)
  - Slides window by `overlap` tokens
  - Returns `list[ChunkResult(text, chunk_index, start_char, page_number)]`

**2.3 — Embedder**
- `src/embeddings/base.py`: `EmbedderBase` ABC with `embed_query()` and `embed_documents()`
- `src/embeddings/openai.py`: `OpenAIEmbedder`
  - Batches `embed_documents()` calls (max 100 per API call)
  - Exponential backoff on rate limit (1s, 2s, 4s, max 3 retries)
  - Raises `EmbedderError` on persistent failure

**2.4 — Vector store (write path)**
- `src/retrieval/vector_store.py`: `VectorStoreBase` ABC + `PgVectorStore`
  - `upsert_chunks(chunks: list[ChunkWithEmbedding], document_id: UUID) -> None`
  - Uses SQLAlchemy bulk insert for efficiency

**2.5 — Ingestion pipeline orchestrator**
- `src/ingestion/pipeline.py`: `IngestionPipeline.ingest(file, filename, session)`
  - Validates file type and size
  - Dispatches to correct parser
  - Detects empty parse result → `PARSE_FAILED`
  - Chunks parsed text
  - Batches embedding calls
  - Writes to DB in a single transaction (upsert document, insert chunks + vectors)
  - Updates document status

**2.6 — API route**
- `src/api/routes/documents.py`: `POST /documents` (multipart file upload, JWT required)
- `src/api/schemas/documents.py`: `IngestResponse(document_id, status, chunk_count, duration_ms)`

**2.7 — CLI script**
- `scripts/ingest_docs.py`: walks a directory, calls ingestion pipeline for each file, prints summary

### Tests for Phase 2
- `tests/unit/test_chunker.py`: empty input, single sentence, paragraph with overlap, token boundary
- `tests/unit/test_parsers.py`: fixture files for each format (tiny PDF, DOCX, HTML, TXT)
- `tests/integration/test_ingestion_pipeline.py`: ingest a real PDF → verify chunks and vectors in DB

### Fixtures required
- `tests/fixtures/sample.pdf` — small multi-page PDF (can be generated)
- `tests/fixtures/sample.docx`
- `tests/fixtures/sample.html`
- `tests/fixtures/sample.txt`
- `tests/fixtures/scanned_only.pdf` — PDF with no text layer (for PARSE_FAILED test)

### Phase 2 done when
- [ ] `POST /documents` with a PDF returns `{"status": "INDEXED"}`
- [ ] Chunks appear in `chunks` table with non-null `embedding` column
- [ ] Scanned PDF returns 422 with `{"status": "PARSE_FAILED", "reason": "empty_extraction"}`
- [ ] `python scripts/ingest_docs.py ./sample_docs/` ingests all documents in a folder
- [ ] All chunker and parser unit tests pass
- [ ] Integration test for ingestion pipeline passes

---

## Phase 3: Retrieval

**Goal:** A query returns the most relevant chunks from the indexed documents. No LLM yet.

**Deliverable:** `POST /query` returns `{"chunks": [...], "retrieval_latency_ms": N}` — raw retrieved chunks, no generation.

### Tasks

**3.1 — Dense retrieval**
- `src/retrieval/vector_store.py`: add `search(query_vector, top_k) -> list[ScoredChunk]`
  - Uses pgvector `<=>` cosine distance operator
  - Returns chunks with their cosine similarity scores

**3.2 — BM25 retrieval**
- `src/retrieval/retriever.py`: `BM25Retriever`
  - Loads all chunk texts from DB (cached with `functools.lru_cache`)
  - Uses `rank_bm25.BM25Okapi` for scoring
  - Returns `list[ScoredChunk]` with BM25 scores

**3.3 — RRF fusion**
- `src/retrieval/retriever.py`: `reciprocal_rank_fusion(ranked_lists, k=60) -> list[ScoredChunk]`
  - Pure function, no DB dependency
  - Fully unit testable

**3.4 — HybridRetriever**
- `src/retrieval/retriever.py`: `HybridRetriever`
  - Calls dense retrieval (top-20) + BM25 (top-20) in parallel (`asyncio.gather`)
  - Applies RRF fusion
  - Returns top-K chunks

**3.5 — Query route (retrieval-only response)**
- `src/api/routes/query.py`: `POST /query` returns retrieved chunks only
  - This is intentionally retrieval-only in Phase 3 — generation added in Phase 4
  - Allows testing and iterating on retrieval quality before adding LLM cost

### Tests for Phase 3
- `tests/unit/test_rrf.py`: RRF with known ranked lists, verify score ordering
- `tests/integration/test_retrieval.py`: ingest sample docs, query, verify relevant chunk in results

### Phase 3 done when
- [ ] `POST /query {"question": "..."}` returns top-5 chunks with scores
- [ ] Cosine similarity search returns sensible results on ingested documents
- [ ] BM25 returns results for exact-term queries
- [ ] Hybrid RRF returns better results than either alone (manual verification on sample queries)
- [ ] RRF unit tests pass
- [ ] Integration test for retrieval passes

---

## Phase 4: Generation + Full RAG

**Goal:** Full end-to-end RAG. Question in, grounded answer with citations out.

**Deliverable:** `POST /query` returns `{"answer": "...", "sources": [...], "trace_id": "...", "latency_ms": N}`.

### Tasks

**4.1 — Prompt builder**
- `src/generation/prompt_builder.py`: `PromptBuilder`
  - `build_system_prompt() -> str`: the fixed system instruction (answer only from context, cite sources)
  - `build_context_block(chunks: list[ScoredChunk]) -> str`: formats chunks with source labels
  - `count_tokens(system, context, question) -> int`: checks total token count before call
  - Truncation logic: if over limit, drop lowest-ranked chunks and log warning

**4.2 — LLM client**
- `src/generation/llm_client.py`: `LLMClientBase` ABC + `AnthropicLLMClient`
  - `generate(system, context, question) -> LLMResponse(text, prompt_tokens, completion_tokens)`
  - 30-second timeout
  - Raises `LLMClientError` on timeout or API error — never returns empty string silently
  - Logs token usage per call

**4.3 — Citation parser**
- `src/generation/response_parser.py`: `ResponseParser`
  - Extracts `[Source: filename]` references from LLM response
  - Validates that cited sources exist in the retrieved chunk set
  - Logs warning if LLM cites a source not in context (potential hallucination signal)

**4.4 — Query pipeline (full)**
- Update `POST /query` to full RAG flow:
  - embed query → hybrid retrieve → build prompt → call LLM → parse citations → log → respond
  - Log `QueryLog` row asynchronously (does not block response)

**4.5 — Structured logging for query**
- Every query logs: `trace_id`, `query_hash`, `retrieved_chunk_count`, `retrieval_latency_ms`, `llm_latency_ms`, `total_latency_ms`, `prompt_tokens`, `completion_tokens`, `model`

### Tests for Phase 4
- `tests/unit/test_prompt_builder.py`: correct structure, source label format, truncation at token limit
- `tests/unit/test_response_parser.py`: citation extraction, invalid citation detection
- `tests/integration/test_query_pipeline.py`: mock LLM + mock embedder, verify full pipeline with real DB

### Phase 4 done when
- [ ] `POST /query` returns answer + sources + trace_id + latency
- [ ] Answer for "answer not in docs" question is "I don't have information about that"
- [ ] Query log row written to DB on every query
- [ ] `X-Trace-ID` header present in every response
- [ ] Prompt builder unit tests pass
- [ ] Citation parser unit tests pass

---

## Phase 5: Evaluation Pipeline

**Goal:** Measurable quality baseline. CI gate on `main`. Retrieval comparison table.

**Deliverable:** `python evaluation/run_eval.py` prints metric table. CI fails on regression. README has numbers.

### Tasks

**5.1 — Golden dataset**
- Create `evaluation/dataset/golden_qa.json` with 50 manually curated Q&A pairs
- Category distribution: factual (15), multi-sentence reasoning (15), exact-match terms (10), negative/no-answer (5), cross-document (5)
- Every pair manually verified against actual document content
- Format: `{id, question, ground_truth_answer, relevant_document, difficulty, category}`

**5.2 — Retrieval metrics**
- `evaluation/retrieval_metrics.py`:
  - `compute_recall_at_k(results, ground_truth, k) -> float`
  - `compute_mrr(results, ground_truth) -> float`
  - Run for all three strategies: BM25-only, dense-only, hybrid RRF
  - Output: comparison table

**5.3 — RAGAS integration**
- `evaluation/ragas_eval.py`:
  - Run full query pipeline on all 50 golden questions
  - Collect: question, generated answer, retrieved contexts, ground truth answer
  - Pass to RAGAS: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`
  - Uses `claude-sonnet-4-6` as judge model

**5.4 — Eval script**
- `evaluation/run_eval.py`:
  - `--ci-mode`: exits with code 1 if thresholds not met
  - `--output-file`: writes results JSON to `evaluation/results/YYYY-MM-DD_HH-MM.json`
  - Prints formatted table to stdout

**5.5 — CI eval job**
- Add `eval` job to `.github/workflows/ci.yml`
  - Runs on push to `main` only
  - Uses `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` from repository secrets
  - Fails pipeline if thresholds breached

**5.6 — README results table**
- Add evaluation results section to `README.md`
- Show retrieval comparison: BM25 vs dense vs hybrid
- Show RAGAS metrics: faithfulness, answer relevancy, context recall, context precision

### Phase 5 done when
- [ ] `python evaluation/run_eval.py` runs to completion and prints metrics
- [ ] Three-way retrieval comparison table is populated with real numbers
- [ ] RAGAS metrics are computed for all 50 golden questions
- [ ] CI eval job runs on `main` push and fails if thresholds are breached
- [ ] README contains real evaluation numbers

---

## Phase 6: Polish and Portfolio Readiness

**Goal:** The repository is in the state a senior engineer would be proud to submit as a portfolio piece.

**Deliverable:** `git clone` → `cp .env.example .env` → add API keys → `docker compose up` → working system.

### Tasks

**6.1 — README completion**
- Project description and architecture diagram (Mermaid)
- Setup instructions (exactly 5 steps from clone to running)
- API documentation (key endpoints with example curl commands)
- Evaluation results table
- "What I deliberately excluded and why" section
- ADR index with one-line summaries

**6.2 — Dockerfile final review**
- Verify multi-stage build produces correct image size
- Verify non-root user is used
- Verify `.dockerignore` excludes tests, docs, `.env`, `.git`

**6.3 — Sample documents**
- Add 5-10 coherent sample documents to `sample_docs/`
- Must be from a single domain (not random)
- Must support the golden dataset questions
- Must be public domain or created for this project

**6.4 — Error response consistency**
- All error responses follow `{"error": {"code": "...", "message": "...", "trace_id": "..."}}`
- No raw exception messages exposed to clients

**6.5 — Final CI verification**
- All CI jobs pass on a clean run from `main`
- No hardcoded secrets, no `.env` in the repo
- `docker compose up` starts cleanly from scratch (no pre-existing data)

**6.6 — Code review pass**
- No TODO comments left in production code
- No commented-out code
- All public functions have docstrings
- Type annotations complete (mypy clean)

### Phase 6 done when
- [ ] `git clone` → setup → `docker compose up` works in under 5 minutes
- [ ] All 6 CI jobs pass on a fresh `main` branch push
- [ ] README is readable by a non-technical recruiter and a senior engineer
- [ ] No hardcoded secrets anywhere in the repository
- [ ] `docker images rag-assistant_api` shows image < 300MB

---

## Dependency List (Approved)

### Runtime dependencies

```toml
[project.dependencies]
# API
fastapi = "^0.115"
uvicorn = {extras = ["standard"], version = "^0.32"}
pydantic = "^2.9"
pydantic-settings = "^2.6"
python-multipart = "^0.0.12"       # file upload support
python-jose = {extras = ["cryptography"], version = "^3.3"}
passlib = {extras = ["bcrypt"], version = "^1.7"}

# Database
sqlalchemy = {extras = ["asyncio"], version = "^2.0"}
asyncpg = "^0.30"                   # async PostgreSQL driver
alembic = "^1.14"
pgvector = "^0.3"                   # SQLAlchemy type for VECTOR column

# Parsing
pdfplumber = "^0.11"
python-docx = "^1.1"
beautifulsoup4 = "^4.12"
lxml = "^5.3"

# AI / ML
openai = "^1.55"
anthropic = "^0.40"
tiktoken = "^0.8"                   # Token counting
nltk = "^3.9"                       # Sentence tokenization
rank-bm25 = "^0.2"                  # BM25 retrieval

# Observability
structlog = "^24.4"
```

### Development and evaluation dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest = "^8.3"",
    "pytest-asyncio = "^0.24"",
    "pytest-cov = "^6.0"",
    "httpx = "^0.27"",              # FastAPI test client
    "ruff = "^0.8"",
    "mypy = "^1.13"",
    "ragas = "^0.2"",               # RAG evaluation framework
]
```

---

## What Is Not Being Built

This is an explicit exclusion list. Each item was considered and rejected with reasoning documented in the ADRs.

| Excluded component | ADR reference | Reason |
|-------------------|--------------|--------|
| Celery + Redis | ADR-001 | FastAPI background tasks sufficient; distributed queue unjustified |
| Kubernetes | ADR-006 | Docker Compose is correct at this scale |
| Qdrant / Weaviate / Pinecone | ADR-002 | pgvector sufficient; abstraction preserves migration path |
| LangChain / LlamaIndex | ADR-004 | Hides the components we want to demonstrate |
| Cross-encoder reranker | ADR-005 | Deferred until eval data justifies the latency cost |
| OCR (Tesseract) | SRS §7 | Detect and fail gracefully; solve if needed as extension |
| Streaming LLM responses | SRS §7 | UX feature, zero AI engineering signal |
| Multi-tenancy / document ACL | SRS §7 | Single-user system; documented extension point |
| Admin UI / frontend | SRS §7 | Backend/ML portfolio; Swagger auto-generated |
| Conversation history | SRS §7 | Stateful multi-turn adds complexity without core RAG signal |
| S3 / MinIO | SRS §7 | Local volume sufficient for portfolio |
| Container registry pipeline | ADR-006 | Build verification in CI is sufficient |

---

## Open Decisions (Require Your Input Before Phase 5)

Only two decisions remain open. Everything else is resolved.

**OD-001 — Sample document domain**  
What domain are your sample documents from? This determines the golden dataset questions. Choose one coherent domain (e.g. FastAPI official docs, PostgreSQL docs, a public company annual report, Wikipedia articles on a single topic). A random mix of unrelated documents produces less meaningful eval results.

*Required before:* Phase 5 (golden dataset creation)

**OD-002 — Live demo hosting**  
Do you want the system accessible at a public URL for demos? If yes, we add a Caddy reverse proxy for HTTPS to the Docker Compose file. If no, `localhost:8000` is sufficient.

*Required before:* Phase 6 (polish)

---

## Implementation Start

Phase 1 can begin immediately. All architecture decisions are closed. All technology choices are approved.

**First implementation task:** `pyproject.toml` + `config.py` + `docker/docker-compose.yml` + `docker/Dockerfile`.
