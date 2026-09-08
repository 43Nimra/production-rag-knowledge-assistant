# ADR-002: PostgreSQL + pgvector Over Separate Vector Database

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

A RAG system requires two categories of storage:

1. **Relational metadata** — documents, chunks, ingestion status, query logs
2. **Vector storage** — high-dimensional embeddings with ANN (approximate nearest neighbor) search capability

The market offers dedicated vector databases (Qdrant, Weaviate, Pinecone, Chroma) specifically built for vector workloads. The question is whether using one of these alongside PostgreSQL provides meaningful benefit at portfolio scale.

---

## Decision

We will use **PostgreSQL 16 with the pgvector extension** for all storage needs — relational metadata and vector embeddings in one database.

No separate vector database will be deployed.

---

## Rationale

### One database, one operational surface

Using a separate vector database alongside PostgreSQL would mean:

- Two database connections to manage
- Two connection pools
- Two sets of credentials in `.env`
- Two backup strategies
- Two containers in Docker Compose
- Two potential failure domains per query

Every one of these costs is real and none of them is offset by a benefit at portfolio scale.

### pgvector is sufficient for this scale

pgvector 0.7+ introduced HNSW indexing. Performance characteristics:

| Vector count | pgvector HNSW | Qdrant HNSW |
|-------------|--------------|-------------|
| < 100K | Excellent | Excellent |
| 100K – 500K | Good | Excellent |
| 500K – 1M | Acceptable | Better |
| > 1M | Degraded | Better |

Portfolio scale is well under 100K vectors. pgvector is not a compromise — it is the correct tool at this scale.

### Joins between metadata and vectors are free

With pgvector, filtering retrieved chunks by `document_id`, `file_type`, or `created_at` is a SQL WHERE clause. With a separate vector database, this requires either:
- Duplicating metadata into the vector DB's payload index
- A two-phase query: retrieve from vector DB, then join in PostgreSQL

Both approaches add complexity. pgvector eliminates the problem entirely.

### The abstraction layer preserves the migration path

The `VectorStoreBase` abstract interface ensures that if scale genuinely requires Qdrant, the migration is localized to one file — `retrieval/vector_store.py`. No application code changes.

This is the correct answer to "wouldn't Qdrant be better?" in an interview: we have measured the trade-off, chosen the simpler tool for the current scale, and preserved the ability to migrate without a rewrite.

---

## Database Schema Design Decisions

### HNSW vs IVFFlat index

pgvector supports two ANN index types:

**HNSW (Hierarchical Navigable Small World)**
- No training/build phase required
- Good recall at low vector counts
- Higher memory usage per vector
- Chosen for this project

**IVFFlat (Inverted File with Flat compression)**
- Requires `VACUUM ANALYZE` to build effectively
- Lower memory footprint at high cardinality
- Less suitable without a training set

HNSW is chosen because it requires no warmup and performs reliably from the first document inserted.

### `filename UNIQUE` constraint

Enforces ingestion idempotency at the database level. If the application-level idempotency check fails (race condition, bug), the database constraint is the last line of defence.

### `embedding VECTOR(1536)`

1536 is the output dimension of `text-embedding-3-small`. This dimension is hardcoded in the schema intentionally:

- If the embedding model changes, the migration is explicit (Alembic migration required)
- There is no silent dimension mismatch
- The schema documents which embedding model was in use

### Separate `query_logs` table

Query logs are append-only and high-volume. Separating them from `chunks` avoids table bloat affecting vector index performance.

---

## Consequences

**Positive:**
- Single `docker compose up` with one database service
- SQL joins work natively between document metadata and retrieved chunks
- One backup strategy, one migration tool (Alembic), one connection pool
- Full ACID compliance for ingestion transactions

**Negative:**
- pgvector's filtered ANN search (WHERE clause + ANN) has known performance degradation at high cardinality with selective filters
- At >500K vectors, query latency under load will be measurably worse than Qdrant
- pgvector does not support approximate filtered search as efficiently as dedicated vector DBs

**Accepted trade-off:** These limitations are irrelevant at portfolio scale. The `VectorStoreBase` abstraction documents and preserves the migration path.

---

## Alternatives Considered

### Qdrant

**Why not chosen:** Requires a second Docker container, a second client library, payload index management for metadata, and either duplicated metadata or a two-phase query pattern. Zero benefit at < 100K vectors. Strong candidate if scale requirement emerges.

### Chroma

**Why not chosen:** Chroma is popular in tutorials but is not production-grade. It lacks the operational maturity, backup tooling, and enterprise adoption of either pgvector or Qdrant. Using it would not be a strong portfolio signal.

### Pinecone / Weaviate

**Why not chosen:** Managed cloud services. Introduce external dependency, account requirement, and cost. Inappropriate for a self-contained portfolio project that must run with `docker compose up`.

### SQLite + sqlite-vec

**Why not considered seriously:** SQLite is not suitable for a service with concurrent writes. pgvector is the correct choice.

---

## Review Trigger

This decision should be revisited if:
- Vector count exceeds 500K in practice
- Filtered ANN queries (by department, date range, document type) with high selectivity show measurable latency degradation
- Profiling shows pgvector as the clear bottleneck under load
