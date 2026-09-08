# ADR-001: Single Service Architecture Over Microservices

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

This is a portfolio project demonstrating AI Engineer / ML Engineer skills. The system needs a document ingestion pipeline and a query/retrieval/generation pipeline. The question is whether these should be separate services or a single deployable unit.

Common patterns seen in enterprise RAG systems include separate services for ingestion workers, retrieval API, generation API, and orchestration. The question is whether that complexity is justified here.

---

## Decision

We will build **one deployable Python service** containing both the ingestion pipeline and the query pipeline, with clean internal module boundaries.

The service runs as a single process. There is no distributed message passing between components. There is no separate worker process.

---

## Rationale

### The problem does not require microservices

Microservices are justified when:
- Individual components need to scale independently (ingestion CPU vs query CPU are different)
- Teams own different components and need deployment independence
- Components have genuinely different runtime requirements

None of these apply here. Ingestion is infrequent. The query pipeline is the hot path. Both fit comfortably in one process.

### Clean module boundaries provide the same conceptual separation

The internal structure mirrors what a microservices split would look like:

```
src/
├── ingestion/      ← would be "ingestion service"
├── retrieval/      ← would be "retrieval service"
├── generation/     ← would be "generation service"
└── api/            ← would be "API gateway"
```

An interviewer can see the boundaries. The code is as readable as a distributed system, without the operational overhead.

### Operational cost of microservices is high for zero portfolio benefit

A microservices split would require:
- Multiple Dockerfiles
- Service discovery or hardcoded URLs
- Network calls between components (with failure modes)
- A more complex Docker Compose or Kubernetes configuration
- More complex local development setup

None of this would be visible in an interview. The complexity is real; the signal is not.

### Monolith-first is the correct starting point

This is established engineering practice (Martin Fowler, Sam Newman). Extract services when you have demonstrated the need — not before. A premature microservices split often obscures the actual data flow, which is exactly what we want to show clearly.

---

## Consequences

**Positive:**
- `docker compose up` starts the entire system
- End-to-end request tracing requires no distributed tracing infrastructure
- All business logic is in one codebase, reviewable in one place
- Failure modes are local, not network failures between services

**Negative:**
- Cannot scale ingestion and query independently
- A bug in the ingestion pipeline can theoretically affect the query API (mitigated by robust error handling at each module boundary)

**Accepted trade-off:** At portfolio scale, independent scaling is not a requirement. The module structure documents the scaling seam — if scale were needed, `ingestion/` would become the ingestion service.

---

## Alternatives Considered

### Separate ingestion worker + API service

Would require: Redis or similar for task handoff, a second Docker container, inter-service communication. The portfolio signal from demonstrating task queue knowledge does not outweigh the complexity introduced and the clarity lost in following the data flow.

### Full microservices (ingestion, retrieval, generation, gateway)

Appropriate for a team of 4-8 engineers with independent deployment requirements. Completely unjustified for a portfolio project. Would obscure the RAG data flow behind network calls.

---

## Review Trigger

This decision should be revisited if:
- The project scope expands to multi-tenant with significant document volume
- Ingestion throughput becomes a bottleneck measurable by profiling
- A second team needs to own a separate component
