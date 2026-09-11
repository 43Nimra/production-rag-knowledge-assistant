# ADR-003: Provider Abstraction for Embeddings and LLM

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

The system depends on two external AI providers:

1. An **embedding model** to convert text into vectors (used at ingestion time and query time)
2. An **LLM** to generate answers from retrieved context

Both providers have real costs, potential outages, and may need to change over the lifetime of the project (cost optimisation, data privacy requirements, model quality improvements).

The question is: should provider selection be hardcoded or abstracted?

---

## Decision

Both the embedding provider and the LLM provider will be hidden behind **abstract base classes** from day one.

```
EmbedderBase          (abstract)
└── OpenAIEmbedder    (default implementation)

LLMClientBase         (abstract)
└── AnthropicLLMClient (default implementation)
```

Provider selection is controlled by environment variables. Swapping providers requires no application code changes.

---

## Embedding Provider

### Chosen: OpenAI `text-embedding-3-small`

**Why:**
- 1536 dimensions — sufficient quality for English enterprise documents
- $0.02 per million tokens — negligible cost at portfolio scale
- No infrastructure to manage
- Consistent, well-documented API with stable versioning

**Performance context:**  
`text-embedding-3-small` achieves 62.3% on MTEB (Massive Text Embedding Benchmark). `text-embedding-3-large` achieves 64.6% at 5× the cost. The quality delta does not justify the cost difference for this project.

### Alternative: `BAAI/bge-small-en-v1.5` (local, via sentence-transformers)

Implemented as `LocalEmbedder`. Use case: if documents contain sensitive data that cannot leave the network.

Swap procedure:
```bash
# .env
EMBEDDER_PROVIDER=local
LOCAL_EMBED_MODEL=BAAI/bge-small-en-v1.5
```

No application code changes.

**Important constraint:** Changing the embedding model requires full re-ingestion. The embedding vectors in `chunks.embedding` are in the new model's vector space; old vectors are incompatible. This is enforced by storing `embedding_model` in the `documents` table — a mismatch between ingested model and configured model raises a startup warning.

### Critical invariant

> The embedding model used at query time MUST be the same model used at ingestion time.

Violation causes silent retrieval failure: the query vector and chunk vectors are in different semantic spaces, producing low similarity scores for all chunks regardless of relevance.

---

## LLM Provider

### Chosen: Anthropic Claude

**Models:**
- `claude-haiku-4-5` — development, integration tests, high-volume calls
- `claude-sonnet-4-6` — evaluation runs, demo mode (higher quality, higher cost)

**Why Anthropic:**
- Strong instruction-following with structured prompts
- Reliable citation adherence — critical for RAG faithfulness
- Context window sufficient for top-5 chunks + system prompt
- Clear pricing tiers enabling conscious cost decisions

**Cost discipline:**  
The model used is configurable via `LLM_MODEL` in `.env`. Development defaults to Haiku. This is documented as a conscious engineering decision, not an accident. It mirrors real production practice: expensive models are reserved for evaluation and production; cheap models are used during development.

### Alternative: OpenAI GPT-4o / GPT-4o-mini

Implemented as `OpenAILLMClient`. Same `LLMClientBase` interface.

Swap procedure:
```bash
# .env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

No application code changes.

### Alternative: Local Ollama (self-hosted)

For fully offline operation. Appropriate if no internet access or data privacy constraints. Lower quality than API models but zero cost.

```bash
# .env
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Why This Abstraction Is Not Premature

This is the most common question about this design: "Isn't this over-engineering?"

No, for three reasons:

1. **The swap scenarios are real and likely.** Data privacy requirements, cost changes, and model quality improvements are not hypothetical. Every production RAG system I've seen has changed one of these providers within 12 months.

2. **The abstraction cost is low.** Two abstract base classes, two concrete implementations. This is ~100 lines of code. The interface is small: `embed_query()`, `embed_documents()`, `generate()`.

3. **It enables testing.** `MockEmbedder` and `MockLLMClient` implement the same interface. Unit tests run without any API calls. This is not possible with hardcoded provider calls.

---

## Interface Contracts

### EmbedderBase

```python
class EmbedderBase(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Returns a vector."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks. Returns a list of vectors.
        Implementations must handle batching internally."""
```

**Contract:**
- Output dimension is fixed per implementation (1536 for OpenAI)
- `embed_query` and `embed_documents` MUST use the same model and same normalisation
- Implementations MUST handle rate limits internally with exponential backoff

### LLMClientBase

```python
class LLMClientBase(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        context: str,
        question: str,
    ) -> LLMResponse:
        """Generate an answer. Returns LLMResponse with text and token counts."""
```

**Contract:**
- `system_prompt`, `context`, and `question` are ALWAYS separate parameters — never pre-concatenated
- Implementations MUST enforce a timeout (default: 30 seconds)
- Implementations MUST return token usage in `LLMResponse`
- On timeout or API error: raise `LLMClientError`, never return partial/empty text silently

---

## Consequences

**Positive:**
- Provider swap requires only `.env` change
- Unit tests run with zero API calls using mock implementations
- Token usage is tracked uniformly regardless of provider
- Cost-tiered model selection (Haiku for dev, Sonnet for eval) is documented and enforced

**Negative:**
- Two additional abstraction layers
- New contributors must understand the interface before adding a provider

**Accepted trade-off:** The abstraction cost (~100 lines) is small. The benefits — testability, swappability, cost control — are large. This is justified abstraction, not premature abstraction.

---

## Alternatives Considered

### Hardcode OpenAI/Anthropic calls directly in pipeline code

**Why not:** Makes testing require live API calls. Makes provider swap a refactor. Misses the opportunity to demonstrate interface-first design — one of the clearest signals of senior engineering thinking.

### Use LiteLLM for unified provider interface

**Why not:** LiteLLM is a legitimate production tool. However, for a portfolio project it introduces a dependency that hides the abstraction design we want to show. Writing the abstraction ourselves is a stronger signal than importing one.

---

## Review Trigger

This decision should be revisited if:
- A new provider offers substantially better cost/quality trade-off
- Embedding model evaluation shows meaningful quality improvement from `text-embedding-3-large`
- Data privacy requirements mandate fully local inference

---

## Addendum: Active Embedding Provider Switched to Local (2026-09-10)

**Status:** Accepted

This ADR's "Alternative" section for the embedding provider — `LocalEmbedder`
via `BAAI/bge-small-en-v1.5` (sentence-transformers) — is now the **active**
configuration for this deployment (`EMBEDDER_PROVIDER=local`), not just a
documented extension point. OpenAI's embedding API was not free
(no free tier; requires a minimum account deposit), and this project has
no funded API budget. `LocalEmbedder` costs nothing, needs no API key,
and has no rate limits — trade-offs against it are documented below.

**This decision was implemented, not just documented, in this pass:**
- `src/embeddings/local.py`: `LocalEmbedder` implementation
- `migrations/versions/002_switch_embedding_dim_local.py`: `chunks.embedding`
  changed from `vector(1536)` to `vector(384)` (bge-small-en-v1.5's output
  dimension) — the exact "explicit Alembic migration, not a silent
  dimension mismatch" this ADR's original text anticipated
- `docker/Dockerfile`: model weights baked into the image at build time
  (same reasoning as ADR-004's NLTK data — a hot path shouldn't depend on
  network access at request time)

**Trade-off accepted:** `sentence-transformers` pulls in `torch` as a
transitive dependency, which is substantial (multi-GB with the default
PyPI wheel, which includes unused CUDA support on this CPU-only
deployment). This works against ADR-006's <300MB image goal. A follow-up
optimization — pinning a CPU-only torch wheel via
`--extra-index-url https://download.pytorch.org/whl/cpu` — was identified
but not applied or verified in this pass (see PHASE2 implementation
report's Known Limitations).

**Consequence for provider swap-back:** Returning to `EMBEDDER_PROVIDER=openai`
requires downgrading the same migration (`alembic downgrade 001`) and
full re-ingestion, per this ADR's original "Important constraint" section
— pgvector requires one fixed dimension per column at a time, so the two
providers are not simultaneously active.
