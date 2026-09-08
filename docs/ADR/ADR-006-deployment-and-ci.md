# ADR-006: Deployment with Docker Compose and GitHub Actions CI

**Status:** Accepted  
**Date:** 2025-09-08  
**Deciders:** Principal AI Architect

---

## Context

The system needs a deployment strategy and a CI pipeline. Two questions:

1. **Deployment:** How does the system run? Who operates it?
2. **CI:** What runs on every commit, and what gates merges to `main`?

For a portfolio project, both must satisfy: works reliably, demonstrable in a live session, and shows production-aware thinking without over-engineering infrastructure.

---

## Decision: Deployment

We will deploy using **Docker Compose with a multi-stage Dockerfile**. No Kubernetes. No container registry pipeline. No managed cloud services.

`docker compose up` is the complete deployment command.

---

## Docker Compose Design

### Services

```yaml
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://rag:rag@postgres:5432/rag_db
    env_file:
      - .env
    depends_on:
      postgres:
        condition: service_healthy
    volumes:
      - ./sample_docs:/app/sample_docs:ro  # Read-only mount for demo docs

  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag_db"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
```

### Why `pgvector/pgvector:pg16` not `postgres:16`

The official `pgvector/pgvector:pg16` image has the pgvector extension pre-installed. Using the base `postgres:16` image would require either a custom Dockerfile for the database container or running `CREATE EXTENSION` setup scripts. The pre-built image eliminates this complexity.

### Health check on postgres

The `api` service uses `depends_on: condition: service_healthy`. This ensures the API container does not start (and fail its database connection) before PostgreSQL is ready to accept connections. Without this, a race condition causes the API to crash on startup in a fresh environment.

This is a detail that separates production-aware Docker Compose from tutorial Docker Compose. It is worth mentioning in an interview.

---

## Multi-Stage Dockerfile Design

```dockerfile
# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
RUN pip install --no-cache-dir build && pip install .

# Stage 2: runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY src/ ./src/
COPY config.py .

RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Why multi-stage

| Approach | Image size | Build tools in prod |
|----------|-----------|---------------------|
| Single stage | ~1.8 GB | Yes (security risk) |
| Multi-stage | ~200 MB | No |

The runtime image contains only:
- Python 3.11 slim base
- Installed package binaries
- Application source code

It does not contain: pip, setuptools, gcc, build headers, or cache files.

**Interview talking point:** "My production image is 200MB because I use a multi-stage build. The builder stage installs everything needed to compile packages; the runtime stage copies only the installed artifacts. This reduces attack surface and speeds up container pulls."

### Non-root user

The runtime container runs as `appuser`, not `root`. This is a security baseline that costs nothing to implement. A container running as root means a container escape vulnerability becomes root-on-host.

---

## Decision: CI Pipeline

We will use **GitHub Actions** with a four-stage pipeline.

```
push / pull_request
       │
       ▼
┌──────────────────────────────────────────────────────┐
│  Job 1: quality (fast, ~2 min)                       │
│  - ruff check (linting)                              │
│  - ruff format --check (formatting)                  │
│  - mypy src/ (type checking)                         │
└──────────────────────┬───────────────────────────────┘
                       │ (on success)
                       ▼
┌──────────────────────────────────────────────────────┐
│  Job 2: test (medium, ~5 min)                        │
│  - spin up postgres:pgvector service container       │
│  - run alembic upgrade head                          │
│  - pytest tests/unit/ (no DB)                        │
│  - pytest tests/integration/ (with DB)               │
└──────────────────────┬───────────────────────────────┘
                       │ (on success)
                       ▼
┌──────────────────────────────────────────────────────┐
│  Job 3: build (fast, ~3 min)                         │
│  - docker build (verify image builds cleanly)        │
│  - (does not push to registry)                       │
└──────────────────────┬───────────────────────────────┘
                       │ (on push to main only)
                       ▼
┌──────────────────────────────────────────────────────┐
│  Job 4: eval (slow, ~10 min, main branch only)       │
│  - spin up postgres:pgvector service container       │
│  - ingest golden dataset documents                   │
│  - python evaluation/run_eval.py --ci-mode           │
│  - fail if recall@5 < 0.70 or faithfulness < 0.80   │
└──────────────────────────────────────────────────────┘
```

### Job 1: Quality

**ruff** replaces flake8 + isort + black. It is 10-100× faster than any combination of those tools and is rapidly becoming the Python standard.

**mypy** catches type errors that tests miss. It also serves as living documentation — typed function signatures make the codebase more readable.

**Why both:** Linting finds style issues and obvious bugs. Type checking finds semantic errors. They complement, not duplicate, each other.

### Job 2: Test

The integration tests use a **PostgreSQL service container** — not a mock or SQLite. Testing against the real database engine ensures:
- pgvector queries actually execute correctly
- HNSW index behaviour is tested
- Migration correctness is verified (`alembic upgrade head` runs in CI)

This is the difference between tests that check your code and tests that check your system.

### Job 3: Build

The Docker image is built in CI on every push. This catches:
- Missing `COPY` statements
- Dependency resolution failures in the build stage
- File permission issues

It does not push to a registry — this is a portfolio project without a registry infrastructure. The build verification is what matters.

### Job 4: Eval (main branch only)

The evaluation job is the CI differentiator. It:
1. Ingests the sample documents into a fresh test database
2. Runs `evaluation/run_eval.py` against the golden dataset
3. Fails the CI pipeline if quality metrics drop below thresholds

**Why only on `main`:**
- Each eval run costs ~$2-5 in API calls (LLM judge + embedding generation)
- Running on every feature branch push is wasteful
- The gate on `main` is where it matters — preventing regressions from being merged

**`--ci-mode` flag:**  
In CI mode, `run_eval.py` exits with code 1 if any threshold is breached. This is what causes the GitHub Actions step to fail. In non-CI mode (local dev), it prints the results without exiting non-zero.

---

## Environment Variable Management

All secrets and configuration are environment variables. `.env` is never committed.

```bash
# .env.example  (committed — shows what variables are needed)
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_db
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5
EMBEDDER_PROVIDER=openai
EMBED_MODEL=text-embedding-3-small
JWT_SECRET_KEY=change-this-to-a-random-string-in-production
JWT_EXPIRE_HOURS=24
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
LOG_LEVEL=INFO
```

**In GitHub Actions:** Secrets are stored as repository secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and injected as environment variables in the eval job. They never appear in logs.

---

## What We Are Not Building

### Kubernetes

Adding Kubernetes to this project would mean:
- 3-5 YAML manifests (Deployment, Service, ConfigMap, Secret, PersistentVolumeClaim)
- A local cluster tool (minikube, kind, k3s)
- Complexity that adds no AI engineering signal

The correct answer to "why not Kubernetes?" in an interview: "The problem doesn't justify it. Docker Compose is the right tool for a single-service, single-host deployment. If I needed horizontal scaling or multi-node deployment, Kubernetes would be the natural next step — and the stateless API service is already designed to scale horizontally."

### Container registry pipeline

No push to Docker Hub or GHCR. The Docker build verification in CI is sufficient. A registry pipeline adds complexity (credentials, tagging strategy, retention policy) that is not relevant to demonstrating AI engineering skills.

### Managed cloud infrastructure (RDS, ECS, etc.)

Managed services add cost and account requirements that prevent others from running the project locally. `docker compose up` is fully self-contained.

---

## Consequences

**Positive:**
- `docker compose up` is the complete setup command
- Multi-stage build produces a lean, secure runtime image
- CI catches quality regressions automatically on every `main` merge
- The health check ensures clean startup ordering

**Negative:**
- Not production-deployable to a public URL out of the box (needs reverse proxy, TLS)
- Eval cost in CI requires secrets management in GitHub Actions

**Accepted trade-offs:** These limitations are appropriate for a portfolio project. Adding a reverse proxy and TLS is a well-understood operational step that is not the focus of this portfolio.

---

## Review Trigger

This decision should be revisited if:
- The project needs public URL hosting for a live demo (add Caddy or nginx reverse proxy)
- The eval CI cost becomes a concern (adjust to run on schedule rather than push)
- A second service is added (at that point, Docker Compose multi-service design is already in place)
