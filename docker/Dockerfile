# syntax=docker/dockerfile:1

# --- Stage 1: builder ---
# Installs build tooling and compiles the package into site-packages.
# None of this stage's tooling (pip, build backend, gcc if needed by a
# dependency) ends up in the runtime image.
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY config.py .
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .

# --- Stage 2: runtime ---
# Only the installed package artifacts and application source are copied
# in. No compiler, no pip cache, no build backend (ADR-006).
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY config.py .
COPY migrations/ ./migrations/
COPY alembic.ini .
COPY scripts/ ./scripts/

# NLTK sentence-tokenizer data is baked into the image at build time
# (ADR-004's chunker needs it) rather than downloaded lazily on first
# request — a hot ingestion path has no business depending on outbound
# network access to a third-party data host, and this keeps the image
# reproducible/deterministic (SRS NFR-08 spirit: no manual setup steps).
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} punkt_tab

# Same reasoning as NLTK above, applied to the local embedding model
# (ADR-003): pre-download BAAI/bge-small-en-v1.5's weights at build time
# so a hot ingestion/query path never depends on outbound network access
# to Hugging Face at request time. HF_HOME is set to a fixed path so the
# runtime process finds the same cache LocalEmbedder's default model
# factory populated here — otherwise sentence-transformers would default
# to a per-user cache dir under $HOME, which appuser (created below, has
# no home directory) can't rely on.
ENV HF_HOME=/usr/local/share/hf-cache
ENV HF_HUB_OFFLINE=1
RUN HF_HUB_OFFLINE=0 python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

RUN useradd --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser /app "${HF_HOME}" "${NLTK_DATA}"
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
