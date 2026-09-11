"""SQLAlchemy ORM models.

Schema matches docs/ARCHITECTURE.md §4.1 exactly. This is a Phase 1
deliverable (task 1.2 in docs/IMPLEMENTATION_PLAN.md): the tables and the
initial migration are created now so the database foundation exists, but
no ingestion, retrieval, or generation code writes to these tables yet —
that begins in Phase 2.
"""

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Vector dimension of the ACTIVE embedding provider. Per ADR-003, this
# project supports OpenAI text-embedding-3-small (1536-dim) and a local
# BAAI/bge-small-en-v1.5 provider (384-dim) as swappable alternatives —
# but pgvector requires ONE fixed dimension per column at a time, so only
# one provider can be active without a migration. This deployment is
# configured for the free/local provider (EMBEDDER_PROVIDER=local), so
# EMBEDDING_DIM matches bge-small-en-v1.5's output. See
# migrations/versions/002_switch_embedding_dim_to_local.py for the change
# from the original 1536, and ADR-003's "Important constraint" section,
# which already anticipates this trade-off: switching providers requires
# an explicit migration and full re-ingestion, by design (no silent
# dimension mismatch).
EMBEDDING_DIM = 384


class Base(DeclarativeBase):
    pass


class Document(Base):
    """One row per ingested file (SRS FR-09; ARCHITECTURE.md §4.1)."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    filename: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    file_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # 'PENDING' | 'INDEXED' | 'PARSE_FAILED' (SRS FR-09)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """One row per text segment (ARCHITECTURE.md §4.1)."""

    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")


class QueryLog(Base):
    """One row per API query request (SRS FR-18; ARCHITECTURE.md §4.1).

    Never stores raw query text — only query_hash (NFR-12, FR-25).
    """

    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    query_hash: Mapped[str] = mapped_column(String(16), nullable=False)
    retrieved_chunk_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=True
    )
    retrieval_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
