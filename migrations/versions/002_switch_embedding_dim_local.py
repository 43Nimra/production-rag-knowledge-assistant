"""switch embedding dimension to local provider (384-dim)

Revision ID: 002
Revises: 001
Create Date: 2026-09-10

Changes chunks.embedding from vector(1536) (OpenAI text-embedding-3-small)
to vector(384) (BAAI/bge-small-en-v1.5, via LocalEmbedder), reflecting the
decision to run the free/local embedding provider (ADR-003) rather than
the paid OpenAI API.

ADR-003 explicitly anticipates this: "changing the embedding model
requires full re-ingestion... the embedding vectors in chunks.embedding
are in the new model's vector space; old vectors are incompatible." This
migration truncates documents/chunks/query_logs rather than attempting a
silent in-place conversion — there is no valid mapping between two
different models' vector spaces, so keeping stale 1536-dim data around
mislabeled as 384-dim would be worse than requiring re-ingestion.
"""

from collections.abc import Sequence

from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

OLD_DIM = 1536
NEW_DIM = 384


def upgrade() -> None:
    # No valid conversion exists between two different embedding models'
    # vector spaces — clear out data built against the old (1536-dim)
    # model rather than leave it silently mismatched.
    op.execute("TRUNCATE documents, chunks, query_logs CASCADE")

    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column(
        "chunks",
        "embedding",
        type_=Vector(NEW_DIM),
        existing_type=Vector(OLD_DIM),
        postgresql_using="embedding::vector(384)",
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("TRUNCATE documents, chunks, query_logs CASCADE")
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding_hnsw")
    op.alter_column(
        "chunks",
        "embedding",
        type_=Vector(OLD_DIM),
        existing_type=Vector(NEW_DIM),
        postgresql_using="embedding::vector(1536)",
    )
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
