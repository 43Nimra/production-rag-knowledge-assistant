"""Vector store write path (task 2.4).

`VectorStoreBase` documents and preserves the migration path ADR-002
describes: if a dedicated vector DB is ever justified, the swap is
localized to this module. The read/search side (`search(query_vector,
top_k)`) is Phase 3 scope (IMPLEMENTATION_PLAN.md task 3.1) — this class
only covers ingestion's write path.
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sqlalchemy import delete, insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk


@dataclass(frozen=True)
class ChunkToStore:
    text: str
    chunk_index: int
    token_count: int
    page_number: int | None
    embedding: list[float]


class VectorStoreBase(ABC):
    @abstractmethod
    async def upsert_chunks(self, document_id: uuid.UUID, chunks: list[ChunkToStore]) -> None:
        """Replaces all chunks for document_id with `chunks` (FR-08
        idempotency: re-ingesting a document replaces its chunk set rather
        than appending duplicates)."""


class PgVectorStore(VectorStoreBase):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def upsert_chunks(self, document_id: uuid.UUID, chunks: list[ChunkToStore]) -> None:
        await self._session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        if not chunks:
            return

        await self._session.execute(
            insert(Chunk),
            [
                {
                    "document_id": document_id,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                    "token_count": c.token_count,
                    "page_number": c.page_number,
                    "embedding": c.embedding,
                }
                for c in chunks
            ],
        )
