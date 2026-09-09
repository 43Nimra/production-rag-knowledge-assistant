"""Repository pattern for database access.

Per docs/ARCHITECTURE.md §2.2 ("No raw SQL in routes or pipelines"), all
DB access goes through repositories. Phase 1 only needs these classes to
exist with basic CRUD primitives — the ingestion, retrieval, and query
logging *business logic* that will call them is built in later phases.
"""

import uuid
from typing import Generic, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Base, Chunk, Document, QueryLog

ModelT = TypeVar("ModelT", bound=Base)


class _BaseRepository(Generic[ModelT]):
    """Shared CRUD primitives. Not exported directly — subclassed per model
    so each repository's public surface stays specific to its table."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, id_: uuid.UUID) -> ModelT | None:
        return await self._session.get(self.model, id_)

    async def add(self, instance: ModelT) -> ModelT:
        self._session.add(instance)
        await self._session.flush()
        return instance


class DocumentRepository(_BaseRepository[Document]):
    model = Document

    async def get_by_filename(self, filename: str) -> Document | None:
        result = await self._session.execute(select(Document).where(Document.filename == filename))
        return result.scalar_one_or_none()


class ChunkRepository(_BaseRepository[Chunk]):
    model = Chunk

    async def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        result = await self._session.execute(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
        )
        return list(result.scalars().all())


class QueryLogRepository(_BaseRepository[QueryLog]):
    model = QueryLog
