"""Ingestion pipeline orchestrator (task 2.5).

Implements the exact state machine documented in ARCHITECTURE.md §5.1
("Ingestion Pipeline: Data Flow") and §5.2 ("Failure Modes and
Handling"):

1. Validate file size (413) and extension (415) — no Document row yet.
2. Upsert a Document row as PENDING and commit immediately, so a crash
   after this point still leaves a queryable status (FR-09).
3. Parse. On ParserError, mark PARSE_FAILED and commit; return that
   outcome (API layer maps this to 422).
4. Chunk. Zero chunks after chunking is treated the same as an empty
   parse (PARSE_FAILED, reason="empty_chunks").
5. Embed (batched internally by the embedder). On EmbedderError, leave
   the already-committed PENDING row as-is and raise (API layer -> 503,
   NFR-06's "never hallucinate success" principle applied to ingestion).
6. Single transaction: replace the document's chunks via the vector
   store and flip status -> INDEXED. On any exception here, roll back —
   the document stays PENDING (NFR-07), and the pipeline raises (API
   layer -> 500).
"""

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document
from src.db.repository import DocumentRepository
from src.embeddings.base import EmbedderBase, EmbedderError
from src.ingestion.chunker import SentenceAwareChunker
from src.ingestion.parsers import ParserError, get_parser
from src.observability.logger import get_logger
from src.retrieval.vector_store import ChunkToStore, VectorStoreBase

logger = get_logger(__name__)

# ARCHITECTURE.md §5.1: "File size check (max 50MB)".
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


class FileTooLargeError(Exception):
    pass


class IngestionError(Exception):
    """Raised for failures that should surface as a specific HTTP status
    at the API layer (503 for embedding failures, 500 for DB failures)."""

    def __init__(self, message: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class IngestOutcome:
    document_id: uuid.UUID
    filename: str
    status: str  # "INDEXED" | "PARSE_FAILED"
    chunk_count: int
    duration_ms: int
    reason: str | None = None  # set when status == "PARSE_FAILED"


class IngestionPipeline:
    def __init__(
        self,
        session: AsyncSession,
        embedder: EmbedderBase,
        vector_store: VectorStoreBase,
        chunker: SentenceAwareChunker | None = None,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunker = chunker or SentenceAwareChunker()
        self._documents = DocumentRepository(session)

    async def ingest(self, file_path: Path, filename: str) -> IngestOutcome:
        start = time.perf_counter()

        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit"
            )

        # Validates extension before any DB write (unsupported formats are
        # a request-validation concern, not a per-document PARSE_FAILED
        # state — ARCHITECTURE.md §5.1 lists format checking under
        # [Validation], before [Parser dispatch]).
        parser = get_parser(filename)
        file_type = Path(filename).suffix.lower().lstrip(".")

        document = await self._upsert_pending_document(filename, file_type)

        try:
            parse_result = parser.parse(file_path)
        except ParserError as exc:
            return await self._mark_parse_failed(document, exc.reason, start)

        chunks = self._chunker.chunk(parse_result)
        if not chunks:
            return await self._mark_parse_failed(document, "empty_chunks", start)

        try:
            embeddings = self._embedder.embed_documents([c.text for c in chunks])
        except EmbedderError as exc:
            logger.warning(
                "ingestion_embedding_failed", document_id=str(document.id), error=str(exc)
            )
            # Document row is already committed as PENDING (NFR-07's
            # "trigger a transaction rollback; the document SHALL be
            # marked PENDING for retry" — nothing to roll back here since
            # no chunks/vectors were written yet).
            raise IngestionError(f"Embedding failed: {exc}", status_code=503) from exc

        chunks_to_store = [
            ChunkToStore(
                text=c.text,
                chunk_index=c.chunk_index,
                token_count=c.token_count,
                page_number=c.page_number,
                embedding=embedding,
            )
            for c, embedding in zip(chunks, embeddings, strict=True)
        ]

        try:
            await self._vector_store.upsert_chunks(document.id, chunks_to_store)
            document.status = "INDEXED"
            document.chunk_count = len(chunks_to_store)
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            logger.error("ingestion_db_write_failed", document_id=str(document.id), error=str(exc))
            raise IngestionError(f"Database write failed: {exc}", status_code=500) from exc

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "ingestion_complete",
            document_id=str(document.id),
            filename=filename,
            chunk_count=len(chunks_to_store),
            duration_ms=duration_ms,
        )
        return IngestOutcome(
            document_id=document.id,
            filename=filename,
            status="INDEXED",
            chunk_count=len(chunks_to_store),
            duration_ms=duration_ms,
        )

    async def _upsert_pending_document(self, filename: str, file_type: str) -> Document:
        """FR-08 idempotency: re-ingesting by filename updates the
        existing row rather than creating a duplicate (the `filename
        UNIQUE` constraint, ADR-002, is the DB-level backstop for this)."""
        existing = await self._documents.get_by_filename(filename)
        if existing is not None:
            existing.status = "PENDING"
            existing.file_type = file_type
            document = existing
        else:
            document = await self._documents.add(
                Document(filename=filename, file_type=file_type, status="PENDING")
            )
        await self._session.commit()
        return document

    async def _mark_parse_failed(
        self, document: Document, reason: str, start: float
    ) -> IngestOutcome:
        document.status = "PARSE_FAILED"
        document.chunk_count = 0
        await self._session.commit()
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.warning(
            "ingestion_parse_failed",
            document_id=str(document.id),
            filename=document.filename,
            reason=reason,
        )
        return IngestOutcome(
            document_id=document.id,
            filename=document.filename,
            status="PARSE_FAILED",
            chunk_count=0,
            duration_ms=duration_ms,
            reason=reason,
        )
