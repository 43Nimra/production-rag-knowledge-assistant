"""IngestionPipeline integration tests against a real PostgreSQL +
pgvector instance (NFR-21 principle; MockEmbedder per ADR-003)."""

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Chunk, Document
from src.ingestion.chunker import SentenceAwareChunker
from src.ingestion.pipeline import IngestionError, IngestionPipeline
from src.retrieval.vector_store import PgVectorStore
from tests.fakes import FailingEmbedder, MockEmbedder

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def _word_count(text: str) -> int:
    """Cheap token counter for tests (see src/ingestion/chunker.py's
    module docstring): avoids a real tiktoken call so these tests don't
    depend on network access to tiktoken's data host. The pipeline's
    default (real tiktoken) is exercised in production and CI, which has
    outbound internet access this sandbox does not."""
    return max(1, len(text.split()))


async def _pipeline(session: AsyncSession, embedder=None) -> IngestionPipeline:
    return IngestionPipeline(
        session=session,
        embedder=embedder or MockEmbedder(),
        vector_store=PgVectorStore(session),
        chunker=SentenceAwareChunker(token_counter=_word_count),
    )


class TestSuccessfulIngestion:
    async def test_ingest_pdf_creates_indexed_document_with_chunks(
        self, db_session: AsyncSession
    ) -> None:
        pipeline = await _pipeline(db_session)
        outcome = await pipeline.ingest(FIXTURES / "sample.pdf", filename="sample.pdf")

        assert outcome.status == "INDEXED"
        assert outcome.chunk_count > 0
        assert outcome.reason is None

        document = await db_session.get(Document, outcome.document_id)
        assert document is not None
        assert document.status == "INDEXED"
        assert document.chunk_count == outcome.chunk_count

        chunks = (
            (
                await db_session.execute(
                    select(Chunk).where(Chunk.document_id == outcome.document_id)
                )
            )
            .scalars()
            .all()
        )
        assert len(chunks) == outcome.chunk_count
        # The done-criteria in IMPLEMENTATION_PLAN.md: "Chunks appear in
        # chunks table with non-null embedding column".
        for chunk in chunks:
            assert chunk.embedding is not None
            assert len(chunk.embedding) == 384
            assert chunk.token_count > 0

    async def test_ingest_docx_and_html_and_txt_succeed(self, db_session: AsyncSession) -> None:
        for filename in ("sample.docx", "sample.html", "sample.txt"):
            pipeline = await _pipeline(db_session)
            outcome = await pipeline.ingest(FIXTURES / filename, filename=filename)
            assert outcome.status == "INDEXED", f"{filename} failed: {outcome.reason}"
            assert outcome.chunk_count > 0


class TestParseFailedPath:
    async def test_scanned_pdf_returns_parse_failed_with_reason(
        self, db_session: AsyncSession
    ) -> None:
        pipeline = await _pipeline(db_session)
        outcome = await pipeline.ingest(FIXTURES / "scanned_only.pdf", filename="scanned_only.pdf")

        assert outcome.status == "PARSE_FAILED"
        assert outcome.reason == "empty_extraction"
        assert outcome.chunk_count == 0

        document = await db_session.get(Document, outcome.document_id)
        assert document is not None
        assert document.status == "PARSE_FAILED"

        chunks = (
            (
                await db_session.execute(
                    select(Chunk).where(Chunk.document_id == outcome.document_id)
                )
            )
            .scalars()
            .all()
        )
        assert chunks == []


class TestIdempotency:
    async def test_reingesting_same_filename_updates_not_duplicates(
        self, db_session: AsyncSession
    ) -> None:
        pipeline1 = await _pipeline(db_session)
        first = await pipeline1.ingest(FIXTURES / "sample.txt", filename="idempotent.txt")

        pipeline2 = await _pipeline(db_session)
        second = await pipeline2.ingest(FIXTURES / "sample.txt", filename="idempotent.txt")

        # FR-08: same document_id, not a new row (filename UNIQUE, ADR-002).
        assert first.document_id == second.document_id

        all_docs = (
            (
                await db_session.execute(
                    select(Document).where(Document.filename == "idempotent.txt")
                )
            )
            .scalars()
            .all()
        )
        assert len(all_docs) == 1

        # Chunks were replaced, not appended.
        chunks = (
            (await db_session.execute(select(Chunk).where(Chunk.document_id == first.document_id)))
            .scalars()
            .all()
        )
        assert len(chunks) == second.chunk_count


class TestEmbeddingFailure:
    async def test_embedder_failure_leaves_document_pending(self, db_session: AsyncSession) -> None:
        pipeline = await _pipeline(db_session, embedder=FailingEmbedder())

        try:
            await pipeline.ingest(FIXTURES / "sample.txt", filename="will_fail_embed.txt")
            raise AssertionError("expected IngestionError")
        except IngestionError as exc:
            assert exc.status_code == 503

        document = (
            await db_session.execute(
                select(Document).where(Document.filename == "will_fail_embed.txt")
            )
        ).scalar_one()
        # ARCHITECTURE.md §5.2: embedding failure leaves the document
        # PENDING, never PARSE_FAILED or a hallucinated INDEXED.
        assert document.status == "PENDING"
