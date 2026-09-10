"""scripts/ingest_docs.py CLI test (UC-08, task 2.7).

Imports `ingest_directory` directly — the same function `python
scripts/ingest_docs.py <dir>` calls — rather than shelling out to a
subprocess, so the test can substitute MockEmbedder (ADR-003) instead of
making real OpenAI API calls, the same pattern used for the POST
/documents route tests. This exercises the real CLI logic end-to-end
against a real database.
"""

import importlib.util
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Document
from tests.fakes import MockEmbedder

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "ingest_docs.py"


def _load_ingest_docs_module():
    spec = importlib.util.spec_from_file_location("ingest_docs_cli", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ingest_docs_cli = _load_ingest_docs_module()


@pytest.fixture(autouse=True)
def _stub_embedder_and_tokenizer(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    import src.ingestion.chunker as chunker_module

    class _FakeEncoding:
        def encode(self, text: str) -> list[str]:
            return text.split()

    monkeypatch.setattr(chunker_module, "_tiktoken_encoding", lambda: _FakeEncoding())
    monkeypatch.setattr(ingest_docs_cli, "get_embedder", lambda: MockEmbedder())
    yield


class TestIngestDirectory:
    async def test_ingests_all_supported_files_in_directory(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        for name in ("sample.pdf", "sample.docx", "sample.html", "sample.txt"):
            (tmp_path / name).write_bytes((FIXTURES / name).read_bytes())
        # A file with an unsupported extension should simply be skipped,
        # not crash the batch (NFR-05: one document's failure doesn't
        # affect others).
        (tmp_path / "notes.xlsx").write_bytes(b"irrelevant")

        exit_code = await ingest_docs_cli.ingest_directory(tmp_path)

        assert exit_code == 0
        for name in ("sample.pdf", "sample.docx", "sample.html", "sample.txt"):
            document = (
                await db_session.execute(select(Document).where(Document.filename == name))
            ).scalar_one()
            assert document.status == "INDEXED"
            assert document.chunk_count and document.chunk_count > 0

    async def test_batch_continues_after_one_parse_failure(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        (tmp_path / "good.txt").write_bytes((FIXTURES / "sample.txt").read_bytes())
        (tmp_path / "scanned.pdf").write_bytes((FIXTURES / "scanned_only.pdf").read_bytes())

        exit_code = await ingest_docs_cli.ingest_directory(tmp_path)

        # At least one document indexed successfully alongside one
        # PARSE_FAILED -> overall exit code should still be 0 (NFR-05).
        assert exit_code == 0

        good = (
            await db_session.execute(select(Document).where(Document.filename == "good.txt"))
        ).scalar_one()
        assert good.status == "INDEXED"

        scanned = (
            await db_session.execute(select(Document).where(Document.filename == "scanned.pdf"))
        ).scalar_one()
        assert scanned.status == "PARSE_FAILED"

    async def test_empty_directory_returns_zero_with_no_documents(
        self, db_session: AsyncSession, tmp_path: Path
    ) -> None:
        exit_code = await ingest_docs_cli.ingest_directory(tmp_path)
        assert exit_code == 0
