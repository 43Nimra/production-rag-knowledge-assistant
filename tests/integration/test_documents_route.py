"""POST /documents integration tests (task 2.6 done-criteria).

Overrides get_embedder_dependency with MockEmbedder (ADR-003's testing
philosophy — no real API calls in tests) via FastAPI's
dependency_overrides. The chunker's tiktoken call is patched at the
module level rather than given a second DI seam in the route: the token
counter is already injectable at the chunker level for exactly this
reason (see src/ingestion/chunker.py), and duplicating that seam through
the route/pipeline would be an unjustified abstraction the approved
architecture doesn't call for (chunking config is env-var driven per
NFR-16, not request-scoped).
"""

from collections.abc import Generator
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import src.ingestion.chunker as chunker_module
from src.api.main import app
from src.api.routes.documents import get_embedder_dependency
from src.db.models import Document
from tests.fakes import MockEmbedder

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


class _FakeEncoding:
    def encode(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture(autouse=True)
def _stub_tiktoken(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    # See module docstring: avoids the route's default (real tiktoken)
    # chunker needing network access to tiktoken's data host in this
    # sandbox. `_tiktoken_encoding` is looked up dynamically inside
    # `default_token_counter` on each call, so patching the module
    # attribute here takes effect without touching production code paths.
    monkeypatch.setattr(chunker_module, "_tiktoken_encoding", lambda: _FakeEncoding())
    yield


@pytest.fixture(autouse=True)
def _override_embedder() -> Generator[None, None, None]:
    app.dependency_overrides[get_embedder_dependency] = lambda: MockEmbedder()
    yield
    app.dependency_overrides.pop(get_embedder_dependency, None)


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    from config import get_settings

    settings = get_settings()
    response = await client.post(
        "/auth/token",
        data={"username": settings.auth_username, "password": settings.auth_password},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def _upload(filename: str, content: bytes, content_type: str = "application/octet-stream"):
    return {"file": (filename, content, content_type)}


class TestAuthRequired:
    async def test_missing_token_returns_401(self, client: AsyncClient) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()
        response = await client.post("/documents", files=_upload("sample.txt", content))
        assert response.status_code == 401


class TestSuccessfulUpload:
    async def test_upload_pdf_returns_indexed(
        self, client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
    ) -> None:
        content = (FIXTURES / "sample.pdf").read_bytes()
        response = await client.post(
            "/documents",
            files=_upload("route_test.pdf", content, "application/pdf"),
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "INDEXED"
        assert body["chunk_count"] > 0
        assert body["duration_ms"] >= 0
        assert "document_id" in body

        document = await db_session.get(Document, body["document_id"])
        assert document is not None
        assert document.status == "INDEXED"

    async def test_upload_txt_returns_indexed(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()
        response = await client.post(
            "/documents",
            files=_upload("route_test.txt", content, "text/plain"),
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["status"] == "INDEXED"


class TestParseFailedResponse:
    async def test_scanned_pdf_returns_422_with_reason(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        content = (FIXTURES / "scanned_only.pdf").read_bytes()
        response = await client.post(
            "/documents",
            files=_upload("scanned_route_test.pdf", content, "application/pdf"),
            headers=auth_headers,
        )

        assert response.status_code == 422
        body = response.json()
        assert body["status"] == "PARSE_FAILED"
        assert body["reason"] == "empty_extraction"


class TestUnsupportedFormat:
    async def test_unsupported_extension_returns_415(
        self, client: AsyncClient, auth_headers: dict[str, str]
    ) -> None:
        response = await client.post(
            "/documents",
            files=_upload("spreadsheet.xlsx", b"irrelevant content"),
            headers=auth_headers,
        )
        assert response.status_code == 415


class TestIdempotentReingest:
    async def test_reuploading_same_filename_reuses_document_id(
        self, client: AsyncClient, auth_headers: dict[str, str], db_session: AsyncSession
    ) -> None:
        content = (FIXTURES / "sample.txt").read_bytes()

        first = await client.post(
            "/documents",
            files=_upload("idempotent_route.txt", content, "text/plain"),
            headers=auth_headers,
        )
        second = await client.post(
            "/documents",
            files=_upload("idempotent_route.txt", content, "text/plain"),
            headers=auth_headers,
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["document_id"] == second.json()["document_id"]

        all_docs = (
            (
                await db_session.execute(
                    select(Document).where(Document.filename == "idempotent_route.txt")
                )
            )
            .scalars()
            .all()
        )
        assert len(all_docs) == 1
