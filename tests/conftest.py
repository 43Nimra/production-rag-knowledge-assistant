"""Shared test fixtures.

Phase 1 integration tests run against a real PostgreSQL + pgvector
instance (NFR-21's principle applied here too, even though NFR-21 itself
is scoped to the retrieval pipeline) — the same instance CI spins up as a
service container, with `alembic upgrade head` already applied before
pytest runs. There is no DB mocking in Phase 1: /health's entire purpose
is to report on a real connection.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.main import app


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
