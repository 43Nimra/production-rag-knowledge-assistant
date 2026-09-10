"""Shared test fixtures.

Integration tests run against a real PostgreSQL + pgvector instance
(NFR-21's principle applied beyond just the retrieval pipeline it's
scoped to) — the same instance CI spins up as a service container, with
`alembic upgrade head` already applied before pytest runs. There is no
DB mocking: /health's entire purpose is to report on a real connection,
and Phase 2's ingestion tests exercise real pgvector inserts.
"""

from collections.abc import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.main import app
from src.db.session import get_engine, get_sessionmaker


@pytest.fixture(autouse=True)
async def _fresh_db_engine_per_test() -> AsyncGenerator[None, None]:
    """Forces a brand-new engine (and thus a brand-new asyncpg connection
    pool) for every test function.

    `get_engine`/`get_sessionmaker` (src/db/session.py) are process-wide
    `lru_cache` singletons — correct in production, where uvicorn runs
    one event loop for the process's entire lifetime. Under pytest,
    each test can run its async body in a different event loop, and an
    asyncpg connection created on one loop cannot be reused on another
    ("Future ... attached to a different loop"). Clearing the cache
    before and after each test sidesteps that entirely: whichever loop
    is active when a test first touches the DB is the loop the engine
    gets bound to, and it's discarded before the next test's loop starts.
    This runs before every other fixture that touches the DB (autouse
    fixtures in a conftest run in file order), including
    _clean_ingestion_tables below.
    """
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    yield
    await get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session


@pytest.fixture(autouse=True)
async def _clean_ingestion_tables() -> AsyncGenerator[None, None]:
    """Truncates documents/chunks/query_logs before each test so tests
    that rely on `filename UNIQUE` (idempotency) or a clean chunk count
    never see another test's leftover rows. Runs before, not after, so a
    failed test's data is left in place for debugging until the next run."""
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        await session.execute(text("TRUNCATE documents, chunks, query_logs CASCADE"))
        await session.commit()
    yield
