"""GET /health integration tests (SRS UC-05, FR-26)."""

from httpx import AsyncClient


async def test_health_returns_ok_and_db_connected(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body == {"status": "ok", "db": "connected"}


async def test_health_requires_no_authentication(client: AsyncClient) -> None:
    """FR-22: /health must be publicly accessible without a JWT."""
    response = await client.get("/health")

    assert response.status_code != 401


async def test_health_response_includes_trace_id_header(client: AsyncClient) -> None:
    """FR-23: every request is assigned a trace_id surfaced via X-Trace-ID."""
    response = await client.get("/health")

    assert "x-trace-id" in response.headers
