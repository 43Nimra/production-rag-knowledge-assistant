"""POST /auth/token integration tests (FR-19, FR-20, FR-21)."""

import pytest
from fastapi import APIRouter, Depends
from httpx import AsyncClient
from jose import jwt

from config import get_settings
from src.api.dependencies import require_auth
from src.api.main import app


@pytest.fixture(scope="module", autouse=True)
def _register_test_protected_route() -> None:
    """FR-19: no protected business routes exist yet in Phase 1 (/documents
    and /query arrive in Phases 2 and 4). This mounts one throwaway route,
    once per test module, so require_auth's behaviour is verified directly
    rather than left untested until a later phase."""
    router = APIRouter()

    @router.get("/_test/protected")
    async def protected_route(subject: str = Depends(require_auth)) -> dict[str, str]:
        return {"subject": subject}

    app.include_router(router)


async def test_valid_credentials_return_valid_jwt(client: AsyncClient) -> None:
    settings = get_settings()

    response = await client.post(
        "/auth/token",
        data={"username": settings.auth_username, "password": settings.auth_password},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in_hours"] == settings.jwt_expire_hours

    # The returned token must actually be a valid, decodable JWT signed
    # with the configured secret (not just an opaque string).
    payload = jwt.decode(
        body["access_token"], settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
    )
    assert payload["sub"] == settings.auth_username


async def test_invalid_credentials_are_rejected(client: AsyncClient) -> None:
    response = await client.post(
        "/auth/token",
        data={"username": "admin", "password": "wrong-password"},
    )

    assert response.status_code == 401


async def test_protected_route_dependency_rejects_missing_token(client: AsyncClient) -> None:
    response = await client.get("/_test/protected")
    assert response.status_code == 401


async def test_protected_route_accepts_valid_token(client: AsyncClient) -> None:
    settings = get_settings()

    token_response = await client.post(
        "/auth/token",
        data={"username": settings.auth_username, "password": settings.auth_password},
    )
    token = token_response.json()["access_token"]

    response = await client.get("/_test/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json() == {"subject": settings.auth_username}
