"""Authentication foundation.

- `create_access_token` / `decode_access_token`: JWT issuance and
  verification (FR-19, FR-21).
- `require_auth`: FastAPI dependency enforcing Bearer auth on protected
  routes (FR-19). Phase 1 has no protected business routes yet (documents
  and query endpoints arrive in Phases 2 and 4) — this dependency exists
  now so Phase 1's "done" criteria ("POST /auth/token returns a valid
  JWT") is verifiable end-to-end, and so later phases can import it
  directly.

Single-user credential model: this system has no user registration or
management (documented out of scope, ARCHITECTURE.md §11). The configured
username/password in Settings are compared using passlib's bcrypt
verification so the comparison pattern matches what a real multi-user
system would use, even though credentials originate from an env var
rather than a database.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from config import get_settings

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer_scheme = HTTPBearer(auto_error=False)


def verify_credentials(username: str, password: str) -> bool:
    settings = get_settings()
    if username != settings.auth_username:
        return False
    expected_hash = _pwd_context.hash(settings.auth_password)
    return bool(_pwd_context.verify(password, expected_hash))


def create_access_token(subject: str) -> tuple[str, int]:
    """Returns (token, expires_in_hours)."""
    settings = get_settings()
    expire_delta = timedelta(hours=settings.jwt_expire_hours)
    expire_at = datetime.now(UTC) + expire_delta
    payload: dict[str, Any] = {"sub": subject, "exp": expire_at}
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, settings.jwt_expire_hours


def decode_access_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return payload


async def require_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> str:
    """FastAPI dependency: verifies the Bearer JWT and returns the subject.

    Used by /documents and /query (FR-19) once those routes exist. /health
    remains public (FR-22) and does not use this dependency.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    subject = payload.get("sub")
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return str(subject)
