"""POST /auth/token — exchange credentials for a JWT (FR-20).

README.md documents this endpoint as form-encoded
(`curl -d "username=admin&password=changeme"`, no explicit Content-Type,
which curl sends as application/x-www-form-urlencoded). The route accepts
Form fields rather than a JSON body so the implementation matches the
approved, published API contract exactly.
"""

from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, status

from src.api.dependencies import create_access_token, verify_credentials
from src.api.schemas.auth import TokenResponse

router = APIRouter(tags=["auth"])


@router.post("/auth/token", response_model=TokenResponse)
async def issue_token(
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
) -> TokenResponse:
    if not verify_credentials(username, password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token, expires_in_hours = create_access_token(subject=username)
    return TokenResponse(access_token=token, expires_in_hours=expires_in_hours)
