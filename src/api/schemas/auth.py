"""Pydantic schemas for POST /auth/token (FR-20).

Request fields are Form(...) parameters on the route itself (README.md
documents this endpoint as form-encoded), so only the response is a JSON
schema here.
"""

from pydantic import BaseModel


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in_hours: int
