"""Pydantic schema for POST /documents (task 2.6)."""

import uuid

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: uuid.UUID
    status: str
    chunk_count: int
    duration_ms: int
