"""POST /documents — upload and index a document (FR-01, FR-19).

Requires JWT Bearer auth (FR-19). Saves the upload to a temp file (parsers
operate on filesystem paths, matching how scripts/ingest_docs.py calls the
same pipeline against real files on disk) and delegates everything else to
IngestionPipeline.
"""

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import require_auth
from src.api.schemas.documents import IngestResponse
from src.db.session import get_session
from src.embeddings import get_embedder
from src.embeddings.base import EmbedderBase
from src.ingestion.parsers import ParserError
from src.ingestion.pipeline import FileTooLargeError, IngestionError, IngestionPipeline
from src.retrieval.vector_store import PgVectorStore

router = APIRouter(tags=["documents"])


def get_embedder_dependency() -> EmbedderBase:
    """A thin wrapper around get_embedder() so tests can override it via
    FastAPI's dependency_overrides without touching real provider config."""
    return get_embedder()


@router.post("/documents", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    subject: Annotated[str, Depends(require_auth)],
    session: Annotated[AsyncSession, Depends(get_session)],
    embedder: Annotated[EmbedderBase, Depends(get_embedder_dependency)],
) -> IngestResponse | JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")

    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await file.read())

    try:
        pipeline = IngestionPipeline(
            session=session, embedder=embedder, vector_store=PgVectorStore(session)
        )
        try:
            outcome = await pipeline.ingest(tmp_path, filename=file.filename)
        except FileTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
            ) from exc
        except ParserError as exc:
            # Unsupported extension — raised before any Document row is
            # written (ARCHITECTURE.md §5.1: format check is [Validation],
            # ahead of [Parser dispatch]).
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
            ) from exc
        except IngestionError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if outcome.status == "PARSE_FAILED":
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "status": "PARSE_FAILED",
                "reason": outcome.reason,
                "document_id": str(outcome.document_id),
            },
        )

    return IngestResponse(
        document_id=outcome.document_id,
        status=outcome.status,
        chunk_count=outcome.chunk_count,
        duration_ms=outcome.duration_ms,
    )
