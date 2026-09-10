"""FastAPI application entrypoint.

Wires together: lifespan DB connectivity check, CORS, the trace_id +
structured-logging middleware (FR-23, FR-24, NFR-12), metrics counting,
and route registration. No business logic lives here — it delegates to
ingestion/retrieval/generation modules in later phases (ARCHITECTURE.md
§2.2: "No business logic; delegates to pipelines").
"""

import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import auth, documents, health
from src.db.session import check_database_connection
from src.observability.logger import configure_logging, get_logger
from src.observability.metrics import metrics_state

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_connected = await check_database_connection()
    if db_connected:
        logger.info("startup_db_check", db="connected")
    else:
        # Do not crash the process on a transient DB outage at boot —
        # /health will report "degraded" and callers can act on that,
        # matching NFR-06's "never fail silently, never hallucinate
        # success" philosophy applied to startup rather than generation.
        logger.warning("startup_db_check", db="disconnected")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Production RAG Knowledge Assistant",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Portfolio-scale, single-consumer API: permissive CORS is acceptable.
    # Documented here rather than silently defaulted so a reviewer can see
    # the trade-off explicitly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def trace_id_and_logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        trace_id = uuid.uuid4()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=str(trace_id))

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)

        metrics_state.record_request()
        logger.info(
            "request_complete",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Trace-ID"] = str(trace_id)
        return response

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(documents.router)

    return app


app = create_app()
