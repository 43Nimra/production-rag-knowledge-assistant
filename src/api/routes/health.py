"""GET /health and GET /metrics.

Both are publicly accessible without authentication (FR-22 explicitly
scopes this to /health; /metrics carries no sensitive data — only
aggregate counts — so the same public-access posture applies).
"""

from fastapi import APIRouter

from src.db.session import check_database_connection
from src.observability.metrics import metrics_state

router = APIRouter(tags=["observability"])


@router.get("/health")
async def health() -> dict[str, str]:
    db_connected = await check_database_connection()
    return {
        "status": "ok" if db_connected else "degraded",
        "db": "connected" if db_connected else "disconnected",
    }


@router.get("/metrics")
async def metrics() -> dict[str, int]:
    return metrics_state.snapshot()
