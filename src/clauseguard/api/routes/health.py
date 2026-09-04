"""Health endpoint — real implementation for scaffold."""

from fastapi import APIRouter
import httpx

from clauseguard.api.schemas import HealthResponse
from clauseguard.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness probe; optionally pings Qdrant."""
    settings = get_settings()
    qdrant_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{settings.qdrant_url.rstrip('/')}/readyz")
            qdrant_status = "ok" if resp.status_code == 200 else f"status_{resp.status_code}"
    except Exception:
        qdrant_status = "unreachable"
    return HealthResponse(status="ok", qdrant=qdrant_status)
