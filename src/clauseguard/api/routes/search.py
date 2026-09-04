"""POST /search stub (Week 2)."""

from fastapi import APIRouter

from clauseguard.api.schemas import SearchRequest, StubResponse

router = APIRouter(tags=["search"])


@router.post("/search", response_model=StubResponse, status_code=501)
async def search(body: SearchRequest) -> StubResponse:
    """Vector search over O-RAN/3GPP chunks. Not implemented in scaffold."""
    return StubResponse(
        status="not_implemented",
        detail="POST /search will return top-k spec chunks",
        week="Week 2",
    )
