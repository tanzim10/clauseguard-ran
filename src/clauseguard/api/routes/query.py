"""POST /query stub (Week 3)."""

from fastapi import APIRouter

from clauseguard.api.schemas import QueryRequest, StubResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=StubResponse, status_code=501)
async def query(body: QueryRequest) -> StubResponse:
    """Grounded QA with citations. Not implemented in scaffold."""
    return StubResponse(
        status="not_implemented",
        detail="POST /query will answer from retrieved chunks only (or abstain)",
        week="Week 3",
    )
