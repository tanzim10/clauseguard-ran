"""POST /rca stub (Week 7)."""

from fastapi import APIRouter

from clauseguard.api.schemas import RcaRequest, StubResponse

router = APIRouter(tags=["rca"])


@router.post("/rca", response_model=StubResponse, status_code=501)
async def rca(body: RcaRequest) -> StubResponse:
    """Multimodal citation-grounded RCA. Not implemented in scaffold."""
    return StubResponse(
        status="not_implemented",
        detail="POST /rca will return fault + KPI/text/spec evidence",
        week="Week 7",
    )
