"""POST /evaluate stub (Week 5)."""

from fastapi import APIRouter

from clauseguard.api.schemas import EvaluateRequest, StubResponse

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=StubResponse, status_code=501)
async def evaluate(body: EvaluateRequest) -> StubResponse:
    """Run golden evaluation. Not implemented in scaffold."""
    return StubResponse(
        status="not_implemented",
        detail="POST /evaluate will run golden eval and write artifacts",
        week="Week 5",
    )
