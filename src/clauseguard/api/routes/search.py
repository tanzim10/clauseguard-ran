"""POST /search route."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from clauseguard.api.schemas import SearchErrorResponse, SearchRequest, SearchResponse
from clauseguard.api.search_service import (
    SearchConfigurationError,
    SearchDependencyError,
    SearchService,
)

router = APIRouter(tags=["search"])


def get_search_service() -> SearchService:
    """Provide the production search service and a test override seam."""
    return SearchService()


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=200,
    responses={
        500: {"model": SearchErrorResponse},
        503: {"model": SearchErrorResponse},
    },
)
def search(
    body: SearchRequest,
    service: Annotated[SearchService, Depends(get_search_service)],
) -> SearchResponse | JSONResponse:
    """Search indexed O-RAN/3GPP passages."""
    try:
        results = service.search(body.query, body.top_k)
    except SearchDependencyError:
        return JSONResponse(
            status_code=503,
            content={"error": "retrieval_unavailable"},
        )
    except SearchConfigurationError:
        return JSONResponse(
            status_code=500,
            content={"error": "search_configuration_error"},
        )
    return SearchResponse(results=results)
