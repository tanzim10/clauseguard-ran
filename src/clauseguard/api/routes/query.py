"""POST /query route and dependency providers."""

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from clauseguard.api.query_service import QueryOutputError, QueryService
from clauseguard.api.routes.search import get_search_service
from clauseguard.api.schemas import QueryErrorResponse, QueryRequest, QueryResponse
from clauseguard.api.search_service import (
    SearchConfigurationError,
    SearchDependencyError,
    SearchService,
)
from clauseguard.config import get_settings
from clauseguard.llm.clients import (
    GenerationUnavailable,
    LLMRouter,
    MalformedGenerationOutput,
)

router = APIRouter(tags=["query"])


def get_llm_router() -> LLMRouter:
    """Provide the NVIDIA router behind a replaceable dependency seam."""
    return LLMRouter(settings=get_settings())


def get_query_service(
    search_service: Annotated[SearchService, Depends(get_search_service)],
    llm_router: Annotated[LLMRouter, Depends(get_llm_router)],
) -> QueryService:
    """Compose query behavior from the shared retrieval and generation services."""
    return QueryService(search_service=search_service, llm_router=llm_router)


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=200,
    responses={
        500: {"model": QueryErrorResponse},
        503: {"model": QueryErrorResponse},
    },
)
def query(
    body: QueryRequest,
    service: Annotated[QueryService, Depends(get_query_service)],
) -> QueryResponse | JSONResponse:
    """Answer a question using only the passages retrieved for this request."""
    try:
        return service.query(body.question, body.top_k)
    except SearchDependencyError:
        return JSONResponse(
            status_code=503,
            content={"error": "retrieval_unavailable"},
        )
    except SearchConfigurationError:
        return JSONResponse(
            status_code=500,
            content={"error": "retrieval_configuration_error"},
        )
    except (GenerationUnavailable, MalformedGenerationOutput, QueryOutputError):
        return JSONResponse(
            status_code=503,
            content={"error": "generation_unavailable"},
        )
