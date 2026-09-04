"""API request/response schemas (placeholders)."""

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class QueryRequest(BaseModel):
    question: str
    top_k: int = 8


class RcaRequest(BaseModel):
    scenario_id: str | None = None
    payload: dict | None = None


class EvaluateRequest(BaseModel):
    golden_path: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: str = "unknown"


class StubResponse(BaseModel):
    status: str = "not_implemented"
    detail: str = ""
    week: str = ""
