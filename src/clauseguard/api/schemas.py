"""API request and response schemas."""

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class SearchHit(BaseModel):
    id: str
    score: float
    text: str
    spec_id: str
    section: str
    source_file: str
    chunk_id: str


class SearchResponse(BaseModel):
    results: list[SearchHit]


class SearchErrorResponse(BaseModel):
    error: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class QueryCitation(BaseModel):
    spec_id: str
    section: str
    source_file: str
    chunk_id: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[QueryCitation]


class QueryErrorResponse(BaseModel):
    error: str


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
