"""API request and response schemas."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    model_config = ConfigDict(extra="forbid")

    scenario_id: str | None = None
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)

    @field_validator("scenario_id", "question")
    @classmethod
    def normalize_nonblank(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @model_validator(mode="after")
    def require_scenario(self) -> "RcaRequest":
        if self.scenario_id is None:
            raise ValueError("scenario_id is required until typed raw observations are supported")
        return self


class RcaClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: Literal["observation", "mechanism", "diagnostic_inference"]
    evidence_ids: list[str] = Field(min_length=1)


class RcaCauseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)


class RcaModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "abstained"]
    final_fault_categories: list[RcaCauseOutput]
    claims: list[RcaClaim]
    uncertainty_reasons: list[str] = Field(default_factory=list)
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def check_status(self) -> "RcaModelOutput":
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim IDs must be unique")
        if self.status == "abstained":
            if self.final_fault_categories or self.claims or not self.abstention_reason:
                raise ValueError("abstention needs a reason and empty causes and claims")
            return self
        if any(set(cause.claim_ids) - claim_ids for cause in self.final_fault_categories):
            raise ValueError("each cause must link to existing claims")
        if len({cause.category for cause in self.final_fault_categories}) != len(
            self.final_fault_categories
        ):
            raise ValueError("cause categories must be unique")
        for cause in self.final_fault_categories:
            linked = [claim for claim in self.claims if claim.claim_id in cause.claim_ids]
            if not any(claim.claim_type == "diagnostic_inference" for claim in linked):
                raise ValueError("each cause must link to a diagnostic inference claim")
        if not self.final_fault_categories or self.abstention_reason is not None:
            raise ValueError("answered output needs causes and no abstention reason")
        return self


class EvaluateRequest(BaseModel):
    golden_path: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    qdrant: str = "unknown"


class StubResponse(BaseModel):
    status: str = "not_implemented"
    detail: str = ""
    week: str = ""
