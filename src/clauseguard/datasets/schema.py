"""Shared dataset schemas (Week 4–5). Placeholder pydantic models."""

from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Spec citation reference (placeholder)."""

    source_id: str = ""
    section: str = ""
    excerpt: str = ""


class ScenarioRecord(BaseModel):
    """One TelecomTS / RCA incident (placeholder)."""

    scenario_id: str
    kpi_window: list[Any] = Field(default_factory=list)
    nl_context: str = ""
    fault_label: str = ""
    rca_label: str = ""
    alarms: list[dict[str, Any]] = Field(default_factory=list)
    spec_refs_golden: list[str] = Field(default_factory=list)


class GoldenEvalRow(BaseModel):
    """Golden eval row for spec Q&A or RCA (placeholder)."""

    id: str
    type: str = "spec_qa"
    question: str | None = None
    scenario_id: str | None = None
    expected_spec_id: str | None = None
    expected_section: str | None = None
    spec_refs_golden: list[str] = Field(default_factory=list)
    is_answerable: bool = True
