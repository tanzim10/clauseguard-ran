"""Dataset contracts; observation values stay separate from reference annotations."""

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


class KpiChannel(BaseModel):
    """An ordered channel with native values; unknown units are explicit."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["numeric", "categorical"]
    values: list[StrictInt | StrictFloat | StrictStr | None] = Field(min_length=1)
    unit: str | None = None

    @model_validator(mode="after")
    def check_values(self) -> "KpiChannel":
        for value in self.values:
            if value is None:
                continue
            if self.kind == "categorical" and not isinstance(value, str):
                raise ValueError("categorical values must be strings or null")
            if self.kind == "numeric" and (
                isinstance(value, str) or not math.isfinite(value)
            ):
                raise ValueError("numeric values must be finite numbers or null")
        return self


class KpiWindow(BaseModel):
    """Source boundaries refer to the first and last sample, not window edges."""

    model_config = ConfigDict(extra="forbid")
    channels: dict[str, KpiChannel] = Field(min_length=1)
    sampling_rate_hz: float = Field(gt=0, allow_inf_nan=False, strict=True)
    start_time: str
    end_time: str

    @property
    def sample_count(self) -> int:
        return len(next(iter(self.channels.values())).values)

    @model_validator(mode="after")
    def check_window(self) -> "KpiWindow":
        if any(not name.strip() for name in self.channels):
            raise ValueError("channel names cannot be blank")
        if len({len(c.values) for c in self.channels.values()}) != 1:
            raise ValueError("channel lengths must match")
        start, end = datetime.fromisoformat(self.start_time), datetime.fromisoformat(self.end_time)
        if (start.tzinfo is None) != (end.tzinfo is None):
            raise ValueError("timestamp timezone conventions must match")
        expected = (self.sample_count - 1) / self.sampling_rate_hz
        if not math.isclose((end - start).total_seconds(), expected, abs_tol=1e-6):
            raise ValueError("source timestamps disagree with sample count and cadence")
        return self


class Citation(BaseModel):
    """Spec citation reference (placeholder)."""

    source_id: str = ""
    section: str = ""
    excerpt: str = ""


class Provenance(BaseModel):
    """Stable source identity; a path alone never establishes recording lineage."""

    model_config = ConfigDict(extra="forbid")
    repository: str = "AliMaatouk/TelecomTS"
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_file: str
    source_row: int = Field(ge=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recording_id: str | None = None
    parent_id: str | None = None
    relationship_status: Literal["unresolved", "verified"] = "unresolved"
    relationship_evidence: str | None = None

    @model_validator(mode="after")
    def check_evidence(self) -> "Provenance":
        if self.relationship_status == "verified" and (
            not self.recording_id or not self.relationship_evidence
        ):
            raise ValueError("verified lineage requires a recording and evidence")
        return self


class TextReview(BaseModel):
    """Only a recorded human approval allows text to enter nl_context."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["pending", "approved", "absent"] = "pending"
    source: str = "description"
    reviewer: str | None = None
    reviewed_at: str | None = None
    approved_tasks: list[Literal["detection", "category"]] = Field(default_factory=list)
    text_origin: Literal[
        "unreviewed", "independent_observation", "kpi_derived", "diagnosis_derived"
    ] = "unreviewed"


class AnomalyAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exists: bool = Field(strict=True)
    source_type: str | None = None
    affected_channels: list[str] = Field(default_factory=list)
    start_index: StrictInt | None = None
    end_index: StrictInt | None = None


class ScenarioRecord(BaseModel):
    """Validated scenario. Reference annotations are not model inputs."""

    model_config = ConfigDict(extra="forbid")
    scenario_id: str = Field(min_length=1)
    kpi_window: KpiWindow
    nl_context: str = ""
    fault_label: Literal["normal", "abnormal"]
    rca_label: str = Field(min_length=1)
    origin: Literal["normal", "real", "synthetic", "unknown"]
    provenance: Provenance
    anomaly: AnomalyAnnotation
    text_review: TextReview = Field(default_factory=TextReview)
    reference: dict[str, Any] = Field(default_factory=dict)
    alarms: list[dict[str, Any]] = Field(default_factory=list)
    spec_refs_golden: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_annotations(self) -> "ScenarioRecord":
        if (self.fault_label == "abnormal") != self.anomaly.exists:
            raise ValueError("fault presence disagrees with anomaly annotation")
        start, end = self.anomaly.start_index, self.anomaly.end_index
        if (start is None) != (end is None):
            raise ValueError("anomaly endpoints must both be present or absent")
        if start is not None and not (0 <= start <= end < self.kpi_window.sample_count):
            raise ValueError("inclusive anomaly bounds fall outside the window")
        if not self.anomaly.exists and (start is not None or self.anomaly.affected_channels):
            raise ValueError("normal records cannot carry anomaly bounds or affected channels")
        if set(self.anomaly.affected_channels) - self.kpi_window.channels.keys():
            raise ValueError("affected channels must exist in the KPI window")
        if self.text_review.status == "approved":
            if not all((self.nl_context.strip(), self.text_review.reviewer,
                        self.text_review.reviewed_at, self.text_review.source)):
                raise ValueError("approved text requires symptoms, reviewer, date and source")
            datetime.fromisoformat(self.text_review.reviewed_at)
        elif self.nl_context:
            raise ValueError("unreviewed text cannot enter nl_context")
        return self


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
