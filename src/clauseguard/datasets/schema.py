"""Dataset contracts; observation values stay separate from reference annotations."""

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    model_validator,
)

NOT_FOUND_ANSWER = "not found"

# Placeholder fields from the original scaffold. Their meaning was never pinned down (for
# example "O-RAN.WG3.E2AP §placeholder" mixes spec and section), so they are rejected rather
# than converted; use `expected_citations` instead.
LEGACY_GOLDEN_FIELDS = (
    "expected_spec_id",
    "expected_section",
    "spec_refs_golden",
    "scenario_id",
)

# Must contain at least one non-whitespace character.
NonEmptyStr = Annotated[str, StringConstraints(pattern=r"\S")]


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


class GoldenCitation(BaseModel):
    """Expected citation for a golden row: manifest `spec_id` plus section identifier."""

    model_config = ConfigDict(extra="forbid")

    spec_id: NonEmptyStr
    section: NonEmptyStr


class GoldenEvalRow(BaseModel):
    """Golden specification Q&A row, matching `data/golden/spec_qa.jsonl`.

    Answerable rows need at least one citation. Unanswerable rows must have
    `expected_answer == "not found"` and no citations. RCA rows use `RcaRow` instead.
    The scaffold's legacy fields (`LEGACY_GOLDEN_FIELDS`) are deprecated and rejected.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    type: Literal["spec_qa"] = "spec_qa"
    question: NonEmptyStr
    expected_answer: NonEmptyStr
    expected_citations: list[GoldenCitation]
    is_answerable: bool = True

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        legacy = [name for name in LEGACY_GOLDEN_FIELDS if name in data]
        if not legacy:
            return data
        names = ", ".join(legacy)
        if "expected_citations" in data:
            raise ValueError(f"legacy field(s) {names} conflict with expected_citations")
        raise ValueError(f"legacy field(s) {names} are unsupported; use expected_citations")

    @model_validator(mode="after")
    def _check_answerability(self) -> "GoldenEvalRow":
        if self.is_answerable:
            if not self.expected_citations:
                raise ValueError("answerable rows need at least one expected citation")
        else:
            if self.expected_answer != NOT_FOUND_ANSWER:
                raise ValueError(
                    f"unanswerable rows must have expected_answer {NOT_FOUND_ANSWER!r}"
                )
            if self.expected_citations:
                raise ValueError("unanswerable rows must not have expected citations")
        return self


class KpiInterval(BaseModel):
    """A window on the scenario's own source timeline (timestamps or sample indices)."""

    model_config = ConfigDict(extra="forbid")

    start: NonEmptyStr
    end: NonEmptyStr


class KpiEvidence(BaseModel):
    """One measured channel: its value over a baseline window versus the incident window."""

    model_config = ConfigDict(extra="forbid")

    channel: NonEmptyStr
    unit: NonEmptyStr | None = None
    baseline: KpiInterval
    comparison: KpiInterval
    baseline_value: float
    comparison_value: float
    direction: Literal["increase", "decrease", "unchanged"]

    @model_validator(mode="after")
    def _check_direction(self) -> "KpiEvidence":
        delta = self.comparison_value - self.baseline_value
        observed = "increase" if delta > 0 else "decrease" if delta < 0 else "unchanged"
        if observed != self.direction:
            raise ValueError(
                f"direction {self.direction!r} disagrees with measured values "
                f"({self.baseline_value} -> {self.comparison_value})"
            )
        return self


class RcaCitation(BaseModel):
    """Expected citation for an RCA row, with the role the clause plays in the explanation.

    `definition` clauses only say what a KPI measures; `mechanism` clauses describe behavior
    that links the symptom to a cause. A definition alone cannot support a diagnosis.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: NonEmptyStr
    section: NonEmptyStr
    role: Literal["definition", "mechanism"]
    explanation: NonEmptyStr


class RcaRow(BaseModel):
    """Golden RCA row, matching `data/golden/rca.jsonl`.

    The question describes symptoms only: it must not contain any source label value, so a
    lexical retriever cannot match the diagnosis from the question text. Source labels are
    recorded as claims to compare against, not as ground truth.
    """

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    type: Literal["rca"] = "rca"
    scenario_id: NonEmptyStr
    origin: Literal["real", "synthetic"]
    inputs: Literal["kpi_only", "multimodal"]
    question: NonEmptyStr
    kpi_evidence: list[KpiEvidence] = Field(min_length=1)
    source_labels: dict[NonEmptyStr, NonEmptyStr] = Field(min_length=1)
    expected_conclusion: NonEmptyStr
    acceptable_conclusions: list[NonEmptyStr] = Field(default_factory=list)
    uncertainty: NonEmptyStr
    expected_citations: list[RcaCitation] = Field(min_length=1)

    @property
    def is_answerable(self) -> bool:
        return True

    @model_validator(mode="after")
    def _check_contract(self) -> "RcaRow":
        question = self.question.casefold()
        leaked = sorted(
            {value for value in self.source_labels.values() if value.casefold() in question}
        )
        if leaked:
            raise ValueError(f"question reveals source label(s) {', '.join(map(repr, leaked))}")
        if not any(citation.role == "mechanism" for citation in self.expected_citations):
            raise ValueError(
                "RCA rows need at least one mechanism citation; definitions alone cannot "
                "support a cause"
            )
        return self
