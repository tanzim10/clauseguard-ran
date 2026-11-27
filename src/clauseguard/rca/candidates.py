"""Candidate records and manifest for SLM preparation (Task 4.6).

A candidate is manifest-side bookkeeping: identity, modality states, a prediction reference,
grouping and eligibility. The ``EvidencePacket`` stays the only model-facing object.
Source labels, cause lists and existing answers live in ``CandidateReference`` rows, which are
written to a separate file so they cannot reach a prompt by construction.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from clauseguard.baselines.contracts import Task
from clauseguard.datasets.schema import NonEmptyStr
from clauseguard.rca.contracts import DetectorPrediction, ModalityState

CANDIDATE_VERSION = "rca-candidate-v1"
REFERENCE_VERSION = "rca-candidate-reference-v1"
MANIFEST_VERSION = "rca-candidate-manifest-v1"

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SourceName = Literal["TelecomTS", "BubbleRAN", "SpotLight", "TeleLogs", "5G core faults"]
LabelStatus = Literal["labelled_fault", "labelled_normal", "unlabelled", "ambiguous"]
EligibilityStatus = Literal["eligible_pilot", "exploratory", "ineligible", "quarantined"]
CheckStatus = Literal["passed", "failed", "not_run"]
# Ordered from most to least restrictive; the most restrictive outcome of all checks wins.
STATUS_ORDER: tuple[EligibilityStatus, ...] = ("quarantined", "ineligible", "exploratory", "eligible_pilot")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceInventory(StrictModel):
    """Access, terms and local state for one source; restricted content is never stored here."""

    dataset: SourceName
    identifier: NonEmptyStr
    revision: NonEmptyStr | None = None
    licence: NonEmptyStr | None = None
    access: Literal["public", "gated", "local_only"]
    redistribution: Literal["allowed", "share_alike", "forbidden_by_request", "unknown"]
    training_use: Literal["cleared", "pending", "blocked"]
    local_status: Literal["present", "absent"]
    local_files: dict[NonEmptyStr, Sha256] = Field(default_factory=dict)
    adapter: NonEmptyStr | None = None
    verified_by: NonEmptyStr
    verified_on: NonEmptyStr
    blockers: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "SourceInventory":
        if self.training_use != "cleared" and not self.blockers:
            raise ValueError("a source not cleared for training must list its blockers")
        if (self.local_status == "present") != bool(self.local_files):
            raise ValueError("local files are listed exactly when the source is present")
        return self


class LeadDecision(StrictModel):
    """A recorded project-lead decision that changes admission; who, when and the exact scope."""

    decision_id: NonEmptyStr
    kind: Literal["family_waiver", "training_terms"]
    sources: list[SourceName] = Field(min_length=1)
    decided_by: NonEmptyStr
    decided_on: NonEmptyStr
    scope: NonEmptyStr
    statement: NonEmptyStr
    # training_terms only: blockers that remain after the terms are cleared (data fit, access).
    remaining_blockers: dict[SourceName, list[NonEmptyStr]] = Field(default_factory=dict)
    identifiers: dict[SourceName, NonEmptyStr] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> "LeadDecision":
        if self.kind == "family_waiver" and (self.remaining_blockers or self.identifiers):
            raise ValueError("a family waiver changes no terms, blockers or identifiers")
        if set(self.remaining_blockers) - set(self.sources) or set(self.identifiers) - set(self.sources):
            raise ValueError("decision details name a source outside its scope")
        return self


class LeadDecisions(StrictModel):
    decisions_version: Literal["rca-lead-decisions-v1"] = "rca-lead-decisions-v1"
    decisions: list[LeadDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "LeadDecisions":
        ids = [d.decision_id for d in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("decision IDs must be unique")
        return self

    def of_kind(self, kind: str, source: str) -> LeadDecision | None:
        return next((d for d in self.decisions if d.kind == kind and source in d.sources), None)


class CandidateIdentity(StrictModel):
    candidate_id: NonEmptyStr
    source_dataset: SourceName
    source_revision: NonEmptyStr
    source_record_id: NonEmptyStr
    source_location: dict[NonEmptyStr, str | int]
    source_sha256: Sha256


class ModalityStates(StrictModel):
    kpi: ModalityState
    text: ModalityState
    # Not part of the rca-v1 packet: settings evidence awaits a contract type (#56).
    settings: ModalityState
    text_reason: str | None = None


class Observations(StrictModel):
    """Link to the built packet; the packet itself is the only model-facing object."""

    packet_status: Literal["built", "not_built"]
    packet_sha256: Sha256 | None = None
    observation_sha256: Sha256 | None = None
    question_template: NonEmptyStr | None = None
    kpi_records: int = Field(default=0, ge=0)
    text_records: int = Field(default=0, ge=0)
    rejected_channels: list[NonEmptyStr] = Field(default_factory=list)
    not_built_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check(self) -> "Observations":
        built = self.packet_status == "built"
        if built != (self.packet_sha256 is not None and self.observation_sha256 is not None):
            raise ValueError("a built packet needs its packet and observation hashes")
        if built == (self.not_built_reason is not None):
            raise ValueError("only an unbuilt packet carries a reason")
        return self


class PredictionRef(StrictModel):
    """Detector output by reference, or an explicit absence; ``packet_head`` enters the packet."""

    packet_head: Task
    status: Literal["available", "absent"]
    file_sha256: Sha256 | None = None
    selection_sha256: Sha256 | None = None
    record_id: NonEmptyStr | None = None
    fold: int | None = None
    heads: dict[Task, DetectorPrediction] = Field(default_factory=dict)
    absence_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def _check(self) -> "PredictionRef":
        if self.status == "available":
            if not (self.file_sha256 and self.record_id and self.packet_head in self.heads):
                raise ValueError("an available prediction needs its file hash, record and packet head")
        elif not self.absence_reason:
            raise ValueError("an absent prediction needs a reason")
        if any(task != p.task for task, p in self.heads.items()):
            raise ValueError("prediction heads are keyed by their own task")
        return self


class Grouping(StrictModel):
    group_id: NonEmptyStr
    group_status: Literal["verified", "unresolved", "not_applicable"]
    event_id: NonEmptyStr | None = None
    grouping_version: NonEmptyStr


class Taxonomy(StrictModel):
    taxonomy_id: NonEmptyStr
    cause_count: int = Field(ge=0)
    label_status: LabelStatus


class Check(StrictModel):
    check_id: NonEmptyStr
    version: NonEmptyStr
    status: CheckStatus
    reasons: list[str] = Field(default_factory=list)


class Eligibility(StrictModel):
    status: EligibilityStatus
    checks: list[Check] = Field(default_factory=list)
    blockers: list[NonEmptyStr] = Field(default_factory=list)
    exclusion_reasons: list[NonEmptyStr] = Field(default_factory=list)
    split: Literal["train", "validation"] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Eligibility":
        if self.status == "eligible_pilot" and (
                self.blockers or any(c.status != "passed" for c in self.checks) or not self.checks):
            raise ValueError("eligible_pilot needs every check passed and no blockers")
        if self.status in ("ineligible", "quarantined") and not (self.blockers or self.exclusion_reasons):
            raise ValueError("an ineligible or quarantined candidate needs a reason")
        if self.status in ("ineligible", "quarantined") and self.split is not None:
            raise ValueError("only admissible candidates receive a split")
        return self


UNEVALUATED = Eligibility(status="ineligible", blockers=["eligibility_not_evaluated"])


class CandidateVersions(StrictModel):
    packet_contract: Literal["rca-v1"] = "rca-v1"
    candidate: Literal["rca-candidate-v1"] = CANDIDATE_VERSION
    adapter: NonEmptyStr
    calculation: NonEmptyStr | None = None
    question_template: NonEmptyStr | None = None


class CandidateRecord(StrictModel):
    """Manifest-side candidate; it has no field for labels, causes or source answers."""

    record_version: Literal["rca-candidate-v1"] = CANDIDATE_VERSION
    identity: CandidateIdentity
    modalities: ModalityStates
    observations: Observations
    prediction_ref: PredictionRef
    grouping: Grouping
    taxonomy: Taxonomy
    eligibility: Eligibility = UNEVALUATED
    versions: CandidateVersions


class CandidateReference(StrictModel):
    """Source labels and existing answers for one candidate; never exported into prompts.

    ``causes`` is the source's own cause list (empty for normal or unlabelled cases). An empty
    list alone cannot tell normal from unlabelled, so ``label_status`` is always explicit.
    """

    record_version: Literal["rca-candidate-reference-v1"] = REFERENCE_VERSION
    candidate_id: NonEmptyStr
    taxonomy_id: NonEmptyStr
    source_labels: dict[NonEmptyStr, list[str]] = Field(default_factory=dict)
    causes: list[NonEmptyStr] = Field(default_factory=list)
    label_status: LabelStatus
    annotation: dict[str, Any] = Field(default_factory=dict)
    source_response: str | None = None
    source_response_sha256: Sha256 | None = None
    source_text_refs: list[NonEmptyStr] = Field(default_factory=list)

    @property
    def cause_count(self) -> int:
        return len(self.causes)

    @model_validator(mode="after")
    def _check(self) -> "CandidateReference":
        if len(set(self.causes)) != len(self.causes):
            raise ValueError("cause lists hold distinct causes")
        if (self.label_status == "labelled_fault") != bool(self.causes):
            raise ValueError("only a labelled fault carries causes, and it needs at least one")
        if self.label_status == "unlabelled" and self.source_labels:
            raise ValueError("an unlabelled candidate has no source labels")
        if (self.source_response is None) != (self.source_response_sha256 is None):
            raise ValueError("a source response and its hash are both present or both absent")
        return self


class FileEntry(StrictModel):
    sha256: Sha256
    rows: int | None = Field(default=None, ge=0)


class BatchRecord(StrictModel):
    selection_rule: NonEmptyStr
    seed: int
    size_limit: int = Field(ge=0)
    candidate_ids: list[NonEmptyStr]
    admitted_statuses: list[EligibilityStatus]

    @model_validator(mode="after")
    def _check(self) -> "BatchRecord":
        if len(self.candidate_ids) > self.size_limit or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("the batch holds distinct candidates within its size limit")
        return self


class CandidateManifest(StrictModel):
    """Versioned release record kept outside model inputs."""

    manifest_version: Literal["rca-candidate-manifest-v1"] = MANIFEST_VERSION
    candidate_release_id: Sha256
    created_on: NonEmptyStr
    sources: list[SourceInventory]
    contract_versions: dict[NonEmptyStr, str | None]
    reservations_sha256: Sha256
    prediction_files: dict[NonEmptyStr, Sha256] = Field(default_factory=dict)
    retrieval: NonEmptyStr
    configuration: dict[str, Any]
    files: dict[NonEmptyStr, FileEntry]
    counts: dict[str, Any]
    batch: BatchRecord

    @model_validator(mode="after")
    def _check(self) -> "CandidateManifest":
        names = [s.dataset for s in self.sources]
        if len(names) != len(set(names)):
            raise ValueError("each source appears once in the inventory")
        return self
