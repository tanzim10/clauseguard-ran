"""Typed RCA evidence records shared by adapters and the generator boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.baselines.contracts import Task
from clauseguard.datasets.schema import NonEmptyStr

ModalityState = Literal["included", "missing", "omitted_by_configuration"]


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: NonEmptyStr
    revision: NonEmptyStr
    case_id: NonEmptyStr
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    recording_family_id: str | None = None
    recording_family_status: Literal["verified", "unresolved", "not_applicable"]
    reuse_terms: NonEmptyStr | None = None


class DetectorPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Task
    predicted_label_id: int | str | None = None
    predicted_label: str | None = None
    label_mapping_version: NonEmptyStr | None = None
    status: Literal["ok", "invalid", "error", "absent"]
    absence_reason: NonEmptyStr | None = None
    model_run_id: NonEmptyStr | None = None
    source_modalities: list[Literal["kpi", "text"]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class KpiStatisticEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: NonEmptyStr
    source_window_id: NonEmptyStr
    channel: NonEmptyStr
    statistic: Literal["mean", "minimum", "maximum", "median", "standard_deviation", "scalar"]
    value: float = Field(allow_inf_nan=False)
    unit: str | None = None
    sample_start: int | None = Field(default=None, ge=0)
    sample_end: int | None = Field(default=None, ge=0)
    time_start: str | None = None
    time_end: str | None = None
    valid_count: int = Field(ge=1)
    sample_count: int = Field(ge=1)
    missing_fraction: float = Field(ge=0, le=1)
    baseline_evidence_id: str | None = None
    calculation_version: NonEmptyStr


class TextEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: NonEmptyStr
    record_id: NonEmptyStr
    text: NonEmptyStr
    origin: Literal["independent_observation", "kpi_derived"]
    approval_status: Literal["approved_for_rca"]
    source: NonEmptyStr
    timestamp: str | None = None


class SpecificationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: NonEmptyStr
    chunk_id: NonEmptyStr
    document_id: NonEmptyStr
    version: NonEmptyStr
    clause: NonEmptyStr
    text: NonEmptyStr
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidencePacket(BaseModel):
    """Inference-only record; reference labels and evaluation data have no fields here."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal["rca-v1"] = "rca-v1"
    source: SourceIdentity
    question: NonEmptyStr
    detector_prediction: DetectorPrediction | None = None
    kpi_state: ModalityState
    text_state: ModalityState
    kpi_evidence: list[KpiStatisticEvidence] = Field(default_factory=list)
    text_evidence: list[TextEvidence] = Field(default_factory=list)
    specification_evidence: list[SpecificationEvidence] = Field(default_factory=list)
    retrieval_status: Literal["completed", "not_run", "failed"]


class RcaServerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    detector_prediction: DetectorPrediction | None = None
    result: RcaModelOutput
    resolved_evidence: dict[NonEmptyStr, SourceIdentity | KpiStatisticEvidence | TextEvidence | SpecificationEvidence]
    structural_checks: list[NonEmptyStr]
    semantic_review_status: Literal["pending", "reviewed"] = "pending"
    human_review_status: Literal["pending", "approved", "rejected"] = "pending"
    execution_status: Literal["complete", "incomplete", "failed"]


DATASET_SOURCE_RULES = {
    "TelecomTS": ("kpi", "reviewed_text", "source_labels", "recording_family_and_text_review"),
    "BubbleRAN": ("radio_kpi", "binary_interference_label", "source_labels", "units_and_capture_review"),
    "SpotLight": ("radio_and_platform", "anomaly_intervals", "source_labels", "alignment_and_terms_review"),
    "TeleLogs": ("text_settings_measurements", "cause_labels", "source_targets", "access_and_taxonomy_review"),
    "5G core faults": ("text", "troubleshooting_response", "source_targets", "accuracy_and_reuse_review"),
}


def validate_output_for_packet(output: RcaModelOutput, packet: EvidencePacket) -> None:
    """Check claim IDs and claim types against the exact evidence packet."""
    evidence_by_id = {
        item.evidence_id: item
        for item in [
            *packet.kpi_evidence,
            *packet.text_evidence,
            *packet.specification_evidence,
        ]
    }
    for claim in output.claims:
        missing = set(claim.evidence_ids) - evidence_by_id.keys()
        if missing:
            raise ValueError(f"claim {claim.claim_id!r} cites unknown evidence IDs: {sorted(missing)}")
        cited = [evidence_by_id[evidence_id] for evidence_id in claim.evidence_ids]
        if claim.claim_type == "observation" and not all(
            isinstance(item, (KpiStatisticEvidence, TextEvidence)) for item in cited
        ):
            raise ValueError("observation claims must cite KPI or text evidence")
        if claim.claim_type == "mechanism" and not all(
            isinstance(item, SpecificationEvidence) for item in cited
        ):
            raise ValueError("mechanism claims must cite specification evidence")
        if claim.claim_type == "diagnostic_inference" and not (
            any(isinstance(item, (KpiStatisticEvidence, TextEvidence)) for item in cited)
            and any(isinstance(item, SpecificationEvidence) for item in cited)
        ):
            raise ValueError("diagnostic claims must cite incident and specification evidence")
