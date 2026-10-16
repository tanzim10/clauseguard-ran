"""Task 2 release contracts. Labels and provenance never belong to predictor inputs."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

SPLITS = ("train", "dev", "test")
TASKS = ("detection", "category")
NUMERIC_CHANNELS = (
    "DL_BLER", "DL_MCS", "DL_NumberOfPackets", "Estimated_UL_Buffer",
    "PRB_Utilization_DL", "PRB_Utilization_UL", "PRBs_DL_Current", "PRBs_UL_Current",
    "RSRP", "RX_Bytes", "TX_Bytes", "UL_BLER", "UL_MCS", "UL_NPRB",
    "UL_NumberOfPackets", "UL_SNR",
)
CATEGORY_LABELS = (
    "Antenna Failure", "Co-Channel Interference (Mild)", "Co-Channel Interference (Severe)",
    "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
    "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)",
    "Resource Allocation Bugs", "High Network Congestion (Gradual Buildup)",
    "High Network Congestion (Sudden Spike)",
)
LABEL_MAPPINGS = {"version": "1", "detection": {"normal": 0, "abnormal": 1},
                  "category": {name: index for index, name in enumerate(CATEGORY_LABELS)}}


class RecordingReview(BaseModel):
    """An explicit human judgment; raw measurement matching cannot create this approval."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["unresolved", "verified"] = "unresolved"
    recording_id: str | None = Field(default=None, min_length=1)
    parent_recording_ids: list[str] = Field(default_factory=list)
    candidate_parent_sources: list[str] = Field(default_factory=list)
    evidence: str = ""
    reviewer: str = ""
    reviewed_at: str | None = None

    @model_validator(mode="after")
    def check_approval(self):
        if any(not item.strip() for item in self.parent_recording_ids):
            raise ValueError("parent recording IDs cannot be blank")
        if self.recording_id is not None and not self.recording_id.strip():
            raise ValueError("recording ID cannot be blank")
        if self.status == "verified" and self.candidate_parent_sources:
            raise ValueError("verified review cannot retain unresolved candidate parents")
        if self.status == "verified":
            if not all((self.recording_id, self.evidence.strip(), self.reviewer.strip(),
                        self.reviewed_at)):
                raise ValueError("verified recording review requires ID, evidence, reviewer and date")
            datetime.fromisoformat(self.reviewed_at)
        return self


class NumericInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    channel_names: list[str]
    sampling_rate_hz: float
    values: list[list[StrictInt | StrictFloat | None]]


class ModelInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kpi: NumericInputs | None = None
    text: str | None = None


class ModelExample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: ModelInputs
    target: int
    scenario_id: str
    group_id: str
    split: Literal["train", "dev", "test"]
    task: Literal["detection", "category"]
