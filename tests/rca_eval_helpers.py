"""Builders for synthetic `rca-run-v1` records used by the Task 4.8 evaluation tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKET = {
    "contract_version": "rca-v1",
    "detector_prediction": {"predicted_label": "normal", "status": "ok"},
    "kpi_evidence": [
        {"evidence_id": "K1", "channel": "UL_SNR", "statistic": "mean", "value": 6.42, "unit": None,
         "time_start": "t0", "time_end": "t1", "baseline_evidence_id": None},
        {"evidence_id": "K2", "channel": "RSRP", "statistic": "mean", "value": -98.1, "unit": None,
         "time_start": "t0", "time_end": "t1", "baseline_evidence_id": None},
    ],
    "specification_evidence": [
        {"evidence_id": "S1", "document_id": "3GPP TS 32.425", "clause": "A.9 Monitoring of RF performance",
         "text": "Uplink interference disturbs the receiver and leads to receive errors."},
        {"evidence_id": "S2", "document_id": "3GPP TS 28.552", "clause": "5.1.1.2.1 DL Total PRB Usage",
         "text": "Defines downlink PRB usage."},
    ],
    "text_evidence": [],
    "question": "What is the most likely root cause?",
}
CLAIMS = [
    {"claim_id": "C1", "claim_type": "observation", "evidence_ids": ["K1"], "text": "UL SNR mean is 6.42."},
    {"claim_id": "C2", "claim_type": "mechanism", "evidence_ids": ["S1"], "text": "Interference disturbs the receiver."},
    {"claim_id": "C3", "claim_type": "diagnostic_inference", "evidence_ids": ["K1", "S1"],
     "text": "Low SNR fits interference."},
]
CAUSES = [{"category": "Co-Channel Interference (Severe)", "claim_ids": ["C1", "C2", "C3"]}]


def make_record(case_id: str = "rca_001", *, run_id: str | None = None, status: str = "complete",
                stage: str | None = None, reason: str | None = None, outcome: str = "answered",
                claims: list[dict[str, Any]] | None = None, causes: list[dict[str, Any]] | None = None,
                packet: dict[str, Any] | None = None, model: str = "gen/model-a", seconds: float = 10.0,
                generator_kind: str = "live") -> dict[str, Any]:
    answered = outcome == "answered"
    parsed = None
    if status == "complete" or stage == "validation":
        parsed = {
            "status": outcome,
            "claims": (CLAIMS if claims is None else claims) if answered else [],
            "final_fault_categories": (CAUSES if causes is None else causes) if answered else [],
            "abstention_reason": None if answered else "insufficient_mechanism",
            "uncertainty_reasons": [],
        }
    result = {"status": outcome} if status == "complete" else None
    return {
        "version": "rca-run-v1",
        "run_id": run_id or f"run-{case_id}-{status}-{stage}-{seconds}",
        "case_id": case_id,
        "generator_kind": generator_kind,
        "requested_model": model,
        "served_model": model,
        "prompt_version": "rca-prompt-v2",
        "config": {"generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 16384}},
        "projected_packet": packet or PACKET,
        "parsed_output": parsed,
        "response": {"execution_status": status, "execution_stage": stage, "execution_reason": reason,
                     "result": result},
        "timings": {"original_retrieval": 0.5, "generation": seconds, "validation": 0.01, "total": seconds + 0.6},
        "usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        "resources": {"peak_rss": 100_000_000, "peak_rss_unit": "bytes", "scope": "whole process, not this run alone"},
    }


def write_runs(directory: Path, records: list[dict[str, Any]]) -> Path:
    runs = directory / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for record in records:
        (runs / f"{record['run_id']}.json").write_text(json.dumps(record))
    return directory
