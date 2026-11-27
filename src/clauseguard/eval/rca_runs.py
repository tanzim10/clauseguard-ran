"""Read saved RCA run records (`rca-run-v1`, Task 4.7) into the small view the evaluator needs.

The evaluator never calls the generator and never reads reference records here: a run record
holds the packet the generator saw, its parsed output and its timings, nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

RUN_RECORD_VERSION = "rca-run-v1"
FACTUAL_CLAIM_TYPES = ("observation", "mechanism", "diagnostic_inference")


class RunRecordError(ValueError):
    """A run record is unreadable or is not an `rca-run-v1` record."""


@dataclass(frozen=True)
class RunView:
    run_id: str
    case_id: str
    execution_status: str
    execution_stage: str | None
    execution_reason: str | None
    result_status: str | None
    abstention_reason: str | None
    claims: tuple[dict[str, Any], ...]
    causes: tuple[dict[str, Any], ...]
    packet: dict[str, Any]
    generator_kind: str
    requested_model: str | None
    served_model: str | None
    prompt_version: str
    generation_settings: dict[str, Any]
    timings: dict[str, float]
    usage: dict[str, Any]
    resources: dict[str, Any]
    detector_label: str | None
    superseded: tuple[str, ...] = ()
    uncertainty_reasons: tuple[str, ...] = ()
    structural_checks: tuple[dict[str, Any], ...] = ()
    numeric_checks: tuple[dict[str, Any], ...] = ()
    parsed_output: dict[str, Any] | None = None

    @property
    def complete(self) -> bool:
        return self.execution_status == "complete"

    @property
    def answered(self) -> bool:
        return self.complete and self.result_status == "answered"

    @property
    def abstained(self) -> bool:
        return self.complete and self.result_status == "abstained"

    @property
    def category_set(self) -> frozenset[str]:
        """Categories of an answered run; failed and abstained runs return an empty set."""
        return frozenset(c["category"] for c in self.causes) if self.answered else frozenset()

    @property
    def packet_evidence_ids(self) -> frozenset[str]:
        packet = self.packet
        records = (
            packet.get("kpi_evidence", [])
            + packet.get("specification_evidence", [])
            + packet.get("text_evidence", [])
        )
        return frozenset(r["evidence_id"] for r in records)


def view_from_record(record: dict[str, Any]) -> RunView:
    """Build a `RunView` from one parsed run record, raising `RunRecordError` if it is malformed."""
    if record.get("version") != RUN_RECORD_VERSION:
        raise RunRecordError(f"not an {RUN_RECORD_VERSION} record: {record.get('version')!r}")
    try:
        response = record["response"]
        parsed = record["parsed_output"] or {}
        result = response.get("result") or {}
        prediction = (record["projected_packet"].get("detector_prediction") or {})
        return RunView(
            run_id=record["run_id"],
            case_id=record["case_id"],
            execution_status=response["execution_status"],
            execution_stage=response.get("execution_stage"),
            execution_reason=response.get("execution_reason"),
            result_status=result.get("status"),
            abstention_reason=parsed.get("abstention_reason"),
            claims=tuple(parsed.get("claims", [])),
            causes=tuple(parsed.get("final_fault_categories", [])),
            packet=record["projected_packet"],
            generator_kind=record["generator_kind"],
            requested_model=record.get("requested_model"),
            served_model=record.get("served_model"),
            prompt_version=record["prompt_version"],
            generation_settings=record["config"]["generation"],
            timings=record.get("timings") or {},
            usage=record.get("usage") or {},
            resources=record.get("resources") or {},
            detector_label=prediction.get("predicted_label"),
            uncertainty_reasons=tuple(parsed.get("uncertainty_reasons", [])),
            structural_checks=tuple(response.get("structural_checks") or []),
            numeric_checks=tuple(response.get("numeric_checks") or []),
            parsed_output=record["parsed_output"],
        )
    except (KeyError, AttributeError, TypeError) as exc:
        raise RunRecordError(f"malformed run record: {exc!r}") from exc


def load_run_records(directories: Sequence[Path | str]) -> list[RunView]:
    """Load run records from one or more directories, one view per case.

    A directory is a `run_rca` output directory (its `runs/` folder is read) or a folder of
    records. When a case appears more than once, the later record wins unless it failed at
    generation while the earlier one did not; replaced run IDs are kept on the surviving view
    as `superseded`.
    """
    chosen: dict[str, RunView] = {}
    for directory in directories:
        root = Path(directory)
        folder = root / "runs" if (root / "runs").is_dir() else root
        files = sorted(folder.glob("*.json"))
        if not files:
            raise RunRecordError(f"no run records in {folder}")
        for file in files:
            try:
                view = view_from_record(json.loads(file.read_text()))
            except json.JSONDecodeError as exc:
                raise RunRecordError(f"{file}: {exc}") from exc
            current = chosen.get(view.case_id)
            if current is None:
                chosen[view.case_id] = view
            elif not _generation_failed(current) and _generation_failed(view):
                chosen[view.case_id] = _supersede(current, view)
            else:
                chosen[view.case_id] = _supersede(view, current)
    return [chosen[key] for key in sorted(chosen)]


def _generation_failed(view: RunView) -> bool:
    return view.execution_status == "failed" and view.execution_stage == "generation"


def _supersede(keep: RunView, dropped: RunView) -> RunView:
    return replace(keep, superseded=keep.superseded + (dropped.run_id,) + dropped.superseded)
