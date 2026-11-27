"""CLI: check and judge teacher drafts for the cited SLM training set (Task 4.9, machine acceptance).

    python -m clauseguard.cli.judge_training --runs <run_rca dir> [--runs ...] \\
        --references <references.jsonl> --judge-model <id> --output-dir <new dir> [--replay <dir>]

For each case the latest run record is checked (structural and numeric results are read from the
record; label and KPI-support checks run here), then judged with prompt rca-judge-v2 by a model that
differs from the teacher. Writes `judgments.jsonl`, `checks.jsonl`, `acceptance.json` and
`replies/<run_id>.txt` (raw judge replies, reusable with `--replay`). Machine acceptance only:
human review is recorded elsewhere. Exit status: 0 on success, 1 when some judge calls failed or
were malformed, 2 when the run could not start.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.eval.rca_judge import (
    JUDGE_PROMPT_VERSION_V2,
    CaseJudgment,
    JudgeCallError,
    JudgeClient,
    JudgePolicy,
    JudgeReply,
    OpenAIJudgeClient,
    judge_case,
)
from clauseguard.eval.rca_runs import RunRecordError, RunView, load_run_records
from clauseguard.rca.candidates import CandidateReference
from clauseguard.rca.checks import CheckResult
from clauseguard.rca.contracts import EvidencePacket
from clauseguard.rca.training_checks import KPI_SUPPORT_CHECK_ID, kpi_support_check, label_check

ACCEPTANCE_VERSION = "rca-acceptance-v1"
ACCEPTANCE_RULE = (
    "accepted iff execution is complete AND every structural and numeric check passed AND "
    "(answered: label.v1 passed AND kpi_support.v1 passed AND every claim judged fully_supported "
    "AND category_supported is true) OR (abstained: abstention_justified is true; an abstention is "
    "accepted only if justified, on any case); all else rejected. Human review is separate."
)


ACCEPTANCE_VERSION_V2 = "rca-acceptance-v2"
ACCEPTANCE_RULE_V2 = (
    "accepted iff execution is complete AND every structural and numeric check passed AND "
    "(answered: label.v1 passed (the category equals the source label) AND kpi_support.v1 passed AND every "
    "observation and mechanism claim judged fully_supported AND every diagnostic_inference claim judged "
    "fully_supported or partially_supported AND no claim judged unsupported, contradicted or cannot_determine; "
    "category_supported is recorded but not required, because the frozen corpus has no passage that names the "
    "mechanism of most source categories) OR (abstained: abstention_justified is true); all else rejected. "
    "Human review is separate."
)
RULES = {"v1": (ACCEPTANCE_VERSION, ACCEPTANCE_RULE), "v2": (ACCEPTANCE_VERSION_V2, ACCEPTANCE_RULE_V2)}


class CaseAcceptance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    machine_status: str
    reasons: list[str] = Field(default_factory=list)
    judge_model: str


def decide(view: RunView, label: CheckResult | None, kpi: CheckResult | None,
           judgment: CaseJudgment | None, rule: str = "v1") -> tuple[str, list[str]]:
    """Apply the acceptance rule (`RULES`); returns the status and every reason for rejection."""
    reasons: list[str] = []
    if not view.complete:
        return "rejected", [f"execution {view.execution_status} at {view.execution_stage}"]
    reasons += [f"{c['check_id']} failed" for c in view.structural_checks if c.get("status") != "passed"]
    reasons += [f"numeric.v2 failed for {n.get('claim_id')}" for n in view.numeric_checks if n.get("status") == "failed"]
    if view.answered:
        reasons += [f"{c.check_id} failed: {c.detail}" for c in (label, kpi) if c is not None and c.status != "passed"]
    if judgment is None:
        reasons.append("not judged")
    elif judgment.status != "judged":
        reasons.append(f"judge {judgment.status}")
    elif view.answered:
        if rule == "v2":
            types = {c["claim_id"]: c.get("claim_type") for c in view.claims}
            for j in judgment.claims:
                allowed = ({"fully_supported", "partially_supported"}
                           if types.get(j.claim_id) == "diagnostic_inference" else {"fully_supported"})
                if j.decision not in allowed:
                    reasons.append(f"claim {j.claim_id} ({types.get(j.claim_id)}) judged {j.decision or j.status}")
        else:
            reasons += [f"claim {j.claim_id} judged {j.decision or j.status}" for j in judgment.claims
                        if j.decision != "fully_supported"]
            if judgment.category_supported is not True:
                reasons.append(f"category_supported is {judgment.category_supported}")
    elif judgment.abstention_justified is not True:
        reasons.append(f"abstention_justified is {judgment.abstention_justified}")
    return ("rejected" if reasons else "accepted"), reasons


class ReplayJudge:
    """Serves recorded replies named `<run_id>.txt`; used in tests, never calls a service."""

    def __init__(self, directory: Path) -> None:
        self._directory, self._current = directory, ""

    def for_run(self, run_id: str) -> None:
        self._current = run_id

    def complete(self, system: str, user: str) -> JudgeReply:
        file = self._directory / f"{self._current}.txt"
        if not file.is_file():
            raise JudgeCallError(f"no recorded reply for {self._current}")
        return JudgeReply(text=file.read_text(), served_model="replay")


class _Recorder:
    """Wraps a judge client, saving each raw reply under the current run id."""

    def __init__(self, inner: JudgeClient, directory: Path) -> None:
        self._inner, self._directory, self.current = inner, directory, ""

    def complete(self, system: str, user: str) -> JudgeReply:
        reply = self._inner.complete(system, user)
        self._directory.mkdir(parents=True, exist_ok=True)
        (self._directory / f"{self.current}.txt").write_text(reply.text)
        return reply


def load_references(path: Path, wanted: set[str]) -> dict[str, CandidateReference]:
    """Stream the references file and parse only the wanted candidate ids."""
    found: dict[str, CandidateReference] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            if data.get("candidate_id") in wanted:
                found[data["candidate_id"]] = CandidateReference.model_validate(data)
    return found


def _parse_output(view: RunView) -> RcaModelOutput | None:
    try:
        return RcaModelOutput.model_validate(view.parsed_output) if view.parsed_output else None
    except ValidationError:
        return None


def _case_checks(view: RunView, output: RcaModelOutput | None, reference: CandidateReference | None,
                 ) -> tuple[CheckResult | None, CheckResult | None]:
    label = label_check(output, reference) if reference is not None else None
    try:
        kpi = kpi_support_check(output, EvidencePacket.model_validate(view.packet))
    except ValidationError:
        kpi = CheckResult(check_id=KPI_SUPPORT_CHECK_ID, status="failed", detail="packet is not a valid evidence packet")
    return label, kpi


def _check_rows(view: RunView, label: CheckResult | None, kpi: CheckResult | None) -> list[dict[str, Any]]:
    rows = [{"check_id": c["check_id"], "status": c["status"], "subject": c.get("subject"), "detail": c.get("detail")}
            for c in view.structural_checks]
    rows += [{"check_id": "numeric.v2", "status": n["status"], "subject": n.get("claim_id"), "detail": None}
             for n in view.numeric_checks]
    rows += [c.model_dump() for c in (label, kpi) if c is not None]
    return [{"case_id": view.case_id, "run_id": view.run_id, **row} for row in rows]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.output_dir.exists():
            raise FileExistsError("output directory exists; judging goes to a new directory")
        views = load_run_records(args.runs)
        generators = {m for v in views for m in (v.requested_model, v.served_model) if m}
        if args.judge_model in generators:
            raise ValueError(f"the judge model {args.judge_model} is also a teacher model; choose a separate judge")
        references = load_references(args.references, {v.case_id for v in views})
        judge_client, policy, bind = _client(
            args, JudgePolicy(model=args.judge_model, prompt_version=JUDGE_PROMPT_VERSION_V2))
    except (FileExistsError, FileNotFoundError, RunRecordError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True)
    judgments, check_rows, cases = [], [], {}
    for view in views:
        output = _parse_output(view)
        label, kpi = _case_checks(view, output, references.get(view.case_id))
        if view.complete and label is None:
            label = CheckResult(check_id="label.v1", status="failed", detail="no reference for this case")
        check_rows += _check_rows(view, label, kpi)
        judgment = None
        if view.complete and output is not None:
            bind(view.run_id)
            judgment = judge_case(judge_client, policy, view)
            judgments.append(judgment)
        status, reasons = decide(view, label, kpi, judgment, args.acceptance_rule)
        cases[view.case_id] = CaseAcceptance(run_id=view.run_id, machine_status=status, reasons=reasons,
                                             judge_model=policy.model).model_dump()
    out = args.output_dir
    (out / "judgments.jsonl").write_text("".join(j.model_dump_json() + "\n" for j in judgments))
    (out / "checks.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in check_rows))
    (out / "acceptance.json").write_text(json.dumps(
        {"version": RULES[args.acceptance_rule][0], "rule": RULES[args.acceptance_rule][1],
         "judge": policy.identity(), "cases": cases},
        indent=2, sort_keys=True))
    accepted = sum(c["machine_status"] == "accepted" for c in cases.values())
    print(f"cases {len(cases)}: accepted {accepted}, rejected {len(cases) - accepted}")
    return 1 if any(j.status != "judged" for j in judgments) else 0


def _client(args: argparse.Namespace, policy: JudgePolicy) -> tuple[JudgeClient, JudgePolicy, Any]:
    """Return the judge client, the policy it runs under and a callback naming the current run."""
    if args.replay:
        replay = ReplayJudge(args.replay)
        return replay, policy, replay.for_run
    from openai import OpenAI

    from clauseguard.config import get_settings, nvidia_api_keys

    settings = get_settings()
    keys = nvidia_api_keys(settings)
    if not keys:
        raise RuntimeError("judging needs NVIDIA_API_KEY (or use --replay)")
    policy = policy.model_copy(update={"max_attempts": max(policy.max_attempts, len(keys))})
    openai_clients = [OpenAI(api_key=k, base_url=settings.nvidia_base_url, timeout=600.0, max_retries=0) for k in keys]
    recorder = _Recorder(OpenAIJudgeClient(openai_clients[0], policy, fallbacks=openai_clients[1:]),
                         args.output_dir / "replies")

    def bind(run_id: str) -> None:
        recorder.current = run_id

    return recorder, policy, bind


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check and judge teacher drafts (machine acceptance only).")
    parser.add_argument("--runs", type=Path, action="append", required=True, help="run_rca output directory; repeatable")
    parser.add_argument("--references", type=Path, required=True, help="candidate references.jsonl")
    parser.add_argument("--judge-model", required=True, help="must differ from the teacher model")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory")
    parser.add_argument("--acceptance-rule", choices=tuple(RULES), default="v2",
                        help="v2 (default): diagnostic claims may be partially supported; v1: all fully supported "
                             "and the category judged supported")
    parser.add_argument("--replay", type=Path, help="directory of recorded judge replies (<run_id>.txt), no network")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
