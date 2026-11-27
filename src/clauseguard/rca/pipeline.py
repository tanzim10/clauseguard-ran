"""Shared RCA core used by FastAPI and MCP: assemble, project, prompt, generate, check, render.

The core takes a saved case (Task 4.4 evidence packet and audits, Task 4.5 retrieval snapshot),
applies the experiment configuration, and records every run. It imports nothing from FastAPI or
MCP. Expected failures never raise from ``run_rca``; they are recorded with their stage.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from clauseguard.api.schemas import RcaModelOutput
from clauseguard.llm.clients import (
    GenerationAttempt,
    GenerationResult,
    GenerationUnavailable,
    MalformedGenerationOutput,
)
from clauseguard.rca.checks import (
    CHECKS_VERSION,
    TELECOMTS_TAXONOMY,
    CheckResult,
    NumericCheckResult,
    OutputParseError,
    Taxonomy,
    numeric_checks,
    parse_model_output,
    structural_checks,
)
from clauseguard.rca.contracts import (
    DetectorPrediction,
    EvidencePacket,
    KpiStatisticEvidence,
    RcaServerResponse,
    SpecificationEvidence,
    TextEvidence,
)
from clauseguard.rca.evidence import (
    EvidenceAudit,
    EvidenceError,
    _digest,
    check_audits,
    check_packet_ids,
)
from clauseguard.rca.prompts import (
    PROMPT_VERSION,
    build_rca_messages,
    prompt_sha256,
    prompt_view,
)
from clauseguard.rca.retrieval import RetrievalSnapshot, SnapshotError, load_snapshot, replay

Stage = Literal["assembly", "retrieval", "context_budget", "generation", "validation"]
Configuration = Literal["base_llm_full", "adapted_slm_full", "text_rag", "kpi_summary_rag"]
RUN_RECORD_VERSION = "rca-run-v1"
NOT_FOUND = "not found"
FULL_CONFIGURATIONS = ("base_llm_full", "adapted_slm_full")
# Placeholder until #65 pins the SLM tokenizer (Qwen3-8B) and measures complete prompts: about
# 20k tokens at ~3 characters per token. A real packet prompt is ~26k characters before passages.
DEFAULT_CONTEXT_BUDGET_CHARS = 60_000


class PipelineFailure(Exception):
    """An expected failure at one stage; ``status`` separates incomplete from failed."""

    def __init__(self, stage: Stage, reason: str, status: Literal["failed", "incomplete"] = "failed"):
        super().__init__(reason)
        self.stage, self.reason, self.status = stage, reason, status


class RcaCase(BaseModel):
    """One saved case: the Task 4.4 packet and audits, plus its Task 4.5 snapshot and manifest."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    packet: EvidencePacket
    audits: list[EvidenceAudit]
    snapshot_path: Path | None
    manifest: dict[str, Any]


def assemble(case: RcaCase) -> tuple[EvidencePacket, RetrievalSnapshot, dict[str, Any]]:
    """Verify evidence IDs, audits, snapshot and packet hashes, then replay the saved passages."""
    try:
        check_packet_ids(case.packet)
        check_audits(case.packet, case.audits)
    except EvidenceError as exc:
        raise PipelineFailure("assembly", f"evidence check failed: {exc}") from exc
    if case.snapshot_path is None or not Path(case.snapshot_path).is_file():
        raise PipelineFailure("retrieval", "no retrieval snapshot exists for this case", "incomplete")
    try:
        snapshot = load_snapshot(case.snapshot_path, case.manifest)
        replayed, timing = replay(case.packet, snapshot)
        check_packet_ids(replayed)
    except (SnapshotError, EvidenceError, ValidationError, KeyError) as exc:
        raise PipelineFailure("assembly", f"snapshot check failed: {exc}") from exc
    return replayed, snapshot, timing


def _without_modality(prediction: DetectorPrediction | None, modality: str) -> DetectorPrediction | None:
    """Keep a prediction only when it provably used none of the omitted modality."""
    if prediction is None or prediction.status == "absent":
        return prediction
    if prediction.source_modalities and modality not in prediction.source_modalities:
        return prediction
    return DetectorPrediction(task=prediction.task, status="absent",
                              absence_reason=f"omitted_by_configuration: prediction may use {modality} input")


def project_packet(packet: EvidencePacket, configuration: str,
                   minimal_question: str | None = None) -> EvidencePacket:
    """Apply an experiment configuration before the prompt is built (``rca-contract.md`` projections)."""
    if configuration in FULL_CONFIGURATIONS:
        return packet
    data = packet.model_dump()
    if configuration == "text_rag":
        data.update(
            kpi_evidence=[],
            kpi_state="omitted_by_configuration" if packet.kpi_state == "included" else packet.kpi_state,
            text_evidence=[t.model_dump() for t in packet.text_evidence if t.origin != "kpi_derived"],
            detector_prediction=_without_modality(packet.detector_prediction, "kpi"))
    elif configuration == "kpi_summary_rag":
        if not minimal_question:
            raise PipelineFailure("assembly", "kpi_summary_rag needs the approved minimal question")
        data.update(
            text_evidence=[], question=minimal_question,
            text_state="omitted_by_configuration" if packet.text_state == "included" else packet.text_state,
            detector_prediction=_without_modality(packet.detector_prediction, "text"))
    else:
        raise PipelineFailure("assembly", f"unknown configuration {configuration!r}")
    return EvidencePacket.model_validate(data)


def check_projection(original: EvidencePacket, projected: EvidencePacket, configuration: str) -> None:
    """Omitted modalities must not return through records, names, predictions or derived text.

    Specification passages are excluded from the scan: shared retrieval may carry information from
    an omitted modality, a limit recorded in the retrieval manifest.
    """
    if configuration in FULL_CONFIGURATIONS:
        return
    view = prompt_view(projected)
    view.pop("specification_evidence")
    blob = json.dumps(view, ensure_ascii=False)
    prediction = projected.detector_prediction
    if configuration == "text_rag":
        omitted, records = "kpi", projected.kpi_evidence
        leaked = sorted({item.channel for item in original.kpi_evidence if item.channel in blob})
        leaked += [item.evidence_id for item in projected.text_evidence if item.origin == "kpi_derived"]
    else:
        omitted, records = "text", projected.text_evidence
        leaked = [item.evidence_id for item in original.text_evidence if item.text in blob]
    if records:
        leaked.append(f"{omitted} records")
    if prediction is not None and prediction.status != "absent" and (
            not prediction.source_modalities or omitted in prediction.source_modalities):
        leaked.append(f"{omitted}-sourced detector prediction")
    if leaked:
        raise PipelineFailure("assembly", f"{configuration} prompt would expose omitted {omitted} input: {leaked}")


class BudgetResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["characters"] = "characters"
    limit: int
    measured: int
    within: bool


def check_context_budget(messages: list[dict[str, str]], limit: int) -> BudgetResult:
    """Measure the whole prompt; overflow is a failure, evidence is never dropped or shortened."""
    measured = sum(len(message["content"]) for message in messages)
    return BudgetResult(limit=limit, measured=measured, within=measured <= limit)



class GenerationSettings(BaseModel):
    """Sent to the provider as given; ``None`` leaves a setting at the provider default."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float | None = Field(default=0.0, ge=0)
    top_p: float | None = Field(default=1.0, gt=0, le=1)
    max_tokens: int | None = Field(default=16384, ge=1)
    seed: int | None = None
    timeout_seconds: float | None = Field(default=240.0, gt=0)


class ExperimentConfig(BaseModel):
    """Fixed before a run and hashed into its record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration: Configuration
    run_kind: Literal["scored", "exploratory"]
    model_id: str = Field(min_length=1)
    fallback_models: tuple[str, ...] = ()
    generation: GenerationSettings = GenerationSettings()
    context_budget_chars: int = Field(default=DEFAULT_CONTEXT_BUDGET_CHARS, ge=1)
    taxonomy: Taxonomy = TELECOMTS_TAXONOMY
    prompt_version: Literal["rca-prompt-v1", "rca-prompt-v2"] = PROMPT_VERSION
    numeric_policy: Literal["fail", "record"] = "fail"
    minimal_question: str | None = None

    @model_validator(mode="after")
    def check_rules(self) -> ExperimentConfig:
        if self.run_kind == "scored" and (self.fallback_models or self.numeric_policy != "fail"):
            raise ValueError("scored runs use one fixed model and fail on numeric mismatches")
        if self.model_id in self.fallback_models:
            raise ValueError("fallback models must differ from the selected model")
        if self.configuration == "kpi_summary_rag" and not self.minimal_question:
            raise ValueError("kpi_summary_rag needs the approved minimal question")
        return self

    def sha256(self) -> str:
        return _digest(self.model_dump(mode="json"))


class Generator(Protocol):
    def complete(self, messages: list[dict[str, str]], *, task: str = "rca",
                 models: tuple[str, ...] | None = None, failover: bool = True,
                 temperature: float | None = None, top_p: float | None = None,
                 max_tokens: int | None = None, seed: int | None = None,
                 timeout: float | None = None) -> GenerationResult: ...


class ReplayGenerator:
    """Fixture generator: returns a saved raw output recorded for the exact prompt hash.

    It does not model anything. A prompt whose hash has no recording is unavailable, so a changed
    prompt can never be answered by an old output.
    """

    kind = "replay"

    def __init__(self, recordings: Mapping[str, Mapping[str, Any]]):
        self.recordings = dict(recordings)

    @classmethod
    def from_directory(cls, directory: Path) -> ReplayGenerator:
        recordings = {}
        for path in sorted(Path(directory).glob("*.json")):
            recording = json.loads(path.read_text(encoding="utf-8"))
            recordings[recording["prompt_sha256"]] = recording
        return cls(recordings)

    def complete(self, messages: list[dict[str, str]], *, task: str = "rca",
                 models: tuple[str, ...] | None = None, failover: bool = True,
                 **settings: Any) -> GenerationResult:
        recording = self.recordings.get(prompt_sha256(messages))
        if recording is None:
            raise GenerationUnavailable("no recorded output exists for this prompt")
        if models and recording["model"] not in models:
            raise GenerationUnavailable("the recorded output came from a different model")
        return GenerationResult(
            content=recording["content"], requested_model=recording["model"],
            served_model=recording["model"],
            attempts=[GenerationAttempt(model=recording["model"], outcome="ok", seconds=0.0)])


Evidence = KpiStatisticEvidence | TextEvidence | SpecificationEvidence


class RcaPipelineResponse(BaseModel):
    """What callers receive: checked result or explicit failure, with review still pending."""

    model_config = ConfigDict(extra="forbid")

    detector_prediction: DetectorPrediction | None = None
    measured_evidence: list[KpiStatisticEvidence | TextEvidence] = Field(default_factory=list)
    result: RcaModelOutput | None = None
    explanation: str | None = None
    resolved_evidence: dict[str, Evidence] = Field(default_factory=dict)
    structural_checks: list[CheckResult] = Field(default_factory=list)
    numeric_checks: list[NumericCheckResult] = Field(default_factory=list)
    semantic_review_status: Literal["pending"] = "pending"
    human_review_status: Literal["pending"] = "pending"
    factual_support: Literal["unverified"] = "unverified"
    golden_eligible: Literal[False] = False
    training_target_eligible: Literal[False] = False
    execution_status: Literal["complete", "incomplete", "failed"]
    execution_stage: Stage | None = None
    execution_reason: str | None = None

    def to_server_response(self) -> RcaServerResponse:
        """The approved ``rca-v1`` server response; only a completed run has one."""
        if self.result is None or self.execution_status != "complete":
            raise ValueError("only a completed run has an rca-v1 server response")
        return RcaServerResponse(
            detector_prediction=self.detector_prediction, result=self.result,
            resolved_evidence=self.resolved_evidence,
            structural_checks=[f"{check.check_id}:{check.status}" for check in self.structural_checks],
            execution_status=self.execution_status)


class RcaRunRecord(BaseModel):
    """Everything needed to reproduce, evaluate and review one run. Never holds credentials."""

    model_config = ConfigDict(extra="forbid")

    version: Literal["rca-run-v1"] = RUN_RECORD_VERSION
    run_id: str = ""
    result_sha256: str = ""
    created_at: str
    case_id: str
    config: ExperimentConfig
    config_sha256: str
    contract_version: str
    checks_version: str = CHECKS_VERSION
    prompt_version: str
    prompt_sha256: str | None = None
    taxonomy_id: str
    source_packet_sha256: str
    audits_sha256: str
    retrieval_manifest_sha256: str | None = None
    snapshot_id: str | None = None
    snapshot_state: str | None = None
    projected_packet_sha256: str | None = None
    projected_packet: EvidencePacket | None = None
    messages: list[dict[str, str]] | None = None
    context_budget: BudgetResult | None = None
    generator_called: bool = False
    generator_kind: Literal["live", "replay"] | None = None
    requested_model: str | None = None
    served_model: str | None = None
    attempts: list[GenerationAttempt] = Field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw_output: str | None = None
    parsed_output: RcaModelOutput | None = None
    response: RcaPipelineResponse
    timings: dict[str, float | None]
    resources: dict[str, Any]


def save_run_record(record: RcaRunRecord, directory: Path) -> Path:
    """Run records are content-addressed by ``run_id``; an earlier record is never overwritten."""
    path = Path(directory) / f"{record.run_id}.json"
    content = json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"refusing to overwrite run record {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def render_explanation(output: RcaModelOutput) -> str:
    """Deterministic text from checked claims only; each line carries its evidence IDs."""
    if output.status == "abstained":
        return NOT_FOUND
    order = {"observation": 0, "mechanism": 1, "diagnostic_inference": 2}
    position = {claim.claim_id: index for index, claim in enumerate(output.claims)}
    claims = {claim.claim_id: claim for claim in output.claims}
    lines = []
    for cause in output.final_fault_categories:
        lines.append("Supported result: normal operation" if cause.category == "normal"
                     else f"Candidate cause: {cause.category}")
        linked = sorted((claims[cid] for cid in cause.claim_ids),
                        key=lambda claim: (order[claim.claim_type], position[claim.claim_id]))
        lines += [f"- {claim.text} [{', '.join(claim.evidence_ids)}]" for claim in linked]
    return "\n".join(lines)


def _abstention(reason: str) -> RcaModelOutput:
    return RcaModelOutput(status="abstained", final_fault_categories=[], claims=[],
                          uncertainty_reasons=[reason], abstention_reason=reason)


def _peak_rss() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss": usage, "peak_rss_unit": "bytes" if sys.platform == "darwin" else "kilobytes",
            "scope": "whole process, not this run alone"}


class RcaPipeline:
    """Single core pipeline: FastAPI and MCP call this instead of duplicating RCA logic."""

    def __init__(self, *, generator: Generator | None = None, search_service: Any | None = None,
                 query_service: Any | None = None, clock: Callable[[], float] = time.perf_counter,
                 now: Callable[[], str] = lambda: datetime.now(UTC).isoformat()):
        self.generator = generator
        self.search_service = search_service
        self.query_service = query_service
        self.clock = clock
        self.now = now

    def search_specs(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Specification search through the shared ``SearchService``."""
        if self.search_service is None:
            from clauseguard.api.search_service import SearchService

            self.search_service = SearchService()
        return [hit.model_dump() for hit in self.search_service.search(query, top_k)]

    def grounded_query(self, question: str, top_k: int = 8) -> dict[str, Any]:
        """Specification questions stay on the existing ``QueryService`` path."""
        if self.query_service is None:
            from clauseguard.api.query_service import QueryService
            from clauseguard.api.search_service import SearchService
            from clauseguard.llm.clients import LLMRouter

            self.query_service = QueryService(self.search_service or SearchService(), LLMRouter())
        return self.query_service.query(question, top_k).model_dump()

    def run_rca(self, case: RcaCase, config: ExperimentConfig) -> RcaRunRecord:
        """Run one case; expected failures are recorded with their stage, never raised."""
        started = self.clock()
        record: dict[str, Any] = {
            "created_at": self.now(), "case_id": case.case_id, "config": config,
            "config_sha256": config.sha256(), "contract_version": case.packet.contract_version,
            "taxonomy_id": config.taxonomy.taxonomy_id, "prompt_version": config.prompt_version,
            "source_packet_sha256": _digest(case.packet.model_dump(mode="json")),
            "audits_sha256": _digest([audit.model_dump(mode="json") for audit in case.audits]),
            "retrieval_manifest_sha256": case.manifest.get("manifest_sha256")}
        timings: dict[str, float | None] = {}
        response: dict[str, Any] = {"execution_status": "complete"}
        try:
            self._execute(case, config, record, response, timings)
        except PipelineFailure as failure:
            response.update(execution_status=failure.status, execution_stage=failure.stage,
                            execution_reason=failure.reason, result=None, explanation=None,
                            resolved_evidence={})
        timings["total"] = self.clock() - started
        body = RcaRunRecord(**record, response=RcaPipelineResponse(**response), timings=timings,
                            resources=_peak_rss())
        # result_sha256 covers what must reproduce; run_id also covers time, timings and resources.
        result = _digest(body.model_dump(mode="json", exclude={"run_id", "result_sha256", "created_at",
                                                               "timings", "resources"}))
        body = body.model_copy(update={"result_sha256": result})
        return body.model_copy(update={"run_id": _digest(body.model_dump(mode="json", exclude={"run_id"}))})

    def _timed(self, timings: dict[str, float | None], name: str, started: float) -> None:
        timings[name] = self.clock() - started

    def _execute(self, case: RcaCase, config: ExperimentConfig, record: dict[str, Any],
                 response: dict[str, Any], timings: dict[str, float | None]) -> None:
        started = self.clock()
        replayed, snapshot, replay_timing = assemble(case)
        self._timed(timings, "assembly", started)
        timings["original_retrieval"] = replay_timing.get("original_retrieval_seconds")
        record.update(snapshot_id=snapshot.snapshot_id, snapshot_state=snapshot.state)
        response.update(detector_prediction=case.packet.detector_prediction,
                        measured_evidence=[*case.packet.kpi_evidence, *case.packet.text_evidence])
        if snapshot.state == "failed":
            raise PipelineFailure("retrieval", f"retrieval failed at {snapshot.stage}: {snapshot.reason}")
        if snapshot.state == "not_run":
            raise PipelineFailure("retrieval", f"retrieval did not run ({snapshot.stage}: {snapshot.reason})",
                                  "incomplete")
        if snapshot.state == "empty":
            # Completed search with no passages: abstain without calling the generator.
            response.update(result=_abstention("no_spec_evidence"), explanation=NOT_FOUND)
            return

        started = self.clock()
        projected = project_packet(replayed, config.configuration, config.minimal_question)
        check_projection(replayed, projected, config.configuration)
        record.update(projected_packet=projected,
                      projected_packet_sha256=_digest(projected.model_dump(mode="json")))
        self._timed(timings, "projection", started)
        if not projected.kpi_evidence and not projected.text_evidence:
            response.update(result=_abstention("insufficient_incident_evidence"), explanation=NOT_FOUND)
            return

        started = self.clock()
        messages = build_rca_messages(projected, config.taxonomy.categories(), config.prompt_version)
        budget = check_context_budget(messages, config.context_budget_chars)
        record.update(messages=messages, prompt_sha256=prompt_sha256(messages), context_budget=budget)
        self._timed(timings, "prompt", started)
        if not budget.within:
            raise PipelineFailure("context_budget", f"prompt has {budget.measured} characters; the limit is "
                                                    f"{budget.limit}; no evidence was dropped")

        generated = self._generate(messages, config, record, timings)

        started = self.clock()
        try:
            output = parse_model_output(generated.content)
        except OutputParseError as exc:
            response["structural_checks"] = [CheckResult(check_id="schema.v1", status="failed", detail=str(exc))]
            self._timed(timings, "validation", started)
            raise PipelineFailure("validation", f"output does not match the RCA schema: {exc}") from exc
        record["parsed_output"] = output
        structural = [CheckResult(check_id="schema.v1", status="passed"),
                      *structural_checks(output, projected, config.taxonomy)]
        numeric = numeric_checks(output, projected)
        response.update(structural_checks=structural, numeric_checks=numeric)
        self._timed(timings, "validation", started)
        failed = sorted({check.check_id for check in structural if check.status == "failed"})
        if config.numeric_policy == "fail" and any(item.status == "failed" for item in numeric):
            failed.append("numeric.v2")
        if failed:
            raise PipelineFailure("validation", f"runtime checks failed: {failed}")

        started = self.clock()
        evidence = {item.evidence_id: item for item in
                    (*projected.kpi_evidence, *projected.text_evidence, *projected.specification_evidence)}
        response.update(result=output, explanation=render_explanation(output),
                        resolved_evidence={eid: evidence[eid] for claim in output.claims
                                           for eid in claim.evidence_ids})
        self._timed(timings, "render", started)

    def _generate(self, messages: list[dict[str, str]], config: ExperimentConfig, record: dict[str, Any],
                  timings: dict[str, float | None]) -> GenerationResult:
        if self.generator is None:
            raise PipelineFailure("generation", "no generator is configured")
        scored = config.run_kind == "scored"
        models = (config.model_id,) if scored else (config.model_id, *config.fallback_models)
        record.update(generator_called=True, generator_kind=getattr(self.generator, "kind", "live"))
        started = self.clock()
        try:
            generated = self.generator.complete(
                messages, task="rca", models=models, failover=not scored and len(models) > 1,
                **config.generation.model_dump(exclude={"timeout_seconds"}),
                timeout=config.generation.timeout_seconds)
        except (GenerationUnavailable, MalformedGenerationOutput) as exc:
            record["attempts"] = list(exc.attempts)
            self._timed(timings, "generation", started)
            raise PipelineFailure("generation", f"{type(exc).__name__}: {exc}") from exc
        self._timed(timings, "generation", started)
        record.update(requested_model=generated.requested_model, served_model=generated.served_model,
                      attempts=generated.attempts, usage=generated.usage, raw_output=generated.content)
        if scored and (generated.requested_model != config.model_id
                       or generated.served_model not in (None, config.model_id)):
            raise PipelineFailure("generation", "scored run was served by a model other than the fixed model")
        return generated
