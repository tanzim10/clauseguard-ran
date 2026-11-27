"""Task 4.7: shared RCA core — assembly, projection, budget, generation, checks and run records."""

import json
import uuid
from types import SimpleNamespace

import pytest

from clauseguard.api.search_service import SearchService
from clauseguard.config import Settings
from clauseguard.llm.clients import (
    GenerationAttempt,
    GenerationResult,
    GenerationUnavailable,
    MalformedGenerationOutput,
)
from clauseguard.rca import pipeline, retrieval
from clauseguard.rca.contracts import DetectorPrediction, SourceIdentity
from clauseguard.rca.evidence import EvidenceRegistry
from clauseguard.rca.summarizer import summarize_channel

SOURCE = SourceIdentity(dataset="TelecomTS", revision="r", case_id="telecomts-1", artifact_sha256="a" * 64,
                        recording_family_status="unresolved")
TEXTS = {"TS 38.300": "1 Scope\nThe scheduler assigns resources.\n2 Interference\nUplink SNR falls under interference.",
         "TS 28.552": "1 Scope\nPerformance measurements are defined here."}
PREDICTION = DetectorPrediction(task="diagnosis", status="ok", predicted_label_id=0, predicted_label="normal",
                                label_mapping_version="1", model_run_id="pipeline:run", source_modalities=["kpi"])
MINIMAL = "Explain the observed KPI changes using the supplied specification evidence."


def _registry(text=True):
    registry = EvidenceRegistry()
    for channel, values in (("UL_SNR", [21.4, 6.4]), ("RSRP", [-87.8, -98.1])):
        for stat in summarize_channel(channel, values, semantics="reported_value")[0]:
            registry.add_kpi(stat, source=SOURCE, location={"row": 1})  # UL_SNR K1-K3, RSRP K4-K6
    if text:
        registry.add_text(record_id="log-1", text="Operator log: uplink calls drop near the stadium.",
                          origin="independent_observation", source="log", permission="approved")
        registry.add_text(record_id="desc-1", text="The UL_SNR series drops in the second half.",
                          origin="kpi_derived", source="description", permission="approved")
    return registry


def _packet(registry=None, prediction=PREDICTION, question="Why did uplink calls drop?"):
    registry = registry or _registry()
    packet = registry.packet(source=SOURCE, question=question, kpi_state="included", text_state="included",
                             retrieval_status="not_run", detector_prediction=prediction)
    return packet, registry.audits()


def _point(chunk, score):
    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["id"])), "score": score,
            "payload": {"text": chunk["text"], **chunk["metadata"]}}


@pytest.fixture
def world(tmp_path):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    rows = ["filename,spec_id,version,sha256,acquired_at"]
    for spec, text in TEXTS.items():
        name = spec.replace(" ", "_")
        (parsed / f"{name}.txt").write_text(text, encoding="utf-8")
        rows.append(f"{name}.pdf,{spec},18.0.0,{'b' * 64},2026-01-01")
    csv = tmp_path / "manifest.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
    settings = Settings(_env_file=None, qdrant_collection="test", embedding_dimensions=4, nvidia_api_key="x")
    manifest = retrieval.freeze_manifest(settings=settings, manifest_path=csv, parsed_dir=parsed, top_k=3,
                                         output=tmp_path / "retrieval-manifest.json", frozen_on="2026-11-17")
    chunks = retrieval.chunk_inventory(csv, parsed)

    def case(packet_and_audits=None, points=None, case_id="rca_t1"):
        packet, audits = packet_and_audits or _packet()
        hits = [_point(chunks[2], 0.9), _point(chunks[1], 0.8)] if points is None else points
        service = SearchService(settings=settings, embedder=lambda texts, **kw: [[0.0] * 4 for _ in texts],
                                qdrant_search=lambda *a, **kw: hits)
        snapshot = retrieval.snapshot_case(packet, manifest, service, case_id=case_id)
        path = retrieval.save_snapshot(snapshot, tmp_path / "snapshots")
        return pipeline.RcaCase(case_id=case_id, packet=packet, audits=audits, snapshot_path=path,
                                manifest=manifest)

    return SimpleNamespace(case=case, manifest=manifest, chunks=chunks, path=tmp_path, settings=settings)


def test_assembly_verifies_audits_snapshot_and_replays_passages(world):
    case = world.case()
    replayed, snapshot, timing = pipeline.assemble(case)
    assert snapshot.state == "completed" and timing["mode"] == "replay"
    assert [s.evidence_id for s in replayed.specification_evidence] == ["S1", "S2"]

    bad_audit = case.audits[0].model_copy(update={"content_sha256": "0" * 64})
    with pytest.raises(pipeline.PipelineFailure) as error:
        pipeline.assemble(case.model_copy(update={"audits": [bad_audit, *case.audits[1:]]}))
    assert error.value.stage == "assembly" and error.value.status == "failed"

    altered = json.loads(case.snapshot_path.read_text())
    altered["passages"][0]["score"] = 0.1
    case.snapshot_path.write_text(json.dumps(altered))
    with pytest.raises(pipeline.PipelineFailure, match="altered") as error:
        pipeline.assemble(case)
    assert error.value.stage == "assembly"


def test_changed_packet_and_missing_snapshot_are_separate_failures(world):
    case = world.case()
    changed = case.packet.model_copy(update={"question": "Another question?"})
    with pytest.raises(pipeline.PipelineFailure, match="differs") as error:
        pipeline.assemble(case.model_copy(update={"packet": changed}))
    assert error.value.stage == "assembly"
    with pytest.raises(pipeline.PipelineFailure) as error:
        pipeline.assemble(case.model_copy(update={"snapshot_path": world.path / "missing.json"}))
    assert (error.value.stage, error.value.status) == ("retrieval", "incomplete")


def test_full_configurations_keep_the_packet_unchanged(world):
    packet, _, _ = pipeline.assemble(world.case())
    for name in pipeline.FULL_CONFIGURATIONS:
        assert pipeline.project_packet(packet, name) is packet
        pipeline.check_projection(packet, packet, name)


def test_text_rag_removes_kpi_records_kpi_derived_text_and_kpi_prediction(world):
    packet, _, _ = pipeline.assemble(world.case())
    projected = pipeline.project_packet(packet, "text_rag")
    assert projected.kpi_evidence == [] and projected.kpi_state == "omitted_by_configuration"
    assert [t.record_id for t in projected.text_evidence] == ["log-1"]
    assert projected.detector_prediction.status == "absent"
    assert "omitted_by_configuration" in projected.detector_prediction.absence_reason
    assert projected.specification_evidence == packet.specification_evidence
    pipeline.check_projection(packet, projected, "text_rag")
    text_only = DetectorPrediction(task="diagnosis", status="ok", predicted_label="normal",
                                   source_modalities=["text"])
    kept = pipeline.project_packet(packet.model_copy(update={"detector_prediction": text_only}), "text_rag")
    assert kept.detector_prediction == text_only


def test_kpi_summary_rag_keeps_kpi_and_only_the_minimal_question(world):
    packet, _, _ = pipeline.assemble(world.case())
    projected = pipeline.project_packet(packet, "kpi_summary_rag", MINIMAL)
    assert projected.question == MINIMAL and projected.text_evidence == []
    assert projected.text_state == "omitted_by_configuration" and projected.kpi_evidence == packet.kpi_evidence
    assert projected.detector_prediction == PREDICTION
    pipeline.check_projection(packet, projected, "kpi_summary_rag")
    with pytest.raises(pipeline.PipelineFailure, match="minimal question"):
        pipeline.project_packet(packet, "kpi_summary_rag")


def test_leak_check_catches_omitted_inputs_returning_through_text(world):
    packet, _, _ = pipeline.assemble(world.case())
    leaky = packet.model_copy(update={"question": "Why did UL_SNR drop?"})
    with pytest.raises(pipeline.PipelineFailure, match="UL_SNR") as error:
        pipeline.check_projection(leaky, pipeline.project_packet(leaky, "text_rag"), "text_rag")
    assert error.value.stage == "assembly"
    projected = pipeline.project_packet(packet, "kpi_summary_rag", MINIMAL)
    echoed = projected.model_copy(update={"question": MINIMAL + " " + packet.text_evidence[0].text})
    with pytest.raises(pipeline.PipelineFailure, match="T1"):
        pipeline.check_projection(packet, echoed, "kpi_summary_rag")
    unknown_source = DetectorPrediction(task="diagnosis", status="ok", predicted_label="normal")
    with pytest.raises(pipeline.PipelineFailure, match="prediction"):
        pipeline.check_projection(packet, pipeline.project_packet(packet, "text_rag").model_copy(
            update={"detector_prediction": unknown_source}), "text_rag")


def test_context_budget_measures_every_character_and_flags_overflow():
    messages = [{"role": "system", "content": "abc"}, {"role": "user", "content": "de"}]
    assert pipeline.check_context_budget(messages, 5).model_dump() == {
        "method": "characters", "limit": 5, "measured": 5, "within": True}
    assert pipeline.check_context_budget(messages, 4).within is False
    assert messages[1]["content"] == "de"


SEVERE = "Co-Channel Interference (Severe)"
MODEL = "model-a"


def _answer(**changes):
    output = {
        "status": "answered",
        "final_fault_categories": [{"category": SEVERE, "claim_ids": ["C1", "C2", "C3"]}],
        "claims": [
            {"claim_id": "C1", "text": "UL_SNR minimum is 6.4 and RSRP minimum is -98.1.",
             "claim_type": "observation", "evidence_ids": ["K2", "K5"]},
            {"claim_id": "C2", "text": "Uplink SNR falls under interference.", "claim_type": "mechanism",
             "evidence_ids": ["S1"]},
            {"claim_id": "C3", "text": "A UL_SNR minimum of 6.4 is consistent with uplink interference.",
             "claim_type": "diagnostic_inference", "evidence_ids": ["K2", "S1"]},
        ],
        "uncertainty_reasons": ["units unknown", "no baseline"],
        "abstention_reason": None,
    }
    return output | changes


class FakeGenerator:
    def __init__(self, *outcomes, served=None):
        self.outcomes, self.calls, self.served = list(outcomes), [], served

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        content = outcome if isinstance(outcome, str) else json.dumps(outcome)
        model = kwargs["models"][0]
        return GenerationResult(content=content, requested_model=model, served_model=self.served or model,
                                attempts=[GenerationAttempt(model=model, outcome="ok", seconds=0.0)])


def _config(**changes):
    return pipeline.ExperimentConfig(**({"configuration": "base_llm_full", "run_kind": "scored",
                                         "model_id": MODEL} | changes))


def _run(case, *outcomes, config=None, served=None):
    generator = FakeGenerator(*outcomes, served=served)
    clock = iter(range(1000))
    rca = pipeline.RcaPipeline(generator=generator, clock=lambda: float(next(clock)), now=lambda: "2026-11-17T00:00:00+00:00")
    return rca.run_rca(case, config or _config()), generator


def test_answered_run_returns_cited_causes_with_review_pending(world):
    record, generator = _run(world.case(), _answer())
    response = record.response
    assert (response.execution_status, response.result.status) == ("complete", "answered")
    assert response.human_review_status == "pending" and response.semantic_review_status == "pending"
    assert response.factual_support == "unverified" and response.golden_eligible is False
    assert response.detector_prediction == PREDICTION  # wrong detector label kept apart from the cause
    assert [c.category for c in response.result.final_fault_categories] == [SEVERE]
    assert response.explanation.splitlines() == [
        f"Candidate cause: {SEVERE}",
        "- UL_SNR minimum is 6.4 and RSRP minimum is -98.1. [K2, K5]",
        "- Uplink SNR falls under interference. [S1]",
        "- A UL_SNR minimum of 6.4 is consistent with uplink interference. [K2, S1]"]
    assert sorted(response.resolved_evidence) == ["K2", "K5", "S1"]
    assert [m.evidence_id for m in response.measured_evidence][:2] == ["K1", "K2"]
    assert {c.status for c in response.structural_checks} == {"passed"}
    assert response.structural_checks[0].check_id == "schema.v1"
    assert {c.status for c in response.numeric_checks} == {"passed"}
    server = response.to_server_response()
    assert server.human_review_status == "pending" and "schema.v1:passed" in server.structural_checks
    call = generator.calls[0]
    assert (call["models"], call["failover"], call["temperature"], call["max_tokens"]) == ((MODEL,), False, 0.0, 16384)
    assert call["timeout"] == 240.0 and "timeout_seconds" not in call
    assert record.messages == call["messages"] and record.prompt_sha256
    assert (record.requested_model, record.served_model, record.generator_kind) == (MODEL, MODEL, "live")
    assert record.raw_output == json.dumps(_answer()) and record.parsed_output == response.result
    assert record.snapshot_state == "completed" and record.context_budget.within
    assert set(record.timings) >= {"assembly", "projection", "prompt", "generation", "validation", "render", "total"}
    assert record.resources["peak_rss"] > 0 and len(record.run_id) == 64


def test_runs_are_reproducible(world):
    case = world.case()
    first, _ = _run(case, _answer())
    again, _ = _run(case, _answer())
    assert first.result_sha256 == again.result_sha256 and first.response == again.response


def test_completed_empty_retrieval_abstains_without_calling_the_generator(world):
    record, generator = _run(world.case(points=[]))
    response = record.response
    assert generator.calls == [] and record.generator_called is False
    assert (response.execution_status, response.execution_stage) == ("complete", None)
    assert response.result.status == "abstained" and response.result.abstention_reason == "no_spec_evidence"
    assert response.result.claims == [] and response.result.final_fault_categories == []
    assert response.explanation == "not found" and response.detector_prediction == PREDICTION
    assert response.measured_evidence and record.snapshot_state == "empty"


def test_failed_and_not_run_retrieval_are_separate_from_abstention(world, monkeypatch):
    case = world.case()
    snapshot = retrieval.load_snapshot(case.snapshot_path, world.manifest)
    for state, status in (("failed", "failed"), ("not_run", "incomplete")):
        changed = retrieval._sealed(snapshot.model_copy(update={
            "state": state, "stage": "configuration" if state == "failed" else "query",
            "reason": "NVIDIA_API_KEY is required for embeddings" if state == "failed" else "empty query",
            "passages": []}))
        path = retrieval.save_snapshot(changed, world.path / state)
        record, generator = _run(case.model_copy(update={"snapshot_path": path}))
        response = record.response
        assert (response.execution_status, response.execution_stage) == (status, "retrieval")
        assert response.result is None and response.explanation is None and generator.calls == []
        assert response.detector_prediction == PREDICTION and response.measured_evidence
    missing, _ = _run(case.model_copy(update={"snapshot_path": None}))
    assert (missing.response.execution_status, missing.response.execution_stage) == ("incomplete", "retrieval")


def test_assembly_failure_is_recorded_before_generation(world):
    case = world.case()
    record, generator = _run(case.model_copy(update={"audits": case.audits[1:]}))
    assert (record.response.execution_status, record.response.execution_stage) == ("failed", "assembly")
    assert generator.calls == [] and record.response.detector_prediction is None


@pytest.mark.parametrize("raw, detail", [
    ("not json", "not valid JSON"),
    ("```json\n" + json.dumps(_answer()) + "\n```", "not valid JSON"),
    (json.dumps(_answer(confidence=0.9)), "confidence"),
    (json.dumps(_answer(final_fault_categories=[])), "answered output needs causes"),
])
def test_malformed_output_fails_validation_and_never_becomes_abstention(world, raw, detail):
    record, _ = _run(world.case(), raw)
    response = record.response
    assert (response.execution_status, response.execution_stage) == ("failed", "validation")
    assert response.result is None and response.explanation is None
    assert record.raw_output == raw and record.parsed_output is None
    assert response.structural_checks[0].check_id == "schema.v1" and detail in response.structural_checks[0].detail


def test_failed_runtime_checks_keep_the_parsed_output_but_return_no_result(world):
    claims = _answer()["claims"]
    claims[2] = {**claims[2], "evidence_ids": ["K2", "S9"]}
    record, _ = _run(world.case(), _answer(claims=claims))
    response = record.response
    assert (response.execution_status, response.execution_stage) == ("failed", "validation")
    assert "evidence_ids_known.v1" in response.execution_reason
    assert record.parsed_output is not None and response.result is None and response.resolved_evidence == {}
    failed = [c for c in response.structural_checks if c.status == "failed"]
    assert {(c.check_id, c.subject) for c in failed} >= {("evidence_ids_known.v1", "C3")}


def test_numeric_mismatch_fails_scored_runs_and_is_recorded_in_exploratory_runs(world):
    claims = _answer()["claims"]
    claims[0] = {**claims[0], "text": "UL_SNR minimum is 5.9."}
    record, _ = _run(world.case(), _answer(claims=claims))
    assert record.response.execution_stage == "validation" and "numeric.v2" in record.response.execution_reason
    assert record.response.numeric_checks[0].status == "failed"
    relaxed = _config(run_kind="exploratory", numeric_policy="record")
    record, _ = _run(world.case(), _answer(claims=claims), config=relaxed)
    assert record.response.execution_status == "complete" and record.response.numeric_checks[0].status == "failed"


def test_conflicting_evidence_requires_abstention(world):
    claims = _answer()["claims"] + [
        {"claim_id": "C4", "text": "Antenna faults reduce received power.", "claim_type": "mechanism",
         "evidence_ids": ["S2"]},
        {"claim_id": "C5", "text": "The same low UL_SNR could come from an antenna fault.",
         "claim_type": "diagnostic_inference", "evidence_ids": ["K2", "S2"]}]
    competing = _answer(claims=claims, final_fault_categories=[
        {"category": SEVERE, "claim_ids": ["C1", "C2", "C3"]},
        {"category": "Antenna Failure", "claim_ids": ["C4", "C5"]}])
    record, _ = _run(world.case(), competing)
    assert record.response.execution_stage == "validation"
    assert "competing_causes_abstain.v1" in record.response.execution_reason
    abstain = {"status": "abstained", "final_fault_categories": [], "claims": [],
               "uncertainty_reasons": ["interference and antenna failure both fit K2"],
               "abstention_reason": "conflicting_evidence"}
    record, _ = _run(world.case(), abstain)
    assert (record.response.execution_status, record.response.result.abstention_reason) == (
        "complete", "conflicting_evidence")
    assert record.response.explanation == "not found" and record.response.resolved_evidence == {}


def test_context_overflow_fails_without_dropping_evidence(world):
    record, generator = _run(world.case(), config=_config(context_budget_chars=1000))
    response = record.response
    assert (response.execution_status, response.execution_stage) == ("failed", "context_budget")
    assert generator.calls == [] and not record.context_budget.within
    assert record.context_budget.measured > 1000
    assert len(record.projected_packet.kpi_evidence) == 6 and len(record.projected_packet.specification_evidence) == 2


def test_generation_failures_record_attempts_and_stage(world):
    attempts = (GenerationAttempt(model=MODEL, outcome="transient_error", error_class="APITimeoutError", seconds=1.0),)
    error = GenerationUnavailable("fixed generation model is unavailable")
    error.attempts = attempts
    record, _ = _run(world.case(), error)
    assert (record.response.execution_status, record.response.execution_stage) == ("failed", "generation")
    assert record.attempts == list(attempts) and record.raw_output is None
    record, _ = _run(world.case(), MalformedGenerationOutput())
    assert record.response.execution_stage == "generation"
    no_generator = pipeline.RcaPipeline().run_rca(world.case(), _config())
    assert no_generator.response.execution_stage == "generation"


def test_scored_runs_fix_the_model_and_reject_substitution(world):
    with pytest.raises(ValueError, match="one fixed model"):
        _config(fallback_models=("model-b",))
    record, _ = _run(world.case(), _answer(), served="model-b")
    assert record.response.execution_stage == "generation" and record.served_model == "model-b"
    exploratory = _config(run_kind="exploratory", fallback_models=("model-b",))
    record, generator = _run(world.case(), _answer(), config=exploratory)
    assert generator.calls[0]["models"] == (MODEL, "model-b") and generator.calls[0]["failover"] is True
    assert record.response.execution_status == "complete"


def test_missing_modalities_and_absent_predictions_stay_explicit(world):
    registry = _registry(text=False)
    absent = DetectorPrediction(task="diagnosis", status="absent", absence_reason="no detector for this source")
    packet = registry.packet(source=SOURCE, question="Why did uplink calls drop?", kpi_state="included",
                             text_state="missing", retrieval_status="not_run", detector_prediction=absent)
    case = world.case((packet, registry.audits()))
    record, generator = _run(case, config=_config(configuration="text_rag"))
    assert generator.calls == [] and record.response.result.abstention_reason == "insufficient_incident_evidence"
    assert record.projected_packet.text_state == "missing" and record.projected_packet.kpi_state == "omitted_by_configuration"
    record, generator = _run(case, _answer())
    user = generator.calls[0]["messages"][1]["content"]
    assert '"text_state":"missing"' in user and '"status":"absent"' in user
    assert record.response.detector_prediction == absent


def test_text_rag_and_kpi_summary_prompts_leave_out_omitted_modalities(world):
    _, generator = _run(world.case(), {"status": "abstained", "final_fault_categories": [], "claims": [],
                                       "uncertainty_reasons": [], "abstention_reason": "insufficient_mechanism"},
                        config=_config(configuration="text_rag"))
    user = generator.calls[0]["messages"][1]["content"]
    assert "UL_SNR" not in user and '"kpi_evidence":[]' in user and "stadium" in user
    _, generator = _run(world.case(), {"status": "abstained", "final_fault_categories": [], "claims": [],
                                       "uncertainty_reasons": [], "abstention_reason": "insufficient_mechanism"},
                        config=_config(configuration="kpi_summary_rag", minimal_question=MINIMAL))
    user = generator.calls[0]["messages"][1]["content"]
    assert "stadium" not in user and MINIMAL in user and "UL_SNR" in user


def test_replay_generator_answers_only_the_recorded_prompt(world):
    case = world.case()
    record, _ = _run(case, _answer())
    replay_gen = pipeline.ReplayGenerator({record.prompt_sha256: {"model": MODEL, "content": record.raw_output}})
    replayed = pipeline.RcaPipeline(generator=replay_gen).run_rca(case, _config())
    assert replayed.generator_kind == "replay" and replayed.response.result == record.response.result
    other = pipeline.RcaPipeline(generator=replay_gen).run_rca(case, _config(configuration="text_rag"))
    assert other.response.execution_stage == "generation" and "no recorded output" in other.response.execution_reason
    wrong_model = pipeline.RcaPipeline(generator=replay_gen).run_rca(case, _config(model_id="model-z"))
    assert "different model" in wrong_model.response.execution_reason


def test_core_does_not_import_fastapi_or_mcp():
    import subprocess
    import sys

    code = ("import sys, clauseguard.rca.pipeline; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('fastapi', 'mcp', 'starlette')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_run_records_are_saved_once_and_hold_no_credentials(world):
    secret = "nvapi-SECRET-test-key"
    record, _ = _run(world.case(), _answer())
    path = pipeline.save_run_record(record, world.path / "runs")
    assert path.name == f"{record.run_id}.json"
    assert pipeline.save_run_record(record, world.path / "runs") == path
    loaded = pipeline.RcaRunRecord.model_validate_json(path.read_text())
    assert loaded == record
    path.write_text(path.read_text().replace("pending", "approved", 1))
    with pytest.raises(FileExistsError):
        pipeline.save_run_record(record, world.path / "runs")

    from clauseguard.llm.clients import LLMRouter

    completions = SimpleNamespace(create=lambda **kw: SimpleNamespace(
        model=MODEL, usage=None,
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(_answer())))]))
    router = LLMRouter(Settings(_env_file=None, nvidia_api_key=secret),
                       client_factory=lambda settings: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    live = pipeline.RcaPipeline(generator=router).run_rca(world.case(), _config())
    assert live.response.execution_status == "complete" and live.generator_kind == "live"
    saved = pipeline.save_run_record(live, world.path / "live").read_text()
    assert secret not in saved and "Authorization" not in saved and "api_key" not in saved


def test_search_and_spec_questions_delegate_to_the_shared_services():
    from clauseguard.api.query_service import QueryService
    from clauseguard.api.schemas import SearchHit

    hit = SearchHit(id="p", score=0.5, text="Uplink SNR falls under interference.", spec_id="TS 38.300",
                    section="2 Interference", source_file="TS_38.300.pdf", chunk_id="c1")
    searches = []
    search = SimpleNamespace(search=lambda query, top_k: searches.append((query, top_k)) or [hit])
    router = SimpleNamespace(generate=lambda prompt, task: json.dumps({"answer": "SNR falls.", "evidence_ids": ["E1"]}))
    rca = pipeline.RcaPipeline(search_service=search, query_service=QueryService(search, router))
    assert rca.search_specs("snr", top_k=3) == [hit.model_dump()]
    answer = rca.grounded_query("What lowers SNR?", top_k=2)
    assert answer["answer"] == "SNR falls." and answer["citations"][0]["chunk_id"] == "c1"
    assert searches == [("snr", 3), ("What lowers SNR?", 2)]
