"""Task 4.7: run_rca CLI over FIXTURE packets, snapshots and recorded raw outputs.

``tests/fixtures/rca_pipeline`` is FIXTURE data: a toy two-document corpus, hand-made rankings and
hand-written raw outputs recorded against the exact prompt hash. It is not a model response. A
prompt change makes the recordings unavailable, so these tests also pin the prompt.
"""

import json
from pathlib import Path

import pytest

from clauseguard.cli import run_rca
from clauseguard.rca.pipeline import RcaRunRecord

FIXTURES = Path(__file__).parent / "fixtures" / "rca_pipeline"


def _args(output, *extra):
    return ["--manifest", str(FIXTURES / "retrieval-manifest.json"), "--packets", str(FIXTURES / "packets.jsonl"),
            "--audits", str(FIXTURES / "audits.jsonl"), "--run-dir", str(FIXTURES / "snapshots"),
            "--output", str(output), *extra]


def _runs(output):
    index = json.loads((output / "index.json").read_text())
    return {row["case_id"]: row for row in index["runs"]}, index


@pytest.fixture(autouse=True)
def _no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.chdir(tmp_path)


def test_replayed_fixture_runs_answer_abstain_and_fail_separately(tmp_path, capsys):
    output = tmp_path / "runs"
    assert run_rca.main(_args(output, "--replay", str(FIXTURES / "recordings"))) == 0
    runs, index = _runs(output)
    assert index["generator"] == "replay" and index["config"]["run_kind"] == "exploratory"
    answered = runs["rca_fx_001"]
    assert (answered["execution_status"], answered["result_status"]) == ("complete", "answered")
    assert answered["categories"] == ["Co-Channel Interference (Severe)"] and answered["detector_label"] == "normal"
    assert answered["human_review_status"] == "pending" and answered["factual_support"] == "unverified"
    empty = runs["rca_fx_002"]
    assert (empty["result_status"], empty["abstention_reason"], empty["generator_called"]) == (
        "abstained", "no_spec_evidence", False)
    failed = runs["rca_fx_003"]
    assert (failed["execution_status"], failed["execution_stage"], failed["result_status"]) == (
        "failed", "retrieval", None)
    assert "configuration" in failed["execution_reason"]
    model_abstains = runs["rca_fx_004"]
    assert (model_abstains["result_status"], model_abstains["abstention_reason"]) == (
        "abstained", "conflicting_evidence")
    record = RcaRunRecord.model_validate_json((output / "runs" / f"{answered['run_id']}.json").read_text())
    assert record.generator_kind == "replay" and record.response.explanation.startswith("Candidate cause:")
    assert all(check.status == "passed" for check in record.response.structural_checks)
    assert json.loads(capsys.readouterr().out)["result_status"] == {"answered": 1, "abstained": 2}


def test_replay_is_reproducible_and_output_is_never_overwritten(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    assert run_rca.main(_args(first, "--replay", str(FIXTURES / "recordings"))) == 0
    assert run_rca.main(_args(second, "--replay", str(FIXTURES / "recordings"))) == 0
    assert ({k: r["result_sha256"] for k, r in _runs(first)[0].items()}
            == {k: r["result_sha256"] for k, r in _runs(second)[0].items()})
    assert run_rca.main(_args(first, "--replay", str(FIXTURES / "recordings"))) == 1


def test_live_mode_without_a_key_is_a_generation_failure_not_an_abstention(tmp_path):
    output = tmp_path / "live"
    assert run_rca.main(_args(output, "--run-kind", "scored", "--case", "rca_fx_001", "--case", "rca_fx_002")) == 0
    runs, index = _runs(output)
    assert index["generator"] == "live" and index["config"]["fallback_models"] == []
    assert (runs["rca_fx_001"]["execution_status"], runs["rca_fx_001"]["execution_stage"]) == ("failed", "generation")
    assert runs["rca_fx_001"]["result_status"] is None
    assert "not configured" in runs["rca_fx_001"]["execution_reason"]
    assert runs["rca_fx_002"]["result_status"] == "abstained"


def test_other_configurations_change_the_prompt_so_recordings_do_not_apply(tmp_path):
    output = tmp_path / "text"
    assert run_rca.main(_args(output, "--configuration", "text_rag", "--case", "rca_fx_001",
                              "--replay", str(FIXTURES / "recordings"))) == 0
    row = _runs(output)[0]["rca_fx_001"]
    assert row["execution_stage"] == "generation" and "no recorded output" in row["execution_reason"]


def test_max_tokens_reaches_the_recorded_configuration(tmp_path):
    output = tmp_path / "tokens"
    assert run_rca.main(_args(output, "--max-tokens", "16384", "--case", "rca_fx_002")) == 0
    assert _runs(output)[1]["config"]["generation"]["max_tokens"] == 16384


def test_prompt_version_is_recorded_and_v1_recordings_do_not_apply_to_v2(tmp_path):
    output = tmp_path / "v2"
    assert run_rca.main(_args(output, "--case", "rca_fx_001", "--replay", str(FIXTURES / "recordings"))) == 0
    runs, index = _runs(output)
    assert index["config"]["prompt_version"] == "rca-prompt-v2" and runs["rca_fx_001"]["result_status"] == "answered"
    record = RcaRunRecord.model_validate_json((output / "runs" / f"{runs['rca_fx_001']['run_id']}.json").read_text())
    assert record.prompt_version == "rca-prompt-v2"
    older = tmp_path / "v1"
    assert run_rca.main(_args(older, "--prompt-version", "rca-prompt-v1", "--case", "rca_fx_001",
                              "--replay", str(FIXTURES / "recordings"))) == 0
    row = _runs(older)[0]["rca_fx_001"]
    assert row["execution_stage"] == "generation" and "no recorded output" in row["execution_reason"]


def test_rca_timeout_and_token_defaults_are_rca_specific(tmp_path, monkeypatch):
    from clauseguard.config import Settings

    settings = Settings(_env_file=None)
    assert (settings.rca_timeout_seconds, settings.rca_max_tokens) == (240.0, 16384)
    assert settings.nvidia_timeout_seconds == 30.0  # grounded_query is unchanged
    output = tmp_path / "defaults"
    assert run_rca.main(_args(output, "--case", "rca_fx_002")) == 0
    generation = _runs(output)[1]["config"]["generation"]
    assert (generation["max_tokens"], generation["timeout_seconds"]) == (16384, 240.0)
    monkeypatch.setenv("RCA_TIMEOUT_SECONDS", "90")
    other = tmp_path / "override"
    assert run_rca.main(_args(other, "--case", "rca_fx_002", "--max-tokens", "8000")) == 0
    generation = _runs(other)[1]["config"]["generation"]
    assert (generation["max_tokens"], generation["timeout_seconds"]) == (8000, 90.0)


def test_top_p_is_recorded_in_the_run_config(tmp_path, capsys):
    output = tmp_path / "runs"
    assert run_rca.main(_args(output, "--replay", str(FIXTURES / "recordings"), "--top-p", "0.95")) == 0
    _, index = _runs(output)
    assert index["config"]["generation"]["top_p"] == 0.95


def test_taxonomy_option_selects_a_source_taxonomy(tmp_path, capsys):
    output = tmp_path / "runs"
    assert run_rca.main(_args(output, "--replay", str(FIXTURES / "recordings"), "--taxonomy", "bubbleran-pi-v1")) == 0
    _, index = _runs(output)
    assert index["config"]["taxonomy"]["taxonomy_id"] == "bubbleran-pi-v1"
    assert index["config"]["taxonomy"]["labels"] == ["persistent_interference"]
