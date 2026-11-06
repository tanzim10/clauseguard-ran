import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIStatusError

from clauseguard.cli import validate_rca
from clauseguard.config import Settings
from clauseguard.datasets.schema import RcaRow
from clauseguard.eval import rca_validation
from clauseguard.eval.annotations import Annotation
from clauseguard.eval.rca_validation import ReviewSettings, build_prompt, review_row, review_rows

ROW = {
    "id": "rca_001",
    "scenario_id": "scn_001",
    "origin": "synthetic",
    "inputs": "kpi_only",
    "question": "Uplink SNR fell from about 21 to about 6. What is the most likely root cause?",
    "kpi_evidence": [
        {
            "channel": "UL_SNR",
            "baseline": {"start": "0", "end": "66"},
            "comparison": {"start": "67", "end": "127"},
            "baseline_value": 21.4,
            "comparison_value": 6.4,
            "direction": "decrease",
        }
    ],
    "source_labels": {"rca_label": "Co-Channel Interference (Severe)"},
    "expected_conclusion": "Uplink radio-quality degradation.",
    "uncertainty": "Synthetic injection.",
    "expected_citations": [
        {
            "spec_id": "3GPP TS 32.425",
            "section": "A.9 Monitoring of RF performance",
            "role": "mechanism",
            "explanation": "Uplink interference degrades the receiver.",
        }
    ],
}
ANNOTATION = {
    "id": "rca_001",
    "membership": "dev",
    "families": ["direct"],
    "evidence": [
        {
            "claim": "Uplink interference degrades the receiver.",
            "alternatives": [
                {
                    "spec_id": "3GPP TS 32.425",
                    "version": "18.0.0",
                    "sha256": "b" * 64,
                    "clause": "A.9",
                    "pdf_page": 105,
                }
            ],
        }
    ],
}
VERDICT = {
    "question_leaks_answer": False,
    "evidence_supports_conclusion": "yes",
    "citations": [
        {
            "spec_id": "3GPP TS 32.425",
            "section": "A.9 Monitoring of RF performance",
            "passage_supports_explanation": "yes",
            "role_correct": True,
        }
    ],
    "has_adequate_mechanism_citation": True,
    "acceptable_conclusions_reasonable": True,
    "uncertainty_honest_and_complete": True,
    "verdict": "valid",
    "confidence": 0.9,
}


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client(*outcomes):
    completions = FakeCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def _reply(content):
    return SimpleNamespace(
        model="served-model", usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _status_error(status, retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after else {}
    response = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://x.test"))
    return APIStatusError("provider detail", response=response, body=None)


def _passage(locator):
    return f"page {locator.pdf_page} text: uplink interference disturbs the base station receiver"


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(rca_validation.time, "sleep", lambda seconds: None)


ROW_MODEL = RcaRow.model_validate(ROW)
ANNOTATION_MODEL = Annotation.model_validate(ANNOTATION)


def test_prompt_carries_evidence_and_cited_page_text():
    prompt = build_prompt(ROW_MODEL, ANNOTATION_MODEL, _passage, "context")
    case = json.loads(prompt.split("\n\n")[0].removeprefix("CASE:\n"))
    assert case["kpi_evidence"][0]["comparison_value"] == 6.4
    assert case["citations"][0]["pdf_page"] == 105
    assert "disturbs the base station receiver" in case["citations"][0]["page_text"]
    assert case["dataset_label_claim"] == ROW["source_labels"]


def test_valid_reply_is_recorded_with_model_and_hashes():
    client, completions = _client(_reply(json.dumps(VERDICT)))
    record = review_row(client, ROW_MODEL, ANNOTATION_MODEL, _passage, ReviewSettings(model="m-1"))
    assert record["status"] == "reviewed"
    assert record["review"]["verdict"] == "valid"
    assert record["model"] == "m-1" and completions.calls[0]["model"] == "m-1"
    assert completions.calls[0]["temperature"] == 0
    assert len(record["prompt_sha256"]) == 64 and len(record["row_sha256"]) == 64


def test_malformed_reply_is_kept_as_failed_review():
    client, _ = _client(_reply('{"verdict": "great"}'))
    record = review_row(client, ROW_MODEL, ANNOTATION_MODEL, _passage, ReviewSettings())
    assert record["status"] == "failed"
    assert record["raw_content"] == '{"verdict": "great"}'


def test_rate_limit_is_retried_then_succeeds():
    client, completions = _client(_status_error(429, "3"), _reply(json.dumps(VERDICT)))
    record = review_row(client, ROW_MODEL, ANNOTATION_MODEL, _passage, ReviewSettings())
    assert record["status"] == "reviewed" and len(completions.calls) == 2


def test_long_rate_limit_wait_fails_without_retrying():
    client, completions = _client(_status_error(429, "2000"), _reply(json.dumps(VERDICT)))
    record = review_row(client, ROW_MODEL, ANNOTATION_MODEL, _passage, ReviewSettings(max_wait_seconds=60))
    assert record == {**record, "status": "failed", "error": "HTTP 429"}
    assert len(completions.calls) == 1


def test_authentication_error_is_not_retried():
    client, completions = _client(_status_error(401), _reply(json.dumps(VERDICT)))
    record = review_row(client, ROW_MODEL, ANNOTATION_MODEL, _passage, ReviewSettings())
    assert record["error"] == "HTTP 401" and len(completions.calls) == 1


def test_review_rows_writes_records_summary_and_resumes(tmp_path):
    client, completions = _client(_reply(json.dumps(VERDICT)))
    rows, annotations = [ROW_MODEL], {"rca_001": ANNOTATION_MODEL}
    summary = review_rows(client, rows, annotations, _passage, tmp_path, ReviewSettings(), log=lambda m: None)
    assert summary["verdicts"] == {"valid": 1} and summary["failed"] == []
    assert json.loads((tmp_path / "rca_001.json").read_text())["review"]["verdict"] == "valid"
    assert json.loads((tmp_path / "summary.json").read_text())["rows"] == 1

    review_rows(client, rows, annotations, _passage, tmp_path, ReviewSettings(), log=lambda m: None)
    assert len(completions.calls) == 1  # unchanged row and model: kept, not re-sent

    client2, completions2 = _client(_reply(json.dumps(VERDICT)))
    review_rows(client2, rows, annotations, _passage, tmp_path, ReviewSettings(model="other"), log=lambda m: None)
    assert len(completions2.calls) == 1  # a different model reviews again


@pytest.mark.parametrize(("provider", "env_var"), [("nvidia", "NVIDIA_API_KEY"), ("gemini", "GEMINI_API_KEY")])
def test_cli_reports_missing_api_key(monkeypatch, capsys, provider, env_var):
    monkeypatch.setattr(validate_rca, "get_settings", lambda: Settings(_env_file=None))
    for name in ("NVIDIA_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert validate_rca.main(["--provider", provider, "--list-models"]) == 1
    assert env_var in capsys.readouterr().err


def test_cli_builds_provider_client_with_its_endpoint():
    settings = Settings(_env_file=None, gemini_api_key="g-key", nvidia_api_key="n-key")
    gemini = validate_rca.build_client("gemini", settings)
    nvidia = validate_rca.build_client("nvidia", settings)
    assert str(gemini.base_url).startswith("https://generativelanguage.googleapis.com/")
    assert str(nvidia.base_url).startswith(settings.nvidia_base_url)
    assert validate_rca.build_client("groq", settings) is None
    openrouter = validate_rca.build_client("openrouter", Settings(_env_file=None, openrouter_api_key="o"))
    assert str(openrouter.base_url).startswith("https://openrouter.ai/api/v1")


def test_records_name_provider_and_resume_is_per_provider(tmp_path):
    client, completions = _client(_reply(json.dumps(VERDICT)), _reply(json.dumps(VERDICT)))
    rows, annotations = [ROW_MODEL], {"rca_001": ANNOTATION_MODEL}
    review_rows(client, rows, annotations, _passage, tmp_path, ReviewSettings(provider="gemini"), log=lambda m: None)
    assert json.loads((tmp_path / "rca_001.json").read_text())["provider"] == "gemini"
    review_rows(client, rows, annotations, _passage, tmp_path, ReviewSettings(provider="groq"), log=lambda m: None)
    assert len(completions.calls) == 2
