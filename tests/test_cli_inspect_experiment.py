import json

from clauseguard.baselines.data import ExperimentCohort
from clauseguard.cli.inspect_experiment import main
from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.release import prepare_release


def test_inspection_saves_reusable_cohorts_and_refuses_overwrite(release_sources, recording_evidence):
    s = release_sources
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    output = s["output"].parent / "inspection"
    args = ["--release-dir", str(s["output"]), "--output-dir", str(output)]
    assert main(args) == 0
    report = json.loads((output / "readiness.json").read_text())
    assert report["release_validated"] and report["ready"]
    assert len(report["cohorts"]) == 6
    name = "detection.kpi.train"
    cohort = ExperimentCohort.model_validate_json((output / f"{name}.json").read_text())
    assert cohort.cohort_id == report["cohorts"][name]["cohort_id"]
    assert len(cohort.scenario_ids) == 13
    assert len(report["combined_diagnosis_ids"]["kpi.train"]) == 12
    before = (output / "readiness.json").read_bytes()
    assert main(args) == 1
    assert (output / "readiness.json").read_bytes() == before


def test_blocked_scored_and_empty_text_views_keep_diagnostics(release_sources):
    s = release_sources
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    prepare_release(s["scenarios"], s["output"], s["output"], trials=128)
    output = s["output"].parent / "inspection"
    assert main(["--release-dir", str(s["output"]), "--output-dir", str(output),
                 "--modality", "text"]) == 3
    report = json.loads((output / "readiness.json").read_text())
    assert report["release_validated"] and not report["ready"]
    assert report["quality"]["release_exclusions"] == {"recording_family_unresolved": 39}
    assert all(not item["ready"] for item in report["cohorts"].values())


def test_exploratory_coverage_does_not_claim_scored_readiness(release_sources):
    s = release_sources
    build_recording_evidence(s["scenarios"], s["acquired"], s["output"])
    prepare_release(s["scenarios"], s["output"], s["output"], mode="exploratory", trials=128)
    output = s["output"].parent / "inspection"
    assert main(["--release-dir", str(s["output"]), "--output-dir", str(output)]) == 3
    report = json.loads((output / "readiness.json").read_text())
    assert report["mode"] == "exploratory" and not report["ready"]
    assert report["quality"]["tasks"]["detection"]["kpi"]["coverage_ready"]


def test_invalid_release_does_not_write_a_validated_report(tmp_path, capsys):
    output = tmp_path / "inspection"
    assert main(["--release-dir", str(tmp_path / "missing"), "--output-dir", str(output)]) == 1
    assert not output.exists()
    assert "inspection failed" in capsys.readouterr().err
