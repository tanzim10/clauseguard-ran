from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from clauseguard.cli.build_golden import main
from clauseguard.datasets.release import prepare_release
from clauseguard.eval import release
from clauseguard.eval.release import (
    ReleasePaths,
    ScenarioMembership,
    build_release,
    check_release,
    load_scenario_membership,
)

SPECS = {
    "3GPP TS 28.552": "a" * 64,
    "3GPP TS 38.331": "b" * 64,
    "3GPP TS 38.300": "c" * 64,
}


def _locator(spec_id: str, clause: str) -> dict:
    return {
        "spec_id": spec_id,
        "version": "18.0.0",
        "sha256": SPECS[spec_id],
        "clause": clause,
        "pdf_page": 10,
    }


def _spec_row(row_id: str, spec_id: str) -> dict:
    return {
        "id": row_id,
        "question": f"Question {row_id}?",
        "expected_answer": "Answer.",
        "expected_citations": [{"spec_id": spec_id, "section": "5.1 Title"}],
    }


def _rca_row(row_id: str, scenario_id: str, inputs: str = "kpi_only") -> dict:
    return {
        "id": row_id,
        "scenario_id": scenario_id,
        "origin": "real",
        "inputs": inputs,
        "question": "Why did downlink throughput fall?",
        "kpi_evidence": [
            {
                "channel": "RRU.PrbUsedDl",
                "baseline": {"start": "0", "end": "59"},
                "comparison": {"start": "60", "end": "119"},
                "baseline_value": 40.0,
                "comparison_value": 95.0,
                "direction": "increase",
            }
        ],
        "source_labels": {"fault_label": "congestion"},
        "expected_conclusion": "PRB saturation.",
        "uncertainty": "One cell only.",
        "expected_citations": [
            {
                "spec_id": "3GPP TS 28.552",
                "section": "5.1 Title",
                "role": "definition",
                "explanation": "Defines PRB usage.",
            },
            {
                "spec_id": "3GPP TS 38.300",
                "section": "10.1 Scheduling",
                "role": "mechanism",
                "explanation": "Scheduler shares PRBs.",
            },
        ],
    }


def _annotation(row_id: str, specs, membership="dev", families=("direct",), **overrides) -> dict:
    clauses = {"3GPP TS 38.300": "10.1"}
    annotation = {
        "id": row_id,
        "status": "approved",
        "reviewer": "Reviewer",
        "reviewed_on": "2026-10-15",
        "membership": membership,
        "families": list(families),
        "evidence": [
            {
                "claim": f"Claim from {spec}.",
                "alternatives": [_locator(spec, clauses.get(spec, "5.1"))],
            }
            for spec in specs
        ],
    }
    if membership is None:
        del annotation["membership"]
    annotation.update(overrides)
    return annotation


def _scenario(split: str, **extra) -> dict:
    """One membership entry shaped like issue #3's `splits.json` (only the fields used here)."""
    no_text = {"detection": "no_approved_text", "category": "no_approved_text"}
    entry = {
        "split": split,
        "group_id": "group_1",
        "exclusion_reason": "",
        "kpi_reason": "",
        "origin": "real",
        "text_reasons": no_text,
    }
    return entry | extra


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_release(
    tmp_path, spec_rows, annotations, rca_rows=(), scenarios=None, monkeypatch=None, mode="scored"
) -> ReleasePaths:
    """Write golden files; `scenarios` stands in for a validated scenario release's membership."""
    golden = tmp_path / "golden"
    golden.mkdir()
    _write_jsonl(golden / "spec_qa.jsonl", spec_rows)
    _write_jsonl(golden / "rca.jsonl", rca_rows)
    _write_jsonl(golden / "annotations.jsonl", annotations)

    corpus = tmp_path / "manifest.csv"
    with open(corpus, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["spec_id", "version", "sha256", "file_format", "page_count"]
        )
        writer.writeheader()
        for spec_id, sha in SPECS.items():
            writer.writerow(
                {
                    "spec_id": spec_id,
                    "version": "18.0.0",
                    "sha256": sha,
                    "file_format": "pdf",
                    "page_count": "100",
                }
            )

    scenario_release = tmp_path / "scenarios"
    if scenarios is not None:
        scenario_release.mkdir()
        (scenario_release / "release.json").write_text("{}", encoding="utf-8")
        membership = ScenarioMembership("fake-release", mode, scenarios)
        monkeypatch.setattr(release, "load_scenario_membership", lambda _: membership)
    return ReleasePaths(golden, corpus, scenario_release)


def _small_release(tmp_path, **changes) -> ReleasePaths:
    """Two dev rows on TS 28.552 and one test row on TS 38.331; valid but incomplete."""
    spec_rows = [
        _spec_row("q_001", "3GPP TS 28.552"),
        _spec_row("q_002", "3GPP TS 28.552"),
        _spec_row("q_003", "3GPP TS 38.331"),
    ]
    annotations = [
        _annotation("q_001", ["3GPP TS 28.552"]),
        _annotation("q_002", ["3GPP TS 28.552"], families=["difficult"]),
        _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
    ]
    args = {"spec_rows": spec_rows, "annotations": annotations} | changes
    return _write_release(tmp_path, **args)


def _complete_release(tmp_path, monkeypatch) -> ReleasePaths:
    """24 spec rows and 6 RCA rows with ordinary, difficult and RCA rows in both splits."""
    spec_rows, annotations = [], []
    for n in range(1, 25):
        row_id = f"q_{n:03d}"
        spec, split = ("3GPP TS 28.552", "dev") if n <= 14 else ("3GPP TS 38.331", "test")
        families = ["difficult"] if n in (1, 15) else ["direct"]
        spec_rows.append(_spec_row(row_id, spec))
        annotations.append(_annotation(row_id, [spec], split, families))

    rca_rows, scenarios = [], {}
    for n in range(1, 7):
        row_id, scenario_id = f"rca_{n:03d}", f"scn_{n:03d}"
        split = "dev" if n <= 3 else "test"
        rca_rows.append(_rca_row(row_id, scenario_id))
        scenarios[scenario_id] = _scenario(split)
        annotations.append(_annotation(row_id, ["3GPP TS 28.552", "3GPP TS 38.300"], split))
    scenarios["scn_train"] = _scenario("train")
    return _write_release(tmp_path, spec_rows, annotations, rca_rows, scenarios, monkeypatch)


def test_complete_release_builds_and_checks(tmp_path, monkeypatch):
    paths = _complete_release(tmp_path, monkeypatch)

    built = build_release(paths)
    checked = check_release(paths)

    assert built.problems == [] and built.incomplete == []
    assert checked.complete
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["counts"] == {
        "approved": 30,
        "by_split": {"dev": {"spec_qa": 14, "rca": 3}, "test": {"spec_qa": 10, "rca": 3}},
    }
    assert manifest["files"]["rca.jsonl"]["rows"] == 6
    assert [s["spec_id"] for s in manifest["sources"]["specs"]] == sorted(SPECS)
    scenario_release = {"release_id": "fake-release", "mode": "scored"}
    assert manifest["sources"]["scenario_release"] == scenario_release


def test_build_is_reproducible(tmp_path, monkeypatch):
    paths = _complete_release(tmp_path, monkeypatch)

    first = build_release(paths).manifest
    second = build_release(paths).manifest

    assert first == second


def test_incomplete_release_is_valid_and_lists_gaps(tmp_path):
    paths = _small_release(tmp_path)

    report = build_release(paths)

    assert report.ok and not report.complete
    assert "approved rows: 3 of 30" in report.incomplete
    assert "approved RCA rows: 0 of 6" in report.incomplete
    assert "test has no difficult rows" in report.incomplete
    assert check_release(paths).ok


def test_missing_annotation_is_rejected(tmp_path):
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
        ],
    )

    assert "q_002: missing annotation" in build_release(paths).problems


def test_orphan_annotation_is_rejected(tmp_path):
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"]),
            _annotation("q_002", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
            _annotation("q_999", ["3GPP TS 28.552"]),
        ],
    )

    assert "q_999: no golden row with this id" in build_release(paths).problems


def test_unresolved_approval_is_rejected(tmp_path):
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"], status="draft"),
            _annotation("q_002", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
        ],
    )

    report = build_release(paths)

    assert "q_001: annotation is not approved (status 'draft')" in report.problems
    assert not paths.manifest.exists()


def test_missing_membership_is_rejected(tmp_path):
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"], membership=None),
            _annotation("q_002", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
        ],
    )

    assert "q_001: annotation has no dev/test membership" in build_release(paths).problems


def test_invalid_membership_value_is_rejected(tmp_path):
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"], membership="train"),
            _annotation("q_002", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
        ],
    )

    [problem] = build_release(paths).problems
    assert "annotations.jsonl:1: membership" in problem


def test_seed_rows_must_stay_in_dev(tmp_path):
    paths = _write_release(
        tmp_path,
        [_spec_row("spec_qa_001", "3GPP TS 28.552")],
        [_annotation("spec_qa_001", ["3GPP TS 28.552"], membership="test")],
    )

    assert "spec_qa_001: seed rows must stay in dev" in build_release(paths).problems


def test_related_spec_rows_cannot_cross_splits(tmp_path):
    # q_002 cites both specs, linking the TS 28.552 rows to the TS 38.331 test row.
    paths = _small_release(
        tmp_path,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"]),
            _annotation("q_002", ["3GPP TS 28.552", "3GPP TS 38.331"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
        ],
    )

    assert build_release(paths).problems == [
        (
            "related spec rows cross splits (shared specs: 3GPP TS 28.552, 3GPP TS 38.331): "
            "dev q_001, q_002; test q_003"
        )
    ]


def test_rca_rows_are_not_tied_to_spec_families(tmp_path, monkeypatch):
    # The test RCA row cites TS 28.552 like the dev spec rows; only its scenario decides.
    paths = _small_release(
        tmp_path,
        rca_rows=[_rca_row("rca_001", "scn_001")],
        scenarios={"scn_001": _scenario("test")},
        monkeypatch=monkeypatch,
        annotations=[
            _annotation("q_001", ["3GPP TS 28.552"]),
            _annotation("q_002", ["3GPP TS 28.552"]),
            _annotation("q_003", ["3GPP TS 38.331"], membership="test"),
            _annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"], membership="test"),
        ],
    )

    assert build_release(paths).problems == []


@pytest.mark.parametrize(
    ("scenarios", "membership", "inputs", "problem"),
    [
        (
            {"scn_other": _scenario("test")},
            "test",
            "kpi_only",
            "rca_001: unknown scenario 'scn_001'",
        ),
        (
            {"scn_001": _scenario("train")},
            "test",
            "kpi_only",
            "rca_001: scenario 'scn_001' is in 'train' and cannot be used in the golden set",
        ),
        (
            {"scn_001": _scenario("excluded", exclusion_reason="overlapping windows")},
            "dev",
            "kpi_only",
            "rca_001: scenario 'scn_001' is in 'excluded' and cannot be used in the golden set",
        ),
        (
            {"scn_001": _scenario("dev")},
            "test",
            "kpi_only",
            "rca_001: membership 'test' does not match scenario 'scn_001' split 'dev'",
        ),
        (
            {"scn_001": _scenario("test", kpi_reason="requires_128_steps")},
            "test",
            "kpi_only",
            "rca_001: scenario 'scn_001' KPI window is unusable (requires_128_steps)",
        ),
        (
            {"scn_001": _scenario("test", origin="synthetic")},
            "test",
            "kpi_only",
            "rca_001: origin 'real' does not match scenario 'scn_001' origin 'synthetic'",
        ),
        (
            {"scn_001": _scenario("test")},
            "test",
            "multimodal",
            "rca_001: scenario 'scn_001' is not ready for multimodal inputs (no_approved_text)",
        ),
    ],
)
def test_rca_scenario_references_are_checked(
    tmp_path, monkeypatch, scenarios, membership, inputs, problem
):
    paths = _write_release(
        tmp_path,
        [],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"], membership)],
        rca_rows=[_rca_row("rca_001", "scn_001", inputs)],
        scenarios=scenarios,
        monkeypatch=monkeypatch,
    )

    assert build_release(paths).problems == [problem]


def test_multimodal_rows_accept_text_approved_for_the_category_task(tmp_path, monkeypatch):
    ready = {"detection": "text_not_approved_for_task", "category": ""}
    paths = _write_release(
        tmp_path,
        [],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"])],
        rca_rows=[_rca_row("rca_001", "scn_001", "multimodal")],
        scenarios={"scn_001": _scenario("dev", text_reasons=ready)},
        monkeypatch=monkeypatch,
    )

    assert build_release(paths).problems == []


def test_exploratory_scenario_releases_cannot_back_rca_rows(tmp_path, monkeypatch):
    paths = _write_release(
        tmp_path,
        [],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"])],
        rca_rows=[_rca_row("rca_001", "scn_001")],
        scenarios={"scn_001": _scenario("dev")},
        monkeypatch=monkeypatch,
        mode="exploratory",
    )

    assert build_release(paths).problems == [
        "scenario release fake-release is 'exploratory', not 'scored'"
    ]


def test_rca_rows_need_the_scenario_release(tmp_path):
    paths = _write_release(
        tmp_path,
        [],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"])],
        rca_rows=[_rca_row("rca_001", "scn_001")],
    )

    [problem] = build_release(paths).problems
    assert problem.startswith("rca.jsonl has rows but the scenario release")


def test_spec_only_release_does_not_need_the_scenario_release(tmp_path):
    paths = _small_release(tmp_path)

    assert not paths.scenario_release.exists()
    assert build_release(paths).problems == []


def test_ids_cannot_repeat_across_spec_and_rca_files(tmp_path, monkeypatch):
    paths = _write_release(
        tmp_path,
        [_spec_row("rca_001", "3GPP TS 28.552")],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"])],
        rca_rows=[_rca_row("rca_001", "scn_001")],
        scenarios={"scn_001": _scenario("dev")},
        monkeypatch=monkeypatch,
    )

    assert (
        "rca_001: id is used by both spec_qa.jsonl and rca.jsonl" in build_release(paths).problems
    )


@pytest.fixture
def scenario_release(release_sources, recording_evidence):
    """A real issue #3 scenario release built offline from the shared test sources."""
    out = release_sources["output"]
    prepare_release(release_sources["scenarios"], out, out, trials=128)
    return out


def test_rca_rows_are_checked_against_a_real_scenario_release(tmp_path, scenario_release):
    scenarios = load_scenario_membership(scenario_release)

    def first(split: str) -> str:
        return next(
            sid
            for sid, m in sorted(scenarios.membership.items())
            if m["split"] == split and not m["kpi_reason"] and m["origin"] in ("real", "synthetic")
        )

    eligible = {split: first(split) for split in ("dev", "test")}
    train = first("train")
    rca_rows, annotations = [], []
    for n, (split, sid) in enumerate([*eligible.items(), ("test", train)], start=1):
        row = _rca_row(f"rca_{n:03d}", sid) | {"origin": scenarios.membership[sid]["origin"]}
        rca_rows.append(row)
        annotations.append(_annotation(row["id"], ["3GPP TS 28.552", "3GPP TS 38.300"], split))
    golden = _write_release(tmp_path, [], annotations, rca_rows)
    paths = ReleasePaths(golden.golden_dir, golden.corpus_manifest, scenario_release)

    report = build_release(paths)

    assert report.problems == [
        f"rca_003: scenario {train!r} is in 'train' and cannot be used in the golden set"
    ]
    assert scenarios.mode == "scored"


def test_tampered_scenario_release_is_rejected(tmp_path, scenario_release):
    splits = scenario_release / "splits.json"
    splits.write_text(splits.read_text(encoding="utf-8") + " ", encoding="utf-8")
    golden = _write_release(
        tmp_path,
        [],
        [_annotation("rca_001", ["3GPP TS 28.552", "3GPP TS 38.300"])],
        rca_rows=[_rca_row("rca_001", "scn_001")],
    )
    paths = ReleasePaths(golden.golden_dir, golden.corpus_manifest, scenario_release)

    [problem] = build_release(paths).problems
    assert problem.startswith(f"scenario release {scenario_release}:")
    assert "checksum mismatch" in problem


def test_check_detects_changed_files(tmp_path):
    paths = _small_release(tmp_path)
    build_release(paths)

    spec_qa = paths.golden_dir / "spec_qa.jsonl"
    spec_qa.write_text(spec_qa.read_text(encoding="utf-8").replace("Answer.", "Edited."))

    problems = check_release(paths).problems
    assert "spec_qa.jsonl does not match the release manifest; rebuild it" in problems
    assert "release manifest 'release_id' is stale; rebuild it" in problems


def test_check_requires_a_manifest(tmp_path):
    paths = _small_release(tmp_path)

    [problem] = check_release(paths).problems
    assert problem.startswith("cannot read release manifest")


def _cli_args(paths: ReleasePaths) -> list[str]:
    return [
        f"--golden-dir={paths.golden_dir}",
        f"--corpus-manifest={paths.corpus_manifest}",
        f"--scenario-release={paths.scenario_release}",
    ]


def test_cli_reports_incomplete_without_failing_unless_strict(tmp_path, capsys):
    paths = _small_release(tmp_path)
    args = _cli_args(paths)

    assert main(["build", *args]) == 0
    assert main(["check", *args]) == 0
    assert main(["check", "--strict", *args]) == 1
    assert "incomplete: approved rows: 3 of 30" in capsys.readouterr().out


def test_cli_fails_on_validation_problems(tmp_path, capsys):
    paths = _small_release(
        tmp_path,
        annotations=[_annotation("q_001", ["3GPP TS 28.552"])],
    )

    assert main(["build", *_cli_args(paths)]) == 1
    assert "error: q_002: missing annotation" in capsys.readouterr().err


def test_repository_release_matches_its_manifest():
    report = check_release(ReleasePaths())

    assert report.problems == []
    assert report.manifest["counts"]["by_split"] == {
        "dev": {"spec_qa": 19},
        "test": {"spec_qa": 5},
    }
