"""Golden release: pin the reviewed golden files and check them together (Week 5).

A release is `data/golden/manifest.json`. It records each golden file's SHA-256 and row count,
the identity (version, hash) of every specification the annotations cite, the scenario release
used for RCA rows, and whether the release meets its coverage targets.

`build` validates the golden files and writes the manifest. `check` validates them again and
also fails when the files no longer match the manifest (`python -m clauseguard.cli.build_golden`).
Validation covers:

- every row has an approved annotation with dev/test membership, and no annotation is orphaned;
- seed rows stay in dev, and spec rows that cite the same specification share a split;
- RCA rows reference a known scenario in issue #3's validated scenario release whose split is
  the same as the row's membership, so training and excluded scenarios never enter the golden
  set. The scenario release is only needed once `rca.jsonl` has rows.

Missing coverage (fewer than 30 approved rows, fewer than 6 RCA rows, or a split without
ordinary and difficult rows) marks the release incomplete. That is reported, not treated as a
validation failure, unless `--strict` is passed.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clauseguard.corpus.manifest import load_manifest
from clauseguard.datasets.scenarios import DEFAULT_RELEASE, ScenarioDataset
from clauseguard.datasets.schema import GoldenEvalRow, RcaRow
from clauseguard.eval.annotations import Annotation, check_against_golden, load_annotations
from clauseguard.eval.golden import GoldenLoadError, load_golden, load_rca

SPEC_QA_FILE = "spec_qa.jsonl"
RCA_FILE = "rca.jsonl"
ANNOTATIONS_FILE = "annotations.jsonl"
MANIFEST_FILE = "manifest.json"
GOLDEN_FILES = (ANNOTATIONS_FILE, RCA_FILE, SPEC_QA_FILE)

# The original seed questions were written while building the pipeline, so they stay in dev.
SEED_IDS = frozenset(f"spec_qa_{n:03d}" for n in range(1, 13))
MIN_APPROVED_ROWS = 30
MIN_APPROVED_RCA_ROWS = 6


@dataclass(frozen=True)
class ScenarioMembership:
    """Scenario membership from issue #3's release, after `ScenarioDataset` has validated it."""

    release_id: str
    mode: str
    membership: dict[str, dict[str, Any]]


def load_scenario_membership(release_dir: Path | str) -> ScenarioMembership:
    """Validate a scenario release (identity, checksums, membership) and return its membership."""
    dataset = ScenarioDataset(Path(release_dir))
    splits = dataset.splits
    return ScenarioMembership(dataset.release_id, splits["mode"], splits["membership"])


@dataclass(frozen=True)
class ReleasePaths:
    golden_dir: Path = Path("data/golden")
    corpus_manifest: Path = Path("data/corpus/manifest.csv")
    scenario_release: Path = DEFAULT_RELEASE

    @property
    def manifest(self) -> Path:
        return self.golden_dir / MANIFEST_FILE


@dataclass
class ReleaseReport:
    problems: list[str] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    manifest: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def complete(self) -> bool:
        return self.ok and not self.incomplete


def evaluate_release(paths: ReleasePaths) -> ReleaseReport:
    """Validate the golden files and compute the manifest they would produce."""
    report = ReleaseReport()
    try:
        corpus = load_manifest(paths.corpus_manifest)
        known_spec_ids = {row["spec_id"] for row in corpus if row.get("spec_id")}
        spec_rows = load_golden(paths.golden_dir / SPEC_QA_FILE, known_spec_ids)
        rca_rows = load_rca(paths.golden_dir / RCA_FILE, known_spec_ids)
        annotations = load_annotations(paths.golden_dir / ANNOTATIONS_FILE, corpus)
    except (GoldenLoadError, OSError, ValueError) as exc:
        report.problems.append(str(exc))
        return report

    scenarios: ScenarioMembership | None = None
    if rca_rows and (paths.scenario_release / "release.json").exists():
        try:
            scenarios = load_scenario_membership(paths.scenario_release)
        except (OSError, KeyError, ValueError) as exc:
            report.problems.append(f"scenario release {paths.scenario_release}: {exc}")
            return report

    by_id = {annotation.id: annotation for annotation in annotations}
    report.problems += _row_problems(spec_rows, rca_rows, annotations, by_id)
    report.problems += _spec_family_problems(spec_rows, by_id)
    report.problems += _rca_problems(rca_rows, by_id, scenarios, paths.scenario_release)
    report.incomplete = _coverage_gaps(spec_rows, rca_rows, by_id)
    report.manifest = _manifest(
        paths, corpus, annotations, spec_rows, rca_rows, by_id, scenarios, report
    )
    return report


def build_release(paths: ReleasePaths) -> ReleaseReport:
    """Validate the golden files and, when they pass, write `manifest.json`."""
    report = evaluate_release(paths)
    if report.ok and report.manifest is not None:
        paths.manifest.write_text(
            json.dumps(report.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return report


def check_release(paths: ReleasePaths) -> ReleaseReport:
    """Validate the golden files and confirm they still match the committed manifest."""
    report = evaluate_release(paths)
    if not report.ok or report.manifest is None:
        return report
    try:
        recorded = json.loads(paths.manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        report.problems.append(f"cannot read release manifest {paths.manifest}: {exc}")
        return report

    expected = report.manifest
    for name in GOLDEN_FILES:
        if recorded.get("files", {}).get(name) != expected["files"][name]:
            report.problems.append(f"{name} does not match the release manifest; rebuild it")
    for key in ("sources", "counts", "complete", "incomplete", "release_id"):
        if recorded.get(key) != expected[key]:
            report.problems.append(f"release manifest {key!r} is stale; rebuild it")
    return report


def _row_problems(
    spec_rows: list[GoldenEvalRow],
    rca_rows: list[RcaRow],
    annotations: list[Annotation],
    by_id: dict[str, Annotation],
) -> list[str]:
    problems: list[str] = []
    spec_ids = {row.id for row in spec_rows}
    for row in rca_rows:
        if row.id in spec_ids:
            problems.append(f"{row.id}: id is used by both {SPEC_QA_FILE} and {RCA_FILE}")

    problems += check_against_golden(annotations, [*spec_rows, *rca_rows])
    for row in [*spec_rows, *rca_rows]:
        annotation = by_id.get(row.id)
        if annotation is None:
            problems.append(f"{row.id}: missing annotation")
            continue
        if annotation.status != "approved":
            problems.append(f"{row.id}: annotation is not approved (status {annotation.status!r})")
        if annotation.membership is None:
            problems.append(f"{row.id}: annotation has no dev/test membership")
        elif row.id in SEED_IDS and annotation.membership != "dev":
            problems.append(f"{row.id}: seed rows must stay in dev")
    return problems


def _spec_family_problems(
    spec_rows: list[GoldenEvalRow], by_id: dict[str, Annotation]
) -> list[str]:
    """Spec rows citing a shared specification must share a split (RCA rows are excluded)."""
    rows_by_spec: dict[str, set[str]] = defaultdict(set)
    for row in spec_rows:
        annotation = by_id.get(row.id)
        if annotation is None or annotation.membership is None:
            continue
        for group in annotation.evidence:
            for locator in group.alternatives:
                rows_by_spec[locator.spec_id].add(row.id)

    parent: dict[str, str] = {}

    def find(row_id: str) -> str:
        parent.setdefault(row_id, row_id)
        while parent[row_id] != row_id:
            parent[row_id] = parent[parent[row_id]]
            row_id = parent[row_id]
        return row_id

    for row_ids in rows_by_spec.values():
        first, *rest = sorted(row_ids)
        for row_id in rest:
            parent[find(row_id)] = find(first)

    families: dict[str, set[str]] = defaultdict(set)
    for row_ids in rows_by_spec.values():
        for row_id in row_ids:
            families[find(row_id)].add(row_id)

    problems = []
    for row_ids in sorted(families.values(), key=min):
        splits = {by_id[row_id].membership for row_id in row_ids}
        if len(splits) > 1:
            specs = sorted(spec for spec, ids in rows_by_spec.items() if ids & row_ids)
            dev = sorted(i for i in row_ids if by_id[i].membership == "dev")
            test = sorted(i for i in row_ids if by_id[i].membership == "test")
            problems.append(
                f"related spec rows cross splits (shared specs: {', '.join(specs)}): "
                f"dev {', '.join(dev)}; test {', '.join(test)}"
            )
    return problems


def _rca_problems(
    rca_rows: list[RcaRow],
    by_id: dict[str, Annotation],
    scenarios: ScenarioMembership | None,
    release_dir: Path,
) -> list[str]:
    if not rca_rows:
        return []
    if scenarios is None:
        return [f"{RCA_FILE} has rows but the scenario release {release_dir} is missing"]
    if scenarios.mode != "scored":
        return [f"scenario release {scenarios.release_id} is {scenarios.mode!r}, not 'scored'"]

    problems = []
    for row in rca_rows:
        scenario = scenarios.membership.get(row.scenario_id)
        if scenario is None:
            problems.append(f"{row.id}: unknown scenario {row.scenario_id!r}")
            continue
        if scenario["split"] in ("train", "excluded"):
            problems.append(
                f"{row.id}: scenario {row.scenario_id!r} is in {scenario['split']!r} and cannot "
                "be used in the golden set"
            )
            continue
        annotation = by_id.get(row.id)
        if annotation and annotation.membership and annotation.membership != scenario["split"]:
            problems.append(
                f"{row.id}: membership {annotation.membership!r} does not match scenario "
                f"{row.scenario_id!r} split {scenario['split']!r}"
            )
        if scenario["kpi_reason"]:
            problems.append(
                f"{row.id}: scenario {row.scenario_id!r} KPI window is unusable "
                f"({scenario['kpi_reason']})"
            )
        if row.origin != scenario["origin"]:
            problems.append(
                f"{row.id}: origin {row.origin!r} does not match scenario {row.scenario_id!r} "
                f"origin {scenario['origin']!r}"
            )
        # Diagnosis-level text must be approved for the category task and must not disclose it.
        text_reason = scenario["text_reasons"]["category"]
        if row.inputs == "multimodal" and text_reason:
            problems.append(
                f"{row.id}: scenario {row.scenario_id!r} is not ready for multimodal inputs "
                f"({text_reason})"
            )
    return problems


def _coverage_gaps(
    spec_rows: list[GoldenEvalRow], rca_rows: list[RcaRow], by_id: dict[str, Annotation]
) -> list[str]:
    approved = [
        (row, by_id[row.id])
        for row in [*spec_rows, *rca_rows]
        if row.id in by_id and by_id[row.id].status == "approved"
    ]
    rca_count = sum(1 for row, _ in approved if isinstance(row, RcaRow))

    gaps = []
    if len(approved) < MIN_APPROVED_ROWS:
        gaps.append(f"approved rows: {len(approved)} of {MIN_APPROVED_ROWS}")
    if rca_count < MIN_APPROVED_RCA_ROWS:
        gaps.append(f"approved RCA rows: {rca_count} of {MIN_APPROVED_RCA_ROWS}")
    for split in ("dev", "test"):
        in_split = [(row, a) for row, a in approved if a.membership == split]
        if not any(_is_ordinary(a) for _, a in in_split):
            gaps.append(f"{split} has no ordinary (non-difficult, answerable) rows")
        if not any("difficult" in a.families for _, a in in_split):
            gaps.append(f"{split} has no difficult rows")
        if not any(isinstance(row, RcaRow) for row, _ in in_split):
            gaps.append(f"{split} has no RCA rows")
    return gaps


def _is_ordinary(annotation: Annotation) -> bool:
    return not {"difficult", "no_answer"} & set(annotation.families)


def _manifest(
    paths: ReleasePaths,
    corpus: list[dict[str, Any]],
    annotations: list[Annotation],
    spec_rows: list[GoldenEvalRow],
    rca_rows: list[RcaRow],
    by_id: dict[str, Annotation],
    scenarios: ScenarioMembership | None,
    report: ReleaseReport,
) -> dict[str, Any]:
    row_counts = {
        ANNOTATIONS_FILE: len(annotations),
        RCA_FILE: len(rca_rows),
        SPEC_QA_FILE: len(spec_rows),
    }
    files = {
        name: {"sha256": _sha256(paths.golden_dir / name), "rows": row_counts[name]}
        for name in GOLDEN_FILES
    }

    cited = {locator.spec_id for a in annotations for g in a.evidence for locator in g.alternatives}
    cited |= {c.spec_id for row in [*spec_rows, *rca_rows] for c in row.expected_citations}
    specs = [
        {"spec_id": row["spec_id"], "version": row.get("version"), "sha256": row.get("sha256")}
        for row in sorted(corpus, key=lambda r: r.get("spec_id") or "")
        if row.get("spec_id") in cited
    ]
    scenario_release = None
    if rca_rows and scenarios is not None:
        scenario_release = {"release_id": scenarios.release_id, "mode": scenarios.mode}
    sources = {"specs": specs, "scenario_release": scenario_release}

    counts: dict[str, Any] = {"approved": 0, "by_split": {}}
    for row in [*spec_rows, *rca_rows]:
        annotation = by_id.get(row.id)
        if annotation is None or annotation.status != "approved":
            continue
        counts["approved"] += 1
        split = counts["by_split"].setdefault(annotation.membership or "none", {})
        split[row.type] = split.get(row.type, 0) + 1
    counts["by_split"] = dict(sorted(counts["by_split"].items()))

    identity = json.dumps({"files": files, "sources": sources}, sort_keys=True)
    return {
        "release_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
        "files": files,
        "sources": sources,
        "counts": counts,
        "complete": not report.problems and not report.incomplete,
        "incomplete": report.incomplete,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
