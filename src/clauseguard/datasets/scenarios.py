"""Validated release loading: one membership, two targets, explicit observation payloads."""

import json
from pathlib import Path

from clauseguard.datasets.contracts import LABEL_MAPPINGS, NUMERIC_CHANNELS, ModelExample
from clauseguard.datasets.evidence import summarize_record
from clauseguard.datasets.release import EVIDENCE_FILES, RELEASE_FILES, _identity, project_inputs
from clauseguard.datasets.schema import ScenarioRecord
from clauseguard.datasets.splits import build_quality, select_task_cohort, validate_membership
from clauseguard.datasets.telecomts import file_sha256

DEFAULT_RELEASE = Path("data/scenarios/release")


class ScenarioDataset:
    """Validate hashes/contracts once, then seek by ID without retaining KPI arrays in memory."""

    def __init__(self, release_dir: Path = DEFAULT_RELEASE, *, allow_exploratory: bool = False,
                 expected_release_id: str | None = None):
        self.release_dir = Path(release_dir)
        manifest = json.loads((self.release_dir / "release.json").read_text(encoding="utf-8"))
        self.release_id = manifest.pop("release_id")
        if manifest.get("version") != "1" or _identity(manifest) != self.release_id:
            raise ValueError("invalid release identity or version")
        if expected_release_id is not None and expected_release_id != self.release_id:
            raise ValueError("unexpected release identity")
        if (set(manifest["files"]) != {*EVIDENCE_FILES, *RELEASE_FILES}
                or manifest["numeric_channels"] != list(NUMERIC_CHANNELS)):
            raise ValueError("invalid release artifact contract")
        for name, expected in manifest["files"].items():
            if file_sha256(self.release_dir / name) != expected:
                raise ValueError(f"release checksum mismatch: {name}")
        self.scenarios_path = (self.release_dir / manifest["scenarios_path"]).resolve()
        self._source_stat = self._stat()
        if file_sha256(self.scenarios_path) != manifest["scenarios_sha256"]:
            raise ValueError("scenario checksum mismatch")
        self.splits = self._read("splits.json")
        self.evidence = self._read("evidence.json")
        self.quality = self._read("quality.json")
        if (manifest["mode"] != self.splits["mode"]
                or manifest["scenarios_sha256"] != self.splits["scenarios_sha256"]
                or manifest["revision"] != self.evidence["revision"]
                or self._read("labels.json") != LABEL_MAPPINGS):
            raise ValueError("release, split, evidence or label identities differ")
        validate_membership(self.splits, self.evidence)
        if build_quality(self.splits) != self.quality:
            raise ValueError("quality report does not match membership")
        self.allow_exploratory = allow_exploratory
        self.offsets = {}
        with self.scenarios_path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                try:
                    record = ScenarioRecord.model_validate_json(line)
                    sid = record.scenario_id
                    if sid in self.offsets or sid not in self.splits["membership"]:
                        raise ValueError(f"duplicate or unknown scenario ID: {sid}")
                    if any(self.splits["membership"][sid][k] != v
                           for k, v in summarize_record(record).items()):
                        raise ValueError(f"scenario differs from membership: {sid}")
                    if record.provenance.revision != manifest["revision"]:
                        raise ValueError(f"scenario revision differs from release: {sid}")
                    self.offsets[sid] = offset
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{self.scenarios_path.name}:{line_number}: {exc}") from exc
        if set(self.offsets) != set(self.splits["membership"]):
            raise ValueError("missing scenario IDs referenced by membership")
        self.check_source()
        self.manifest = manifest

    def _read(self, name):
        return json.loads((self.release_dir / name).read_text(encoding="utf-8"))

    def _stat(self):
        stat = self.scenarios_path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    def check_source(self):
        """Reject changes to the validated source, including for empty experiment views."""
        if self._stat() != self._source_stat:
            raise ValueError("scenario source changed since release validation")

    def load_scenario(self, scenario_id: str) -> ScenarioRecord:
        """Full record for inspection/golden review, including explicitly excluded records."""
        self.check_source()
        if scenario_id not in self.offsets:
            raise KeyError(f"unknown scenario ID: {scenario_id}")
        with self.scenarios_path.open("rb") as stream:
            stream.seek(self.offsets[scenario_id])
            return ScenarioRecord.model_validate_json(stream.readline())

    def iter_examples(self, *, split: str, task: str, modality: str = "kpi"):
        """Default scored iteration refuses exploratory releases and incomplete task coverage."""
        ids = select_task_cohort(self.splits, split=split, task=task, modality=modality)
        status = self.quality["tasks"][task][modality]
        if self.splits["mode"] == "exploratory" and not self.allow_exploratory:
            raise ValueError("exploratory release: explicit allow_exploratory=True is required")
        if not status["coverage_ready"] or (
            self.splits["mode"] == "scored" and not status["ready"]
        ):
            raise ValueError(f"{task}/{modality} is not ready: {status['reasons']}")
        self.check_source()
        with self.scenarios_path.open("rb") as stream:
            for sid in ids:
                stream.seek(self.offsets[sid])
                record = ScenarioRecord.model_validate_json(stream.readline())
                member = self.splits["membership"][sid]
                yield ModelExample(
                    inputs=project_inputs(record, modality=modality, task=task),
                    target=member[f"{task}_target"], scenario_id=sid, group_id=member["group_id"],
                    split=split, task=task,
                )
        self.check_source()


def load_scenario(scenario_id: str, *, release_dir: Path = DEFAULT_RELEASE) -> ScenarioRecord:
    return ScenarioDataset(release_dir).load_scenario(scenario_id)


def iter_examples(*, split: str, task: str, release_dir: Path = DEFAULT_RELEASE,
                  modality: str = "kpi", allow_exploratory: bool = False):
    dataset = ScenarioDataset(release_dir, allow_exploratory=allow_exploratory)
    yield from dataset.iter_examples(split=split, task=task, modality=modality)
