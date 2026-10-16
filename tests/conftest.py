"""Offline, measurement-backed source families for dataset release tests."""

import csv
import json
from datetime import datetime, timedelta

import pytest

from clauseguard.datasets.acquisition import acquire_supplements, acquire_telecomts
from clauseguard.datasets.contracts import CATEGORY_LABELS, NUMERIC_CHANNELS
from clauseguard.datasets.evidence import build_recording_evidence
from clauseguard.datasets.telecomts import build_scenarios, file_sha256


@pytest.fixture
def release_sources(tmp_path):
    local, acquired, scenarios = (tmp_path / name for name in ("local", "acquired", "scenarios"))
    checksums, processed, reviews = {}, [], {}
    for group in range(3):
        for kind in ("normal", "synthetic", "jammer"):
            family = f"{kind}/{group}" if kind == "normal" else f"anomalous/{kind}/{group}"
            source = family + "/processed/chunked.jsonl"
            labels = ["normal"] if kind == "normal" else (
                list(CATEGORY_LABELS) if kind == "synthetic" else ["Jamming"])
            count = 160 if kind == "normal" else 128 * len(labels)
            start = datetime.fromisoformat(f"2025-01-{group + 1:02d} 00:00:00")
            raw = []
            for i in range(count):
                raw.append({"timestamp": str(start + timedelta(seconds=i / 10)),
                            **{c: group * 10000 + i + j / 10 for j, c in enumerate(NUMERIC_CHANNELS)},
                            "UL_Protocol": "TCP", "DL_Protocol": ""})
            offsets = [0, 32] if kind == "normal" else list(range(0, count, 128))
            rows = []
            for number, offset in enumerate(offsets):
                label = "normal" if kind == "normal" else labels[number]
                rows.append({"start_time": raw[offset]["timestamp"],
                             "end_time": raw[offset + 127]["timestamp"], "sampling_rate": 10,
                             "KPIs": {c: [r[c] for r in raw[offset:offset + 128]]
                                      for c in raw[0] if c != "timestamp"},
                             "anomalies": {"exists": kind != "normal", "type": label},
                             "description": "There is a jammer. Revealing source text.",
                             "QnA": {"answer": label}})
            # Source order deliberately differs from time order for overlapping normal windows.
            target = local / source
            target.parent.mkdir(parents=True)
            target.write_text("".join(json.dumps(r) + "\n" for r in reversed(rows)))
            processed.append(source)
            for asset in ("metrics.csv", "description.txt"):
                path = local / family / "raw" / asset
                path.parent.mkdir(parents=True, exist_ok=True)
                if asset.endswith("csv"):
                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
                        writer.writeheader()
                        writer.writerows(raw)
                else:
                    path.write_text("There is a jammer.")
                checksums[str(path.relative_to(local))] = file_sha256(path)
            checksums[source] = file_sha256(target)
            reviews[source] = {
                "status": "verified", "recording_id": f"session-{group}-{kind}",
                "parent_recording_ids": [f"session-{group}-normal"] if kind != "normal" else [],
                "evidence": f"Fixture session ledger {group}", "reviewer": "fixture reviewer",
                "reviewed_at": "2026-10-15",
            }
    acquire_telecomts(revision="a" * 40, output_dir=acquired, local_dir=local,
                      checksums=checksums, files=processed)
    acquire_supplements(revision="a" * 40, output_dir=acquired, sources=processed,
                        local_dir=local, checksums=checksums)
    build_scenarios({s: local / s for s in processed}, revision="a" * 40,
                    checksums=checksums, output_dir=scenarios)
    return {"scenarios": scenarios / "scenarios.jsonl", "acquired": acquired,
            "local": local, "checksums": checksums, "reviews": reviews,
            "output": tmp_path / "release"}


@pytest.fixture
def recording_evidence(release_sources):
    s = release_sources
    return build_recording_evidence(s["scenarios"], s["acquired"], s["output"], reviews=s["reviews"])
