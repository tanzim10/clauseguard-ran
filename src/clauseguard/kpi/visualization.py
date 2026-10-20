"""Offline plots of saved release metadata and unchanged source observations."""

import json
from collections import Counter
from pathlib import Path

from clauseguard.datasets.telecomts import file_sha256

SPLITS = ("train", "dev", "test")
COLORS = ("#2471a3", "#e67e22", "#229954")
SHORT_LABELS = {
    "Antenna Failure": "Antenna failure",
    "Co-Channel Interference (Mild)": "Interference: mild",
    "Co-Channel Interference (Severe)": "Interference: severe",
    "Faulty RF Filters (Temporal)": "RF filters",
    "Doppler Shift (Severe)": "Doppler shift",
    "Faulty Handover Algorithm (Too Frequent)": "Frequent handover",
    "Buffer Overflow (Gradual Buildup)": "Buffer overflow",
    "Resource Allocation Bugs": "Resource allocation",
    "High Network Congestion (Gradual Buildup)": "Congestion: gradual",
    "High Network Congestion (Sudden Spike)": "Congestion: spike",
}


def _metadata(release_dir):
    manifest = json.loads((release_dir / "release.json").read_text())
    loaded = {}
    for name in ("splits.json", "labels.json", "evidence.json"):
        path = release_dir / name
        if file_sha256(path) != manifest["files"][name]:
            raise ValueError(f"release metadata checksum mismatch: {name}")
        loaded[name] = json.loads(path.read_text())
    return manifest, loaded


def _selection(path, manifest, members):
    selection = json.loads(path.read_text())
    if selection["release_id"] != manifest["release_id"]:
        raise ValueError("selection belongs to a different release")
    ids = selection["scenario_ids"]
    if len(ids) != len(set(ids)) or set(ids) - members.keys():
        raise ValueError("selection contains duplicate or unknown scenario IDs")
    selected = set(ids)
    if any(sid not in selected for sid, row in members.items()
           if row["split"] != "train" or row["detection_target"] == 1):
        raise ValueError("training selection removed evaluation cases or faults")
    return selected


def _representatives(members, evidence, raw_dir):
    examples = []
    root = raw_dir.resolve()
    for origin in ("normal", "synthetic", "real"):
        candidates = [r for r in members.values() if r["origin"] == origin]
        if not candidates:
            continue
        row = min(candidates, key=lambda r: (r["source_file"], r["source_row"]))
        path = (root / row["source_file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source path escapes acquisition directory")
        if file_sha256(path) != evidence["traces"][row["source_file"]]["processed_sha256"]:
            raise ValueError("representative source checksum mismatch")
        with path.open() as stream:
            for index, line in enumerate(stream, 1):
                if index == row["source_row"]:
                    observation = json.loads(line)
                    break
            else:
                raise ValueError("representative source row is absent")
        if (observation["start_time"] != row["start_time"]
                or observation["end_time"] != row["end_time"]):
            raise ValueError("representative source timeline differs from metadata")
        examples.append((row, observation))
    return examples


def visualize_release(
    release_dir: Path, output_dir: Path, *, selection_path: Path, raw_dir: Path | None = None,
) -> dict[str, Path]:
    """Save PNG/SVG figures; verify metadata and plotted source files, not release readiness.

    Requires the analysis extra. Selection IDs must retain every fault and evaluation case.
    Existing matching filenames are replaced so the notebook can be rerun.
    """
    import matplotlib.pyplot as plt

    release_dir, output_dir = Path(release_dir), Path(output_dir)
    manifest, metadata = _metadata(release_dir)
    members = metadata["splits.json"]["membership"]
    labels = metadata["labels.json"]["category"]
    selected = _selection(Path(selection_path), manifest, members)
    examples = _representatives(
        members, metadata["evidence.json"],
        Path(raw_dir) if raw_dir is not None else release_dir.parents[1] / "telecomts/raw",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    mode = metadata["splits.json"]["mode"]
    unresolved = any(row.get("evidence_status") != "verified" for row in members.values())
    review = "recording families unresolved" if unresolved else "recording groups verified"
    disclaimer = f"{mode.capitalize()} data · {review} · before model training"

    def save(fig, name):
        fig.text(0.5, 0.005, disclaimer, ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.035, 1, 0.97))
        for extension in ("png", "svg"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            paths[f"{name}.{extension}"] = path
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, target in enumerate((0, 1)):
        counts = [sum(r["split"] == split and r["detection_target"] == target
                      for r in members.values()) for split in SPLITS]
        bars = ax.bar([i + (offset - 0.5) * 0.35 for i in range(3)], counts, 0.35,
                      label=("Normal", "Fault")[offset], color=COLORS[offset])
        ax.bar_label(bars, padding=3, fmt="%d")
    ax.set(xticks=range(3), xticklabels=SPLITS, ylabel="Windows (log scale)",
           title="Detection: saved split counts", yscale="log", ylim=(1, 50000))
    ax.legend()
    save(fig, "detection_counts")

    fig, ax = plt.subplots(figsize=(10, 6))
    ordered = sorted(labels, key=labels.get)
    for index, split in enumerate(SPLITS):
        counts = Counter(r["category_target"] for r in members.values() if r["split"] == split)
        bars = ax.barh([i + (index - 1) * 0.25 for i in range(len(ordered))],
                       [counts[labels[label]] for label in ordered], 0.25,
                       label=split, color=COLORS[index])
        ax.bar_label(bars, padding=2, fontsize=8)
    ax.set(yticks=range(len(ordered)), yticklabels=[SHORT_LABELS.get(x, x) for x in ordered],
           xlabel="Windows", title="Category: ten synthetic fault classes by saved split")
    ax.invert_yaxis()
    ax.legend()
    ax.margins(x=0.12)
    save(fig, "category_counts")

    fig, ax = plt.subplots(figsize=(8, 4))
    for index, retained in enumerate((False, True)):
        counts = [sum(r["split"] == split and (not retained or sid in selected)
                      for sid, r in members.items()) for split in SPLITS]
        bars = ax.bar([i + (index - 0.5) * 0.35 for i in range(3)], counts, 0.35,
                      label=("Original", "Normal thinning in training")[index], color=COLORS[index])
        ax.bar_label(bars, padding=3, fmt="%d")
    ax.set(xticks=range(3), xticklabels=SPLITS, ylabel="Windows",
           title="Training reduction: all faults and evaluation cases retained", ylim=(0, 24000))
    ax.legend()
    save(fig, "training_reduction")

    if examples:
        channels = ("RSRP", "DL_BLER", "UL_SNR")
        fig, axes = plt.subplots(len(examples), len(channels), figsize=(13, 8), squeeze=False)
        for row_index, (member, observation) in enumerate(examples):
            origin = {"normal": "Normal", "synthetic": "Synthetic fault", "real": "Jamming"}[
                member["origin"]]
            for col_index, channel in enumerate(channels):
                ax = axes[row_index, col_index]
                values = observation["KPIs"][channel]
                if len(values) != 128:
                    raise ValueError("representative window must contain 128 samples")
                rate = observation["sampling_rate"]
                ax.plot([i / rate for i in range(128)], values, color=COLORS[row_index % 3])
                ax.set(xlabel="Seconds from window start", ylabel=channel, xlim=(0, 127 / rate))
                if col_index == 0:
                    ax.set_title(f"{origin}: {SHORT_LABELS.get(member['rca_label'], member['rca_label'])}",
                                 fontsize=10, loc="left")
                ax.grid(alpha=0.2)
        source_notes = "\n".join(
            f"{r['origin']}: {r['source_file']} · row {r['source_row']}" for r, _ in examples)
        fig.suptitle("Representative unchanged observations: 128 samples at 10 Hz", fontsize=13)
        fig.text(0.02, 0.045, source_notes, fontsize=7)
        fig.tight_layout(rect=(0, 0.12, 1, 0.96))
        # Keep the source notes inside the final layout.
        fig.text(0.5, 0.005, disclaimer, ha="center", fontsize=9, color="#555555")
        for extension in ("png", "svg"):
            path = output_dir / f"representative_kpis.{extension}"
            fig.savefig(path, dpi=120, bbox_inches="tight")
            paths[f"representative_kpis.{extension}"] = path
        plt.close(fig)
    return paths
