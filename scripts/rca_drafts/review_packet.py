"""Render the human review packet for the draft RCA cases.

Run from the repo root: uv run python scripts/rca_drafts/review_packet.py <rows> <annotations> data/golden/rca_drafts/measured_events.json <out.md> <label=dir> [<label=dir> ...]
"""
# ruff: noqa: SIM115, ISC004  (one-off report script)

import json
import sys
from pathlib import Path

from clauseguard.corpus.manifest import load_manifest
from clauseguard.eval.annotations import load_annotations
from clauseguard.eval.golden import load_rca
from clauseguard.eval.rca_validation import pdf_passage_reader

ROWS, ANNS, MEAS, OUT, *REVIEWS = sys.argv[1:]
manifest = load_manifest("data/corpus/manifest.csv")
rows = load_rca(ROWS, known_spec_ids={r["spec_id"] for r in manifest if r.get("spec_id")})
anns = {a.id: a for a in load_annotations(ANNS, manifest)}
meas = {r["event_id"]: r for r in json.load(open(MEAS))}
read = pdf_passage_reader(manifest, Path("data/corpus/raw"))
reviewers = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in REVIEWS]


def tag(a, key):
    return next(t.split(":", 1)[1] for t in a.tags if t.startswith(key + ":"))


def review(directory, row_id):
    path = directory / f"{row_id}.json"
    return json.loads(path.read_text()) if path.exists() else None


def short(rec):
    if not rec:
        return "not run"
    if rec["status"] != "reviewed":
        return f"failed ({rec['error']})"
    v = rec["review"]
    return f"**{v['verdict']}** (evidence {v['evidence_supports_conclusion']}, mechanism {'ok' if v['has_adequate_mechanism_citation'] else 'weak'}, conf {v['confidence']})"


L = [
    "# RCA golden cases: human review packet",
    "",
    "Drafted by Claude from measured TelecomTS data; every KPI value re-computed by an independent script (all 192 matched); "
    "reviewed by models from other families through `python -m clauseguard.cli.validate_rca`. "
    "**A model verdict is an audit, not an approval. Only a person sets `approved`.**",
    "",
    "TelecomTS records carry no alarms or logs (`alarms` is empty in all 32,000), so each case's evidence is measured KPI statistics plus specification excerpts. The citations were proposed with AI help and machine-checked on their PDF pages; golden citations must be human-selected, so the reviewer confirms or replaces each one before approval.",
    "",
    "For each case: open the cited PDF page, confirm the passage says what the citation claims, check the conclusion against the KPI numbers, then mark the decision.",
    "",
    "| Case | Split | Dataset label | " + " | ".join(label for label, _ in reviewers) + " |",
    "|---|---|---|" + "---|" * len(reviewers),
]
for r in rows:
    L.append(
        f"| {r.id} | {anns[r.id].membership} | {r.source_labels['rca_label']} | "
        + " | ".join(short(review(d, r.id)) for _, d in reviewers)
        + " |"
    )
for r in rows:
    a = anns[r.id]
    m = meas[tag(a, "event")]
    L += [
        "",
        "---",
        "",
        f"## {r.id} ({a.membership}, {r.source_labels['rca_label']})",
        "",
        f"- **Scenario:** `{r.scenario_id}`",
        f"- **Source:** `{tag(a, 'source_file')}`, raw rows {tag(a, 'raw_rows')}, anomaly samples {tag(a, 'anomaly_index')} (event {tag(a, 'event')})",
        f"- **Recording family:** `{tag(a, 'recording_family')}` (unresolved, like all 33 recordings)",
        "- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)",
        f"- **Annotation families:** {', '.join(a.families)}",
        "",
        "**Question (what the model under test sees):**",
        "",
        f"> {r.question}",
        "",
        "**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):",
        "",
        "| Channel | Baseline | Comparison | Direction | Baseline std |",
        "|---|---|---|---|---|",
    ]
    for k in r.kpi_evidence:
        L.append(
            f"| {k.channel} | {k.baseline_value} | {k.comparison_value} | {k.direction} | {m['channels'][k.channel]['baseline_std']:.4g} |"
        )
    k0 = r.kpi_evidence[0]
    L += [
        "",
        f"Baseline {k0.baseline.start} to {k0.baseline.end}; comparison {k0.comparison.start} to {k0.comparison.end}.",
        "",
        f"**Expected conclusion:** {r.expected_conclusion}",
        "",
        "**Acceptable conclusions:** " + "; ".join(r.acceptable_conclusions),
        "",
        f"**Uncertainty:** {r.uncertainty}",
        "",
        "**Citations:**",
        "",
    ]
    locs = {(loc.spec_id, loc.clause): loc for g in a.evidence for loc in g.alternatives}
    for c in r.expected_citations:
        loc = locs[(c.spec_id, c.section.split()[0])]
        L += [
            f"- **{c.spec_id} §{c.section}** ({c.role}), version {loc.version}, PDF page {loc.pdf_page}",
            f"  - Why cited: {c.explanation}",
            f"  - Page text (start): “{read(loc)[:600]}…”",
        ]
    L += ["", "**Model reviews:**", ""]
    for label, d in reviewers:
        rec = review(d, r.id)
        L.append(f"- {label}: {short(rec)}")
        if rec and rec["status"] == "reviewed":
            v = rec["review"]
            if v.get("evidence_note"):
                L.append(f"  - Evidence note: {v['evidence_note']}")
            for x in v.get("unsupported_claims_in_conclusion", []):
                L.append(f"  - Unsupported claim flagged: {x}")
            for x in v.get("required_changes", []):
                L.append(f"  - Requested change: {x}")
    L += [
        "",
        "**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____",
    ]
Path(OUT).write_text("\n".join(L) + "\n")
print("wrote", OUT, len(L), "lines")
