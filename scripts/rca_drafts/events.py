"""Merge overlapping dev/test synthetic fault windows into distinct anomaly events (raw-row spans),
and dump each window's KPI arrays for the event's candidate windows."""
# ruff: noqa: SIM115  (one-off build script: streaming file reads; per-row helpers used inside the loop)

import collections
import csv
import json
import re
import sys

REL, SC, OUT = sys.argv[1:4]
mem = {
    r["scenario_id"]: r
    for r in csv.DictReader(open(f"{REL}/membership.csv"))
    if r["split"] in ("dev", "test") and r["origin"] == "synthetic"
}
src = {
    r["scenario_id"]: r
    for r in csv.DictReader(open(f"{REL}/window_sources.csv"))
    if r["scenario_id"] in mem
}
wins = []
with open(SC) as f:
    for line in f:
        m = re.search(r'"scenario_id":\s*"([^"]+)"', line[:300])
        if not m or m.group(1) not in mem:
            continue
        r = json.loads(line)
        sid = r["scenario_id"]
        a = r["anomaly"]
        w = src[sid]
        base = int(w["raw_start_row"])
        ch = {
            k: v["values"] for k, v in r["kpi_window"]["channels"].items() if v["kind"] == "numeric"
        }
        wins.append(
            {
                "scenario_id": sid,
                "split": mem[sid]["split"],
                "group_id": mem[sid]["group_id"],
                "source": mem[sid]["source_file"],
                "label": r["rca_label"],
                "start": a["start_index"],
                "end": a["end_index"],
                "raw_start_row": base,
                "raw_end_row": int(w["raw_end_row"]),
                "span": (base + a["start_index"], base + a["end_index"]),
                "start_time": r["kpi_window"]["start_time"],
                "end_time": r["kpi_window"]["end_time"],
                "rate": r["kpi_window"]["sampling_rate_hz"],
                "source_row": r["provenance"]["source_row"],
                "revision": r["provenance"]["revision"],
                "source_sha256": r["provenance"]["source_sha256"],
                "affected": a["affected_channels"],
                "channels": ch,
            }
        )
# merge spans per (source,label)
by = collections.defaultdict(list)
for w in wins:
    by[(w["source"], w["label"])].append(w)
events = []
for (s, lab), ws in by.items():
    ws.sort(key=lambda w: w["span"])
    cur = None
    for w in ws:
        if cur and w["span"][0] <= cur["span"][1] + 1:
            cur["span"] = (cur["span"][0], max(cur["span"][1], w["span"][1]))
            cur["windows"].append(w["scenario_id"])
        else:
            cur = {
                "source": s,
                "label": lab,
                "split": w["split"],
                "group_id": w["group_id"],
                "span": w["span"],
                "windows": [w["scenario_id"]],
            }
            events.append(cur)
for i, e in enumerate(
    sorted(events, key=lambda e: (e["split"], e["label"], e["source"], e["span"]))
):
    e["event_id"] = f"E{i + 1:02d}"
json.dump({"events": events, "windows": {w["scenario_id"]: w for w in wins}}, open(OUT, "w"))
c = collections.Counter((e["split"], e["label"]) for e in events)
print("windows", len(wins), "events", len(events))
for k in sorted(c):
    print(k, c[k])
