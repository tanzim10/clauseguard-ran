"""Rules 3-5: pick each event's representative window and measure baseline vs anomaly from raw metrics.csv."""
# ruff: noqa: SIM115, B023  (one-off build script: streaming file reads; per-row helpers used inside the loop)

import csv
import json
import statistics as st
import sys

EV, RAWROOT, OUT = sys.argv[1:4]
CH = [
    "RSRP",
    "UL_SNR",
    "DL_BLER",
    "UL_BLER",
    "DL_MCS",
    "UL_MCS",
    "PRB_Utilization_DL",
    "PRB_Utilization_UL",
    "PRBs_DL_Current",
    "PRBs_UL_Current",
    "UL_NPRB",
    "RX_Bytes",
    "TX_Bytes",
    "DL_NumberOfPackets",
    "UL_NumberOfPackets",
    "Estimated_UL_Buffer",
]
d = json.load(open(EV))
W = d["windows"]
cache = {}


def raw(source):
    if source not in cache:
        cache[source] = list(
            csv.DictReader(
                open(f"{RAWROOT}/{source.replace('processed/chunked.jsonl', 'raw/metrics.csv')}")
            )
        )
    return cache[source]


out = []
for e in d["events"]:
    cands = []
    for sid in e["windows"]:
        w = W[sid]
        pre = w["start"]
        an = w["end"] - w["start"] + 1
        if pre >= 16 and an >= 8:
            cands.append((-min(pre, an), w["raw_start_row"], sid))
    rec = {
        "event_id": e["event_id"],
        "split": e["split"],
        "label": e["label"],
        "source": e["source"],
        "group_id": e["group_id"],
        "event_raw_span": e["span"],
        "n_windows": len(e["windows"]),
    }
    if not cands:
        rec["rejected"] = "rule3: no window with >=16 pre rows and >=8 anomaly rows"
        out.append(rec)
        continue
    sid = min(cands)[2]
    w = W[sid]
    rows = raw(e["source"])
    b0 = w["raw_start_row"]

    def vals(c, i0, i1):
        return [float(rows[b0 - 1 + i][c]) for i in range(i0, i1 + 1)]

    mism = 0
    chans = {}
    for c in CH:
        win = w["channels"][c]
        rv = vals(c, 0, 127)
        mism += sum(1 for a, b in zip(win, rv) if a is None or abs(a - b) > 1e-9)
        b = rv[: w["start"]]
        a = rv[w["start"] : w["end"] + 1]
        bm, am, sd = st.mean(b), st.mean(a), st.pstdev(b)
        shift = am - bm
        changed = abs(shift) > 2 * sd and (abs(shift) > 0.05 * abs(bm) if bm else abs(shift) > 0)
        chans[c] = {
            "baseline": bm,
            "comparison": am,
            "baseline_std": sd,
            "n_base": len(b),
            "n_cmp": len(a),
            "shift": shift,
            "ratio": (am / bm if bm else None),
            "changed": changed,
        }
    t0 = rows[b0 - 1]["timestamp"]
    rec.update(
        scenario_id=sid,
        window_raw_rows=[b0, w["raw_end_row"]],
        window_start_time=w["start_time"],
        window_end_time=w["end_time"],
        start_index=w["start"],
        end_index=w["end"],
        baseline_interval=[rows[b0 - 1]["timestamp"], rows[b0 - 1 + w["start"] - 1]["timestamp"]],
        comparison_interval=[
            rows[b0 - 1 + w["start"]]["timestamp"],
            rows[b0 - 1 + w["end"]]["timestamp"],
        ],
        raw_vs_window_mismatches=mism,
        revision=w["revision"],
        source_sha256=w["source_sha256"],
        affected=w["affected"],
        channels=chans,
        changed=[c for c in CH if chans[c]["changed"]],
    )
    out.append(rec)
json.dump(out, open(OUT, "w"), indent=1)
for r in sorted(out, key=lambda r: (r["split"], r["label"])):
    if "rejected" in r:
        print(r["event_id"], r["split"], r["label"][:28], "REJECT", r["rejected"])
        continue
    ch = r["channels"]
    s = "; ".join(f"{c} {ch[c]['baseline']:.3g}->{ch[c]['comparison']:.3g}" for c in r["changed"])
    print(
        r["event_id"],
        r["split"],
        r["label"][:28],
        r["source"].split("/")[2] + "/" + r["source"].split("/")[3],
        f"pre{r['start_index']} an{r['end_index'] - r['start_index'] + 1}",
        "mism",
        r["raw_vs_window_mismatches"],
        "|",
        s,
    )
