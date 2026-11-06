"""Rules 5-7: signature agreement per category (ticket 'Symptoms' recipe), spec support, split balance."""
# ruff: noqa: SIM115  (one-off build script: streaming file reads; per-row helpers used inside the loop)

import collections
import json
import sys

M, OUT = sys.argv[1:3]
U, D = 1, -1
SIG = {  # designed directions from the ticket Symptoms recipe (research/telecomts.md §3b)
    "Co-Channel Interference (Mild)": {
        "RSRP": D,
        "UL_SNR": D,
        "DL_BLER": U,
        "UL_BLER": U,
        "RX_Bytes": D,
        "TX_Bytes": D,
        "PRB_Utilization_DL": U,
        "PRB_Utilization_UL": U,
        "UL_NPRB": U,
    },
    "Co-Channel Interference (Severe)": {
        "RSRP": D,
        "UL_SNR": D,
        "DL_BLER": U,
        "UL_BLER": U,
        "RX_Bytes": D,
        "TX_Bytes": D,
        "PRB_Utilization_DL": U,
        "PRB_Utilization_UL": U,
        "UL_NPRB": U,
        "DL_NumberOfPackets": D,
        "UL_NumberOfPackets": D,
    },
    "High Network Congestion (Gradual Buildup)": {
        "Estimated_UL_Buffer": U,
        "DL_BLER": U,
        "UL_BLER": U,
        "PRB_Utilization_DL": U,
        "PRB_Utilization_UL": U,
        "UL_NPRB": U,
        "RX_Bytes": D,
        "TX_Bytes": D,
        "DL_NumberOfPackets": U,
        "UL_NumberOfPackets": U,
    },
    "High Network Congestion (Sudden Spike)": {
        "Estimated_UL_Buffer": U,
        "DL_BLER": U,
        "UL_BLER": U,
        "PRB_Utilization_DL": U,
        "PRB_Utilization_UL": U,
        "UL_NPRB": U,
        "RX_Bytes": D,
        "TX_Bytes": D,
        "DL_NumberOfPackets": D,
        "UL_NumberOfPackets": D,
    },
    "Buffer Overflow (Gradual Buildup)": {
        "Estimated_UL_Buffer": U,
        "DL_BLER": U,
        "UL_BLER": U,
        "PRB_Utilization_DL": U,
        "PRB_Utilization_UL": U,
        "UL_NPRB": U,
        "RX_Bytes": D,
        "TX_Bytes": D,
        "DL_NumberOfPackets": U,
        "UL_NumberOfPackets": U,
    },
}
PRIMARY = {
    "Co-Channel Interference (Mild)": ["UL_SNR"],
    "Co-Channel Interference (Severe)": ["UL_SNR"],
    "High Network Congestion (Gradual Buildup)": ["PRB_Utilization_DL", "PRB_Utilization_UL"],
    "High Network Congestion (Sudden Spike)": ["Estimated_UL_Buffer", "UL_NPRB"],
    "Buffer Overflow (Gradual Buildup)": [
        "PRB_Utilization_DL",
        "PRB_Utilization_UL",
        "UL_NumberOfPackets",
    ],
}
NO_MECH = {
    "Antenna Failure": "rule6: no mechanism clause in corpus",
    "Faulty RF Filters (Temporal)": "rule6: no mechanism clause in corpus",
    "Doppler Shift (Severe)": "rule6: no mechanism clause in corpus",
    "Resource Allocation Bugs": "rule6: weak mechanism support (scheduler describes correct behaviour only)",
    "Faulty Handover Algorithm (Too Frequent)": "rule6: handover signature (0.1-0.3 Hz RSRP/SNR oscillation) not observable in span; single-gNB testbed",
}
rows = json.load(open(M))
passed, rejected = [], []
for r in rows:
    if "rejected" in r:
        rejected.append((r, r["rejected"]))
        continue
    if r["label"] in NO_MECH:
        rejected.append((r, NO_MECH[r["label"]]))
        continue
    sig = SIG[r["label"]]
    ch = r["channels"]
    COUNTS = {
        "Estimated_UL_Buffer",
        "DL_NumberOfPackets",
        "UL_NumberOfPackets",
        "RX_Bytes",
        "TX_Bytes",
    }
    changed = [
        c
        for c in r["changed"]
        if not (c.endswith("_BLER") and abs(ch[c]["shift"]) < 0.01)
        and not (c in COUNTS and abs(ch[c]["baseline"]) < 1 and abs(ch[c]["comparison"]) < 1)
    ]  # rule 5c
    r["changed_material"] = changed
    agree = [c for c in changed if c in sig and (ch[c]["shift"] > 0) == (sig[c] > 0)]
    contra = [c for c in changed if c in sig and (ch[c]["shift"] > 0) != (sig[c] > 0)]
    r["agree"], r["contra"] = agree, contra
    prim = PRIMARY[r["label"]]
    if not any(c in agree for c in prim):
        rejected.append((r, f"rule5b: primary symptom {prim} not observed; agree={agree}"))
        continue
    if len(agree) >= 2 and len(agree) > len(contra):
        passed.append(r)
    else:
        rejected.append((r, f"rule5: agree={agree} contra={contra}"))


def score(r):
    return (
        len(r["agree"]) - len(r["contra"]),
        max(
            abs(r["channels"][c]["shift"]) / (r["channels"][c]["baseline_std"] or 1e-9)
            for c in r["agree"]
        ),
    )


picked = []
for split in ("dev", "test"):
    pool = sorted(
        [r for r in passed if r["split"] == split],
        key=lambda r: (r["label"], [-x for x in score(r)], r["event_id"]),
    )
    chosen = []
    want = 6 if split == "dev" else 12 - len(picked)
    while len(chosen) < want and len(chosen) < len(pool):
        cnt = collections.Counter(r["label"] for r in chosen)
        fams = collections.defaultdict(set)
        for r in chosen:
            fams[r["label"]].add(r["group_id"])
        rest = [r for r in pool if r not in chosen]
        rest.sort(
            key=lambda r: (
                cnt[r["label"]],
                r["group_id"] in fams[r["label"]],
                [-x for x in score(r)],
                r["event_id"],
            )
        )
        chosen.append(rest[0])
    picked += chosen
json.dump(
    {
        "picked": [r["event_id"] for r in picked],
        "agree": {
            r["event_id"]: {
                "agree": r["agree"],
                "contra": r["contra"],
                "changed_material": r["changed_material"],
            }
            for r in passed
        },
        "passed": [r["event_id"] for r in passed],
        "rejected": [(r["event_id"], r["split"], r["label"], why) for r, why in rejected],
    },
    open(OUT, "w"),
    indent=1,
)
for r in picked:
    print(
        "PICK",
        r["event_id"],
        r["split"],
        r["label"],
        r["source"].split("/")[2:4],
        "agree",
        r["agree"],
        "contra",
        r["contra"],
    )
print("passed not picked:", [r["event_id"] for r in passed if r not in picked])
for r, why in rejected:
    print("REJ", r["event_id"], r["split"], r["label"][:30], why)
