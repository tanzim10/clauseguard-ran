"""Build draft RcaRow + Annotation records for the 12 selected events (status=draft; a human approves)."""
# ruff: noqa: SIM115  (one-off build script: streaming file reads; per-row helpers used inside the loop)

import csv
import json
import sys

M, SEL, MAN, OUT_ROWS, OUT_ANN = sys.argv[1:6]
man = {r["spec_id"]: r for r in csv.DictReader(open(MAN))}


def loc(spec, clause, title, page):
    m = man[spec]
    return {
        "spec_id": spec,
        "version": m["version"],
        "sha256": m["sha256"],
        "clause": clause,
        "title": title,
        "pdf_page": page,
    }


C = {  # (spec, clause, title, page, role, explanation, claim)
    "sinr": (
        "3GPP TS 28.552",
        "A.78",
        "Monitoring of SS-SINR",
        352,
        "definition",
        "Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.",
        "SS-SINR is the ratio of the received signal level to the sum of interference and noise.",
    ),
    "ulbler": (
        "3GPP TS 28.552",
        "5.1.1.7.7",
        "Error number of UL initial TBs",
        92,
        "definition",
        "Defines the uplink transport-block error count behind the UL_BLER channel.",
        "The measurement counts initial faulty uplink transport blocks in a cell.",
    ),
    "rf": (
        "3GPP TS 32.425",
        "A.9",
        "Monitoring of RF performance",
        105,
        "mechanism",
        "States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).",
        "Uplink interference disturbs the base station receiver and leads to receiving code errors and degraded service.",
    ),
    "blermcs": (
        "3GPP TS 28.552",
        "A.18",
        "Monitor of BLER performance",
        337,
        "mechanism",
        "Links the transport-block error rate to MCS selection and user throughput, explaining why higher BLER coincides with lower MCS or throughput (informative annex).",
        "The TB error rate reflects BLER and influences MCS selection and user throughput.",
    ),
    "prbdl": (
        "3GPP TS 28.552",
        "5.1.1.2.1",
        "DL Total PRB Usage",
        38,
        "definition",
        "Defines downlink PRB usage, the measured counterpart of PRB_Utilization_DL.",
        "DL Total PRB Usage is the percentage of downlink physical resource blocks used for any purpose.",
    ),
    "prbul": (
        "3GPP TS 28.552",
        "5.1.1.2.2",
        "UL Total PRB Usage",
        38,
        "definition",
        "Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.",
        "UL Total PRB Usage is the percentage of uplink physical resource blocks used for any purpose.",
    ),
    "bsr": (
        "3GPP TS 38.300",
        "10.4",
        "Measurements to Support Scheduler Operation",
        121,
        "definition",
        "Defines uplink buffer status reports, the concept behind the Estimated_UL_Buffer channel (no PM counter for UE buffer size exists in the corpus).",
        "Uplink buffer status reports describe the data buffered in the UE for a logical channel group.",
    ),
    "prbulmean": (
        "3GPP TS 28.552",
        "5.1.1.2.7",
        "Mean UL PRB used for data traffic",
        41,
        "definition",
        "Defines the mean number of uplink PRBs used for data traffic. UL_NPRB counts the uplink PRBs allocated to the UE per TTI, so this is the nearest spec measurement, not an exact match.",
        "The measurement gives the average number of PRBs used in uplink for data traffic.",
    ),
    "sched": (
        "3GPP TS 38.300",
        "10.1",
        "Basic Scheduler Operation",
        119,
        "mechanism",
        "The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).",
        "Schedulers assign resources between UEs taking into account UE buffer status and QoS requirements.",
    ),
    "mlb": (
        "3GPP TS 38.300",
        "15.5.1.1",
        "General",
        140,
        "mechanism",
        "Describes congested cells and load balancing as the remedy, supporting load (not radio quality) as the fault class.",
        "Mobility load balancing distributes load among cells or transfers traffic from a congested cell.",
    ),
    "load": (
        "3GPP TS 38.300",
        "15.5.1.2",
        "Load reporting for intra-RAT and intra-system inter-RAT load balancing",
        140,
        "mechanism",
        "Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.",
        "Radio resource usage, including per-cell DL/UL total PRB usage, is reported as load information for load balancing.",
    ),
}
CAT_NOTE = (
    "Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. "
    "PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale."
)
CAT = {
    "cci": {
        "cites": ["sinr", "ulbler", "rf", "blermcs"],
        "conclusion": "Uplink radio-quality degradation consistent with co-channel interference: the uplink SNR falls sharply while traffic demand does not rise, so the receiver sees a worse signal-to-interference ratio and decodes more blocks in error.",
        "acceptable": [
            "Uplink interference on the serving channel lowering the received SINR",
            "Co-channel interference from another transmitter on the same frequency",
            "Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)",
        ],
        "unc": "Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference.",
    },
    "congG": {
        "cites": ["prbdl", "prbul", "sched", "load"],
        "conclusion": "Traffic load increase saturating radio resources (cell congestion): DL and UL PRB usage and packet counts rise sharply while the radio quality indicators stay near baseline.",
        "acceptable": [
            "Network congestion building up from rising traffic",
            "Traffic surge driving PRB utilization up",
            "Load-related degradation rather than a radio-quality fault",
        ],
        "unc": "Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 describes PRB usage as the load indicator and load balancing as the remedy but does not state that load degrades per-UE throughput. A buffer-overflow fault produces a similar load signature in this data,. Absolute PRB utilization stays low, so this is a relative load rise, not a saturated cell.",
    },
    "congS": {
        "cites": ["bsr", "ulbler", "sched", "mlb"],
        "conclusion": "Uplink demand exceeding the granted resources (congestion): the estimated UL buffer backlog grows several-fold and uplink block errors rise, while signal strength does not explain the change.",
        "acceptable": [
            "Sudden uplink congestion",
            "Uplink buffer backlog from demand exceeding scheduled resources",
            "Uplink buffer build-up / overflow at the UE",
        ],
        "unc": "Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 10.1 says the scheduler allocates by buffer status but does not state that unmet demand causes backlog (inferred). PRB usage and the uplink PRB allocation (UL_NPRB) rise far less than the buffer and stay within the noise of the baseline (shift below 2x baseline std), so congestion vs. a scheduling fault is not fully separable from KPIs; BSR/MAC detail (TS 38.321) is not in the corpus.",
    },
    "buf": {
        "cites": ["prbul", "bsr", "sched", "load"],
        "conclusion": "Traffic load surge: packet counts and PRB usage rise well above baseline while radio quality stays stable, pointing to a demand/queuing problem rather than a radio fault.",
        "acceptable": [
            "Network congestion from increased traffic load",
            "Traffic surge raising resource usage and packet counts",
            "Load-related degradation rather than a radio-quality fault",
        ],
        "unc": "Synthetic injection; the label (buffer overflow) is a dataset claim. The estimated uplink buffer does not rise in this window ({buf}), so the evidence supports a load surge but does not support an overflow; buffer overflow is therefore not an acceptable conclusion here, and congestion is equally consistent. Spec support is partial (scheduler and BSR concepts in 38.300; MAC/RLC buffer and discard behaviour live in TS 38.321/38.322, not in the corpus).",
    },
}
LAB2CAT = {
    "Co-Channel Interference (Mild)": "cci",
    "Co-Channel Interference (Severe)": "cci",
    "High Network Congestion (Gradual Buildup)": "congG",
    "High Network Congestion (Sudden Spike)": "congS",
    "Buffer Overflow (Gradual Buildup)": "buf",
}
NAMES = {
    "RSRP": "RSRP",
    "UL_SNR": "uplink SNR",
    "DL_BLER": "downlink BLER",
    "UL_BLER": "uplink BLER",
    "UL_MCS": "uplink MCS",
    "PRB_Utilization_DL": "downlink PRB utilization",
    "PRB_Utilization_UL": "uplink PRB utilization",
    "UL_NPRB": "uplink PRB count (UL_NPRB)",
    "RX_Bytes": "uplink bytes (RX_Bytes)",
    "TX_Bytes": "downlink bytes (TX_Bytes)",
    "DL_NumberOfPackets": "downlink packet count",
    "UL_NumberOfPackets": "uplink packet count",
    "Estimated_UL_Buffer": "estimated uplink buffer",
}


def fmt(x):
    return float(f"{x:.4g}")


def human(x):
    a = abs(x)
    return f"{x:,.0f}" if a >= 100 else f"{x:.1f}" if a >= 1 else f"{x:.3f}" if a >= 0.001 else "0"


LOAD = ["PRB_Utilization_DL", "PRB_Utilization_UL", "DL_NumberOfPackets", "UL_NumberOfPackets"]
RADIO = ["RSRP", "UL_SNR"]


def stable(r, chans):
    mat = set(r.get("changed_material", r["changed"]))
    return [c for c in chans if c not in mat]


def observed(r, agree):
    ch = r["channels"]
    return "; ".join(
        f"{NAMES[c]} {human(ch[c]['baseline'])} -> {human(ch[c]['comparison'])}" for c in agree
    )


CONTEXT = {
    "cci": ["PRB_Utilization_DL", "PRB_Utilization_UL", "DL_NumberOfPackets", "UL_NumberOfPackets"],
    "congG": ["RSRP", "UL_SNR"],
    "buf": ["RSRP", "UL_SNR"],
    "congS": ["UL_NPRB", "PRB_Utilization_UL", "RSRP", "UL_SNR"],
}


def is_stable(r, c):
    x = r["channels"][c]
    b, a = x["baseline"], x["comparison"]
    rel_ok = abs(a - b) < 0.2 * abs(b) if abs(b) >= 1 else abs(a - b) < 0.2
    return c not in r["changed"] and rel_ok


def context(cat, r):
    return [c for c in CONTEXT[cat] if is_stable(r, c)]


def unstable_context(cat, r):
    return [c for c in CONTEXT[cat] if not is_stable(r, c) and c not in r["changed"]]


def conclusion(cat, r, agree):
    obs = observed(r, agree)
    cx = context(cat, r)
    within = "; ".join(
        f"{NAMES[c]} {human(r['channels'][c]['baseline'])} -> {human(r['channels'][c]['comparison'])}"
        for c in cx
    )
    if cat == "cci":
        ctx = f" Roughly flat: {within}." if cx else ""
        d = r["channels"]["UL_SNR"]["baseline"] - r["channels"]["UL_SNR"]["comparison"]
        return (
            f"Uplink radio-quality degradation: the uplink SNR falls by about {d:.1f} dB together with RSRP. The cited mechanism supports uplink "
            "(co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, "
            f"so the KPIs alone cannot separate the two. Observed: {obs}.{ctx}"
        )
    ctx = (
        f" Roughly flat: {within}, so the radio link itself does not explain the change."
        if cx
        else ""
    )
    prb = [r["channels"][c]["comparison"] for c in ("PRB_Utilization_DL", "PRB_Utilization_UL")]
    lvl = f" PRB utilization peaks near {human(max(prb))} % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown."
    if cat == "congG":
        return f"A traffic-load increase consistent with congestion building up: resource usage and traffic rise sharply relative to baseline. Observed: {obs}.{ctx}{lvl}"
    if cat == "congS":
        g = f" Roughly flat: {within}." if cx else ""
        ch = r["channels"]
        bx = ch["Estimated_UL_Buffer"]["comparison"] / ch["Estimated_UL_Buffer"]["baseline"]
        gx = ch["UL_NPRB"]["comparison"] / ch["UL_NPRB"]["baseline"]
        return (
            f"Uplink backlog consistent with congestion: the estimated uplink buffer grows about {bx:.1f}x while the uplink PRB allocation (UL_NPRB) grows only "
            f"about {gx:.1f}x, so offered uplink traffic outpaces what is being scheduled. Observed: {obs}; UL_NPRB "
            f"{human(ch['UL_NPRB']['baseline'])} -> {human(ch['UL_NPRB']['comparison'])}.{g} Uplink BLER also rises; the cited clauses do not explain that part."
        )
    return f"A traffic-load surge rather than a radio fault: resource usage and packet counts rise well above baseline. Observed: {obs}.{ctx}{lvl}"


meas = {r["event_id"]: r for r in json.load(open(M))}
S = json.load(open(SEL))
sel = S["picked"]
rows, anns = [], []
for i, ev in enumerate(sorted(sel, key=lambda e: (meas[e]["split"] != "dev", e)), start=1):
    r = meas[ev]
    cat = LAB2CAT[r["label"]]
    T = CAT[cat]
    ch = r["channels"]
    agree = S["agree"][ev]["agree"]
    if cat == "cci":
        agree = sorted(agree, key=lambda c: c != "UL_SNR")
    ke = []
    extra = (
        ["UL_NPRB"]
        if cat == "congS" and "UL_NPRB" not in context(cat, r) and "UL_NPRB" not in agree
        else []
    )
    if cat == "buf" and "Estimated_UL_Buffer" not in agree:
        extra = ["Estimated_UL_Buffer"]
    for c in agree + extra + context(cat, r):
        b, a = fmt(ch[c]["baseline"]), fmt(ch[c]["comparison"])
        ke.append(
            {
                "channel": c,
                "unit": None,
                "baseline": {"start": r["baseline_interval"][0], "end": r["baseline_interval"][1]},
                "comparison": {
                    "start": r["comparison_interval"][0],
                    "end": r["comparison_interval"][1],
                },
                "baseline_value": b,
                "comparison_value": a,
                "direction": "increase" if a > b else "decrease" if a < b else "unchanged",
            }
        )
    onset = r["start_index"] / 10
    parts = [
        f"{NAMES[c]} went from about {human(ch[c]['baseline'])} to about {human(ch[c]['comparison'])}"
        for c in agree
    ]
    q = (
        f"In a 12.8-second KPI window from a single 5G NR cell, a change starts about {onset:.1f} s in: "
        + "; ".join(parts)
        + ". What is the most likely root cause, and which specification clauses support that explanation?"
    )
    rid = f"rca_{i:03d}"
    keys = list(T["cites"])
    if cat == "cci" and "UL_BLER" not in agree:
        keys = [k for k in keys if k not in ("ulbler", "blermcs")]
    if cat == "buf":
        keys = [k for k in keys if k != "bsr"]
        if "PRB_Utilization_DL" in agree:
            keys.insert(0, "prbdl")
    if cat == "congG" and "PRB_Utilization_DL" not in agree:
        keys = [k for k in keys if k != "prbdl"]
    if cat == "congS":
        keys.insert(2, "prbulmean")
    cites = [
        {
            "spec_id": C[k][0],
            "section": f"{C[k][1]} {C[k][2]}",
            "role": C[k][4],
            "explanation": C[k][5],
        }
        for k in keys
    ]
    r["changed_material"] = S["agree"][ev].get("changed_material", r["changed"])
    rows.append(
        {
            "id": rid,
            "type": "rca",
            "scenario_id": r["scenario_id"],
            "origin": "synthetic",
            "inputs": "kpi_only",
            "question": q,
            "kpi_evidence": ke,
            "source_labels": {"rca_label": r["label"], "fault_label": "abnormal"},
            "expected_conclusion": conclusion(cat, r, agree),
            "acceptable_conclusions": T["acceptable"],
            "uncertainty": (
                T["unc"]
                + (
                    " Other channels that also changed but are not used in this conclusion: "
                    + ", ".join(
                        f"{c} {human(ch[c]['baseline'])} -> {human(ch[c]['comparison'])}"
                        for c in r["changed_material"]
                        if c not in agree
                    )
                    + "."
                    if [c for c in r["changed_material"] if c not in agree]
                    else ""
                )
                + (
                    " Channels with a large relative move that is not significant at 2x baseline std: "
                    + ", ".join(
                        f"{c} {human(ch[c]['baseline'])} -> {human(ch[c]['comparison'])}"
                        for c in unstable_context(cat, r)
                        if c not in extra
                    )
                    + "."
                    if [c for c in unstable_context(cat, r) if c not in extra]
                    else ""
                )
                + " "
                + CAT_NOTE
            ).replace(
                "{buf}",
                f"{human(ch['Estimated_UL_Buffer']['baseline'])} -> {human(ch['Estimated_UL_Buffer']['comparison'])}",
            ),
            "expected_citations": cites,
        }
    )
    weak = cat in ("congS", "buf") or r["label"].endswith("(Mild)")
    anns.append(
        {
            "id": rid,
            "status": "draft",
            "membership": r["split"],
            "families": ["multi_clause"] + (["difficult"] if weak else []),
            "tags": [
                f"event:{ev}",
                f"recording_family:{r['group_id']}",
                f"source_file:{r['source']}",
                f"source_revision:{r['revision']}",
                f"source_sha256:{r['source_sha256']}",
                f"raw_rows:{r['window_raw_rows'][0]}-{r['window_raw_rows'][1]}",
                f"anomaly_index:{r['start_index']}-{r['end_index']}",
                "inputs:kpi_only (no approved text; tickets leak the label)",
                "selection_rule:data/golden/rca_drafts/selection_rule.md",
            ],
            "evidence": [
                {"claim": C[k][6], "alternatives": [loc(C[k][0], C[k][1], C[k][2], C[k][3])]}
                for k in keys
            ],
        }
    )
with open(OUT_ROWS, "w") as f:
    f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in rows)
with open(OUT_ANN, "w") as f:
    f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in anns)
for x in rows:
    print(x["id"], x["scenario_id"][-8:], x["source_labels"]["rca_label"], "|", x["question"][:200])
