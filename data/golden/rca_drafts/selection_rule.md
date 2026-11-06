# RCA case selection rule (fixed before measuring candidates; no model outputs consulted)

Source: exploratory release `data/scenarios/release` (f3bd4d46…), dev and test splits only. Train and excluded windows are never used.

1. **Unit = event.** Overlapping windows (stride 32, 128 rows) of one source file and one label are merged into one anomaly event by raw-row span. At most one case per event.
2. **Origin.** Synthetic events only. Real jamming is rejected: every window is labelled 0–127, no in-window baseline exists, and jammer-on and jammer-off rows show no measurable KPI difference (research/telecomts.md §5).
3. **Representative window.** Among the event's windows, keep those with ≥ 16 pre-anomaly rows (index < start_index) and ≥ 8 anomaly rows inside the window. Choose the one maximising min(pre rows, anomaly rows); ties go to the earliest raw row. Events with no such window are rejected (no same-timeline baseline).
4. **Evidence from source.** Baseline = the window's pre-anomaly rows; comparison = its anomaly rows (start_index..end_index). Values are recomputed from `data/telecomts/raw/<source>/raw/metrics.csv` (1-based data row = `raw_start_row` + index) and must equal the release window values.
5. **Measurable signature.** A channel counts as changed when |mean shift| > 2 × baseline std and > 5 % of |baseline mean| (or an absolute change when the baseline mean is 0). The event needs ≥ 2 changed channels that agree with the category's designed signature (ticket "Symptoms" recipe, research/telecomts.md §3b).
6. **Spec support.** The category must have at least a partial mechanism clause in the corpus (research/spec_clauses.md). Antenna Failure, Faulty RF Filters and Doppler have none and are rejected. Faulty Handover has a strong clause, but its signature (0.1–0.3 Hz oscillation) is not observable in its short spans and the testbed has one gNB, so it is rejected unless rule 5 passes on the handover-specific channels.
7. **Split balance.** Prefer 6 dev + 6 test. Within a split, prefer different categories and different source families before repeating one.
8. **Reported, not hidden.** Every rejected event is listed with the rule it failed.

## Amendment 5b (added 2026-11-04 after measuring candidates, before drafting or any model check)

Rule 5 counted any two agreeing channels, which let E24 (CCI Mild) pass on an RSRP drop plus a 5 % UL_NPRB rise while UL_SNR did not change. Interference lowers the signal-to-interference ratio (28.552 A.78), so a case without an SNR drop cannot support an interference conclusion. Each category's primary symptom must therefore be among the changed, agreeing channels:

| Category | Primary symptom required |
|---|---|
| Co-Channel Interference (Mild/Severe) | UL_SNR decrease |
| High Network Congestion (Gradual Buildup) | PRB_Utilization_DL or PRB_Utilization_UL increase |
| High Network Congestion (Sudden Spike) | Estimated_UL_Buffer or UL_NPRB increase |
| Buffer Overflow (Gradual Buildup) | PRB_Utilization_DL/UL or UL_NumberOfPackets increase (the buffer channel itself barely moves in this data, research/telecomts.md §3b, so these cases conclude a load surge; buffer overflow is not an accepted answer because the buffer channel falls in both selected windows) |

## Amendment 5c (added 2026-11-04 after reading the first drafts, before any model check)

BLER channels are zero-inflated in this data (research/telecomts.md §3c), so rule 5 counted moves such as 0 → 0.001 (0.1 %) as changes. That is far below operational significance (link adaptation typically targets around 10 % BLER). A DL_BLER or UL_BLER change now counts only when |shift| ≥ 0.01 (one percentage point). Effect: E05 (dev, Buffer Overflow) keeps only one agreeing channel (UL packets) and is rejected under rule 5. Dev then has five eligible events, so rule 7 yields 5 dev + 7 test; both splits still contain RCA cases.

Addition to 5c: a count or volume channel (Estimated_UL_Buffer, packet counts, byte counts) whose baseline mean is below 1 counts only if its comparison mean reaches at least 1, so near-zero moves such as an estimated buffer of 0 → 0.475 are not treated as evidence. No selection changed.

## Drafting note: absolute load level

PRB_Utilization_* is a percentage of PRBs (paper App. B). In the congestion and buffer-overflow events the injected rise starts from a near-idle base, so utilization stays in the low single digits. The drafts therefore conclude a relative traffic-load increase (a demand-side fault rather than a radio fault) and state that saturation is not shown; congestion and buffer overflow are listed as acceptable alternatives.
