# RCA golden cases: human review packet

Drafted by Claude from measured TelecomTS data; every KPI value re-computed by an independent script (all 192 matched); reviewed by models from other families through `python -m clauseguard.cli.validate_rca`. **A model verdict is an audit, not an approval. Only a person sets `approved`.**

TelecomTS records carry no alarms or logs (`alarms` is empty in all 32,000), so each case's evidence is measured KPI statistics plus specification excerpts. The citations were proposed with AI help and machine-checked on their PDF pages; golden citations must be human-selected, so the reviewer confirms or replaces each one before approval.

For each case: open the cited PDF page, confirm the passage says what the citation claims, check the conclusion against the KPI numbers, then mark the decision.

**Reading the model verdicts.** gpt-oss-120b and qwen3.8-27b reviewed the final text of all 12 cases. gemini-3.6-flash reviewed an earlier draft (before three small wording fixes) and hit its free daily quota on two cases. gpt-oss-120b rates rca_001 invalid because it reads the dataset label "Co-Channel Interference (Severe)" as the case's claim; the expected conclusion already says interference and a received-power loss cannot be separated from these KPIs, so this needs a human call. Its other recurring request, an NR-specific mechanism clause for interference, cannot be met from the current corpus.

| Case | Split | Dataset label | gpt-oss-120b (OpenRouter) | qwen3.8-27b (Groq) | gemini-3.6-flash (earlier draft, 10 of 12) |
|---|---|---|---|---|---|
| rca_001 | dev | Co-Channel Interference (Severe) | **invalid** (evidence partial, mechanism weak, conf 0.78) | **valid** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 0.95) |
| rca_002 | dev | Co-Channel Interference (Severe) | **valid_with_changes** (evidence partial, mechanism weak, conf 0.86) | **valid** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 1.0) |
| rca_003 | dev | High Network Congestion (Gradual Buildup) | **valid_with_changes** (evidence yes, mechanism ok, conf 0.93) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.9) | **valid** (evidence yes, mechanism ok, conf 0.95) |
| rca_004 | dev | High Network Congestion (Sudden Spike) | **valid_with_changes** (evidence yes, mechanism weak, conf 0.78) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 0.95) |
| rca_005 | dev | High Network Congestion (Sudden Spike) | **valid_with_changes** (evidence yes, mechanism weak, conf 0.78) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 0.95) |
| rca_006 | test | Buffer Overflow (Gradual Buildup) | **valid_with_changes** (evidence yes, mechanism ok, conf 0.93) | **valid** (evidence yes, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 1.0) |
| rca_007 | test | Buffer Overflow (Gradual Buildup) | **valid_with_changes** (evidence yes, mechanism ok, conf 0.93) | **valid** (evidence yes, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 0.95) |
| rca_008 | test | Co-Channel Interference (Mild) | **valid_with_changes** (evidence partial, mechanism weak, conf 0.86) | **valid** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 1.0) |
| rca_009 | test | Co-Channel Interference (Severe) | **valid_with_changes** (evidence partial, mechanism weak, conf 0.78) | **valid** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 1.0) |
| rca_010 | test | Co-Channel Interference (Severe) | **valid_with_changes** (evidence partial, mechanism weak, conf 0.78) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.95) | **valid** (evidence yes, mechanism ok, conf 1.0) |
| rca_011 | test | High Network Congestion (Gradual Buildup) | **valid** (evidence yes, mechanism ok, conf 0.93) | **valid** (evidence yes, mechanism ok, conf 0.95) | failed (HTTP 429) |
| rca_012 | test | High Network Congestion (Gradual Buildup) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.78) | **valid_with_changes** (evidence partial, mechanism ok, conf 0.9) | failed (HTTP 429) |

---

## rca_001 (dev, Co-Channel Interference (Severe))

- **Scenario:** `telecomts-61a3b56949b3d19375eb5d474a8728cc3640aa3e4c32f05bf5fa813ec7bc1af9`
- **Source:** `anomalous/synthetic/Zone_A/Twitch/processed/chunked.jsonl`, raw rows 5249-5376, anomaly samples 67-127 (event E09)
- **Recording family:** `family-6aaa4378e87449bd2ac2d04e5cc5e62dffdfdc9050806e23083ee4640185a981` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 6.7 s in: uplink SNR went from about 21.4 to about 6.4; RSRP went from about -87.8 to about -98.1; uplink BLER went from about 0.087 to about 0.189. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_SNR | 21.37 | 6.422 | decrease | 0.5376 |
| RSRP | -87.81 | -98.11 | decrease | 0.3955 |
| UL_BLER | 0.08657 | 0.1892 | increase | 0.03312 |

Baseline 2026-09-02 00:08:44.800 to 2026-09-02 00:08:51.400; comparison 2026-09-02 00:08:51.500 to 2026-09-02 00:08:57.500.

**Expected conclusion:** Uplink radio-quality degradation: the uplink SNR falls by about 14.9 dB together with RSRP. The cited mechanism supports uplink (co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, so the KPIs alone cannot separate the two. Observed: uplink SNR 21.4 -> 6.4; RSRP -87.8 -> -98.1; uplink BLER 0.087 -> 0.189.

**Acceptable conclusions:** Uplink interference on the serving channel lowering the received SINR; Co-channel interference from another transmitter on the same frequency; Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)

**Uncertainty:** Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference. Channels with a large relative move that is not significant at 2x baseline std: PRB_Utilization_DL 6.6 -> 2.2, PRB_Utilization_UL 3.5 -> 0.630, DL_NumberOfPackets 49.6 -> 2.4, UL_NumberOfPackets 54.6 -> 1.6. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §A.78 Monitoring of SS-SINR** (definition), version 18.7.0, PDF page 352
  - Why cited: Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3513GPP TS 28.552 version 18.7.0 Release 18 A.76 Monitoring of subscriber profile sizes in UDM The subscriber profiles maintained by UDM have two parts - static and dynamic data. Dynamic part of the subscriber profile grows over time (e.g. due to roaming conditions, dynamic dereferencing of provisioning data, addition/enabling of new features). The uncontrolled and unnoticed growth of subscriber profiles may result in degraded system performance and lead to service impacts. Various corrective actions varying from subscriber profile optimization (clean-up/c…”
- **3GPP TS 28.552 §5.1.1.7.7 Error number of UL initial TBs** (definition), version 18.7.0, PDF page 92
  - Why cited: Defines the uplink transport-block error count behind the UL_BLER channel.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)913GPP TS 28.552 version 18.7.0 Release 18 d) Each measurement is an integer. e) TB.ErrTotNbrDl.X. Where X identified by DL MU-MIMO maximum layer. f) NRCellDU. g) Valid for packet switched traffic . h) 5GS. 5.1.1.7.5 Residual error number of DL TBs a) This measurement provides the number of final faulty TBs transmitted on the downlink in a cell at last HARQ re-transmissions. b) CC. c) On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at the last HARQ feedback during the period of measurement. d) A single integer …”
- **3GPP TS 32.425 §A.9 Monitoring of RF performance** (mechanism), version 18.0.0, PDF page 105
  - Why cited: States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).
  - Page text (start): “ETSI ETSI TS 132 425 V18.0.0 (2024-05)1043GPP TS 32.425 version 18.0.0 Release 18 Monitoring interference, both uplink and downlink, is an important aspect of monitoring RF performance. For this purpose, monitoring of UL interference power is useful. Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference. By efficient frequency planning and careful Selection of base station locations, the impact of interference signal can be effectively contr…”
- **3GPP TS 28.552 §A.18 Monitor of BLER performance** (mechanism), version 18.7.0, PDF page 337
  - Why cited: Links the transport-block error rate to MCS selection and user throughput, explaining why higher BLER coincides with lower MCS or throughput (informative annex).
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3363GPP TS 28.552 version 18.7.0 Release 18 A.18 Monitor of BLER performance The TB Error Rate in UL/DL can directly reflect the BLER, and has an influence on MCS selection and user throughput. It can be helpful to estimate the performance of radio resource management like radio resource schedulein transport layer and be helpful in trouble shooting. Furthermore, they should be taken into account to optimize the system performance. To obtain TB Error Rate by calculating, the number of total and error TBs transmitted in a cell should be monitored. A.19 Monit…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **invalid** (evidence partial, mechanism weak, conf 0.78)
  - Evidence note: The KPI trends clearly show uplink radio‑quality degradation, but the evidence cannot uniquely distinguish between co‑channel interference and a loss of received power.
  - Unsupported claim flagged: The definitive attribution to "Co‑Channel Interference (Severe)" is not fully supported by the KPI evidence, which also matches a received‑power loss scenario.
  - Requested change: Remove the definitive statement that the fault is "Co‑Channel Interference (Severe)"; instead state that the observed KPI pattern is consistent with either uplink interference or a loss of received power.
  - Requested change: Add a note that the current citations do not provide an NR‑specific mechanism linking uplink interference to the UL_SNR KPI.
  - Requested change: If possible, cite an NR‑specific clause (e.g., 3GPP TS 38.331 or 38.133) that defines UL SINR/SNR or describes interference impact on uplink radio quality.
- qwen3.8-27b (Groq): **valid** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a simultaneous drop in UL SNR and RSRP with an increase in BLER. While this is consistent with interference, the drop in RSRP (received signal power) is not explained by interference alone (which typically affects SINR/SNR but not the desired signal power RSRP). The conclusion correctly identifies this ambiguity, noting that a received-power loss (antenna/RF/coverage) is an equally consistent explanation. Therefore, the evidence supports the *phenomenon* described in the conclusion, but does not uniquely support the 'Co-Channel Interference' label as the sole root cause.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence directly supports the numerical shifts and direction of degradation detailed in the conclusion (UL_SNR dropping ~15 dB, RSRP dropping ~10.3 dB, UL_BLER doubling).

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_002 (dev, Co-Channel Interference (Severe))

- **Scenario:** `telecomts-0ef60c5b2a02754e4e9af4a1f44a63aad5a76f53a9ea63f4d5ef48a9a4dc4e56`
- **Source:** `anomalous/synthetic/Zone_C/Twitch/processed/chunked.jsonl`, raw rows 5313-5440, anomaly samples 73-127 (event E10)
- **Recording family:** `family-128a65d178ae86f23b074e237b0b7ac42fc3d9c9a7d374b107f69069e9a1d710` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 7.3 s in: uplink SNR went from about 20.6 to about 9.3; RSRP went from about -117 to about -127; uplink BLER went from about 0.098 to about 0.177. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_SNR | 20.59 | 9.33 | decrease | 0.8121 |
| RSRP | -117.1 | -126.5 | decrease | 0.2276 |
| UL_BLER | 0.09781 | 0.1767 | increase | 0.03024 |

Baseline 2037-04-08 00:08:51.200 to 2037-04-08 00:08:58.400; comparison 2037-04-08 00:08:58.500 to 2037-04-08 00:09:03.900.

**Expected conclusion:** Uplink radio-quality degradation: the uplink SNR falls by about 11.3 dB together with RSRP. The cited mechanism supports uplink (co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, so the KPIs alone cannot separate the two. Observed: uplink SNR 20.6 -> 9.3; RSRP -117 -> -127; uplink BLER 0.098 -> 0.177.

**Acceptable conclusions:** Uplink interference on the serving channel lowering the received SINR; Co-channel interference from another transmitter on the same frequency; Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)

**Uncertainty:** Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference. Channels with a large relative move that is not significant at 2x baseline std: PRB_Utilization_DL 15.7 -> 20.5, PRB_Utilization_UL 0.916 -> 1.3, DL_NumberOfPackets 59.2 -> 33.1, UL_NumberOfPackets 22.8 -> 8.5. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §A.78 Monitoring of SS-SINR** (definition), version 18.7.0, PDF page 352
  - Why cited: Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3513GPP TS 28.552 version 18.7.0 Release 18 A.76 Monitoring of subscriber profile sizes in UDM The subscriber profiles maintained by UDM have two parts - static and dynamic data. Dynamic part of the subscriber profile grows over time (e.g. due to roaming conditions, dynamic dereferencing of provisioning data, addition/enabling of new features). The uncontrolled and unnoticed growth of subscriber profiles may result in degraded system performance and lead to service impacts. Various corrective actions varying from subscriber profile optimization (clean-up/c…”
- **3GPP TS 28.552 §5.1.1.7.7 Error number of UL initial TBs** (definition), version 18.7.0, PDF page 92
  - Why cited: Defines the uplink transport-block error count behind the UL_BLER channel.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)913GPP TS 28.552 version 18.7.0 Release 18 d) Each measurement is an integer. e) TB.ErrTotNbrDl.X. Where X identified by DL MU-MIMO maximum layer. f) NRCellDU. g) Valid for packet switched traffic . h) 5GS. 5.1.1.7.5 Residual error number of DL TBs a) This measurement provides the number of final faulty TBs transmitted on the downlink in a cell at last HARQ re-transmissions. b) CC. c) On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at the last HARQ feedback during the period of measurement. d) A single integer …”
- **3GPP TS 32.425 §A.9 Monitoring of RF performance** (mechanism), version 18.0.0, PDF page 105
  - Why cited: States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).
  - Page text (start): “ETSI ETSI TS 132 425 V18.0.0 (2024-05)1043GPP TS 32.425 version 18.0.0 Release 18 Monitoring interference, both uplink and downlink, is an important aspect of monitoring RF performance. For this purpose, monitoring of UL interference power is useful. Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference. By efficient frequency planning and careful Selection of base station locations, the impact of interference signal can be effectively contr…”
- **3GPP TS 28.552 §A.18 Monitor of BLER performance** (mechanism), version 18.7.0, PDF page 337
  - Why cited: Links the transport-block error rate to MCS selection and user throughput, explaining why higher BLER coincides with lower MCS or throughput (informative annex).
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3363GPP TS 28.552 version 18.7.0 Release 18 A.18 Monitor of BLER performance The TB Error Rate in UL/DL can directly reflect the BLER, and has an influence on MCS selection and user throughput. It can be helpful to estimate the performance of radio resource management like radio resource schedulein transport layer and be helpful in trouble shooting. Furthermore, they should be taken into account to optimize the system performance. To obtain TB Error Rate by calculating, the number of total and error TBs transmitted in a cell should be monitored. A.19 Monit…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence partial, mechanism weak, conf 0.86)
  - Evidence note: The observed drops in UL_SNR and RSRP together with an increase in UL_BLER clearly indicate uplink radio‑quality degradation, but the KPI pattern is compatible with both co‑channel interference and a loss of received power; the evidence alone cannot distinguish them nor assess severity.
  - Unsupported claim flagged: Co-Channel Interference (Severe) as the definitive root cause
  - Requested change: Do not assert 'Severe' co‑channel interference as the definitive root cause; note that the KPI pattern is also consistent with a received‑power loss.
  - Requested change: Acknowledge that the mechanism citation (3GPP TS 32.425 A.9) is LTE‑specific and that no NR‑specific clause directly links uplink interference to UL_SNR, so the support is only partial.
- qwen3.8-27b (Groq): **valid** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a simultaneous drop in UL SNR and RSRP. While this is consistent with interference (lowering SNR), the drop in RSRP (signal strength) is not explained by interference alone; it suggests a loss of received power (e.g., antenna, cable, or coverage issue). The conclusion correctly identifies this ambiguity, stating that KPIs alone cannot separate the two. However, the dataset label claims 'Co-Channel Interference', which is only one of the possible explanations supported by the evidence.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 1.0)
  - Evidence note: The KPI evidence confirms UL_SNR decrease, RSRP decrease, and UL_BLER increase, matching the observations reported in the expected conclusion.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_003 (dev, High Network Congestion (Gradual Buildup))

- **Scenario:** `telecomts-8ac8201b86308141f6cd34339b2274de0e38edddb0bc749d447a0b2b38f26221`
- **Source:** `anomalous/synthetic/Zone_A/Twitch/processed/chunked.jsonl`, raw rows 4129-4256, anomaly samples 66-127 (event E14)
- **Recording family:** `family-6aaa4378e87449bd2ac2d04e5cc5e62dffdfdc9050806e23083ee4640185a981` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 6.6 s in: downlink PRB utilization went from about 1.2 to about 3.6; uplink PRB utilization went from about 0.152 to about 1.1; downlink packet count went from about 5.9 to about 20.8; uplink packet count went from about 2.8 to about 17.4. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| PRB_Utilization_DL | 1.205 | 3.611 | increase | 1.191 |
| PRB_Utilization_UL | 0.1519 | 1.076 | increase | 0.2178 |
| DL_NumberOfPackets | 5.879 | 20.8 | increase | 3.342 |
| UL_NumberOfPackets | 2.788 | 17.43 | increase | 1.237 |
| RSRP | -87.61 | -87.89 | decrease | 0.4858 |
| UL_SNR | 20.98 | 21.09 | increase | 0.8268 |

Baseline 2026-09-02 00:06:52.800 to 2026-09-02 00:06:59.300; comparison 2026-09-02 00:06:59.400 to 2026-09-02 00:07:05.500.

**Expected conclusion:** A traffic-load increase consistent with congestion building up: resource usage and traffic rise sharply relative to baseline. Observed: downlink PRB utilization 1.2 -> 3.6; uplink PRB utilization 0.152 -> 1.1; downlink packet count 5.9 -> 20.8; uplink packet count 2.8 -> 17.4. Roughly flat: RSRP -87.6 -> -87.9; uplink SNR 21.0 -> 21.1, so the radio link itself does not explain the change. PRB utilization peaks near 3.6 % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown.

**Acceptable conclusions:** Network congestion building up from rising traffic; Traffic surge driving PRB utilization up; Load-related degradation rather than a radio-quality fault

**Uncertainty:** Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 describes PRB usage as the load indicator and load balancing as the remedy but does not state that load degrades per-UE throughput. A buffer-overflow fault produces a similar load signature in this data,. Absolute PRB utilization stays low, so this is a relative load rise, not a saturated cell. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §5.1.1.2.1 DL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines downlink PRB usage, the measured counterpart of PRB_Utilization_DL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 28.552 §5.1.1.2.2 UL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.2 Load reporting for intra-RAT and intra-system inter-RAT load balancing** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence yes, mechanism ok, conf 0.93)
  - Evidence note: All KPI metrics show a clear increase in PRB utilization and packet counts while radio quality remains flat, indicating a traffic load increase.
  - Unsupported claim flagged: Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change.
  - Unsupported claim flagged: PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.
  - Requested change: Remove or qualify the unsupported methodological claim about the statistical thresholds for 'changed' and 'roughly flat'.
  - Requested change: Remove the unsupported statement about PRB_Utilization being a percent of 106 PRBs and the exact scaling relationship, unless a citation is added.
- qwen3.8-27b (Groq): **valid_with_changes** (evidence partial, mechanism ok, conf 0.9)
  - Evidence note: The KPI evidence shows a sharp increase in DL/UL PRB utilization and packet counts, while RSRP and UL SNR remain stable. This supports a load-related event. However, the absolute PRB utilization remains very low (3.6% DL, 1.1% UL), which contradicts the label 'High Network Congestion'. The conclusion correctly notes this is a 'relative load rise' and not saturation, but the dataset label 'High Network Congestion' is misleading given the low absolute values. The evidence supports 'traffic surge' or 'load increase' more than 'high congestion'.
  - Unsupported claim flagged: The dataset label 'High Network Congestion' is not fully supported by the low absolute PRB utilization values (3.6% and 1.1%), which indicate the cell is far from saturated.
  - Requested change: Clarify that the label 'High Network Congestion' is potentially misleading given the low absolute PRB utilization, and that 'Traffic Surge' or 'Load Increase' is a more accurate description based on the evidence.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The evidence shows a simultaneous increase in uplink and downlink PRB utilization and packet counts while channel quality metrics (RSRP and UL SNR) remain virtually unchanged, supporting a traffic load / congestion buildup conclusion.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_004 (dev, High Network Congestion (Sudden Spike))

- **Scenario:** `telecomts-b0685933f73c9be86af5b68aa71efe8f34487a9fc4a5dac1a84b67910b4a1283`
- **Source:** `anomalous/synthetic/Zone_A/Twitch/processed/chunked.jsonl`, raw rows 833-960, anomaly samples 57-127 (event E15)
- **Recording family:** `family-6aaa4378e87449bd2ac2d04e5cc5e62dffdfdc9050806e23083ee4640185a981` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause, difficult

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 5.7 s in: uplink BLER went from about 0.087 to about 0.174; estimated uplink buffer went from about 862 to about 3,944. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_BLER | 0.08688 | 0.1739 | increase | 0.02739 |
| Estimated_UL_Buffer | 861.6 | 3944.0 | increase | 643 |
| UL_NPRB | 5.544 | 10.96 | increase | 2.955 |
| RSRP | -88.0 | -87.75 | increase | 0 |
| UL_SNR | 21.69 | 21.73 | increase | 0.4589 |

Baseline 2026-09-02 00:01:23.200 to 2026-09-02 00:01:28.800; comparison 2026-09-02 00:01:28.900 to 2026-09-02 00:01:35.900.

**Expected conclusion:** Uplink backlog consistent with congestion: the estimated uplink buffer grows about 4.6x while the uplink PRB allocation (UL_NPRB) grows only about 2.0x, so offered uplink traffic outpaces what is being scheduled. Observed: uplink BLER 0.087 -> 0.174; estimated uplink buffer 862 -> 3,944; UL_NPRB 5.5 -> 11.0. Roughly flat: RSRP -88.0 -> -87.7; uplink SNR 21.7 -> 21.7. Uplink BLER also rises; the cited clauses do not explain that part.

**Acceptable conclusions:** Sudden uplink congestion; Uplink buffer backlog from demand exceeding scheduled resources; Uplink buffer build-up / overflow at the UE

**Uncertainty:** Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 10.1 says the scheduler allocates by buffer status but does not state that unmet demand causes backlog (inferred). PRB usage and the uplink PRB allocation (UL_NPRB) rise far less than the buffer and stay within the noise of the baseline (shift below 2x baseline std), so congestion vs. a scheduling fault is not fully separable from KPIs; BSR/MAC detail (TS 38.321) is not in the corpus. Channels with a large relative move that is not significant at 2x baseline std: PRB_Utilization_UL 0.147 -> 0.416. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 38.300 §10.4 Measurements to Support Scheduler Operation** (definition), version 18.1.0, PDF page 121
  - Why cited: Defines uplink buffer status reports, the concept behind the Estimated_UL_Buffer channel (no PM counter for UE buffer size exists in the corpus).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1203GPP TS 38.300 version 18.1.0 Release 18 The UE may be configured with up to 12 active configured uplink grants for a given BWP of a serving cell. When more than one is configured, the network decides which of these configured uplink grants are active at a time (including all of them). Each configured uplink grant can either be of Type 1 or Type 2. For Type 2, activation and deactivation of configured uplink grants are independent among the serving cells. When more than one Type 2 configured grant is configured, each configured grant is activated separa…”
- **3GPP TS 28.552 §5.1.1.7.7 Error number of UL initial TBs** (definition), version 18.7.0, PDF page 92
  - Why cited: Defines the uplink transport-block error count behind the UL_BLER channel.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)913GPP TS 28.552 version 18.7.0 Release 18 d) Each measurement is an integer. e) TB.ErrTotNbrDl.X. Where X identified by DL MU-MIMO maximum layer. f) NRCellDU. g) Valid for packet switched traffic . h) 5GS. 5.1.1.7.5 Residual error number of DL TBs a) This measurement provides the number of final faulty TBs transmitted on the downlink in a cell at last HARQ re-transmissions. b) CC. c) On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at the last HARQ feedback during the period of measurement. d) A single integer …”
- **3GPP TS 28.552 §5.1.1.2.7 Mean UL PRB used for data traffic** (definition), version 18.7.0, PDF page 41
  - Why cited: Defines the mean number of uplink PRBs used for data traffic. UL_NPRB counts the uplink PRBs allocated to the UE per TTI, so this is the nearest spec measurement, not an exact match.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)403GPP TS 28.552 version 18.7.0 Release 18 h) 5GS. i) One usage of this measurement is for monitoring the total number of available PRBs in average for DL traffic. 5.1.1.2.7 Mean UL PRB used for data traffic a) This measurement provides the number of physical resource blocks (PRBs) in average used in uplink for data traffic. The measurement is optionally split into subcounters per QoS level (mapped 5QI or QCI in NR option 3) and subcounters per supported S-NSSAI and subcounters per supported PLMN ID. b) SI c) Each measurement is obtained as the average num…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.1 General** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Describes congested cells and load balancing as the remedy, supporting load (not radio quality) as the fault class.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence yes, mechanism weak, conf 0.78)
  - Evidence note: The observed increase in UL BLER and a ~4.6× rise in estimated UL buffer together with a modest increase in UL PRB allocation indicate that traffic demand exceeds scheduled resources, consistent with sudden uplink congestion.
  - Requested change: Add a citation to a specification clause that explicitly describes the relationship between uplink buffer backlog, PRB allocation limits, and congestion (e.g., 3GPP TS 38.321 MAC scheduling behavior or a more detailed part of TS 38.300).
  - Requested change: Clarify that the current citations only define measurements and give a high‑level scheduler description, and therefore do not fully explain the observed BLER increase.
- qwen3.8-27b (Groq): **valid_with_changes** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence strongly supports the 'congestion' or 'backlog' hypothesis: UL_BLER doubled, Estimated_UL_Buffer increased ~4.6x, and UL_NPRB increased ~2x. The fact that the buffer grew much faster than the allocated resources (NPRB) suggests demand outpacing supply. RSRP and SNR are flat, ruling out radio link degradation. However, the conclusion claims the cited clauses support the explanation, but the provided spec text (38.300 10.1, 10.4) only defines the scheduler's inputs (BSR) and general operation; it does not explicitly state that 'unmet demand causes backlog' or define the specific failure mode of congestion. The link is inferential.
  - Unsupported claim flagged: The claim that the cited specification clauses 'support that explanation' is partially unsupported because the provided text defines the mechanisms (BSR, scheduling) but does not explicitly describe the causal chain of congestion leading to buffer overflow in the manner implied by the conclusion's confidence.
  - Requested change: Clarify in the conclusion or uncertainty that the specification citations define the relevant KPIs and general scheduler behavior, but the specific causal link between 'demand > supply' and 'buffer overflow' is an inference from the KPI trends, not an explicit statement in the provided spec excerpts.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The dramatic increase in estimated UL buffer (~4.6x) paired with a smaller increase in allocated PRBs (UL_NPRB ~2x) while channel quality metrics (RSRP, UL_SNR) remain flat strongly points to uplink traffic demand exceeding scheduled resources (congestion/backlog).

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_005 (dev, High Network Congestion (Sudden Spike))

- **Scenario:** `telecomts-ec8abd176a3001dd9eebde990fe8fccb10d7f13daf9f87afe95e6462c6189032`
- **Source:** `anomalous/synthetic/Zone_A/Twitch/processed/chunked.jsonl`, raw rows 3809-3936, anomaly samples 53-127 (event E16)
- **Recording family:** `family-6aaa4378e87449bd2ac2d04e5cc5e62dffdfdc9050806e23083ee4640185a981` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause, difficult

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 5.3 s in: uplink BLER went from about 0.088 to about 0.170; estimated uplink buffer went from about 1,174 to about 8,071. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_BLER | 0.08796 | 0.1704 | increase | 0.01932 |
| Estimated_UL_Buffer | 1174.0 | 8071.0 | increase | 862.5 |
| UL_NPRB | 7.104 | 13.78 | increase | 4.332 |
| RSRP | -87.66 | -87.96 | decrease | 0.4732 |
| UL_SNR | 21.36 | 20.81 | decrease | 0.9481 |

Baseline 2026-09-02 00:06:20.800 to 2026-09-02 00:06:26.000; comparison 2026-09-02 00:06:26.100 to 2026-09-02 00:06:33.500.

**Expected conclusion:** Uplink backlog consistent with congestion: the estimated uplink buffer grows about 6.9x while the uplink PRB allocation (UL_NPRB) grows only about 1.9x, so offered uplink traffic outpaces what is being scheduled. Observed: uplink BLER 0.088 -> 0.170; estimated uplink buffer 1,174 -> 8,071; UL_NPRB 7.1 -> 13.8. Roughly flat: RSRP -87.7 -> -88.0; uplink SNR 21.4 -> 20.8. Uplink BLER also rises; the cited clauses do not explain that part.

**Acceptable conclusions:** Sudden uplink congestion; Uplink buffer backlog from demand exceeding scheduled resources; Uplink buffer build-up / overflow at the UE

**Uncertainty:** Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 10.1 says the scheduler allocates by buffer status but does not state that unmet demand causes backlog (inferred). PRB usage and the uplink PRB allocation (UL_NPRB) rise far less than the buffer and stay within the noise of the baseline (shift below 2x baseline std), so congestion vs. a scheduling fault is not fully separable from KPIs; BSR/MAC detail (TS 38.321) is not in the corpus. Other channels that also changed but are not used in this conclusion: UL_MCS 27.0 -> 23.5. Channels with a large relative move that is not significant at 2x baseline std: PRB_Utilization_UL 0.425 -> 1.1. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 38.300 §10.4 Measurements to Support Scheduler Operation** (definition), version 18.1.0, PDF page 121
  - Why cited: Defines uplink buffer status reports, the concept behind the Estimated_UL_Buffer channel (no PM counter for UE buffer size exists in the corpus).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1203GPP TS 38.300 version 18.1.0 Release 18 The UE may be configured with up to 12 active configured uplink grants for a given BWP of a serving cell. When more than one is configured, the network decides which of these configured uplink grants are active at a time (including all of them). Each configured uplink grant can either be of Type 1 or Type 2. For Type 2, activation and deactivation of configured uplink grants are independent among the serving cells. When more than one Type 2 configured grant is configured, each configured grant is activated separa…”
- **3GPP TS 28.552 §5.1.1.7.7 Error number of UL initial TBs** (definition), version 18.7.0, PDF page 92
  - Why cited: Defines the uplink transport-block error count behind the UL_BLER channel.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)913GPP TS 28.552 version 18.7.0 Release 18 d) Each measurement is an integer. e) TB.ErrTotNbrDl.X. Where X identified by DL MU-MIMO maximum layer. f) NRCellDU. g) Valid for packet switched traffic . h) 5GS. 5.1.1.7.5 Residual error number of DL TBs a) This measurement provides the number of final faulty TBs transmitted on the downlink in a cell at last HARQ re-transmissions. b) CC. c) On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at the last HARQ feedback during the period of measurement. d) A single integer …”
- **3GPP TS 28.552 §5.1.1.2.7 Mean UL PRB used for data traffic** (definition), version 18.7.0, PDF page 41
  - Why cited: Defines the mean number of uplink PRBs used for data traffic. UL_NPRB counts the uplink PRBs allocated to the UE per TTI, so this is the nearest spec measurement, not an exact match.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)403GPP TS 28.552 version 18.7.0 Release 18 h) 5GS. i) One usage of this measurement is for monitoring the total number of available PRBs in average for DL traffic. 5.1.1.2.7 Mean UL PRB used for data traffic a) This measurement provides the number of physical resource blocks (PRBs) in average used in uplink for data traffic. The measurement is optionally split into subcounters per QoS level (mapped 5QI or QCI in NR option 3) and subcounters per supported S-NSSAI and subcounters per supported PLMN ID. b) SI c) Each measurement is obtained as the average num…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.1 General** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Describes congested cells and load balancing as the remedy, supporting load (not radio quality) as the fault class.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence yes, mechanism weak, conf 0.78)
  - Evidence note: The observed increase in UL BLER and large rise in Estimated UL Buffer together with modest UL PRB increase indicate uplink congestion/backlog.
  - Requested change: Add a mechanism citation that directly ties uplink buffer backlog to increased UL BLER (e.g., TS 38.321 on MAC HARQ retransmissions or UL transport block error handling).
  - Requested change: Clarify that the current mechanism citations only partially support the congestion explanation and do not explain the BLER increase.
- qwen3.8-27b (Groq): **valid_with_changes** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence strongly supports the 'congestion' label: UL_BLER increased (0.088 -> 0.170), Estimated_UL_Buffer increased significantly (1174 -> 8071), and UL_NPRB increased (7.1 -> 13.8). The ratio of buffer growth (~6.9x) to PRB growth (~1.9x) supports the inference that demand outpaced scheduling. However, the conclusion states that 'cited clauses do not explain' the BLER rise. While true that the provided spec text doesn't explicitly link congestion to BLER, the mechanism of congestion (resource contention) inherently leads to higher error rates in practical systems, though strictly speaking, the provided text only defines the measurements and general scheduling principles. The evidence supports the *label* of congestion well, but the *explanation* of BLER is weak based solely on the provided text.
  - Unsupported claim flagged: The conclusion states 'cited clauses do not explain that part [BLER rise]'. This is a meta-comment on the citations rather than a claim about the fault itself, but it is accurate based on the provided text. However, the conclusion implies the BLER rise is consistent with congestion. While physically true, the provided spec text (38.300 10.1, 10.4, 28.552) does not explicitly state that congestion causes BLER to rise. It defines BLER and scheduling. Therefore, the link between the specific KPI change (BLER) and the root cause (congestion) is inferred, not directly supported by the cited text.
  - Requested change: The conclusion should explicitly state that the link between the observed BLER increase and the congestion root cause is an inference based on general network behavior, as the provided specification text defines the metrics and scheduling principles but does not explicitly describe the causal chain from resource contention to increased BLER.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a large increase in uplink buffer backlog (1,174 to 8,071, ~6.9x) paired with a much smaller relative increase in scheduled UL resource blocks (7.1 to 13.8, ~1.9x), while RF conditions (RSRP, UL SNR) remain essentially stable. This strongly supports uplink buffer backlog and network congestion.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_006 (test, Buffer Overflow (Gradual Buildup))

- **Scenario:** `telecomts-27aa7e0653b962a96812e3c35b973596058dad2806a6a9d7c0782811fa01d3ea`
- **Source:** `anomalous/synthetic/Zone_A/YouTube/processed/chunked.jsonl`, raw rows 4161-4288, anomaly samples 75-127 (event E21)
- **Recording family:** `family-eafe6e671551d71f8dfe46141dfed768e35d8c42cd1635144b888eb523cd13fa` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause, difficult

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 7.5 s in: downlink PRB utilization went from about 0.007 to about 1.9; uplink PRB utilization went from about 0.025 to about 1.2; downlink packet count went from about 0 to about 16.6; uplink packet count went from about 0 to about 25.6. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| PRB_Utilization_DL | 0.006504 | 1.899 | increase | 0.01761 |
| PRB_Utilization_UL | 0.02505 | 1.229 | increase | 0.03922 |
| DL_NumberOfPackets | 0.0 | 16.62 | increase | 0 |
| UL_NumberOfPackets | 0.0 | 25.57 | increase | 0 |
| Estimated_UL_Buffer | 0.6587 | 2.664e-08 | decrease | 5.232 |
| RSRP | -92.82 | -92.69 | increase | 0.3166 |
| UL_SNR | 19.63 | 19.66 | increase | 1.797 |

Baseline 2026-01-28 00:06:56.000 to 2026-01-28 00:07:03.400; comparison 2026-01-28 00:07:03.500 to 2026-01-28 00:07:08.700.

**Expected conclusion:** A traffic-load surge rather than a radio fault: resource usage and packet counts rise well above baseline. Observed: downlink PRB utilization 0.007 -> 1.9; uplink PRB utilization 0.025 -> 1.2; downlink packet count 0 -> 16.6; uplink packet count 0 -> 25.6. Roughly flat: RSRP -92.8 -> -92.7; uplink SNR 19.6 -> 19.7, so the radio link itself does not explain the change. PRB utilization peaks near 1.9 % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown.

**Acceptable conclusions:** Network congestion from increased traffic load; Traffic surge raising resource usage and packet counts; Load-related degradation rather than a radio-quality fault

**Uncertainty:** Synthetic injection; the label (buffer overflow) is a dataset claim. The estimated uplink buffer does not rise in this window (0.659 -> 0), so the evidence supports a load surge but does not support an overflow; buffer overflow is therefore not an acceptable conclusion here, and congestion is equally consistent. Spec support is partial (scheduler and BSR concepts in 38.300; MAC/RLC buffer and discard behaviour live in TS 38.321/38.322, not in the corpus). Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §5.1.1.2.1 DL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines downlink PRB usage, the measured counterpart of PRB_Utilization_DL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 28.552 §5.1.1.2.2 UL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.2 Load reporting for intra-RAT and intra-system inter-RAT load balancing** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence yes, mechanism ok, conf 0.93)
  - Evidence note: Increases in DL/UL PRB utilization and packet counts together with flat RSRP/SNR and a decrease in estimated UL buffer indicate a traffic load surge rather than a radio‑quality or buffer‑overflow problem.
  - Unsupported claim flagged: Buffer Overflow (Gradual Buildup) – not supported by the observed decrease in Estimated_UL_Buffer and lack of buffer growth.
  - Requested change: Replace the dataset label claim "Buffer Overflow (Gradual Buildup)" with the evidence‑supported conclusion of a traffic‑load surge / network congestion.
  - Requested change: Remove any statements that imply a buffer overflow, since Estimated_UL_Buffer decreases during the event.
  - Requested change: Ensure the final answer cites the definition clauses (TS 28.552) for PRB utilization and the mechanism clauses (TS 38.300 10.1 and 15.5.1.2) that explain why increased PRB usage indicates higher traffic load.
- qwen3.8-27b (Groq): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence shows a simultaneous, sharp increase in DL/UL PRB utilization and packet counts from near-zero baselines, while radio quality metrics (RSRP, UL_SNR) remain stable. This pattern strongly supports a traffic load surge. The decrease in Estimated_UL_Buffer to near-zero is consistent with the buffer being drained by the increased transmission activity, or simply reflects the low absolute volume relative to the scale, but it does not contradict the load surge hypothesis. The conclusion correctly identifies that saturation is not shown (1.9% is low) and that radio faults are unlikely.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 1.0)
  - Evidence note: The KPI evidence shows significant increases in downlink/uplink PRB utilization and packet counts while radio link metrics (RSRP and UL_SNR) remain flat, directly supporting a traffic load surge over a radio-quality degradation.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_007 (test, Buffer Overflow (Gradual Buildup))

- **Scenario:** `telecomts-e9d3c94262c9a437c27459583cc334db52da00d723867bced91417ba7593a8bb`
- **Source:** `anomalous/synthetic/Zone_B/YouTube/processed/chunked.jsonl`, raw rows 8673-8800, anomaly samples 66-127 (event E23)
- **Recording family:** `family-e6c16ffb8df2ca8e08581ab6b3bf8360800c9ecca81bae8d2fb823809d8710f7` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause, difficult

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 6.6 s in: uplink PRB utilization went from about 0.073 to about 1.5; uplink packet count went from about 1.2 to about 19.0. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| PRB_Utilization_UL | 0.07335 | 1.54 | increase | 0.3307 |
| UL_NumberOfPackets | 1.152 | 19.02 | increase | 3.73 |
| Estimated_UL_Buffer | 7.997 | 0.0 | decrease | 54.46 |
| RSRP | -109.0 | -108.4 | increase | 4.255 |
| UL_SNR | 18.93 | 19.75 | increase | 3.274 |

Baseline 2027-07-07 00:14:27.200 to 2027-07-07 00:14:33.700; comparison 2027-07-07 00:14:33.800 to 2027-07-07 00:14:39.900.

**Expected conclusion:** A traffic-load surge rather than a radio fault: resource usage and packet counts rise well above baseline. Observed: uplink PRB utilization 0.073 -> 1.5; uplink packet count 1.2 -> 19.0. Roughly flat: RSRP -109 -> -108; uplink SNR 18.9 -> 19.8, so the radio link itself does not explain the change. PRB utilization peaks near 1.7 % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown.

**Acceptable conclusions:** Network congestion from increased traffic load; Traffic surge raising resource usage and packet counts; Load-related degradation rather than a radio-quality fault

**Uncertainty:** Synthetic injection; the label (buffer overflow) is a dataset claim. The estimated uplink buffer does not rise in this window (8.0 -> 0), so the evidence supports a load surge but does not support an overflow; buffer overflow is therefore not an acceptable conclusion here, and congestion is equally consistent. Spec support is partial (scheduler and BSR concepts in 38.300; MAC/RLC buffer and discard behaviour live in TS 38.321/38.322, not in the corpus). Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §5.1.1.2.2 UL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.2 Load reporting for intra-RAT and intra-system inter-RAT load balancing** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence yes, mechanism ok, conf 0.93)
  - Evidence note: The KPI evidence shows a clear increase in UL PRB utilization and packet count while radio quality metrics (RSRP, SNR) remain flat and the estimated UL buffer drops to zero, indicating a traffic load surge rather than a buffer overflow.
  - Unsupported claim flagged: Buffer Overflow (Gradual Buildup)
  - Requested change: Remove or correct the dataset_label_claim 'Buffer Overflow (Gradual Buildup)' because it is not supported by the KPI evidence; the appropriate root cause is a traffic load surge/congestion.
- qwen3.8-27b (Groq): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence shows a significant increase in UL PRB utilization (0.073 to 1.54) and UL packet count (1.15 to 19.02), while radio quality metrics (RSRP, UL_SNR) remain stable. The Estimated_UL_Buffer decreases to 0, which contradicts a 'buffer overflow' scenario but supports a transient traffic surge where the scheduler successfully clears the buffer. The conclusion correctly identifies this as a load/traffic surge rather than a radio fault or buffer overflow.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The evidence shows uplink PRB utilization rising from 0.073% to 1.54% and uplink packet count rising from 1.15 to 19.02, while radio quality KPIs (RSRP, UL_SNR) remain stable. This supports a traffic load surge over a radio quality issue.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_008 (test, Co-Channel Interference (Mild))

- **Scenario:** `telecomts-2510347a7fdef5a421825133cdaa33062d97a8919d92363decbb7c4fb5faca25`
- **Source:** `anomalous/synthetic/Zone_B/YouTube/processed/chunked.jsonl`, raw rows 321-448, anomaly samples 55-127 (event E26)
- **Recording family:** `family-e6c16ffb8df2ca8e08581ab6b3bf8360800c9ecca81bae8d2fb823809d8710f7` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause, difficult

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 5.5 s in: uplink SNR went from about 21.5 to about 16.7; RSRP went from about -106 to about -115; uplink PRB count (UL_NPRB) went from about 5.0 to about 5.8. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_SNR | 21.5 | 16.7 | decrease | 2.361 |
| RSRP | -106.0 | -114.8 | decrease | 0 |
| UL_NPRB | 5.0 | 5.837 | increase | 0 |
| PRB_Utilization_DL | 0.005149 | 0.005209 | increase | 0.01471 |
| PRB_Utilization_UL | 0.02403 | 0.02388 | decrease | 0.02681 |
| DL_NumberOfPackets | 0.0 | 0.0 | unchanged | 0 |
| UL_NumberOfPackets | 0.0 | 0.0 | unchanged | 0 |

Baseline 2027-07-07 00:00:32.000 to 2027-07-07 00:00:37.400; comparison 2027-07-07 00:00:37.500 to 2027-07-07 00:00:44.700.

**Expected conclusion:** Uplink radio-quality degradation: the uplink SNR falls by about 4.8 dB together with RSRP. The cited mechanism supports uplink (co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, so the KPIs alone cannot separate the two. Observed: uplink SNR 21.5 -> 16.7; RSRP -106 -> -115; uplink PRB count (UL_NPRB) 5.0 -> 5.8. Roughly flat: downlink PRB utilization 0.005 -> 0.005; uplink PRB utilization 0.024 -> 0.024; downlink packet count 0 -> 0; uplink packet count 0 -> 0.

**Acceptable conclusions:** Uplink interference on the serving channel lowering the received SINR; Co-channel interference from another transmitter on the same frequency; Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)

**Uncertainty:** Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference. Other channels that also changed but are not used in this conclusion: DL_MCS 7.0 -> 5.5, UL_MCS 9.0 -> 7.5. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §A.78 Monitoring of SS-SINR** (definition), version 18.7.0, PDF page 352
  - Why cited: Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3513GPP TS 28.552 version 18.7.0 Release 18 A.76 Monitoring of subscriber profile sizes in UDM The subscriber profiles maintained by UDM have two parts - static and dynamic data. Dynamic part of the subscriber profile grows over time (e.g. due to roaming conditions, dynamic dereferencing of provisioning data, addition/enabling of new features). The uncontrolled and unnoticed growth of subscriber profiles may result in degraded system performance and lead to service impacts. Various corrective actions varying from subscriber profile optimization (clean-up/c…”
- **3GPP TS 32.425 §A.9 Monitoring of RF performance** (mechanism), version 18.0.0, PDF page 105
  - Why cited: States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).
  - Page text (start): “ETSI ETSI TS 132 425 V18.0.0 (2024-05)1043GPP TS 32.425 version 18.0.0 Release 18 Monitoring interference, both uplink and downlink, is an important aspect of monitoring RF performance. For this purpose, monitoring of UL interference power is useful. Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference. By efficient frequency planning and careful Selection of base station locations, the impact of interference signal can be effectively contr…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence partial, mechanism weak, conf 0.86)
  - Evidence note: The KPI pattern (UL SNR and RSRP decrease, slight UL PRB increase, other metrics flat) indicates uplink radio‑quality degradation, but it cannot uniquely distinguish co‑channel interference from a simple received‑power loss.
  - Unsupported claim flagged: The claim that the fault is specifically "Co‑Channel Interference (Mild)" is not uniquely supported by the presented KPIs.
  - Unsupported claim flagged: Citation of 3GPP TS 28.552 A.78 as evidence for uplink interference is inappropriate because it defines downlink SS‑SINR, not UL SNR.
  - Requested change: Rephrase the root‑cause conclusion to state that the observed KPI changes indicate uplink radio‑quality degradation that could be caused by either co‑channel interference or a loss of received power (e.g., antenna or path loss), rather than asserting "Co‑Channel Interference (Mild)" as the definitive cause.
  - Requested change: Replace or supplement the 3GPP TS 28.552 citation with a clause that actually defines uplink SNR/SINR for NR (e.g., 3GPP TS 38.133 §5.2.2) or note the lack of an appropriate NR uplink SINR definition in the corpus.
  - Requested change: Clarify that the 3GPP TS 32.425 citation is LTE‑specific and only provides a generic mechanism; indicate the limitation for NR applicability.
- qwen3.8-27b (Groq): **valid** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a simultaneous drop in UL_SNR and RSRP. While the conclusion correctly notes that interference lowers SNR, the concurrent drop in RSRP (received power) is not explained by interference alone (which typically affects SINR/SNR while leaving RSRP relatively stable or increasing it if the interferer is strong). The conclusion honestly admits this limitation, stating that a received-power loss is an equally consistent reading. Thus, the evidence supports the *observation* of degradation but does not uniquely support the *label* of 'Co-Channel Interference' over 'Antenna/RF Failure'. The conclusion text itself is accurate in describing the ambiguity.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 1.0)
  - Evidence note: The KPI evidence table directly supports all stated numerical changes and baseline-to-comparison comparisons in the expected conclusion.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_009 (test, Co-Channel Interference (Severe))

- **Scenario:** `telecomts-a6f309dcd341b8d2436497a1c8a399d14f0dcfce3dabbe725e195c4eb862299e`
- **Source:** `anomalous/synthetic/Zone_A/YouTube/processed/chunked.jsonl`, raw rows 1633-1760, anomaly samples 51-127 (event E28)
- **Recording family:** `family-eafe6e671551d71f8dfe46141dfed768e35d8c42cd1635144b888eb523cd13fa` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 5.1 s in: uplink SNR went from about 21.6 to about 10.9; RSRP went from about -87.0 to about -99.8; uplink PRB count (UL_NPRB) went from about 5.0 to about 7.3; uplink bytes (RX_Bytes) went from about 209 to about 116. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_SNR | 21.6 | 10.88 | decrease | 1.361 |
| RSRP | -87.0 | -99.81 | decrease | 0 |
| UL_NPRB | 5.0 | 7.251 | increase | 0 |
| RX_Bytes | 208.8 | 116.1 | decrease | 45.09 |
| PRB_Utilization_DL | 0.003702 | 0.006632 | increase | 0.01269 |
| PRB_Utilization_UL | 0.02591 | 0.04373 | increase | 0.02528 |
| DL_NumberOfPackets | 0.09804 | 0.0 | decrease | 0.2974 |
| UL_NumberOfPackets | 0.1373 | 0.0 | decrease | 0.8406 |

Baseline 2026-01-28 00:02:43.200 to 2026-01-28 00:02:48.200; comparison 2026-01-28 00:02:48.300 to 2026-01-28 00:02:55.900.

**Expected conclusion:** Uplink radio-quality degradation: the uplink SNR falls by about 10.7 dB together with RSRP. The cited mechanism supports uplink (co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, so the KPIs alone cannot separate the two. Observed: uplink SNR 21.6 -> 10.9; RSRP -87.0 -> -99.8; uplink PRB count (UL_NPRB) 5.0 -> 7.3; uplink bytes (RX_Bytes) 209 -> 116. Roughly flat: downlink PRB utilization 0.004 -> 0.007; uplink PRB utilization 0.026 -> 0.044; downlink packet count 0.098 -> 0; uplink packet count 0.137 -> 0.

**Acceptable conclusions:** Uplink interference on the serving channel lowering the received SINR; Co-channel interference from another transmitter on the same frequency; Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)

**Uncertainty:** Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §A.78 Monitoring of SS-SINR** (definition), version 18.7.0, PDF page 352
  - Why cited: Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3513GPP TS 28.552 version 18.7.0 Release 18 A.76 Monitoring of subscriber profile sizes in UDM The subscriber profiles maintained by UDM have two parts - static and dynamic data. Dynamic part of the subscriber profile grows over time (e.g. due to roaming conditions, dynamic dereferencing of provisioning data, addition/enabling of new features). The uncontrolled and unnoticed growth of subscriber profiles may result in degraded system performance and lead to service impacts. Various corrective actions varying from subscriber profile optimization (clean-up/c…”
- **3GPP TS 32.425 §A.9 Monitoring of RF performance** (mechanism), version 18.0.0, PDF page 105
  - Why cited: States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).
  - Page text (start): “ETSI ETSI TS 132 425 V18.0.0 (2024-05)1043GPP TS 32.425 version 18.0.0 Release 18 Monitoring interference, both uplink and downlink, is an important aspect of monitoring RF performance. For this purpose, monitoring of UL interference power is useful. Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference. By efficient frequency planning and careful Selection of base station locations, the impact of interference signal can be effectively contr…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence partial, mechanism weak, conf 0.78)
  - Evidence note: The KPI changes clearly show uplink radio‑quality degradation, but they do not uniquely identify severe co‑channel interference; a loss of received power could produce the same pattern.
  - Unsupported claim flagged: The root cause is specifically "Co‑Channel Interference (Severe)" – the evidence does not distinguish severity nor confirm co‑channel origin.
  - Unsupported claim flagged: That the observed RSRP drop is explained by interference alone – the KPI pattern is also consistent with a received‑power loss.
  - Requested change: Qualify the root‑cause claim: replace "Co‑Channel Interference (Severe)" with a statement that the evidence only indicates uplink radio‑quality degradation, which could be caused by interference or a received‑power loss.
  - Requested change: Add or replace citations with NR‑specific clauses that directly link UL_SNR degradation to uplink interference (e.g., 3GPP TS 38.133 or 38.331).
  - Requested change: Note that the provided 28.552 citation does not support the UL_SNR KPI and therefore should not be used as primary evidence.
  - Requested change: Clarify that the severity (mild vs severe) cannot be inferred from the KPI changes alone.
- qwen3.8-27b (Groq): **valid** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a simultaneous drop in UL SNR and RSRP. While this is consistent with interference, it is equally consistent with a loss of received power (e.g., antenna failure or coverage hole). The conclusion correctly notes that KPIs alone cannot distinguish between these two causes. The increase in UL_NPRB and decrease in RX_Bytes are consistent with degraded link quality leading to retransmissions or lower throughput, but do not uniquely identify the root cause.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 1.0)
  - Evidence note: The measured KPIs match all values and directions cited in the expected conclusion. The conclusion correctly notes that both RSRP and UL_SNR drop significantly alongside decreased RX_Bytes and increased UL_NPRB.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_010 (test, Co-Channel Interference (Severe))

- **Scenario:** `telecomts-8f3838af972511f0903f2a5480d6c77df8be12f11ab0036af8229a0f2a2f8ce0`
- **Source:** `anomalous/synthetic/Zone_A/YouTube/processed/chunked.jsonl`, raw rows 8545-8672, anomaly samples 74-127 (event E29)
- **Recording family:** `family-eafe6e671551d71f8dfe46141dfed768e35d8c42cd1635144b888eb523cd13fa` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 7.4 s in: uplink SNR went from about 21.5 to about 7.8; RSRP went from about -87.0 to about -102; uplink PRB count (UL_NPRB) went from about 5.0 to about 7.0; uplink bytes (RX_Bytes) went from about 211 to about 108. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| UL_SNR | 21.51 | 7.764 | decrease | 1.614 |
| RSRP | -87.0 | -101.5 | decrease | 0 |
| UL_NPRB | 5.0 | 6.977 | increase | 0 |
| RX_Bytes | 211.2 | 107.9 | decrease | 33.33 |
| PRB_Utilization_DL | 0.004465 | 0.00934 | increase | 0.01381 |
| PRB_Utilization_UL | 0.0236 | 0.03443 | increase | 0.02603 |
| DL_NumberOfPackets | 0.09459 | 0.0 | decrease | 0.4083 |
| UL_NumberOfPackets | 0.09459 | 0.0 | decrease | 0.8082 |

Baseline 2026-01-28 00:14:14.400 to 2026-01-28 00:14:21.700; comparison 2026-01-28 00:14:21.800 to 2026-01-28 00:14:27.100.

**Expected conclusion:** Uplink radio-quality degradation: the uplink SNR falls by about 13.7 dB together with RSRP. The cited mechanism supports uplink (co-channel) interference as the explanation, but a loss of received power (antenna, RF path or coverage) produces the same KPI pattern, so the KPIs alone cannot separate the two. Observed: uplink SNR 21.5 -> 7.8; RSRP -87.0 -> -102; uplink PRB count (UL_NPRB) 5.0 -> 7.0; uplink bytes (RX_Bytes) 211 -> 108. Roughly flat: downlink PRB utilization 0.004 -> 0.009; uplink PRB utilization 0.024 -> 0.034; downlink packet count 0.095 -> 0; uplink packet count 0.095 -> 0.

**Acceptable conclusions:** Uplink interference on the serving channel lowering the received SINR; Co-channel interference from another transmitter on the same frequency; Uplink radio-quality degradation from interference or a received-power loss (antenna, RF path or coverage)

**Uncertainty:** Synthetic injection (TelecomTS label is a dataset claim, not an observed cause). RSRP also falls, which interference alone does not explain, so a received-power loss is an equally consistent reading; the corpus has no mechanism clause for that alternative. The only mechanism text is an informative annex of the LTE PM spec (32.425 A.9): the corpus contains no NR-specific clause that links interference to uplink SNR, no uplink SINR counter, and nothing that separates mild from severe interference. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §A.78 Monitoring of SS-SINR** (definition), version 18.7.0, PDF page 352
  - Why cited: Closest SINR definition in the corpus: it defines SINR as the received signal level over interference plus noise, so added interference lowers it. It is the downlink SS-SINR measurement; the dataset's UL_SNR is an uplink measurement and the corpus has no uplink SINR or SNR counter, so this clause defines the concept, not the exact KPI.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)3513GPP TS 28.552 version 18.7.0 Release 18 A.76 Monitoring of subscriber profile sizes in UDM The subscriber profiles maintained by UDM have two parts - static and dynamic data. Dynamic part of the subscriber profile grows over time (e.g. due to roaming conditions, dynamic dereferencing of provisioning data, addition/enabling of new features). The uncontrolled and unnoticed growth of subscriber profiles may result in degraded system performance and lead to service impacts. Various corrective actions varying from subscriber profile optimization (clean-up/c…”
- **3GPP TS 32.425 §A.9 Monitoring of RF performance** (mechanism), version 18.0.0, PDF page 105
  - Why cited: States the causal chain: uplink interference disturbs the base-station receiver and leads to receive errors and degraded service (informative annex, LTE PM spec).
  - Page text (start): “ETSI ETSI TS 132 425 V18.0.0 (2024-05)1043GPP TS 32.425 version 18.0.0 Release 18 Monitoring interference, both uplink and downlink, is an important aspect of monitoring RF performance. For this purpose, monitoring of UL interference power is useful. Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference. By efficient frequency planning and careful Selection of base station locations, the impact of interference signal can be effectively contr…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence partial, mechanism weak, conf 0.78)
  - Evidence note: The KPI changes (UL SNR and RSRP drop, UL traffic drop, slight PRB utilization increase) indicate uplink quality degradation, but they do not uniquely identify co‑channel interference versus a simple received‑power loss.
  - Unsupported claim flagged: The definitive labeling of the root cause as "Co‑Channel Interference (Severe)" is not fully supported by the KPI evidence alone.
  - Requested change: Temper the root‑cause label to reflect that the evidence only indicates uplink quality degradation, not definitively severe co‑channel interference.
  - Requested change: State explicitly that the KPI pattern is also consistent with a received‑power loss (e.g., antenna or RF path issue).
  - Requested change: Acknowledge the lack of an NR‑specific mechanism clause linking interference to the observed UL SNR drop.
- qwen3.8-27b (Groq): **valid_with_changes** (evidence partial, mechanism ok, conf 0.95)
  - Evidence note: The KPIs show a simultaneous drop in UL_SNR and RSRP. While this is consistent with interference (lowering SNR), the drop in RSRP (received power) is not explained by interference alone (which typically affects the noise/interference floor, not the signal power). The conclusion correctly identifies that the KPIs alone cannot distinguish between interference and a received-power loss (antenna/RF/coverage). The 'expected_conclusion' text is honest about this ambiguity, but the dataset label 'Co-Channel Interference' is a specific claim that the evidence does not uniquely support over the alternative of power loss.
  - Unsupported claim flagged: The dataset label 'Co-Channel Interference (Severe)' is presented as the ground truth, but the KPI evidence (specifically the RSRP drop) is equally consistent with a received-power loss, which the conclusion text admits but the label does not reflect.
  - Requested change: Clarify that the dataset label 'Co-Channel Interference' is a synthetic injection claim, not a conclusion derived solely from the KPIs, as the KPIs (specifically RSRP drop) are ambiguous between interference and power loss.
  - Requested change: Note that the mechanism citation (TS 32.425) is from an LTE specification, which may not be strictly applicable to 5G NR uplink interference mechanisms, although the general principle holds.
- gemini-3.6-flash (earlier draft, 10 of 12): **valid** (evidence yes, mechanism ok, conf 1.0)
  - Evidence note: The baseline and comparison values cited in the conclusion accurately match the provided KPI evidence table.

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_011 (test, High Network Congestion (Gradual Buildup))

- **Scenario:** `telecomts-315d93f3ea5ab054268d482755084bdfcc398c90c369b0770894aea2c67d1eea`
- **Source:** `anomalous/synthetic/Zone_B/YouTube/processed/chunked.jsonl`, raw rows 161-288, anomaly samples 84-122 (event E36)
- **Recording family:** `family-e6c16ffb8df2ca8e08581ab6b3bf8360800c9ecca81bae8d2fb823809d8710f7` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 8.4 s in: downlink PRB utilization went from about 0.008 to about 0.853; uplink PRB utilization went from about 0.031 to about 0.799; uplink packet count went from about 0.286 to about 14.2; estimated uplink buffer went from about 0.758 to about 18.9. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| PRB_Utilization_DL | 0.00802 | 0.8525 | increase | 0.02211 |
| PRB_Utilization_UL | 0.0309 | 0.7988 | increase | 0.06951 |
| UL_NumberOfPackets | 0.2857 | 14.19 | increase | 1.985 |
| Estimated_UL_Buffer | 0.7582 | 18.87 | increase | 5.241 |
| RSRP | -110.7 | -110.4 | increase | 0.45 |
| UL_SNR | 18.53 | 19.32 | increase | 2.807 |

Baseline 2027-07-07 00:00:16.000 to 2027-07-07 00:00:24.300; comparison 2027-07-07 00:00:24.400 to 2027-07-07 00:00:28.200.

**Expected conclusion:** A traffic-load increase consistent with congestion building up: resource usage and traffic rise sharply relative to baseline. Observed: downlink PRB utilization 0.008 -> 0.853; uplink PRB utilization 0.031 -> 0.799; uplink packet count 0.286 -> 14.2; estimated uplink buffer 0.758 -> 18.9. Roughly flat: RSRP -111 -> -110; uplink SNR 18.5 -> 19.3, so the radio link itself does not explain the change. PRB utilization peaks near 0.853 % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown.

**Acceptable conclusions:** Network congestion building up from rising traffic; Traffic surge driving PRB utilization up; Load-related degradation rather than a radio-quality fault

**Uncertainty:** Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 describes PRB usage as the load indicator and load balancing as the remedy but does not state that load degrades per-UE throughput. A buffer-overflow fault produces a similar load signature in this data,. Absolute PRB utilization stays low, so this is a relative load rise, not a saturated cell. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §5.1.1.2.1 DL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines downlink PRB usage, the measured counterpart of PRB_Utilization_DL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 28.552 §5.1.1.2.2 UL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.2 Load reporting for intra-RAT and intra-system inter-RAT load balancing** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid** (evidence yes, mechanism ok, conf 0.93)
  - Evidence note: All KPI metrics (DL/UL PRB utilization, UL packet count, UL buffer) show a sharp increase while radio quality (RSRP, UL SNR) remains flat, which aligns with a traffic‑load surge causing congestion.
- qwen3.8-27b (Groq): **valid** (evidence yes, mechanism ok, conf 0.95)
  - Evidence note: The KPI evidence shows a sharp increase in DL/UL PRB utilization, UL packet count, and UL buffer size, while RSRP and UL SNR remain stable. This pattern is consistent with a traffic load increase (congestion) rather than a radio link quality issue. The conclusion correctly identifies this as a load-related change and notes that absolute utilization is still low (0.85%), indicating a relative rise from a near-idle state rather than saturation.
- gemini-3.6-flash (earlier draft, 10 of 12): failed (HTTP 429)

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____

---

## rca_012 (test, High Network Congestion (Gradual Buildup))

- **Scenario:** `telecomts-866959b53f6d54ac7f403ce05692b6e2f215a842c6b405060f7172b87839edc7`
- **Source:** `anomalous/synthetic/Zone_B/YouTube/processed/chunked.jsonl`, raw rows 3713-3840, anomaly samples 51-127 (event E38)
- **Recording family:** `family-e6c16ffb8df2ca8e08581ab6b3bf8360800c9ecca81bae8d2fb823809d8710f7` (unresolved, like all 33 recordings)
- **Inputs:** kpi_only (no approved text; TelecomTS ticket text leaks the label)
- **Annotation families:** multi_clause

**Question (what the model under test sees):**

> In a 12.8-second KPI window from a single 5G NR cell, a change starts about 5.1 s in: downlink PRB utilization went from about 0.005 to about 1.5; uplink PRB utilization went from about 0.020 to about 1.2; uplink PRB count (UL_NPRB) went from about 5.0 to about 6.0; downlink packet count went from about 0.078 to about 16.6; uplink packet count went from about 0.078 to about 15.6. What is the most likely root cause, and which specification clauses support that explanation?

**KPI evidence** (baseline = pre-fault samples, comparison = fault samples of the same window; units are null in the repo):

| Channel | Baseline | Comparison | Direction | Baseline std |
|---|---|---|---|---|
| PRB_Utilization_DL | 0.004627 | 1.533 | increase | 0.01404 |
| PRB_Utilization_UL | 0.02036 | 1.18 | increase | 0.02518 |
| UL_NPRB | 5.0 | 6.023 | increase | 0 |
| DL_NumberOfPackets | 0.07843 | 16.65 | increase | 0.4358 |
| UL_NumberOfPackets | 0.07843 | 15.65 | increase | 0.5546 |
| RSRP | -106.0 | -106.2 | decrease | 0 |
| UL_SNR | 21.67 | 21.07 | decrease | 2.645 |

Baseline 2027-07-07 00:06:11.200 to 2027-07-07 00:06:16.200; comparison 2027-07-07 00:06:16.300 to 2027-07-07 00:06:23.900.

**Expected conclusion:** A traffic-load increase consistent with congestion building up: resource usage and traffic rise sharply relative to baseline. Observed: downlink PRB utilization 0.005 -> 1.5; uplink PRB utilization 0.020 -> 1.2; uplink PRB count (UL_NPRB) 5.0 -> 6.0; downlink packet count 0.078 -> 16.6; uplink packet count 0.078 -> 15.6. Roughly flat: RSRP -106 -> -106; uplink SNR 21.7 -> 21.1, so the radio link itself does not explain the change. PRB utilization peaks near 1.5 % of the cell's PRBs, so the rise is relative to a near-idle base and saturation is not shown.

**Acceptable conclusions:** Network congestion building up from rising traffic; Traffic surge driving PRB utilization up; Load-related degradation rather than a radio-quality fault

**Uncertainty:** Synthetic injection; the label is a dataset claim. Spec support is partial: 38.300 describes PRB usage as the load indicator and load balancing as the remedy but does not state that load degrades per-UE throughput. A buffer-overflow fault produces a similar load signature in this data,. Absolute PRB utilization stays low, so this is a relative load rise, not a saturated cell. Method: 'changed' means |shift| > 2x baseline std and > 5 % of |baseline| (BLER also >= 0.01); 'roughly flat' also needs < 20 % relative change. PRB_Utilization is a percent of the cell's 106 PRBs per the TelecomTS paper; in unaffected rows PRBs_*_Current = 1.06 x PRB_Utilization_* exactly, confirming the scale.

**Citations:**

- **3GPP TS 28.552 §5.1.1.2.1 DL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines downlink PRB usage, the measured counterpart of PRB_Utilization_DL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 28.552 §5.1.1.2.2 UL Total PRB Usage** (definition), version 18.7.0, PDF page 38
  - Why cited: Defines uplink PRB usage, the measured counterpart of PRB_Utilization_UL.
  - Page text (start): “ETSI ETSI TS 128 552 V18.7.0 (2024-07)373GPP TS 28.552 version 18.7.0 Release 18 GTP.DelayDlPsaUpfNgranDist.SNSSAI.bin, Where Bin indicates a delay range which is vendor specific, and SNSSAI identifies the S-NSSAI. f) EP_N3 (contained by GNBCUUPFunction). g) Valid for packet switched traffic. h) 5GS. 5.1.1.2 Radio resource utilization 5.1.1.2.1 DL Total PRB Usage a) This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose. The measurement is optionally split into subcounters per PLMN ID and per supported S-NSSAI and per QoS resource…”
- **3GPP TS 38.300 §10.1 Basic Scheduler Operation** (mechanism), version 18.1.0, PDF page 119
  - Why cited: The scheduler assigns resources according to UE buffer status and QoS, so more offered traffic raises PRB use and, once grants cannot keep up, leaves data waiting in the buffer (the last step is inferred, not stated).
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1183GPP TS 38.300 version 18.1.0 Release 18 PLMN), the source gNB shall replace the serving PLMN with the identity of the target PLMN and move the serving PLMN to the equivalent PLMN list, before propagating the roaming and access restriction information. If the Xn handover results in a change of serving SNPN (to an equivalent SNPN), the source gNB shall replace the serving SNPN with the identity of the target SNPN and move the serving SNPN to the equivalent SNPN list, before propagating the roaming and access restriction information. If NG-RAN nodes with …”
- **3GPP TS 38.300 §15.5.1.2 Load reporting for intra-RAT and intra-system inter-RAT load balancing** (mechanism), version 18.1.0, PDF page 140
  - Why cited: Treats per-cell PRB usage as the radio-resource load indicator, so a sustained PRB-usage rise is read as rising load.
  - Page text (start): “ETSI ETSI TS 138 300 V18.1.0 (2024-05)1393GPP TS 38.300 version 18.1.0 Release 18 15.4.3 O&M requirements Operators should be able to configure the energy saving function. The configured information should include: - The ability of an NG-RAN node to perform autonomous cell switch-off; - The ability of an NG-RAN node to request the re-activation of a configured list of inactive cells owned by a peer NG-RAN node. O&M may also configure: - policies used by the NG-RAN node for cell switch-off decision; - policies used by peer NG-RAN nodes for requesting the re-activation of an inactive cell; - The…”

**Model reviews:**

- gpt-oss-120b (OpenRouter): **valid_with_changes** (evidence partial, mechanism ok, conf 0.78)
  - Evidence note: KPI data clearly shows a traffic surge (large increase in packet counts and PRB utilisation) while radio quality remains flat, supporting a load increase. However, the claim of "high network congestion" is not substantiated because PRB utilisation stays very low (~1.5%).
  - Unsupported claim flagged: High Network Congestion (Gradual Buildup) – utilisation remains far from saturation.
  - Requested change: Remove or qualify the claim of "high network congestion" because PRB utilisation remains far below saturation.
  - Requested change: Clarify that the evidence points to a traffic surge causing a modest increase in load, not necessarily congestion.
  - Requested change: Optionally add a citation that directly links traffic volume increase to PRB utilisation (e.g., 3GPP TS 38.321 on MAC scheduling) to strengthen the mechanism argument.
- qwen3.8-27b (Groq): **valid_with_changes** (evidence partial, mechanism ok, conf 0.9)
  - Evidence note: The KPI evidence shows a sharp increase in DL/UL PRB utilization and packet counts, while RSRP and UL_SNR remain roughly flat. This supports a load-related change rather than a radio-quality degradation. However, the absolute PRB utilization values (1.5% and 1.2%) are very low, which contradicts the label 'High Network Congestion'. The conclusion correctly notes this is a 'relative load rise' and not saturation, but the dataset label 'High Network Congestion' is misleading given the low absolute values. The evidence supports 'traffic surge' or 'load increase' but not necessarily 'high congestion' in the traditional sense of resource exhaustion.
  - Unsupported claim flagged: The dataset label 'High Network Congestion' is not fully supported by the low absolute PRB utilization values (1.5% DL, 1.2% UL), which indicate the cell is far from saturated.
  - Requested change: Clarify that the label 'High Network Congestion' is potentially misleading given the low absolute PRB utilization values, and emphasize that the evidence supports a 'relative load increase' or 'traffic surge' rather than saturation-level congestion.
- gemini-3.6-flash (earlier draft, 10 of 12): failed (HTTP 429)

**Human decision:** ☐ approve ☐ approve with edits ☐ reject — reviewer: ____ — date: ____ — notes: ____
