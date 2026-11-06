# Spec clauses for golden RCA citations

Corpus: `data/corpus` (12 docs). Generated 2026-11-04. Every quote below was machine-checked against the pypdf text of the cited **physical** PDF page (whitespace-normalised). The quote **starts** on `pdf_page`; clauses whose heading is on an earlier page are cited at the page holding the quoted sentence (e.g. 38.331 5.5.4.1 heading p246, quote p248; 28.552 A.27 heading p338, quote p339). The page convention (1-based physical index) was verified against the approved annotation `3GPP TS 38.300 9.2.8 -> pdf_page 112` and `28.552 5.1.1.2.5 -> 40` (printed page 39). The E2SM-KPM source is a DOCX with no PDF pages; existing annotations cite it with `docx_heading` instead of `pdf_page`.

**Role rules applied.** `definition` says what a KPI or measurement is. `mechanism` is used only when the text states, or directly describes, how a condition leads to an effect. **Strength:** `strong` means the text states the causal link in its own words. `partial` means the text covers a related mechanism, a narrower scope (IAB, NTN, LTE, inter-system), or only part of the chain, so you have to infer the rest. `none` means nothing usable was found.

**Normative status.** Annex A of TS 28.552 and of TS 32.425 is *informative* ('Use cases for performance measurements'). Those clauses carry most of the mechanism language for interference, BLER and MRO, so cite them as explanatory, not normative. TS 32.425 is LTE (E-UTRAN).

## Spec identities (from manifest.csv)

| spec_id | version | sha256 |
|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` |
| 3GPP TS 38.331 | 18.1.0 | `939befcf81d76ee493a45d8c9a47d32b0863bb0941a7b3ecf63f5907cf61ea3e` |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` |
| O-RAN.WG3.TS.E2SM-KPM | 08.00 | `174c856f804c7af10310be03ecd9879e0c27998c26dea5282624df6fae6dc8b7` |
| O-RAN.WG2.TS.A1TD | 12.00 | `218367734581ba93b0b512608b0fca582322b09684123262220c78dd72074d6b` |

## KPI channel -> definition clause (cross-category)

| Dataset KPI | Best definition clause in corpus | Fit |
|---|---|---|
| RSRP | 28.552 5.1.1.22.1 SS-RSRP distribution per SSB (p117); KPM 7.10.7 | good (distribution/average of UE-reported L1 SS-RSRP) |
| UL_SNR | **none**. Only DL SS-SINR (28.552 5.1.1.32.1 p135; KPM 7.10.8) and UL SRS-RSRP (KPM 7.10.9) | gap |
| DL_BLER / UL_BLER | 28.552 5.1.1.7.2/5.1.1.7.4 (DL, p91) and 5.1.1.7.7 (UL, p92); 28.552 A.18 links TB error rate to BLER | good (ratio of error TBs to total TBs) |
| DL_MCS / UL_MCS | 28.552 5.1.1.12.1 / 5.1.1.12.2 (p100) | good (distribution) |
| PRB_Utilization_DL/UL | 28.552 5.1.1.2.1 / 5.1.1.2.2 (p38) | good |
| PRBs_DL/UL_Current, UL_NPRB | 28.552 5.1.1.2.5 (p40) / 5.1.1.2.7 (p41) | good (mean PRBs used for data) |
| TX_Bytes / RX_Bytes | 28.552 5.1.2.1.1.1 DL PDCP SDU data volume (p142); 5.1.1.3.1/5.1.1.3.3 throughput (p49/p51); KPM 7.10.1/7.10.2 | approximate (layer differs) |
| DL/UL_NumberOfPackets | none as a packet count; drop/loss rates only (28.552 5.1.3.x) | gap |
| Estimated_UL_Buffer | 38.300 10.4 BSR concept (p121). No PM counter | weak |

## 1. Co-Channel Interference (Mild / Severe)

Evidence KPIs: UL_SNR, DL/UL_BLER, DL/UL_MCS, RSRP (flat while quality falls), throughput.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.32.1 | SS-SINR distribution in gNB | 135 | definition | strong | “This measurement provides the distribution of SS-SINR received by gNB from UEs in the cell.” | DL SS-SINR distribution; there is no UL SINR/SNR PM in the corpus, so this is only a proxy for UL_SNR. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.78 | Monitoring of SS-SINR | 352 | definition | strong | “SS-SINR is the ratio of the received signal level and the sum of interference and noise, which is used in 5G NR networks to determine the quality of the radio channel.” | Says SINR includes interference; it does not explain a cause. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.7.2 | Intial error number of DL TBs | 91 | definition | strong | “On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at first HARQ feedback during the period of measurement.” | DL BLER numerator (initial TB errors). |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.7.7 | Error number of UL initial TBs | 92 | definition | strong | “This measurement provides the number of initial faulty TBs on the uplink in a cell.” | UL BLER numerator. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.12.1 | MCS Distribution in PDSCH | 100 | definition | strong | “This measurement provides the distribution of the MCS scheduled for PDSCH RB by NG-RAN.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.12.2 | MCS Distribution in PUSCH | 100 | definition | strong | “This measurement provides the distribution of the MCS scheduled for PUSCH RB by NG-RAN.” |  |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.9 | Monitoring of RF performance | 105 | mechanism | strong | “uplink interference disturbs the base station receiver. Once the base station receiver is compromised, it leads to receiving code errors for the whole base station and the site’s entire service area is degraded including its network and service.” | UL interference -> receiver code errors -> degraded service. Informative annex of the LTE PM spec. |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.7 | Monitoring of interference situation | 104 | mechanism | partial | “In the LTE radio technology interference has to be coordinated on the basis uplink and downlink i.e. in a coordinated usage of the UL resources (Physical Resource Blocks, PRBs) and DL Transmitted Power, which lead to improve SIR and corresponding throughput.” | Coordinating PRBs and power improves SIR and throughput (the reverse of the fault). LTE ICIC context. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.18 | Monitor of BLER performance | 337 | mechanism | partial | “The TB Error Rate in UL/DL can directly reflect the BLER, and has an influence on MCS selection and user throughput.” | BLER -> MCS selection and throughput. Explains the BLER->MCS->throughput step, not interference itself. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 5.2.5.1 | Link adaptation | 43 | mechanism | partial | “For channel state estimation purposes, the UE may be configured to measure CSI-RS and estimate the downlink channel state based on the CSI-RS measurements. The UE feeds the estimated channel state back to the gNB to be used in link adaptation.” | Channel state feedback drives DL link adaptation, so MCS follows channel quality. It never says that lower SINR means lower MCS. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 5.3.5.1 | Link adaptation | 47 | mechanism | partial | “For channel state estimation purposes, the UE may be configured to transmit SRS that the gNB may use to estimate the uplink channel state and use the estimate in link adaptation.” | UL link adaptation is based on the SRS channel estimate. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.27 | Monitoring of RF measurements | 339 | mechanism | partial | “The MCS scheduling strategies of MU-MIMO and SU-MIMO are different due to factors such as user pairing and interference.” | Says interference affects the MCS scheduling strategy (MU-MIMO context). |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 17.2 | Cross-Link Interference Management | 231 | mechanism | partial | “When different TDD DL/UL patterns are used between neighbouring cells, UL transmission in one cell may interfere with DL reception in another cell: this is referred to as Cross Link Interference (CLI).” | A specific inter-cell interference mechanism (TDD pattern mismatch); not generic co-channel. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.104 | Use case of UL remote interference identification | 358 | mechanism | partial | “This type of interference affects the effective reception of SRSs, resulting in a series of problems affecting user experience, such as call drops, access failures, and low rates.” | Remote interference -> call drops, access failures, low rates; specific to atmospheric ducting. |

**Verdict:** ADEQUATE (partial-to-strong). Cite 32.425 A.9 (UL interference -> receiver errors), chained with 28.552 A.18 (BLER -> MCS/throughput) and 38.300 5.3.5.1 (UL link adaptation). Gap: no UL-SINR/UL_SNR PM definition, and no spec separates Mild from Severe.

## 2. High Network Congestion (Gradual Buildup / Sudden Spike)

Evidence KPIs: PRB_Utilization_DL/UL, PRBs_DL/UL_Current, RX/TX_Bytes, packet counts, Estimated_UL_Buffer, throughput per UE.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.1 | DL Total PRB Usage | 38 | definition | strong | “This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.2 | UL Total PRB Usage | 38 | definition | strong | “This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the uplink for any purpose.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.5 | Mean DL PRB used for data traffic | 40 | definition | strong | “This measurement provides the number of physical resource blocks (PRBs) in average used in downlink for data traffic.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.7 | Mean UL PRB used for data traffic | 41 | definition | strong | “This measurement provides the number of physical resource blocks (PRBs) in average used in uplink for data traffic.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.2.1.1.1 | DL Cell PDCP SDU Data Volume | 142 | definition | strong | “This measurement provides the Data Volume (amount of PDCP SDU bits) in the downlink delivered to PDCP layer.” | Volume counterpart of TX_Bytes. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.3.1 | Average DL UE throughput in gNB | 49 | definition | strong | “This measurement provides the average UE throughput in downlink.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.3.3 | Average UL UE throughput in gNB | 51 | definition | strong | “This measurement provides the average UE throughput in uplink.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.4.1 | Mean number of RRC Connections | 62 | definition | strong | “This measurement provides the mean number of users in RRC connected mode for each NR cell during each granularity period.” | Load-side counter (users in connected mode). |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 3.1 | Definitions | 28 | definition | partial | “Packet drop rate: share of packets that were not sent to the target due to high traffic load or traffic management and should be seen as a part of the packet loss rate.” | The definition names high traffic load as a cause of drops. It is still a definition, so it cannot be the sole support for a cause. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.1.1 | General | 140 | mechanism | partial | “The objective of mobility load balancing is to distribute load evenly among cells and among areas of cells, or to transfer part of the traffic from congested cell or from congested areas of cells” | Under 15.5.1 Support for MLB. Describes a congested cell and the load-balancing remedy; it says nothing about how load degrades KPIs. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.1.2 | Load reporting for intra-RAT and intra-system inter-RAT load balancing | 140 | mechanism | partial | “Radio resource usage (per-cell and per SSB area PRB usage: DL/UL GBR PRB usage, DL/UL non-GBR PRB usage, DL/UL total PRB usage, and DL/UL scheduling PDCCH CCE usage;” | PRB usage is the load indicator exchanged between nodes. This supports PRB usage as a congestion signal. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.1.3 | Load balancing action based on handovers | 141 | mechanism | partial | “The source cell may initiate handover due to load. The target cell performs admission control for the load balancing handovers.” | Load-triggered handover plus target admission control. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.1 | Basic Scheduler Operation | 119 | mechanism | partial | “Taking into account the UE buffer status and the QoS requirements of each UE and associated radio bearers, schedulers assign resources between UEs; - Schedulers may assign resources taking account the radio conditions at the UE identified through measurements made at the gNB and/or reported by the UE;” | The scheduler shares resources among UEs by buffer status and QoS. With more demand, each UE gets a smaller share; this is implied, not stated. |
| O-RAN.WG2.TS.A1TD | 12.00 | `218367734581ba93b0b512608b0fca582322b09684123262220c78dd72074d6b` | 6.3.3.6 | Load balancing targets | 28 | definition | partial | “The target downlink PRB usage in percent. The denominator is the total number of PRBs in the cell, and the numerator is the number of PRBs specified by dlPrbUsgType.” | O-RAN A1 policy target for PRB usage (a policy definition, not a mechanism). |

**Verdict:** PARTIAL. Definitions are strong. The only mechanism text links congestion to PRB usage and load balancing (38.300 15.5.1.x) and to drops (28.552 3.1/5.1.3.2.1). No clause says that high load lowers per-UE throughput or raises delay, and nothing separates Gradual Buildup from Sudden Spike.

## 3. Buffer Overflow (Gradual Buildup)

Evidence KPIs: Estimated_UL_Buffer, UL_NPRB, UL_NumberOfPackets, RX_Bytes, drops.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.4 | Measurements to Support Scheduler Operation | 121 | definition | strong | “Uplink buffer status reports (BSR) are needed to provide support for QoS-aware packet scheduling. In NR, uplink buffer status reports refer to the data that is buffered in for a group of logical channels (LCG) in the UE.” | The closest definition for Estimated_UL_Buffer (the BSR concept). No PM counter in the corpus reports UE buffer size. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.4 | Measurements to Support Scheduler Operation | 121 | mechanism | partial | “When a BSR is triggered (e.g. when new data arrives in the transmission buffers of the UE), a Scheduling Request (SR) can be transmitted by the UE (e.g. when no resources are available to transmit the BSR).” | Data arrival triggers BSR/SR; this is the buffer -> scheduling-request linkage. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.1 | Basic Scheduler Operation | 119 | mechanism | partial | “Taking into account the UE buffer status and the QoS requirements of each UE and associated radio bearers, schedulers assign resources between UEs; - Schedulers may assign resources taking account the radio conditions at the UE identified through measurements made at the gNB and/or reported by the UE;” | UL grants depend on the reported buffer status, so a growing buffer with flat UL_NPRB means demand is not being served (inferred). |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.3.2.1 | DL PDCP SDU Drop rate in gNB-CU-UP | 148 | definition | partial | “This measurement provides the fraction of PDCP SDU packets which are dropped on the downlink, due to high traffic load, traffic management etc in the gNB-CU-UP.” | DL PDCP drops 'due to high traffic load'. DL, gNB-side. |
| O-RAN.WG3.TS.E2SM-KPM | 08.00 | `174c856f804c7af10310be03ecd9879e0c27998c26dea5282624df6fae6dc8b7` | 7.10.5 | Distribution of DL Packet Drop Rate | n/a (DOCX; cite `docx_heading`) | definition | partial | “This measurement provides the fraction of RLC SDU packets which are dropped on the downlink, due to high traffic load, traffic management etc in the gNB-DU.” | O-RAN RLC-level DL drop-rate counter, same wording as above. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 4.7.3.2 | Flow and Congestion Control | 34 | mechanism | partial | “Flow and congestion control can be supported in both upstream and downstream directions in order to avoid congestion-related packet drops on IAB-nodes and IAB-donor-DU:” | Flow control exists 'to avoid congestion-related packet drops', and feedback is sent when 'buffer load exceeds a certain threshold' (p35). IAB-only scope. |

**Verdict:** PARTIAL/WEAK. Nothing in the corpus states that a UE or gNB buffer overflows and then discards data. The only explicit buffer-load -> drop link is IAB-specific (38.300 4.7.3.2). BSR and MAC buffer semantics live in TS 38.321 (missing); RLC/PDCP discard live in TS 38.322/38.323 (missing).

## 4. Resource Allocation Bugs

Evidence KPIs: PRBs_DL/UL_Current vs PRB_Utilization, UL_NPRB vs Estimated_UL_Buffer, MCS vs SINR inconsistency.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.1 | Basic Scheduler Operation | 119 | mechanism | partial | “Taking into account the UE buffer status and the QoS requirements of each UE and associated radio bearers, schedulers assign resources between UEs; - Schedulers may assign resources taking account the radio conditions at the UE identified through measurements made at the gNB and/or reported by the UE;” | States the intended behaviour (allocate by buffer status, QoS and radio conditions). A bug is a deviation from it; the spec does not describe bugs. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 10.4 | Measurements to Support Scheduler Operation | 121 | mechanism | partial | “Uplink buffer status reports (BSR) are needed to provide support for QoS-aware packet scheduling. In NR, uplink buffer status reports refer to the data that is buffered in for a group of logical channels (LCG) in the UE.” | BSR exists to support QoS-aware scheduling, so an allocation that ignores BSR contradicts the intended behaviour. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 5.2.5.1 | Link adaptation | 43 | mechanism | partial | “For channel state estimation purposes, the UE may be configured to measure CSI-RS and estimate the downlink channel state based on the CSI-RS measurements. The UE feeds the estimated channel state back to the gNB to be used in link adaptation.” | MCS should follow the channel-state feedback. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.5 | Mean DL PRB used for data traffic | 40 | definition | strong | “This measurement provides the number of physical resource blocks (PRBs) in average used in downlink for data traffic.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.7 | Mean UL PRB used for data traffic | 41 | definition | strong | “This measurement provides the number of physical resource blocks (PRBs) in average used in uplink for data traffic.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.2.1 | DL Total PRB Usage | 38 | definition | strong | “This measurement provides the usage (in percentage) of physical resource blocks (PRBs) on the downlink for any purpose.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.12.1 | MCS Distribution in PDSCH | 100 | definition | strong | “This measurement provides the distribution of the MCS scheduled for PDSCH RB by NG-RAN.” |  |

**Verdict:** WEAK. Specs describe correct scheduler behaviour only; no clause describes misallocation or its symptoms. The best you can cite is 'expected behaviour' (38.300 10.1 + 10.4) next to the definitions. Implementation faults are out of scope for 3GPP/O-RAN text.

## 5. Faulty Handover Algorithm (Too Frequent)

Evidence KPIs: RSRP oscillation, throughput dips, BLER spikes around HO. The dataset has no HO counters, so HO events have to be inferred.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.25.1 | Handover failures related to MRO for intra-system mobility | 121 | definition | strong | “The measurement includes separate counters for various handover failure types, classified as "Intra-system too early handover", "Intra-system too late handover" and "Intra-system handover to wrong cell".” | HO.IntraSys.TooEarly/TooLate/ToWrongCell counters; not in the dataset. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.25.4 | Handover ping-pong for inter-system mobility | 123 | definition | partial | “An example of handover ping-pong occurred when a UE is handed over from a cell in a source system (e.g. NG-RAN) to a cell in a target system different from the source system (e.g. E-UTRAN), then within a predefined limited time the UE is handed over back to a cell in the source system” | Ping-pong counter, inter-system only. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.22.1 | SS-RSRP distribution per SSB | 117 | definition | strong | “This measurement provides the distribution of SS-RSRP per SSB (see TS 38.215 [34]) received by gNB from UEs in the cell” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.66 | Monitoring of MRO performance | 349 | mechanism | strong | “5G NR cells may experience issues, such as too early or too late handover, handover to wrong cell, ping-pong handover, that not only impact user experience, but also waste network resources, if handover parameters are not set properly.” | Explicit: improperly set HO parameters -> too early/late, wrong cell, ping-pong -> user impact and wasted resources. Informative annex. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.2.2.2 | Connection failure due to intra-system mobility | 142 | mechanism | strong | “Intra-system Too Early Handover: an RLF occurs shortly after a successful handover from a source cell to a target cell or a handover failure occurs during the handover procedure; the UE attempts to re-establish the radio link connection in the source cell.” | Too-early HO -> RLF shortly after HO (normative definition of the failure type). |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.2.2.2 | Connection failure due to intra-system mobility | 142 | mechanism | strong | “Intra-system Too Late Handover: an RLF occurs after the UE has stayed for a long period of time in the cell; the UE attempts to re-establish the radio link connection in a different cell.” |  |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.2.4 | Inter-system Ping-pong | 145 | mechanism | partial | “A UE is handed over from a cell in a source system (e.g. 5GS) to a cell in a target system different from the source system (e.g. EPS), then within a predefined limited time the UE is handed over back to a cell in the source system” | The ping-pong problem is defined for inter-system mobility only; there is no intra-NR ping-pong clause. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 15.5.2.1 | General | 142 | mechanism | partial | “MRO provides means to distinguish the above problems from NR coverage related problems and other problems, not related to mobility.” | Under 15.5.2 Support for MRO. MRO separates mobility problems from coverage problems; useful for differential diagnosis. |
| 3GPP TS 38.331 | 18.1.0 | `939befcf81d76ee493a45d8c9a47d32b0863bb0941a7b3ecf63f5907cf61ea3e` | 5.5.4.4 | Event A3 (Neighbour becomes offset better than SpCell) | 255 | mechanism | partial | “Inequality A3-1 (Entering condition) Mn + Ofn + Ocn – Hys > Mp + Ofp + Ocp + Off” | Shows that HO triggering depends on Hys/offsets. 'Small Hys -> more frequent triggering' follows from the inequality but is not stated. |
| 3GPP TS 38.331 | 18.1.0 | `939befcf81d76ee493a45d8c9a47d32b0863bb0941a7b3ecf63f5907cf61ea3e` | 5.5.4.1 | General | 248 | mechanism | partial | “is fulfilled for one or more applicable cells for all measurements after layer 3 filtering taken during timeToTrigger defined for this event within the VarMeasConfig” | Under 5.5.4 Measurement report triggering (heading p246). Triggering requires the condition to hold for the whole timeToTrigger; the same inference applies to short TTT. |

**Verdict:** ADEQUATE (strong). Cite 28.552 A.66 together with 38.300 15.5.2.2.2. Caveat: 'too frequent' (intra-NR ping-pong) is only covered by A.66's list; 38.300 defines ping-pong for inter-system mobility only. The dataset has no HO counters, so the definition side relies on RSRP/throughput.

## 6. Antenna Failure

Evidence KPIs: RSRP drop (step), UL_SNR drop, MCS collapse, throughput collapse.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.22.1 | SS-RSRP distribution per SSB | 117 | definition | strong | “This measurement provides the distribution of SS-RSRP per SSB (see TS 38.215 [34]) received by gNB from UEs in the cell” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.32.1 | SS-SINR distribution in gNB | 135 | definition | strong | “This measurement provides the distribution of SS-SINR received by gNB from UEs in the cell.” |  |
| 3GPP TS 38.331 | 18.1.0 | `939befcf81d76ee493a45d8c9a47d32b0863bb0941a7b3ecf63f5907cf61ea3e` | 5.3.10.1 | Detection of physical layer problems in RRC_CONNECTED | 187 | mechanism | partial | “upon receiving N310 consecutive "out-of-sync" indications for the SpCell from lower layers while neither T300, T301, T304, T311, T316 nor T319 are running: 2> start timer T310 for the corresponding SpCell.” | Physical-layer out-of-sync -> T310 started; generic radio-problem detection. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 9.2.7 | Radio Link Failure | 111 | mechanism | partial | “The UE declares Radio Link Failure (RLF) when one of the following criteria are met: - Expiry of a radio problem timer started after indication of radio problems from the physical layer (if radio problems are recovered before the timer is expired, the UE stops the timer); or” | Radio problems -> RLF. Generic; antennas are not mentioned. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 9.2.8 | Beam failure detection and recovery | 112 | mechanism | partial | “the UE declares beam failure when the number of beam failure instance indications from the physical layer reaches a configured threshold before a configured timer expires.” | Degraded beam -> beam failure declared. Beam-level only, no hardware cause. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.64 | Monitoring of RF performance | 348 | mechanism | partial | “In 5G NR, gNB cells transmit many narrow beams targeting UEs in the cell that result in better link budget and lower interference. However, some areas between beams of neighbouring NR cells , or between the NR cell and the neighbor E-UTRA cell may experience poor coverage or coverage holes.” | Narrow beams improve link budget, and gaps between beams give coverage holes. This is a coverage mechanism, not a failure one. |

**Verdict:** NOT ADEQUATE. No clause links antenna or hardware faults to RF symptoms. Only generic 'signal degrades -> RLF/beam failure' clauses (partial) exist. Equipment alarms (FM, e.g. TS 28.545 / 32.111) and BS RF specs (TS 38.104) are not in the corpus.

## 7. Faulty RF Filters (Temporal)

Evidence KPIs: UL_SNR / BLER degradation in time windows, RSRP mostly unaffected.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.7.7 | Error number of UL initial TBs | 92 | definition | strong | “This measurement provides the number of initial faulty TBs on the uplink in a cell.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.32.1 | SS-SINR distribution in gNB | 135 | definition | strong | “This measurement provides the distribution of SS-SINR received by gNB from UEs in the cell.” | DL proxy only. |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.7 | Monitoring of interference situation | 104 | mechanism | partial | “Use cases for the related interference measurements are e.g. optimisation of ICIC related RRM functionality, the detection of long distance interferer and the interference due to spurious emissions of neighbour cells.” | Mentions 'interference due to spurious emissions of neighbour cells' as a use case. Closest to a filtering defect, but it is a phrase only, with no causal description. |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.9 | Monitoring of RF performance | 105 | mechanism | partial | “uplink interference disturbs the base station receiver. Once the base station receiver is compromised, it leads to receiving code errors for the whole base station and the site’s entire service area is degraded including its network and service.” | Interference -> receiver code errors; applies if a filter fault lets interference in. Inferred. |

**Verdict:** NOT ADEQUATE. Nothing in the corpus describes RF filter behaviour or failure. BS/UE RF requirements (ACLR, blocking, selectivity, spurious emissions) are in TS 38.104 / 38.101-1, which are missing. Treat any filter cause as an uncited hypothesis.

## 8. Doppler Shift (Severe)

Evidence KPIs: DL/UL_BLER up, MCS down, SNR down, RSRP roughly unchanged.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.7.2 | Intial error number of DL TBs | 91 | definition | strong | “On receipt by the gNB of a NACK or DTX from UE which indicates a faulty reception of TB by UE at first HARQ feedback during the period of measurement.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.12.1 | MCS Distribution in PDSCH | 100 | definition | strong | “This measurement provides the distribution of the MCS scheduled for PDSCH RB by NG-RAN.” |  |
| 3GPP TS 38.331 | 18.1.0 | `939befcf81d76ee493a45d8c9a47d32b0863bb0941a7b3ecf63f5907cf61ea3e` | 6.3.2 | HighSpeedConfig | 786 | mechanism | partial | “If the field is present and UE supports demodulationEnhancement-r16, the UE shall apply the enhanced demodulation processing for HST-SFN joint transmission scheme with velocity up to 500km/h as specified in TS 38.101-4 [59].” | IE inside 6.3.2 RRC information elements. Shows that high-speed (up to 500 km/h) operation needs an enhanced-demodulation configuration. It implies a speed-related impairment but never says Doppler degrades BLER. Lean toward none. |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 16.14.2.2 | Timing Advance and Frequency Pre-compensation | 210 | mechanism | partial | “The UE shall compute the frequency Doppler shift of the service link, and autonomously pre-compensate for it in the uplink transmissions, by considering UE position and the ephemeris.” | Doppler pre-compensation, but for NTN satellite links only; not terrestrial mobility. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.18 | Monitor of BLER performance | 337 | mechanism | partial | “The TB Error Rate in UL/DL can directly reflect the BLER, and has an influence on MCS selection and user throughput.” | BLER -> MCS/throughput (downstream step only). |

**Verdict:** NOT ADEQUATE. No terrestrial clause explains Doppler -> channel ageing -> BLER/MCS. 38.331 HighSpeedConfig and 38.300 16.14.2.2 (NTN) only show that Doppler or speed is handled. Requirements and channel models (TS 38.101-4, 38.133, 38.211, TR 38.901) are missing.

## 9. Jamming (real)

Evidence KPIs: UL_SNR collapse, UL_BLER up, UL_MCS down, possibly RSRP unaffected.

| spec_id | version | sha256 | clause | title | pdf_page | role | strength | verbatim quote (≤60 words) | note |
|---|---|---|---|---|---|---|---|---|---|
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.32.1 | SS-SINR distribution in gNB | 135 | definition | strong | “This measurement provides the distribution of SS-SINR received by gNB from UEs in the cell.” | DL proxy. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.7.7 | Error number of UL initial TBs | 92 | definition | strong | “This measurement provides the number of initial faulty TBs on the uplink in a cell.” |  |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | 5.1.1.12.2 | MCS Distribution in PUSCH | 100 | definition | strong | “This measurement provides the distribution of the MCS scheduled for PUSCH RB by NG-RAN.” |  |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.9 | Monitoring of RF performance | 105 | definition | partial | “Interference signals in a number of different ways, including frequency, frequency band, direction of interfering signals, etc. can also be classified by System internal interference and External interference.” | Classifies interference into system-internal and external; jamming fits 'external' but is not named. |
| 3GPP TS 32.425 | 18.0.0 | `b3e04ba07ffbcb1d6963abe2e90cd7876de6ab61dbd847f5cd2e8c7e085e0fd9` | A.9 | Monitoring of RF performance | 105 | mechanism | strong | “uplink interference disturbs the base station receiver. Once the base station receiver is compromised, it leads to receiving code errors for the whole base station and the site’s entire service area is degraded including its network and service.” | UL interference compromises the BS receiver -> code errors -> whole service area degraded. Strongest generic support for an external-interference cause. |
| 3GPP TS 28.552 | 18.7.0 | `9785644b0c5a3060ced870ba3470ea3f02212262d518b2b919b1c40ec0e9b5b6` | A.104 | Use case of UL remote interference identification | 358 | mechanism | partial | “This type of interference affects the effective reception of SRSs, resulting in a series of problems affecting user experience, such as call drops, access failures, and low rates.” | Interference -> call drops, access failures, low rates (remote-interference context). |
| 3GPP TS 38.300 | 18.1.0 | `058188999eb469ae95097a3b29b3abd24e65470a10f60d9d8965f7610c0de8f1` | 16.18.5 | Interference Detection and Mitigation for Aerial UE Communication | 219 | mechanism | partial | “For interference detection, an Aerial UE can be configured with RRM event A3, A4, A5 or AxHy that triggers measurement report when individual (per cell) RSRP/RSRQ/SINR values (for events A3, A4, A5)” | An interference-detection mechanism (aerial UE) via A3/A4/A5 reporting; detection only. |

**Verdict:** PARTIAL. 'Jamming' / intentional interference does not appear anywhere in the corpus. You can cite 32.425 A.9 (external UL interference -> receiver errors) for the effect, but not for intent or source. The spec text cannot tell jamming from co-channel interference.

## Summary of mechanism adequacy

| Category | Mechanism support | Best definition + mechanism pair |
|---|---|---|
| Co-Channel Interference | adequate (partial-strong) | 28.552 5.1.1.7.7 (UL TB errors, p92) + 32.425 A.9 (p105), chained with 28.552 A.18 (p337) |
| High Network Congestion | partial | 28.552 5.1.1.2.1 (p38) + 38.300 15.5.1.1/15.5.1.2 (p140) |
| Buffer Overflow | partial/weak | 38.300 10.4 BSR (p121) + 38.300 10.1 (p119); 4.7.3.2 (p34) is IAB-only |
| Resource Allocation Bugs | weak (expected behaviour only) | 28.552 5.1.1.2.7 (p41) + 38.300 10.1 (p119) |
| Faulty HO (Too Frequent) | adequate (strong) | 28.552 5.1.1.25.1 (p121) + 28.552 A.66 (p349) / 38.300 15.5.2.2.2 (p142) |
| Antenna Failure | not adequate | 28.552 5.1.1.22.1 (p117) + 38.300 9.2.7 (p111) (generic only) |
| Faulty RF Filters | not adequate | 28.552 5.1.1.7.7 (p92) + 32.425 A.7 (p104) ('spurious emissions' mention only) |
| Doppler Shift | not adequate | 28.552 5.1.1.7.2 (p91) + 38.331 6.3.2 HighSpeedConfig (p786) (config only) |
| Jamming | partial | 28.552 5.1.1.7.7 (p92) + 32.425 A.9 (p105) |

## Relevant specs MISSING from the corpus (cannot be cited)

| Spec | Why it matters |
|---|---|
| TS 38.321 (NR MAC) | BSR triggering/format, SR, HARQ, beam-failure MAC procedure. The definition and mechanism for Estimated_UL_Buffer and buffer overflow |
| TS 38.322 (RLC) / TS 38.323 (PDCP) | RLC max-retx, SDU discard timer, i.e. the actual buffer-overflow and discard mechanism |
| TS 38.214 (PHY data procedures) | MCS tables, CQI/CSI, link adaptation; MCS<->SINR/BLER relationship |
| TS 38.215 (PHY measurements) | Normative definitions of SS-RSRP, SS-SINR, SRS-RSRP (28.552 and KPM only reference it) |
| TS 38.213 (PHY control) | RLM in-sync/out-of-sync generation (clause 5), beam failure detection, UL power control |
| TS 38.133 (RRM requirements) | Qout/Qin thresholds, HST measurement requirements, RSRP/SINR reporting ranges |
| TS 38.104 / TS 38.101-1 / 38.101-4 (BS/UE RF and demod) | Receiver blocking/selectivity, ACLR, spurious emissions (RF filters, jamming), HST/Doppler demodulation |
| TS 38.211 / TR 38.901 | Numerology/Doppler tolerance; channel models with Doppler |
| TS 28.554 (5G E2E KPIs) | KPI formulas built on the 28.552 counters |
| TS 28.313 / TS 28.627-628 / TS 32.500-522 (SON/MRO/MLB management) | Operational MRO and MLB detection logic referenced by 28.552 5.1.1.25 |
| TS 38.423 (XnAP) | Resource Status Reporting (load info for MLB), Failure Indication / HO Report (MRO) |
| TS 37.817 / TR 38.866 (RIM) | Remote interference details |
| 36-series (36.300, 36.331, 36.321, 36.214, 36.133) | LTE counterparts; 32.425 LTE annex text refers to them |
| TS 28.545 / TS 32.111 (fault supervision, alarms) | Needed for hardware/antenna failure evidence (FM, not PM) |
| O-RAN WG1 Use Cases Analysis Report / WG2 AI-ML / WG3 RIC use cases (e.g. traffic steering, interference, QoE) | O-RAN use-case narratives with fault->KPI mechanisms. The corpus O-RAN docs (A1AP, A1TD, E2AP, E2GAP, OAD, KPM, Overview) are interface, type and architecture specs with **no** causal mechanism text |

## Notes and caveats
- 38.300 15.5.2.4 (and 28.552 5.1.1.25.4) define ping-pong for **inter-system** mobility only. Intra-NR 'too frequent' handover is covered only by the informative list in 28.552 A.66.
- 28.552 3.1 and 5.1.3.2.1 phrase drops as 'due to high traffic load'. That wording sits inside a definition; under the role rules it still cannot be the sole support for a congestion or overflow cause.
- O-RAN corpus docs searched: none contain interference, congestion, Doppler or jamming mechanism text. E2SM-KPM 7.10.x gives counter definitions only; A1TD 6.3.3.6 gives load-balancing policy targets only.
- Corpus-wide searches for 'jamm', 'Doppler' (terrestrial), 'RF filter' and 'antenna fail' returned no usable hits.
