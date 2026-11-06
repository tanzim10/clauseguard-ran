# TelecomTS — research notes for writing evidence-grounded RCA cases

Date: 2026-11-04. Read-only study of the local TelecomTS data revision
`1b3a88a440f4922edfcd1e230e42c003c0e13bdf`, the TelecomTS paper (arXiv 2510.06063v3, full HTML
text incl. Appendices B–F) and the HF card `AliMaatouk/TelecomTS`.

Evidence tags: **[paper]** = quoted or paraphrased from the paper; **[card]** = HF card;
**[repo]** = repo code/docs; **[measured]** = computed from local data (method stated).
Ephemeral analysis scripts used for the measured tags can be re-derived from the methods described
here.

---

## 1. Testbed and collection

| Item | Value | Source |
|---|---|---|
| RAT | **5G NR standalone**, n78 TDD, 38.16 MHz BW centred at 3.319 GHz | [paper] App. B.1 |
| Stack | **OpenAirInterface (OAI)** gNB (monolithic) and OAI 5G core (AMF/SMF/UPF/AUSF/NRF/UDR/UDM on Kubernetes). Not srsRAN, not Amarisoft | [paper] |
| Radio | USRP N300 with 2× UBX boards, 2×2 beamforming over a 10 GbE fronthaul. Jammer: USRP X310 running GNU Radio | [paper] |
| UE | Google Pixel 6 / 7 with programmable SIMs. **One gNB, one cell** | [paper] |
| Zones | ~70 m² lab. Zone A < 3 m from RU, B 3–6 m, C > 6 m. Raw `description.txt` calls them "zone 1/2/3" | [paper], [measured] |
| Mobility | Static, or walking at 5 km/h ("mobile" recordings have no zone) | [paper] |
| Apps | YouTube (buffered video), Twitch (live), HTTP large-file download. Congestion = a second UE running concurrent downloads | [paper] |
| KPIs | 18 (16 numeric + `UL_Protocol`/`DL_Protocol` categorical), from gNB logs, the UE (RSRP) and a Wireshark capture at the core (protocol and packet counts). The two traces were aligned by histogram/KL matching of TX bytes vs DL packet counts (offset ≈ 2 s) | [paper] App. B.2 |
| Sampling | **100 ms (10 Hz)**. `kpi_window.sampling_rate_hz = 10.0` in every record | [paper], [measured] |
| Window | **128 steps = 12.8 s**. The record's `end_time - start_time` = 12.7 s | [paper], [measured] |
| Stride (paper) | "sliding window of length 128 time steps with a stride of 32" | [paper] §3.3 |
| Volume | 32,000 windows: 30,765 normal, 956 synthetic-fault, 279 real-jamming | [repo] quality.json |
| Timestamps | Not wall-clock. They are relative or synthetic: jammer files start 2025-01-01/02/03, some congestion files 2036-xx, and synthetic files 2026-xx | [measured] |

Session lengths in the raw data: normal no-congestion recordings have 99,999 rows (≈2.8 h). Congestion,
mobile, jammer and synthetic recordings have 9,999 rows each (≈16.7 min). This matches Table 8
("100k/zone" and "10k"). The paper's "four hours" per session does not match the published raw
files [measured].

### Stride and overlap, verified from `release/window_sources.csv` [measured]

The check uses consecutive `raw_start_row` differences per source file. All windows are 128 raw rows,
and all 32,000 windows matched their raw `metrics.csv` rows with max abs error 0.

| Source type | Dominant stride | Overlap of consecutive windows | Notes |
|---|---|---|---|
| Normal (all 21 files) | 32 rows (3.2 s) | **96/128 rows = 75 %** | About 3–4 % of gaps are 64 or 96 (dropped windows) |
| Synthetic (9 files) | 32 rows | **75 %** | Many larger gaps, because only windows containing an anomaly were kept and windows with two anomaly types were dropped [paper App. C] |
| Jammer (3 files) | **100 rows (10 s)** | **28/128 rows = 22 %** | Windows are phase-locked to the jammer cycle (see §5). This is not the paper's stride 32 |

So any one synthetic anomaly event appears in several overlapping windows. In dev/test, 393
synthetic windows come from only **41 distinct anomaly events** (§3).

---

## 2. Channel table (16 numeric channels = `contracts.NUMERIC_CHANNELS`)

Every channel has `unit: null` in the repo: `csv_schema.json`, each record's
`kpi_window.channels.*.unit`, and `unknown_units` in `evidence.json`/`splits.json`. The adapter only
fills units from an optional sidecar (`telecomts.py:52`), and none was supplied. The units below come
from the paper's App. B KPI table plus inference, marked as such.

| Channel | Meaning (paper) | Layer / reporter | Unit (paper range) | Direction semantics |
|---|---|---|---|---|
| RSRP | Reference Signal Received Power | PHY / UE | dBm (paper says only "[-140, -45]"; the descriptions mix "dB" and "dBm") | Higher (less negative) = better coverage. Recording-level medians are a zone/position proxy (normal Zone A -87 to -102, Zone B -104 to -113, jammer "zone 1" -73 to -78) |
| UL_SNR | Uplink SNR | PHY (paper says "reported by UE", but UL SNR is measured at the gNB) | dB, [-3.5, 60] | Higher = better |
| DL_BLER / UL_BLER | Block error rate | MAC / gNB | fraction [0, 1] | Higher = worse. **Zero-inflated**: DL_BLER median is 0 in normal Zone A recordings |
| DL_MCS / UL_MCS | Average MCS index | MAC / gNB | index [0, 27] (float averages) | Higher = better link. DL_MCS is ~constant 7 in almost all data |
| UL_NPRB | UL PRBs allocated to the UE per TTI | MAC / gNB | PRBs [0, 105]. Paper says int, but data has floats (e.g. 20.77) | Higher = more UL grant |
| Estimated_UL_Buffer | BSR-estimated UL buffer at the UE | MAC / gNB | presumably bytes, [0, 250k]. LLM descriptions call it "KB" (likely wrong) | Higher = UL backlog |
| PRBs_DL_Current / PRBs_UL_Current | PRBs allocated in DL/UL this TTI | MAC / gNB | PRBs [0, 105] | Higher = more load or allocation |
| PRB_Utilization_DL / _UL | % of PRBs used by the UE | MAC / gNB | % [0, 100] | Higher = more load. **Identity in real data: `PRBs_X_Current = 1.06 × PRB_Utilization_X`** (106 PRBs at 38.16 MHz/30 kHz) [measured] |
| TX_Bytes | Bytes transmitted by the gNB = **downlink** volume per 100 ms sample | MAC / gNB | bytes. Paper range "[0, 450M]" reads like a total, but the values are per sample (File DL ≈ 440 kB/100 ms ≈ 35 Mb/s) | Higher = more DL throughput |
| RX_Bytes | Bytes received by the gNB = **uplink** volume per sample | MAC / gNB | bytes per 100 ms | Higher = more UL throughput |
| DL_NumberOfPackets / UL_NumberOfPackets | User-plane packets seen at the UPF | Network / UPF | packets per 100 ms, [0, 10k] | Higher = more traffic |

TX = DL and RX = UL follows from the gNB perspective. The paper aligned "transmitted byte counts
(from the Physical layer) with the downlink packet counts", and File-download windows show TX ≈ 440 kB
vs RX ≈ 6 kB. **The GPT-generated `reference.description` gets this wrong**: it calls TX_Bytes "uplink"
and RX_Bytes "downlink" (e.g. the jamming example below). Do not copy direction semantics from the
descriptions.

Paper Table 1 says "10 float, 6 integer, 2 categorical". The per-KPI types in App. B do not quite add
up to that, and the data stores everything as floats.

---

## 3. Fault categories: generation, designed signature, and what the data actually shows

### 3a. How synthetic anomalies were made [paper App. C]

- Real recordings were transformed: *"we alternate between sampling from two exponential
  distributions, one to get the anomaly inter-arrival time … and one to get the anomaly duration …
  For each timestamp (s_i, t_i), we will assign it some anomaly type a_i. Then, for the relevant
  affected KPIs … we apply a function to get our transformed data."*
- There are 8 function types (Table 9): constant addition, constant multiplication, linear growth,
  exponential growth, logistic growth, logarithmic decay, additive sinusoid and multiplicative sinusoid.
  Values are clipped to sampled "soft bounds" (based on the 20th lowest/highest point), and noise is
  injected at saturation. Exponential growth is applied to a kernel-smoothed trend with the residuals
  added back.
- *"Given an affected KPI of an anomaly, we use GPT-4.1 to generate a range of feasible values for each
  parameter … at generation time, we will uniformly sample within this range … We repeatedly verify
  these parameters using human feedback."*
- The paper's examples of the mappings: *"co-channel interference is modeled to degrade SNR, BLER, and
  PRB utilization …; congestion and buffer overflow exhibit PRB saturation and declining throughput …;
  Doppler-induced channel aging leads to lower MCS and elevated BLER …; and ping-pong handover results
  in oscillatory signal quality and increased packet loss."*
- Windowing: *"we slice all anomalous time series sequences into sequences of length 128 with a stride
  of 32. We remove all samples that no longer contain anomalous data points. We also remove all samples
  containing two different types of anomalies."*
- Table 10 (paper names differ from data labels): Antenna Failure (Hardware, temporal, 13 KPIs);
  Buffer Overflow (SW/Infra, temporal, 10); CCI Mild (Infra, **static**, 9); CCI Severe (Infra,
  **static**, 11); Doppler Shift (Env, temporal, 7); Faulty Handover (Frequent) (SW, temporal, 9);
  Faulty RF Filters (HW, temporal, 9); High Network Congestion (Static) (Usage, **static**, 10)
  = data "Sudden Spike"; High Network Congestion (Temporal) (Usage, temporal, 10) = data "Gradual
  Buildup"; Resource Allocation Bugs (SW, temporal, 9). The KPI counts match the stored
  `affected_channels` lengths.
- The paper does **not** publish per-category function/parameter tables. The closest thing to the
  injection recipe is the ticket's **Symptoms** line. App. C: *"The anomaly impact variable inputs a
  fixed textual description for each anomaly that lists how every affected metric changes under the
  anomaly."* I verified that each category has exactly **one** distinct Symptoms string and one Root
  Cause string across all its windows [measured]. These strings are quoted in the table below.

### 3b. Per-category table

The "Designed signature" column quotes the ticket Symptoms line (= the paper's fixed "anomaly impact"
text). "Measured" is from dev/test windows with ≥ 8 pre-anomaly rows: per window, mean(span) vs
mean(pre = rows < start_index) for each channel. up%/down% = share of windows where the shift exceeds
0.5 × pre-std (floored at 5 % of |pre mean|). Ratios are medians of mean(span)/mean(pre). Δ is the
median absolute difference. "n pre" is the number of windows usable for this (many start at index 0).
"Events" counts distinct raw-row anomaly spans after merging overlapping windows (dev+test).

| Category (data label) | dev+test windows / events | Designed signature (ticket "Symptoms", verbatim-ish) | Measured (dev/test, n pre) | Consistency flags |
|---|---|---|---|---|
| **Antenna Failure** | 42 / 6 | RSRP ↓ 10–20; UL SNR ↓ 5–15; UL&DL BLER "surged exponentially"; UL&DL MCS ↓ 5–15; PRBs DL ↓ 30–80 %, UL ↓ 50–80 %; TX/RX bytes "declined logarithmically"; UL buffer ↑ exponentially; UL packets ↑ 50–200 %, DL packets ↑ 30–100 % | n=21. RSRP ↓ in 100 % (Δ ≈ -15 dB). UL_SNR ↓ 90 % (Δ -7.3 dB). DL_MCS ↓ 95 % (**ratio 0: 7 → 0**, clipped at the floor). UL_MCS ↓ 95 % (×0.55). UL packets ↑ 100 % (×3.2), DL packets ↑ 95 % (×2). PRBs_DL ↓ 48 % (×0.51). DL_BLER ↑ 62 % (×1.6), UL_BLER ↑ 57 % (×1.25). Bytes ↓ only 19–29 %. UL buffer ↑ only 38 % | **Packets rise while bytes fall** (designed, physically odd). "Exponential" BLER surge is modest. DL_MCS goes to 0 because a 5–15 drop from a base of 7 is clipped |
| **Buffer Overflow (Gradual Buildup)** | 63 / 5 | UL buffer and BLER ↑ exponentially; TX/RX bytes log-decay; UL/DL packets ↑ linearly; PRB utilization and "PRBs allocated" logistic growth | n=19. UL packets ↑ 94 % (×2.1), DL packets ↑ 72 % (×1.5). PRB_Util_UL ↑ 94 % (×13.5), DL ↑ 56 % (×6.2). DL_BLER ↑ 78 % (×2.2), UL_BLER ↑ 67 %. RX ↓ 44 %, TX ↓ 28 %. **Estimated_UL_Buffer: median Δ 0, ↑ only 39 %** | **Headline KPI (UL buffer) barely moves.** "PRBs allocated" is in the recipe but PRBs_*_Current is not in `affected_channels` and does not change, so the PRB identity breaks (§3c) |
| **Co-Channel Interference (Mild)** | 61 / 5 | RSRP ↓ 3–8; UL SNR ↓ 1–3; UL&DL BLER ↑ 10–40 %; TX/RX bytes ↓ 5–15 %; PRB util/alloc ↑ 5–20 % | n=16. RSRP ↓ 93 % (Δ -4.8). UL_SNR ↓ 93 % (Δ -2.3). UL_NPRB ↑ 67 %. Everything else is in the noise: **DL_BLER ↑27 %/↓27 %, median Δ 0**; UL_BLER median Δ 0 | Static/sudden (paper). In practice only an RSRP+SNR step. **RSRP falling under interference is counter-intuitive** (interference lowers SINR, not RSRP). BLER effect invisible because BLER is zero-inflated and multiplicative changes on 0 stay 0 |
| **Co-Channel Interference (Severe)** | 66 / 5 | RSRP ↓ 8–15 dB; UL SNR ↓ 8–15 dB; TX/RX bytes ↓ 30–80 %; UL/DL packets ↓ 40–70 %; UL/DL BLER ↑ 50–200 %; PRB util ↑ 30–80 %; PRB alloc ↑ 20–70 % | n=17. RSRP ↓ 100 % (Δ -12.7). UL_SNR ↓ 100 % (Δ -11.3, ×0.45). UL_NPRB ↑ 88 % (×1.4). RX ↓ 76 % (×0.51), TX ↓ 41 % (×0.67). PRB_Util ↑ 41–53 %. Packets ↓ 41–47 %. **DL_BLER ↓ in 47 % of windows (ratio 0.38)**; UL_BLER median Δ 0 (ratio 1.8) | Same RSRP-falls caveat. **DL_BLER contradicts the recipe** (zero-inflation). Most distinctive cue: a large simultaneous RSRP + SNR step |
| **Doppler Shift (Severe)** | 20 / 2 | Periodic RSRP (3–8 dB) and UL SNR (2–5 dB) fluctuations at **2–3 Hz**; DL/UL MCS ↓ 1–2; multiplicative oscillation of TX/RX bytes (×0.2–0.4) and PRBs allocated (×0.3–0.6) at 2–3 Hz | n=5 only. DL_MCS ↓ 100 % (Δ -1). UL_MCS ↓ 80 % (Δ -1). UL_SNR ↓ 100 % (Δ -1.9). RSRP ↓ 60 %. Diff-std ratio (span/pre): RSRP ×27 (n=2), UL_SNR ×2.5, TX ×1.4 | **No BLER effect**, although the paper text says "Doppler … elevated BLER". 2–3 Hz at 10 Hz sampling is near Nyquist, so it shows as jitter. Recipe says "PRBs allocated", but `affected_channels` has UL_NPRB and not PRBs_*_Current. The testbed is pedestrian 5 km/h, so a "severe Doppler" at 3.3 GHz is physically implausible. Only 2 events in dev/test |
| **Faulty Handover Algorithm (Too Frequent)** | 16 / 4 | Periodic RSRP (amp 2–5, **0.1–0.3 Hz**) and UL SNR (amp 1–3, 0.1–0.3 Hz); UL/DL BLER ↑ 30–150 %; TX/RX bytes ↓ 10–50 %; UL buffer ↑ 50–200 %; UL/DL packets ↑ 50–200 % | n=11. DL_BLER ↑ 75 % (×6.3), UL_BLER ↑ 75 % (×1.7). UL buffer ↑ 62 % (×7.7). RSRP ↓ 62 % (Δ -5.5). UL_SNR ↓ 75 %. Bytes mixed. **Spans are short: median 12 rows (1.2 s), max 41.** A 0.1–0.3 Hz oscillation (3–10 s period) cannot complete a cycle; diff-std ratios are < 1 | **There is a single gNB, so there are no real handovers.** The label is purely a synthetic recipe. Oscillation is not observable in these spans, so it effectively looks like a short RSRP/SNR dip with BLER/buffer up |
| **Faulty RF Filters (Temporal)** | 26 / 3 | Linear ↓ RSRP and UL SNR; exponential ↑ UL/DL BLER and PRB utilization; logarithmic decay of TX/RX bytes; linear ↑ PRBs allocated | n=8. DL_BLER ↑ 88 % (×112), UL_BLER ↑ 88 %. UL_SNR ↓ 88 % (Δ -2.5). PRB_Util_UL ↑ 62 %. RSRP ↓ only 25 %. **TX ↑ (×1.2), RX ≈ flat** | Bytes do not decay, contrary to the recipe. "PRBs allocated" not in `affected_channels`. Gradual (ramp) shape, so the pre-window mean understates late-span change |
| **High Network Congestion (Gradual Buildup)** | 42 / 4 | Exponential ↑ UL buffer and BLER; logistic ↑ PRB utilization and allocation; log-decay of TX/RX bytes; **exponential ↑ UL/DL packets** | n=15. PRB_Util_DL ↑ 100 % (×106 from a near-idle Twitch base), UL ↑ 93 % (×26). UL packets ↑ 100 % (×50), DL ↑ 80 % (×4.5). DL/UL BLER ↑ 93 %. UL buffer ↑ 73 % (×1.35). UL_NPRB ↑ 87 %. **RX_Bytes ↑ 60 % (×2.05), TX flat** | Bytes rise or stay flat while the recipe says decay. Packets ×50 while bytes are flat means tiny packets, an injection artifact. RSRP/SNR unchanged (good discriminator vs radio faults) |
| **High Network Congestion (Sudden Spike)** | 29 / 5 | UL buffer ↑ 100–400 %; PRB util UL/DL ↑ 30–80 %; BLER ↑ 50–150 %; TX/RX bytes ↓ 20–50 %; **packets ↓ 40–60 %**; PRBs allocated ↑ 40–100 % | n=17. UL_NPRB ↑ 100 % (×1.9). UL buffer ↑ 65 % (×3.7). PRB_Util ↑ 41–47 %. UL packets ↓ 41 %. RX ↓ 29 %. UL_BLER ↑ 53 %. **DL_BLER: 0 % up, 24 % down** | **Packet direction is opposite to Gradual Buildup** (↓ vs exponential ↑). Paper Table 10 calls this "Static"; the data label says "Sudden Spike". DL_BLER contradicts the recipe |
| **Resource Allocation Bugs** | 28 / 2 | DL/UL PRBs, utilizations, allocated PRBs and TX/RX bytes multiplicative oscillation (amp 0.30–1.00, 0.3–1.0 Hz); UL/DL BLER ↑ 20–100 % | n=8. **RX ↓ 100 % (×0.36), TX ↓ 62 % (×0.16)**. UL_BLER ↑ 50 % (×1.7). PRB means ~unchanged (an oscillation averages out); PRB/Util ratio spreads 0.3–5.4. DL_BLER mixed (ratio 0.38) | A multiplicative sine with amplitude up to 1 drives bytes toward 0 half the time. Look for oscillation and variance, not level. Only 2 events in dev/test |
| **Jamming (real)** | 185 / n.a. (2 recordings) | Ticket (one fixed template): "DL_BLER and UL_BLER increase, UL_MCS decreases, and TX_Bytes and RX_Bytes experience drops." Root cause: "Active jammer generating **continuous** radio interference" | Every window is labelled 0–127. Jammer-on vs jammer-off rows in the raw CSV are **indistinguishable** for every KPI (§5) | **The ticket is a template, not an observation.** It says "continuous", but the paper says 2 s on / 8 s off. No measurable phase-locked KPI effect |

`affected_channels` is constant within each category (one variant each) and equals the paper's
Table 10 KPI count. It is a **per-category recipe list**, not a per-window observation [measured].

### 3c. Cross-cutting injection artifacts [measured]

1. **The PRB identity breaks inside synthetic spans.** In all pre-anomaly rows and in jamming windows,
   median `PRBs_X_Current / PRB_Utilization_X` = 1.060 exactly. Inside spans the ratio drops for
   Congestion-Gradual (DL 0.001, UL 0.017), Buffer Overflow (0.205/0.022), Antenna (0.61/0.42),
   CCI-Severe (0.70/0.65) and Congestion-Sudden (0.67/0.70). It spreads widely for Resource Allocation
   (0.30–5.4). It is preserved for Doppler and Handover, which do not touch PRB channels. This is a
   synthetic fingerprint, not network physics. Do **not** write RCA narratives that read physical
   meaning into "utilization rose but allocated PRBs didn't".
2. **Zero-inflated channels are not transformed.** DL_BLER, packet counts and bytes at 0 stay ≈ 0 under
   multiplicative effects. The paper adds noise only for exponential growth. So "BLER ↑" often does
   not appear in YouTube/Twitch windows with idle links.
3. **Floor clipping**: Antenna Failure drives DL_MCS from 7 to 0.
4. **Pre-anomaly context is scarce.** In dev/test, start_index = 0 for 21/42 Antenna, 43/63 Buffer,
   45/61 CCI-M, 48/66 CCI-S, 14/20 Doppler, 3/16 Handover, 18/26 RF, 26/42 Cong-G, 12/29 Cong-S and
   20/28 RAB windows. For those, take the baseline from the preceding overlapping window or from the raw
   `metrics.csv` rows before `raw_start_row - 1 + start_index`. Caveat: raw rows outside any released
   window may still carry a dropped (two-type) anomaly.

---

## 4. Family structure (release/membership.csv) [measured]

A **family** (`group_id`) is the unit that `splits.assign_splits` keeps whole within a split. It
connects traces through shared recording IDs, parent IDs, candidate-parent links or observed duplicate
windows. No such links exist in this data: all 33 recording reviews are `unresolved`, provenance
`recording_id`/`parent_id` are null, and the audit found no shared raw segments. So **each family is
exactly one processed source file** (33 families = 33 source traces: 22 train, 6 dev, 5 test).

Synthetic categories are **not** separate families. All categories injected into one base recording
(e.g. Zone_A/Twitch) share that file and therefore one family. The base 10k-row recording itself is not
released separately. It is plausibly from the same physical session/position as the matching normal
no-congestion file: identical `description.txt` hash and the same RSRP medians (e.g. -102 for Zone A
File, -88 for Zone A Twitch/YouTube). However, no raw segment matches it, so parentage is
**unverified** (`IMPLEMENTATION_SUMMARY.md` "Recording review audit";
`artifacts/experiments/recording-family-audit-2026-10-18-v2/report.md`).

### Dev and test families with fault windows

| Split | group_id (prefix) | Source file | Labels (windows) | Fault windows |
|---|---|---|---|---|
| dev | family-128a65d178ae86f2 | anomalous/synthetic/Zone_C/Twitch | Antenna 6, Buffer Overflow 26, CCI-Mild 15, CCI-Severe 8, Doppler 2, Cong-Sudden 4, Resource Alloc 18 | 79 |
| dev | family-6aaa4378e87449bd | anomalous/synthetic/Zone_A/Twitch | Antenna 27, CCI-Mild 25, CCI-Severe 8, Handover 4, RF Filters 9, Cong-Gradual 11, Cong-Sudden 14 | 98 |
| dev | family-bfe199279c7d4946 | anomalous/jammer/YouTube | Jamming 92 | 92 |
| test | family-e6c16ffb8df2ca8e | anomalous/synthetic/Zone_B/YouTube | Antenna 3, Buffer Overflow 16, CCI-Mild 5, Handover 3, Cong-Gradual 31, Cong-Sudden 11 | 69 |
| test | family-eafe6e671551d71f | anomalous/synthetic/Zone_A/YouTube | Antenna 6, Buffer Overflow 21, CCI-Mild 16, CCI-Severe 50, Doppler 18, Handover 9, RF Filters 17, Resource Alloc 10 | 147 |
| test | family-fbb769b7b5e578dc | anomalous/jammer/File | Jamming 93 | 93 |

Dev also has 3 normal-only families (Zone_C no-cong Twitch 3006, Zone_C cong YouTube 309, Zone_B
no-cong Twitch 3006). Test has 2 (Zone_C no-cong File 3006, Zone_B no-cong YouTube 3006). Train holds
the other 5 synthetic files, jammer/Twitch, and the remaining normals.

**How many independent recordings are behind the dev/test fault windows?** At most **6 traces**:
4 synthetic base recordings and 2 jammer recordings, with independence unverified. These contain
**41 distinct synthetic anomaly events** (merged raw spans): Antenna 6, Buffer 5, CCI-M 5, CCI-S 5,
Doppler 2, Handover 4, RF 3, Cong-G 4, Cong-S 5, RAB 2. Per category, the dev/test fault windows come
from 2–4 recordings and 2–6 events. Two of the dev/test synthetic traces (Zone_A Twitch and Zone_A
YouTube) share zone and position with the train-side Zone_A File synthetic and the normal Zone A files.

---

## 5. Real jamming

**The raw jammer `metrics.csv` has an extra `Jamming` 0/1 column** (the synthetic and normal CSVs do
not) [measured]. The jammer `description.txt` reads: "User is … static at zone 1. There is a jammer.
The network is not congested."

| Recording | Rows | Jam fraction | First on-row (pre-jam baseline) | Pattern |
|---|---|---|---|---|
| jammer/File | 9,999 | 0.190 | row 330 → **33.0 s clean** | 20 on / 80 off (2 s / 8 s), 94 full bursts |
| jammer/Twitch | 9,999 | 0.190 | row 260 → **26.0 s clean** | same |
| jammer/YouTube | 9,999 | 0.187 | row 250 → **25.0 s clean** | same (one 286-row and one 352-row gap) |

- **The recording is not jammed throughout.** There is a clean lead-in of 25–33 s, and the jammer is
  idle 80 % of the time after that. Both could serve as same-timeline baselines.
- **Windows start exactly at a jammer onset and step 100 rows.** Each window therefore has rows 0–19
  on, 20–99 off, 100–119 on and 120–127 off. Jammer-on fraction per window is 0.30 (min 0.15). The
  first window starts at the first on-row (0-based 330/260/250), so **none of the clean lead-in rows
  are in any released window**. Of the raw rows, 543–715 per file (incl. the lead-in) are in no window.
- **The stored labels say the whole window is anomalous.** All 279 jamming windows (all splits) have
  `start_index=0, end_index=127`. The true on-rows are about 40 of 128.
- **No measurable KPI effect is tied to the `Jamming` flag.** Jam-on vs jam-off medians and means are
  essentially equal for all 16 KPIs in all three recordings. Example (File): UL_BLER 0.080 on vs 0.090
  off; UL_MCS 16 vs 16; TX 440 kB vs 440 kB. A phase fold over the 100-row cycle and lagged correlation
  (-60..+60 rows) give |r| ≤ 0.08 for every channel. The phase-binned profiles are flat.
- **The jammer recordings differ from normal Zone A recordings mainly in level, which looks like
  position or setup.** RSRP medians are -73/-78/-76 (jammer File/Twitch/YouTube) vs -102/-88/-87 in
  normal Zone A no-cong. That is 10–30 dB *stronger*, consistent with the paper's statement that the
  jammer was "positioned at varying locations", or with a different UE spot. DL_BLER median is nonzero
  (0.002–0.004) vs 0 in normal Zone A. `IMPLEMENTATION_SUMMARY.md` (2026-10-20 note) already found that
  RSRP/UL_SNR levels act as a recording-location shortcut that also "carries" the jammer label.
- Possible explanations, unresolved: (a) the `Jamming` column is the scheduled pattern and the residual
  ~2 s clock offset between PHY and network traces (paper App. B.2) blurs it, although the lag scan
  should catch a pure shift; (b) the pulsed/narrowband jammer barely affected KPIs at 100 ms resolution;
  (c) the "jammer" signature is a whole-session effect.

**For RCA cases:** within a jamming window there is no defensible KPI-level onset to point at. The
honest evidence is "the dataset labels this session as jammed (zone 1, static) and the per-row jammer
schedule is known". A baseline from the same timeline exists (rows 0–249/259/329, and the off-phases),
but it does **not** differ from the on-phases.

---

## 6. Text fields

| Field | Content | Status |
|---|---|---|
| `nl_context` | **Empty for all 32,000 records.** The adapter fills it only when `text_review.status == "approved"` (`telecomts.py:94`) | No text is model-eligible |
| `text_review` | `status: pending` for **32,000/32,000**, `source: description`, reviewer null. `inputs.text.jsonl` is 0 bytes; quality.json says no text cohort is ready | — |
| `reference.description` | A paragraph of KPI ranges, means, jumps and trends per channel. App. F: generated by **GPT-o3** "as a surface realization step … prompted with structured summaries of KPI behavior, task-specific metadata, **and anomaly symptoms derived from troubleshooting tickets**" | Conditioned on the label. It rarely names the category (keyword hits: Resource Allocation 31/60, Cong-Gradual 4/93, others 0), but the symptom wording is steered by the ticket. It contains factual errors (TX called "uplink", buffer in "KB", RSRP in "dB") |
| `reference.anomalies.troubleshooting_tickets` | **GPT-4.1**: one fixed ticket per category (Issue / Symptoms / Root Cause / Resolution), conditioned on type, "anomaly impact" and time bounds | Direct label: Issue and Root Cause name the fault |
| `reference.anomalies.{type, anomaly_duration, affected_kpis}` | Gold label, span, per-category KPI list | Labels |
| `reference.QnA.anomalies` | Synthetic: 3 Qs ("Was an anomaly detected?", "What is the root cause?", "What is the index range?"). Jamming: 2 Qs (no index range). Normal: 1 | **Answers are the label** (root-cause answer matches the category in 100 % of windows). Reasoning traces from **Qwen3-32B**, conditioned on "Ground-truth anomaly type, network description, affected KPIs, and observed symptoms from the troubleshooting tickets" (App. F); 91–100 % name the category |
| `reference.QnA.network` | 4 template Qs (zone, app, congestion, mobility) with reasoning traces | Context labels |
| `reference.QnA.timeseries` | 64 programmatic Qs (mean/var/trend/periodicity per KPI) | Computed statistics |
| `reference.statistics`, `reference.labels` | Per-KPI stats; zone/application/mobility/congestion/anomaly_present | Labels and metadata |
| `alarms`, `spec_refs_golden` | Empty for all records | — |

### Leakage example

Dev, `telecomts-61a3b569…af9`, Zone_A/Twitch synthetic, CCI-Severe, span 67–127:

- Ticket: *"**Issue:** Severe co-channel interference impacting wireless network performance. …
  **Root Cause:** Overlapping frequency channels between neighboring cells causing interference."*
  (The testbed has one cell.)
- QnA: *"What is the root cause of the anomaly?" → "co-channel interference (severe)"*; *"What is the
  index range …?" → "67, 127"*.
- Description (no category name, but recipe-shaped): *"RSRP weakened notably …, falling by about
  10.4 dB … UL SNR dropped sharply to a low near 5.87 dB … UL BLER experienced spikes …"*. These
  phrases mirror the ticket's "RSRP ↓ 8–15 dB, UL SNR ↓ 8–15 dB".
- Jamming example (test, `telecomts-b2342446…459`): the description says *"Traffic volume on the uplink
  (TX_Bytes) was high … Downlink throughput (RX_Bytes) was much lower"*, which reverses the directions.
  The QnA root cause → "jamming".

---

## 7. Caveats for writing RCA cases

1. **Synthetic labels are dataset claims about an injection recipe, not observed faults.** The "root
   cause" was chosen first and the KPIs were transformed to match a GPT-4.1-proposed parameter range.
   Ground truth = "this transform was applied", not "this physical fault occurred". Several labels are
   physically impossible on this testbed: handover with one gNB, "neighboring cells" co-channel
   interference with one cell, severe Doppler at 5 km/h. Write cases as "signature consistent with the
   dataset's <category> injection", and cite the measured KPIs.
2. **Measured signatures often diverge from the designed recipe** (§3b flags): Buffer Overflow's buffer
   barely moves; CCI and Cong-Sudden DL_BLER doesn't rise; Cong-Gradual bytes don't decay; RF-filter
   bytes rise; Handover oscillation is unobservable in 1–4 s spans; Doppler has no BLER effect despite
   the paper text. Ground each case in the window's actual numbers, not in the ticket.
3. **Discriminating cues that hold up empirically:**
   - RSRP + UL_SNR step down: Antenna (with MCS collapse and packet rise), CCI-S (large step, RX ↓),
     CCI-M (small step, little else).
   - BLER ramp + SNR down: RF filters.
   - PRB util/packets/UL_NPRB ↑ with RSRP/SNR flat: congestion or buffer overflow.
   - Bytes ↓ with oscillating PRB/bytes: Resource Allocation.
   - MCS −1 with RSRP/SNR jitter: Doppler.
   - Short BLER/buffer bump with an RSRP/SNR dip: Handover.
   The categories are separable mostly through injection fingerprints (including the broken PRB
   identity). That is not evidence of real-network diagnosability.
4. **Real jamming has no KPI evidence inside windows** (§5). Its labels span whole windows, the ticket is
   a template ("continuous"), and the recording-level differences confound jammer with location. Treat
   jamming cases as low-evidence, or frame them around session context.
5. **Overlap and independence.** 75 % overlap for normal/synthetic windows and 22 % for jammer windows.
   393 dev/test synthetic windows = 41 events in 4 recordings, and only 2 events each for Doppler and
   Resource Allocation. Pick one representative window per event, and do not present several overlapping
   windows as independent cases. Recording independence across splits is unverified.
6. **Units are unverified in the repo** (`unit: null` everywhere). Use paper units (dBm, dB, fraction,
   MCS index, PRBs, %, bytes/100 ms, packets/100 ms), mark them as "per paper", and never use the
   description's TX/RX direction or "KB" buffer.
7. **Baselines.** Many windows start at index 0, so there is no in-window pre-fault context. Use the
   previous overlapping window or raw CSV rows. Raw rows outside released windows may still contain
   unreleased (two-type) anomalies. Cross-recording baselines confound zone and position (RSRP differs
   by 10–30 dB between "Zone A" recordings).
8. **Text is not usable as independent evidence.** Descriptions, tickets and reasoning traces were all
   generated with the label in the prompt. Any case that quotes them inherits label leakage.
   `nl_context` is empty and no text review is approved.
9. **Timestamps are synthetic or relative** (2025-01-01 for jammer, 2036 for some congestion). Do not
   infer time-of-day or cross-file chronology.
10. **Naming mismatch between paper and data**: Static ↔ Sudden Spike, Temporal ↔ Gradual Buildup,
    "Doppler Shift" ↔ "(Severe)", "(Frequent)" ↔ "(Too Frequent)". The paper also says "ten" and
    "11" anomaly types. Use the data labels (`contracts.CATEGORY_LABELS` + "Jamming").
