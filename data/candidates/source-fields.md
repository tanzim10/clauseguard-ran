# Candidate sources: fields and mappings (Task 4.6)

This document lists, for each of the five planned sources, the actual fields, the unit status, what becomes a model observation and what stays a reference, the label and taxonomy mapping, the grouping unit, and the candidate status. The code is `src/clauseguard/rca/candidate_pool.py` and `src/clauseguard/rca/sources.py`.

**Verification basis.**
- TelecomTS, BubbleRAN and SpotLight were checked against the local files.
- TeleLogs and 5G core faults are **card-based and unverified**: their files are not local, so field names come from the dataset cards and one public sample.

**Rules shared by all sources.**
- **Observations** are the `EvidencePacket` built by the Task 4.4 adapter: KPI statistics (`kpi-stats-v1`: mean, minimum and maximum per numeric channel), plus approved text, which no source has today.
- **References** go to `references.jsonl` and never reach a prompt.
- **Units** stay `null` unless verified. No source has a verified unit today.
- **Baselines** stay `null`. No verified baseline exists, so change claims are not allowed.
- **Detector predictions** exist only for TelecomTS. Every other source carries the documented absence (`rca.detector.DATASET_COMPATIBILITY`).
- **Labels are never mapped across sources.** Each candidate keeps its source taxonomy (`taxonomy_id`). Combining category names across sources needs a reviewed mapping.
- **Label status:**
  - `labelled_fault` carries one or more causes.
  - `labelled_normal`, `unlabelled` and `ambiguous` carry none.
  - None of these is an abstention; abstention belongs to a model output.

## TelecomTS

**Status:** local (release `f3bd4d46…`, revision `1b3a88a4…`), MIT licence. Candidates are `exploratory`.

### Fields

| Field | Contents | Unit status |
|---|---|---|
| `kpi_window` | 128 samples at 10 Hz, 16 numeric channels | Unknown for all 18 channels (release `unknown_units`) |
| Numeric channels | `DL_BLER`, `DL_MCS`, `DL_NumberOfPackets`, `Estimated_UL_Buffer`, `PRB_Utilization_DL`, `PRB_Utilization_UL`, `PRBs_DL_Current`, `PRBs_UL_Current`, `RSRP`, `RX_Bytes`, `TX_Bytes`, `UL_BLER`, `UL_MCS`, `UL_NPRB`, `UL_NumberOfPackets`, `UL_SNR` | Unknown |
| Categorical channels | `DL_Protocol`, `UL_Protocol` | Not applicable |
| `nl_context` | Free text | Not applicable |
| `provenance` | Source file, row, file hash, revision | Not applicable |

### Observations and references

- **Observations:**
  - 48 KPI records: 16 channels × mean, minimum and maximum, with sample and time bounds. The timezone is unknown.
  - The two protocol channels are rejected (categorical).
  - `nl_context` enters only with RCA approval. None exists, so the text state is `missing: no_approved_text`.
- **References:**
  - `rca_label` and `fault_label`.
  - `anomaly.source_type`, `anomaly.start_index`, `anomaly.end_index`, `anomaly.affected_channels` and `origin`.
- **Reference text is listed by field name only and never copied:**
  - `reference.description`: KPI-derived.
  - `reference.anomalies`: holds the troubleshooting tickets, which state the root cause.
  - `reference.QnA`: contains reasoning.
  - `reference.statistics`, `reference.labels`, and the time fields.
- **Existing answers:** the tickets and QnA are existing answers that leak the label. They stay unapproved references.

### Taxonomy and grouping

- **Taxonomy `telecomts-v1`:** the 10 synthetic categories plus `Jamming` are causes (`labelled_fault`, one cause each); `normal` is `labelled_normal`. No source label has more than one cause.
- **Grouping:**
  - The family is the release `group_id`. All families are `unresolved`, which blocks `eligible_pilot`.
  - An event is the set of overlapping windows of one source file with the same labels.
  - Overlap across labels is reported, not merged.
  - Only training-split windows become candidates. Dev and test families are reserved.

## BubbleRAN

**Status:** local copy of `bubbleran/open-telco-datasets`, `anomaly-detection-persistent-interference`, licence CC BY-SA 4.0. Candidates are `ineligible` because training terms are pending.

### Fields

- **Columns:** 104 per row, at a 1 s cadence, one UE (`ran_ue_id=2`) on one gNB (`e2node_nb_id=50`).
- **Files:** `anomalous_data.csv` (1,597 rows) and `clean_data.csv` (3,601 rows).
- **Column groups** (local `data_dictionary.md`): MAC `mac_*`, RLC `rlc_*`, PDCP `pdcp_*`, KPM `kpm_*`.

### Unit status

- `data_dictionary.md` gives units (for example `mac_pusch_snr` dB, `mac_dl_bler` ratio).
- It also says `*_tbs` may be bytes or bits.
- The units are not verified against the exporter, so every unit stays `null`.

### Observations and references

- **Observations:** 77 numeric channels × 3 statistics (231 KPI records per window).
- **Rejected channels:**
  - Cumulative counters (`*_aggr_*`, `*_harq0..3`, `*_errors`): counter semantics are unverified, so they are not differenced.
  - Identifiers: `timestamp`, `timestamp_iso`, `ran_ue_id`, `e2node_nb_id`, `mac_frame`, `mac_slot`.
- **References:**
  - `persistent_anomaly` (binary): the distinct values in the window.
  - Anomalous-row count, first and last anomalous offsets, and the anomaly episode.

### Taxonomy and grouping

- **Taxonomy `bubbleran-pi-v1`:**
  - A window with any anomalous row is `labelled_fault` with cause `persistent_interference`. The interference source and type are not described by the source.
  - Other windows are `labelled_normal`.
- **Windows:** a fixed grid of 60 rows at stride 60 inside one capture segment. A segment ends at a time gap, a timestamp reset or a UE/gNB change. Labels never move a boundary.
- **Grouping:**
  - One family for both files: one UE and one session.
  - An event is a contiguous anomaly episode, or the window itself for normal windows.

## SpotLight

**Status:** local `sample/` of `netsys-edinburgh/SpotLight`. No licence is published. Candidates are `ineligible`.

### Fields

- **`radio.csv`:** 208 columns at about 0.15 s. Groups: `mac_*` 61, `l1_*` 24, fronthaul `BHTX/BHRX/MHTX/MHRX_*` 88, `rlc_*` 22, `f1u_*` 11, plus `timestamp` and `anomaly`.
- **`platform.csv`:** 249 columns at about 0.3 s, with a leading unnamed index column. Groups: per-thread CPU statistics `gnb_*` 132, `l1app_*` 66, `dumgr_*` 22; PTP `ptp4l_*` and `phc2sys_*` 11 each; network `in_*` and `out_*`.
- **Captures:** both files hold two captures, separated by a repeated header row.
- **`label`:** 60 intervals in Unix nanoseconds.
- **`raw_sample`:** a codelet trace. It is not used.

### Unit status

Units are not stated. Every unit stays `null`.

### Observations and references

- **Observations:**
  - One stream per candidate. Radio and platform are never joined, because their alignment is unverified.
  - Numeric columns × 3 statistics.
  - Columns with no values in the window are rejected (`no_valid_values`).
  - Identifiers are excluded.
- **References:**
  - The in-file `anomaly` column: distinct values. They are 0 throughout, apart from blank and `0.0` values in platform capture 2.
  - Whether the window overlaps a `label` interval.
  - The number of overlapping intervals.

### Taxonomy and grouping

- **Taxonomy `spotlight-v1`:**
  - The anomaly type is not stated.
  - The CSV timestamps are naive. Overlap with the label intervals is computed assuming UTC, which is unverified.
  - The in-file column disagrees with the label file.
  - Every window is therefore `ambiguous`, with no cause.
- **Windows:** a fixed grid of 100 rows at stride 100 inside one capture of one stream. A repeated header starts a new capture.
- **Grouping:**
  - One family for the sample experiment, covering both streams and both captures.
  - Each window is its own event.

## TeleLogs (card-based, unverified)

**Status:** `netop/TeleLogs` on Hugging Face. Gated; the card states MIT and asks users not to share, re-upload or distribute the benchmark. Not local. No adapter, no candidates.

### Fields

- **Per the card and the public GSMA `ot-lite` sample of 100 items:** each item is a `question` string with an answer from causes `C1`…`C8`.
- **The `question` string serialises:**
  - **a user-plane table:** timestamp, GPS, speed, PCI, SS-RSRP, SS-SINR, DL throughput, top-5 neighbour PCI and BRSRP, DL RB
  - **engineering parameters:** tilt, azimuth, beam scenario, height, and others
- **Splits:** `train` and `test`.

### Unit status

Units are not verified.

### Observations and references (proposed)

- **Observations:**
  - Measurements as scalar KPI evidence.
  - Settings need a contract type, which does not exist yet; see `contract-updates.md`.
- **References:** the cause set.
- **Excluded:** the official `test` split and the 100 GSMA `ot-lite` items. Both are reserved.

### Taxonomy and grouping

- **Taxonomy `telelogs-v1`:**
  - The card says "one or more among 8 predefined causes", but the public sample items have exactly one answer each.
  - Whether the files contain multiple causes is unverified.
- **Grouping:** unknown until the files are inspected. The proposed unit is the scenario or base-station configuration.

### Blockers

- The gate has not been accepted.
- The card asks users not to redistribute the dataset.
- Fields and families are unverified.
- There is no settings evidence type.

## 5G core faults (card-based, unverified)

**Status:** identified by name only as `greenwich157/telco-5G-core-faults` on Hugging Face (Apache-2.0 on the card). The identity is unconfirmed by the project. Not local. No adapter, no candidates.

### Fields

- 1,323 rows with three columns: `instruction`, `input` (the issue text) and `output` (remediation steps).
- One `train` split.

### Unit status

Not applicable: the rows are text only.

### Observations and references (proposed)

- **Observations:** `input` as text, but only when it does not state the cause. 790 of 1,323 inputs state it.
- **References:** `output` as an unapproved existing response. It is a remediation, not an RCA.
- **Detector prediction:** absent.

### Taxonomy and grouping

- **Taxonomy:** `core-faults-none`. The source has no labels.
- **Grouping:** near-duplicate clusters. The inputs include templated variants and exact duplicates.

### Blockers

- The dataset identity is unconfirmed.
- Inputs leak the cause.
- Most topics are core-network, outside the RAN corpus.
