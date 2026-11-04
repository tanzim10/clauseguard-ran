# RCA input and output contract

Version: `rca-v1`  
Status: approved for implementation; example review recorded in `rca-contract.example.json`.

## Five records and owners

| Record | Owner | Contract |
|---|---|---|
| Request | Caller | `RcaRequest`; scenario ID, question, and bounded `top_k`. |
| Evidence packet | Shared RCA core | `EvidencePacket`; allowed observations, prediction, modality states, and retrieved excerpts. |
| Model output | Generator | `RcaModelOutput`; causes, claims, evidence IDs, uncertainty, and abstention. |
| Server response | Shared RCA core | `RcaServerResponse`; checked output, source links, run status, and review state. |
| Reference record | Evaluator | `RcaRow`; reference causes, labels, citations, uncertainty, and split-specific annotation. |

Requests reject unknown keys. The first request path requires a saved scenario ID. Raw observations need a typed adapter before the request can accept them.
The evaluator reference record must never enter `EvidencePacket` or the generator prompt.

## Field rules

All records reject unknown fields. Required text fields reject empty or whitespace-only values. Hashes use 64 lowercase hexadecimal characters.

`RcaRequest` requires `scenario_id` and `question`. `top_k` defaults to 8 and accepts integers from 1 through 50. The server selects the experiment configuration.

`DetectorPrediction` preserves the detector task, label ID, exact mapped label, mapping version, run ID, source modalities, and status. Detection maps `normal=0` and `abnormal=1`; category and diagnosis use their own mappings. Confidence is optional and must be finite within [0, 1]. An absent detector requires an absence reason. A failed call uses `error` and is not an absence.

`KpiStatisticEvidence` requires a finite value, valid count, sample count, missing fraction, and calculation version. Units and time bounds may be null. Compute statistics from observed values without classifier imputation. Missing values differ from zero. Do not claim a change without a baseline or another reviewed comparison rule. Do not copy reference anomaly bounds into observed evidence.

`TextEvidence` accepts only text approved for RCA. It records the source, record ID, timestamp when available, and whether it came from an independent observation or KPI description. Exclude diagnosis-derived text and text that exposes a reference label.

`SpecificationEvidence` records the exact retrieved chunk, document, version, clause, text, and content hash. The core assigns evidence IDs and source metadata. The generator cannot create source metadata.

Each claim cites one or more evidence IDs from its packet. Observation claims cite KPI or approved text. Mechanism claims cite specifications. Diagnostic claims cite both incident evidence and a specification passage. Every returned cause links to a diagnostic claim. The server renders explanations from checked claims.

An answered output has one or more unique fault categories and no abstention reason. A cause may link to other claim types as context, but it must link to at least one diagnostic claim. An abstained output has no causes or claims and has an abstention reason. Normal operation is a mapped result when supported; it is not an abstention.

## Dataset source matrix

The source matrix describes candidate records. It does not assert that every dataset has passed review or is ready for inference.

| Dataset | Candidate input | Separate reference | Confirmed gap or required check |
|---|---|---|---|
| TelecomTS | KPI windows and eligible reviewed text | Fault labels and reviewed RCA targets | Recording families remain unresolved in the current local release; RCA text approval is required. |
| BubbleRAN | Radio KPI measurements | Binary persistent-interference label | Verify metric units, capture boundaries, baselines, and reuse terms. |
| SpotLight | Radio and platform measurements | Anomaly intervals and run labels | Verify run identity, timestamp alignment, repeated headers, and reuse terms. |
| TeleLogs | Case text, network settings, and numeric measurements | Cause labels and any available explanation targets | Inspect authorized files; confirm exact fields, taxonomy, access, and redistribution terms. |
| 5G core faults | Text description | Troubleshooting response | Check answer correctness, target leakage, RAN relevance, and reuse terms. |

Keep incidents from different sources separate. Preserve source taxonomies and add cross-source mappings only after review. Record unknown units, missing modalities, unresolved groups, and unknown reuse terms explicitly.

## Source identity and eligibility

`SourceIdentity` records dataset, revision, case ID, artifact hash, recording-family status, and known reuse terms. A verified family needs a stable family ID. Unresolved family membership blocks scored split claims and approved training use.

Keep labels, expected answers, golden citations, split membership, and reviewer decisions in evaluator records. Keep detector predictions separate from RCA categories. Record a case's training, development, or evaluation eligibility outside the prompt.

## Experiment input projections

| Configuration | Prompt inputs |
|---|---|
| Base LLM, full evidence | KPI evidence, eligible text, detector prediction, and retrieved specifications. |
| Adapted SLM, full evidence | The same logical evidence and prompt as the base LLM. |
| Text + RAG | Approved text and retrieved specifications; omit KPI summaries and KPI-derived predictions. |
| KPI summary + RAG | KPI evidence, retrieved specifications, and the fixed minimal text fields approved before the run. |

Shared retrieval makes these prompt-input comparisons. It may still carry information from omitted modalities. Complete modality removal requires a separate retrieval-query policy. Filenames and derived text cannot restore omitted inputs.

## Result and failure states

| State | Meaning |
|---|---|
| `answered` | Checked claims support one or more returned causes. |
| `abstained` | Evidence cannot support a grounded cause; cause and claim lists are empty. |
| `incomplete` | Required assembly or retrieval did not finish. |
| `failed` | A retrieval, generation, or validation dependency failed. |

Completed empty retrieval requires abstention with `no_spec_evidence`. Search that did not run is incomplete. Invalid output is a validation failure. Do not report any of these conditions as a correct diagnosis.

## Sixteen acceptance checks

1. Reject requests with both observation modes or neither mode. The initial endpoint accepts a saved scenario ID only.
2. Reject reference fields and unknown keys at input boundaries.
3. Resolve labels with the task and mapping version.
4. Reproduce KPI statistics from source measurements.
5. Reject invalid intervals, non-finite values, and unverified units.
6. Reject unknown or duplicate evidence IDs.
7. Require each cause to link to a diagnostic claim with evidence.
8. Require abstention to have no category or diagnostic claims.
9. Preserve execution errors separately from abstentions.
10. Ensure omitted modalities cannot return through predictions, names, or derived descriptions.
11. Record structural checks separately from semantic review.
12. Keep human approval pending until a reviewer records it.
13. Check source identity and modality states for each adapter.
14. Accept valid scalar or text-only cases without inventing a KPI time series.
15. Keep absent predictions separate from failed detector calls.
16. Check source-specific mappings and keep related recording families in one split.

## Reviewed example and limits

`rca-contract.example.json` preserves the actual exploratory source observation and the saved prediction. Its expected abstention uses an injected empty retrieval result. It does not represent a live search or generator response.

The review record approves the contract example only. It does not approve the incident as a golden RCA case. The source recording family remains unresolved, the retrieval status is `not_run`, and no RCA reference is available. The KPI units are unknown and no baseline supports abnormality claims.

The runtime schemas and packet checks live in `src/clauseguard/api/schemas.py` and `src/clauseguard/rca/contracts.py`. Structural checks do not establish factual support. Human review of actual golden RCA cases remains separate work.
