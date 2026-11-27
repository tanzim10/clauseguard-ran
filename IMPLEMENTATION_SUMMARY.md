# Implementation Summary

This document records completed feature work by implementation phase. Each phase is focused on
the corresponding PR or issue: what it delivered, how it was built, how it was verified, and what
remains outside its scope.

## Table of Contents

- [Phase 1 — Corpus Manifest Validation](#phase-1--corpus-manifest-validation)
- [Phase 2 — Clause-Aware Corpus Chunking](#phase-2--clause-aware-corpus-chunking)
- [Phase 3 — Qdrant Corpus Indexing](#phase-3--qdrant-corpus-indexing)
- [Phase 4 — POST /search Retrieval API](#phase-4--post-search-retrieval-api)
- [Phase 5 — Manifest-Validated Corpus Parsing](#phase-5--manifest-validated-corpus-parsing)
- [Phase 6 — Grounded POST /query and NVIDIA Models](#phase-6--grounded-post-query-and-nvidia-models)
- [Phase 8 — Project Plan and Weekly Timeline](#phase-8--project-plan-and-weekly-timeline)
- [Shared Detection and Category Dataset Release](#shared-detection-and-category-dataset-release--issue-22)
- [Phase 9 — Golden Q&A Schema and Loader](#phase-9--golden-qa-schema-and-loader)
- [Phase 10 — Reviewed Golden Q&A with Source Evidence](#phase-10--reviewed-golden-qa-with-source-evidence)
- [Phase 11 — Golden RCA Contract and Release Manifest](#phase-11--golden-rca-contract-and-release-manifest)
- [Phase 12 — Golden Evaluation Runner](#phase-12--golden-evaluation-runner)
- [Task 3.1 — Validated Experiment Data](#task-31--validated-experiment-data)
- [Task 3.8 — Held-Out Evaluation of the KPI Baselines](#task-38--held-out-evaluation-of-the-kpi-baselines--2026-11-05)
- [Task 3.9 — Frozen MantisV2 Baseline](#task-39--frozen-mantisv2-baseline--2026-10-28)
- [Task 3.10 — Frozen Toto 2.0 Detection Baseline](#task-310--frozen-toto-20-detection-baseline--2026-11-05)
- [Task 3.11 — PatchTST Baseline](#task-311--patchtst-baseline--2026-11-06)
- [Task 4.1 — RCA input and output contract](#task-41--rca-input-and-output-contract--2026-11-03)
- [Task 4.3 — Draft RCA golden cases and independent model review](#task-43--draft-rca-golden-cases-and-independent-model-review--2026-11-04)
- [Task 4.2 — Detector predictions for RCA and SLM data](#task-42--detector-predictions-for-rca-and-slm-data--2026-11-09)
- [Task 4.4 — Traceable evidence records](#task-44--traceable-evidence-records--2026-11-09)
- [Task 4.5 — Frozen retrieval and specification snapshots](#task-45--frozen-retrieval-and-specification-snapshots--2026-11-09)
- [Task 4.6 — SLM candidate pool, manifest and gap report](#task-46--slm-candidate-pool-manifest-and-gap-report--2026-11-17)
- [Task 4.7 — Base LLM pipeline for cited RCA](#task-47--base-llm-pipeline-for-cited-rca--2026-11-17)
- [Task 4.8 — Evaluation of base RCA results](#task-48--evaluation-of-base-rca-results--2026-11-18)
- [Task 4.9 — Cited SLM training examples](#task-49--cited-slm-training-examples--2026-11-25)
- [Adding a New Phase](#adding-a-new-phase)

## Phase 1 — Corpus Manifest Validation

**Focus:** Corpus acquisition milestone — establish a verified, reproducible inventory of the
starter O-RAN/3GPP corpus.

**Status:** Complete — 2026-09-10

**PR:** TBD

### Delivered

The repository now has a real 12-document manifest, with validation for required provenance,
document type, acquisition state, text-layer metadata, file existence, and SHA-256 integrity.

### Technical implementation

- Implemented `load_manifest` and `validate_manifest_row` in
  `src/clauseguard/corpus/manifest.py`.
- Added `data/corpus/manifest.csv` with the starter corpus rows.
- Preserved raw corpus files in ignored local storage under `data/corpus/raw/`.
- Added fixture tests for valid rows, missing fields, invalid document types, checksum mismatches,
  documented blockers, text-layer values, and multi-row CSV loading.

### Verification

- Manifest tests pass.
- Local raw files were checked against their recorded hashes.

### Out of scope

PDF parsing, chunking, embeddings, Qdrant indexing, and retrieval endpoints.

## Phase 2 — Clause-Aware Corpus Chunking

**Focus:** Issue #5 / [PR #9](https://github.com/tanzim10/clauseguard-ran/pull/9) — produce stable,
metadata-bearing passages for retrieval.

**Status:** Complete — 2026-09-16

### Delivered

Specification text can now be split into repeatable chunks that preserve numbered sections and
paragraphs where possible while respecting the configured token budget.

### Technical implementation

- Added numbered-heading and paragraph-aware chunking in
  `src/clauseguard/corpus/chunker.py`.
- Added sentence-level and word-level splitting for oversized content.
- Added deterministic chunk IDs and metadata for `spec_id`, `source_file`, `section`, and
  `chunk_id`.
- Added nine contract tests for section boundaries, fallback paragraphs, metadata, stable IDs,
  and token budgets.

### Verification

- `uv run pytest tests/test_corpus_chunker.py` — 9 passed.

### Out of scope

PDF parsing, embeddings, Qdrant indexing, retrieval endpoints, and end-to-end RCA.

## Phase 3 — Qdrant Corpus Indexing

**Focus:** Issue #6 — embed parsed chunks and store citation-ready points in Qdrant.

**Status:** Complete — 2026-09-20

**PR:** [#10 — Implement RAG embeddings and Qdrant indexing](https://github.com/tanzim10/clauseguard-ran/pull/10)

**Commits:** `a461b3a`, `14a3a3f`, `396718b`, `667b2a8`, `79597ed`, `7b85af6`

### Delivered

Existing parsed O-RAN/3GPP text can now be indexed with `make index`. The pipeline batches
OpenAI embeddings, validates the Qdrant collection schema, writes deterministic point IDs, and
stores the original passage plus provenance metadata for later citation-grounded retrieval.

### Technical implementation

- Added `openai` and `qdrant-client` dependencies and explicit settings for model, dimensions,
  batch sizes, distance, collection, and parsed-corpus location.
- Implemented injectable batched embeddings with response-order and vector-dimension validation.
- Implemented idempotent Qdrant collection management with dimension and distance compatibility
  checks.
- Converted arbitrary chunk IDs to deterministic UUIDs accepted by Qdrant.
- Upserted citation-ready payloads containing `text`, `spec_id`, `section`, `source_file`, and
  `chunk_id`.
- Added a low-level `query_points` search wrapper for the later `/search` endpoint.
- Implemented the parsed-text indexer and `index_corpus` CLI.
- Documented the parsed-file convention: `data/corpus/parsed/<manifest stem>.txt`.
- Added fixture-only tests for configuration, embedding batching, Qdrant behavior, indexing, and
  CLI success/failure paths.

### Verification

- `make test` — 37 passed, with one existing deprecation warning.
- Targeted Ruff checks for changed files passed.
- `uv run python -m clauseguard.cli.index_corpus --help` passed.
- Missing parsed inputs produce a clear non-zero CLI failure before any embedding or write.

### Out of scope and follow-ups

- PDF parsing remains a prerequisite and is not invoked by `make index`.
- `/search`, grounded `/query`, BM25/hybrid retrieval, stale-point deletion, and dual indexes
  remain future work.
- Full-repository Ruff still reports two pre-existing findings in
  `src/clauseguard/api/routes/health.py`.

### PR #10 record

#### In simple words

Issue #6 is now proposed for review as the RAG embedding and Qdrant indexing feature.

#### Technical details

- Pull request: [#10](https://github.com/tanzim10/clauseguard-ran/pull/10).
- The PR closes Issue #6 and preserves the parsed-text prerequisite and retrieval follow-ups
  described above.

## Phase 4 — POST /search Retrieval API

**Focus:** Issue #7 — expose indexed specification retrieval through a typed API contract.

**Status:** Complete — 2026-09-20

### Delivered

Users can now submit a natural-language specification query to `POST /search` and receive
ranked, citation-ready passages from the indexed Qdrant collection. Valid searches with no
matches return an explicit empty result list, while invalid blank queries are rejected.

### Technical implementation

- Added bounded and normalized search request validation with a default `top_k` of 8 and a
  maximum of 50.
- Added typed search-hit and search-response models containing passage text, score, and all
  required provenance fields.
- Added a shared search service that composes the existing embedding adapter and Qdrant search
  helper without duplicating retrieval logic.
- Added strict projection of Qdrant payloads so internal fields and incomplete metadata are not
  returned.
- Added safe `500` configuration/data errors and `503` retrieval-dependency errors.
- Added FastAPI dependency overrides for fixture-only API tests.
- Kept answer generation, `/query`, and generated citations out of scope.

### Verification

- Focused search/API tests pass with fake embedding and Qdrant dependencies.
- OpenAPI response and request models are generated from the typed route contract.
- Ruff and `git diff --check` pass for changed files.

### Out of scope and follow-ups

- The endpoint requires an indexed parsed corpus and running Qdrant for a real demo.
- PDF parsing remains a separate prerequisite.
- `/query` answer generation, hybrid retrieval, reranking, and full citation-grounded RAG remain
  future work.
- Docker readiness improvements such as pinning Qdrant and adding readiness-based startup are
  tracked in `docs/todo.md`.

## Phase 5 — Manifest-Validated Corpus Parsing

**Focus:** Parser enhancement — complete the raw corpus to retrieval pipeline for live demos.

**Status:** Complete — 2026-09-21

### Delivered

The acquired 12-document O-RAN/3GPP corpus can now be parsed directly into the text files used by
indexing. The operational flow is now raw source -> manifest validation -> parsed text ->
embeddings -> Qdrant retrieval.

### Technical implementation

- Added PDF extraction with `pypdf` and DOCX paragraph extraction using the Python standard
  library.
- Added batch parsing that validates every acquired source against its manifest hash before any
  document is parsed.
- Added the `make parse` CLI, with manifest, raw-directory, and output-directory overrides.
- Added explicit failure behavior for missing, mismatched, unsupported, and text-empty sources.
- Added fixture-only unit and CLI tests for PDF, DOCX, batch validation, and command status.

### Verification

- Parsed all 12 local manifest sources into non-empty, manifest-aligned text files.
- Indexed the parsed corpus into the dedicated `oran_specs_demo` Qdrant collection.
- Verified the collection contains 10,715 points and that live `POST /search` returns
  provenance-bearing E2SM-KPM and throughput results.

### Out of scope and follow-ups

- `/query`, generated answers, reranking, and RCA remain unimplemented.
- Parsed corpus text remains ignored local data and is not committed.

## Phase 6 — Grounded POST /query and NVIDIA Models

**Focus:** Issue #15 — answer specification questions from retrieved evidence with citations
and bounded NVIDIA model failover.

**Status:** Complete — 2026-09-28

**PR:** [#17](https://github.com/tanzim10/clauseguard-ran/pull/17)

### Delivered

`POST /query` now retrieves specification passages, answers only from those passages, maps
citations to retrieved metadata, and returns `not found` when retrieval is empty or evidence is
insufficient. Corpus indexing and query embedding now use NVIDIA's Nemotron VL embedding model.

### Technical implementation

- Added a typed query request/response contract and a shared query service over `SearchService`.
- Added evidence IDs, structured answer parsing, citation allow-listing, and abstention behavior.
- Added bounded NVIDIA failover across Nemotron Lightning, GPT-OSS, Muse Glimmer, and
  DiffusionGemma, with transient-error cooldown and `Retry-After` handling.
- Added model-specific prompt behavior for DiffusionGemma, which does not accept the provider's
  JSON-mode option; its output remains checked by the same strict parser.
- Replaced OpenAI corpus embeddings with `nvidia/llama-nemotron-embed-vl-1b-v2`, using
  `passage` for indexing and `query` for retrieval, with a 2048-dimension Qdrant contract and a
  dedicated collection name.
- Documented setup, migration/reindex requirements, and an opt-in NVIDIA failover smoke script.

### Verification

- `uv run pytest` — 111 passed; Ruff and `git diff --check` passed.
- Rebuilt and ran the Docker API and Qdrant services; both were healthy.
- Indexed all 12 parsed documents into the new collection (10,715 chunks).
- Live `POST /search` and `POST /query` both returned HTTP 200; the query returned an answer
  with a citation to the retrieved O-RAN A1 passage.
- Live Muse Glimmer and DiffusionGemma checks passed against the production prompt and output
  parser. The opt-in failover smoke simulated a primary-model 429 and verified a live Muse
  Glimmer fallback response against the structured evidence-ID contract.

### Out of scope and follow-ups

- The prompt and citation allow-list do not independently verify semantic entailment.
- NVIDIA-only routing cannot recover from account-wide quota exhaustion; cooldown state is
  process-local, and answer wording can vary between models.
- `/rca`, KPI analysis, UI, LoRA, MCP, hybrid retrieval, and golden Q&A evaluation remain out
  of scope.
  
  ## Phase 7 — Golden Specification Q&A Dataset

**Focus:** Issue #16 — curate an initial, manually verified golden dataset for specification Q&A.

**Status:** Dataset artifact prepared — 2026-09-28

**PR:** [#18 — Add the initial verified specification Q&A dataset](https://github.com/tanzim10/clauseguard-ran/pull/18)

### Delivered

Added a tracked JSONL seed dataset with ten answerable O-RAN/3GPP questions and two unanswerable
questions represented by the required `not found` response and no citations.

### Technical implementation

- Added `data/golden/spec_qa.jsonl` with stable record IDs and the agreed question, expected answer,
  expected citation, and answerability fields.
- Citations use specification and section identifiers, not unstable Qdrant point or chunk IDs.
- Kept this change data-only; it does not connect the seed dataset to `/query` or an evaluation
  runner.

### Verification

- Validated all 12 JSONL records, required fields, unique IDs, answerability counts, and citation
  specification IDs against the corpus manifest.
- The expected answers and references were manually checked against the available parsed corpus.

### Out of scope and follow-ups

- Typed dataset loading and schema validation, evaluation execution, and runtime integration remain
  separate work.
- This initial seed is not a comprehensive evaluation of all standards or all valid answer wording.

## Phase 8 — Project Plan and Weekly Timeline

**Focus:** Documentation — publish the agreed project plan and dated weekly timeline.

**Status:** Complete — 2026-10-01

**PR:** [#19 — Add project plan PDF with dated weekly timeline](https://github.com/tanzim10/clauseguard-ran/pull/19)

### Delivered

In simple words: the team now has one short PDF that explains what the project does, lists every
task, and shows week by week what happens between September 1 and the December 11 submission.

### Technical implementation

- Added `docs/progress/ClauseGuardRAN_Project_Plan.pdf` and un-ignored only that file in
  `.gitignore`; the rest of `docs/progress/` stays local.
- The plan swaps two tasks: the KPI–text fusion model is now Task 5, and FastAPI + MCP + demo
  packaging is now Backup Task 1.
- Task 4 lists four RAG-based ablation configurations; no-RAG unimodal baselines stay in Task 3.
- The timeline uses Tuesday–Monday weeks from Sep 1, 2026, marks Weeks 1–3 delivered and Week 4
  slipped, skips Thanksgiving week, and ends on Friday, Dec 11, 2026.
- Linked the PDF from `README.md`.

### Verification

- Rendered the PDF with headless Chrome (4 pages) and checked the Task 4 pipeline line prints in
  full.

### Out of scope and follow-ups

- No code, API, or test changes.
- `.context/merge.md` and the README deliverables table still describe the original task order.

## TelecomTS scenario preparation — issue #21

**Status:** Complete for acquisition/conversion and analysis — 2026-10-14

### Delivered

- Named KPI windows preserve numeric and categorical values, missing samples and unknown units.
- Scenario records separate observations, labels, source provenance and human text approval.
- Streaming conversion preserves source references, stable identities and native annotations;
  every input line becomes a validated scenario or a rejection with its source location.
- Pinned acquisition verifies LFS SHA256 or Git blob identity and records SHA256 for every file.
  The ingestion CLI also supports checksum-verified offline source trees.
- The CLI exports a scenario overview and sample-level KPI CSV by default, with a schema/hash
  sidecar and explicit missing-value masks; these are analysis tables, not model inputs.
- One offline notebook analyzes coverage, missingness, sampled distributions/correlations,
  representative traces and text/lineage status, rendering and saving seven figures.

### Technical implementation

- Validate matching channel lengths, first/last sample timestamps and cadence.
- Use inclusive anomaly indices, explicit unresolved recording lineage, and reviewed symptom text.
- Record conversion counts and exact observation duplicates without claiming recording identity;
  checksum failures preserve diagnostics and existing outputs are never silently overwritten.
- `datasets/acquisition.py` performs immutable-revision downloads, cache verification, safe source
  path handling and retained failure manifests. The CLI records review/lineage/unit sidecar hashes.
- Optional analysis dependencies and a locked environment support the notebook. CSV scanning is
  chunked and distribution/correlation plots use a deterministic illustrative selection.
- The README documents preparation and notebook setup. The analysis notebook explains CSV
  masks, sampled plots and the boundary with #22.
- Keep the existing golden-evaluation contract unchanged; examples are illustrative local fixtures.

### Verification

- Local schema fixtures cover fractional values, missing/category values, invalid channels/times,
  annotation bounds, provenance evidence and text approval.
- Conversion fixtures cover source aliases, contextual versus injected labels, unknown causes,
  duplicate observations, invalid JSON, deterministic output and failed checksums. An actual pinned
  jamming row also passed conversion with all 18 channels preserved.
- Mocked acquisition and offline CLI tests cover failed checksums, wrong revisions, path escape,
  cache reuse, rejected rows and nonzero exit status.
- CSV fixtures verify deterministic output, source value preservation, scenario joins and null
  versus literal categorical strings. A pinned three-file run accepted all 468 source rows.
- Full acquisition at revision `1b3a88a440f4922edfcd1e230e42c003c0e13bdf` verified all 33 processed
  files and accepted all 32,000 rows with no rejections: 30,765 normal, 279 jamming and 956
  synthetic. The sample CSV contains 4,096,000 rows. All generated data remains ignored.
- Notebook execution is checked with an isolated local-fixture kernel and the full local release.

### Out of scope and follow-ups

- Splits, safe inference exports and release readiness remain in #22; baseline training remains in
  Task 3. Recording lineage and source text require review; all acquired records initially retain
  unresolved relationships and pending text review. No release-readiness claim is made.

## TelecomTS notebook readability — 2026-10-15

**Pull request:** [#41](https://github.com/tanzim10/clauseguard-ran/pull/41)

### Delivered

- Explain origin, fault presence and root-cause labels in plain language, with a meaning and
  window count for each supported root-cause label.
- Preview both CSV files and show a window's KPI names, types, sample values and time steps.
- Replace the coverage chart with labelled horizontal bars and show abnormal categories separately
  from normal windows. Explain length, cadence, channel types and units with bullets and a diagram.
- Add an example KPI figure with the source-labelled fault interval shaded when bounds are known.

### Technical implementation

- Keep CSV previews bounded to five rows and reuse the existing deterministic window selection.
- Read the selected scenario's native annotation from JSONL and display inclusive sample bounds
  in relative seconds. Preserve the notebook's local kernel choice and offline execution.
- The notebook now saves eight figures; the fixture check verifies their expected names.

### Verification

- The notebook executes on the offline fixture and the full local 32,000-window export.
- Checked the coverage, sampling and annotated-window figures visually; test lint and diff checks pass.

## Shared detection and category dataset release — issue #22

**Pull request:** [#42](https://github.com/tanzim10/clauseguard-ran/pull/42)

**Date:** 2026-10-15

**Status:** Implementation delivered; real-data scored readiness remains blocked by unresolved
recording relationships and missing task-approved text. No baseline model was trained.

### Delivered

One saved split now supplies both prediction tasks. Detection retains normal, synthetic-fault and
real-jamming examples. Category prediction selects ten supported synthetic classes without changing
membership. Recording evidence and split assignment are separate functions. Overlapping windows,
known recording aliases and related synthetic traces travel together as whole families.

The CLI writes a recording evidence table, window-to-raw mapping, membership CSV, quality report,
versioned targets and separate inference exports. The shared loader checks release integrity and
returns permitted observations separately from targets. The notebook previews the new tables and
adds four figures for overlap, binary counts, category/group coverage and exclusions/text coverage.

### Technical implementation

- `datasets/acquisition.py`: acquire checksum-verified supplemental CSV/description files in a
  separate `supplements.json`; preserve the processed-source manifest and existing conversion path.
- `datasets/contracts.py`: named 16-channel order, native ten-category mapping, explicit recording
  review, and typed predictor/example envelopes. `TextReview` now records approved tasks and whether
  wording came from independent observations, KPIs or diagnosis material.
- `datasets/evidence.py`: `build_recording_evidence` checks acquisition hashes, full-window values,
  source timestamps/cadence and categorical values, with a fixed absolute numeric tolerance. It
  records raw matching separately from human recording approval. Synthetic approval needs parent
  IDs. Existing provenance and candidate links are retained rather than silently treated as proof.
- `datasets/splits.py`: `assign_splits` connects whole traces and parent families, then uses seeded
  bounded search plus whole-group moves/swaps. Required label coverage precedes a size objective
  targeting 60/20/20 over the full release. No model scores or class resampling select membership.
  `select_task_cohort` derives detection/category and KPI/text/paired views; `build_quality` reports
  coverage separately from scored readiness. Missing support does not imply exhaustive proof that
  every possible assignment is infeasible.
- `datasets/release.py`: save authoritative `splits.json`, derived `membership.csv`, `labels.json`,
  `quality.json`, targets and safe inputs with a hashed release identity. Keep the scenario JSONL
  external by a recorded relative path. Existing artifacts are protected from overwrite.
- `datasets/scenarios.py`: `ScenarioDataset`, `load_scenario` and `iter_examples` validate hashes,
  source identity, complete membership, family separation, targets and quality before model use.
  File offsets support ID lookup without keeping KPI arrays in memory. Source edits invalidate
  the instance. Inspection can retrieve excluded records; model iteration cannot include them.
- `cli/prepare_scenarios.py`: separate evidence, release or combined stages; default scored mode
  refuses unreviewed releases. Explicit exploratory mode checks coverage but retains unverified
  status. Exit 3 preserves readiness diagnostics; exit 1 reports preparation errors. Ingestion can
  run this stage with `--prepare-release`.
- Task-specific human text review is required. Labels, reference answers, source paths, origin and
  diagnosis annotations stay outside predictor payloads. Text-only inputs contain no KPI array;
  paired inputs require both modalities. A secondary lexical guard rejects direct disclosures,
  but human provenance review remains necessary.
- README documents commands, review fields, outputs and loader use. Generated release analysis
  artifacts are ignored; golden curation and training remain downstream work.

### Verification

- Repository suite: `make test` — 178 passed. Fixture coverage includes raw checksums, tiny rounding
  differences, categorical/timestamp mismatches, malformed/duplicate IDs, whole-family membership,
  synthetic descendants, unrelated clocks, reproducibility, both targets, approved text/paired
  views, revealing text, artifact corruption and blocked readiness.
- Ruff checks and `git diff --check` pass for the change.
- Pinned production acquisition verified all 66 supplemental assets at revision
  `1b3a88a440f4922edfcd1e230e42c003c0e13bdf`. All 32,000 windows matched their raw sources.
- Executed the notebook with and without a fixture release, and against all 32,000 real windows.
  The full run saved 12 figures and an executed notebook under ignored
  `artifacts/telecomts_release_eda/`; overlap and category/group figures were visually inspected.
- Full-data loader inspection validated all 32,000 records. The held-out detection view returned
  6,321 examples, and the category view returned its 216-example subset with all ten labels.
  Default model loading rejected the exploratory release without explicit opt-in.

The current **exploratory source-trace** release has the following counts. They are preparation
results, not model scores or verified independent recordings:

| Split | Normal | Synthetic faults | Real jamming | Detection total | Category total |
|---|---:|---:|---:|---:|---:|
| Train | 18,432 | 563 | 94 | 19,089 | 563 |
| Dev | 6,321 | 177 | 92 | 6,590 | 177 |
| Test | 6,012 | 216 | 93 | 6,321 | 216 |

`data/scenarios/release/` contains this explicit exploratory release. A separate scored preparation
under `data/scenarios/release-scored/` correctly exited 3 and retained all 32,000 records as excluded:
no recording-family approvals were supplied. No text cohort is ready. Generated source assets,
CSV/JSONL exports, releases and analysis output remain local and ignored.

### Out of scope and follow-ups

Human recording-session/parent verification is still needed before scoring unseen-recording
performance. Task 3 owns training features, fitted preprocessing, label balancing and model
comparisons. Task 2.2 owns reviewed golden answers/citations; this release only reports eligible
scenario support. Do not close #22 as scored-data ready on the strength of source-window matching.

## Phase 9 — Golden Q&A Schema and Loader

**Focus:** Task 2.2a — make the golden row model match the real seed data and load it with clear
validation errors.

**Status:** Complete — 2026-10-18

**PR:** https://github.com/tanzim10/clauseguard-ran/pull/43

### Delivered

In simple words: the code can now read the golden Q&A file without dropping the expected answers
or citations, and it stops with a precise `file:line: what's wrong` message when a row is
malformed, so scoring work later can trust what it loads.

### Technical implementation

- `datasets/schema.py`: added `GoldenCitation` (non-blank `spec_id`, `section`) and rewrote
  `GoldenEvalRow` with required `question`, `expected_answer` and `expected_citations`,
  `type: Literal["spec_qa"]` (default kept), and `extra="forbid"`. TelecomTS scenario models are
  unchanged.
- Answerability rules live on the model: answerable rows need at least one citation; unanswerable
  rows need exactly `"not found"` and an empty citation list.
- Legacy placeholder fields (`expected_spec_id`, `expected_section`, `spec_refs_golden`,
  `scenario_id`) are rejected with a specific message — a conflict when combined with
  `expected_citations`, unsupported otherwise — instead of being guessed into citations.
- `eval/golden.py`: `load_golden(path, known_spec_ids=None)` checks JSON, schema, unique IDs and
  manifest references, raising `GoldenLoadError` with file, line and reason;
  `load_known_spec_ids(manifest)` reads every manifest `spec_id`, including documented blockers.
- Replaced `golden_eval.example.jsonl` with one answerable and one unanswerable row in the new
  shape. Documented loading in `README.md`.

### Verification

- `uv run pytest tests/test_golden_loader.py -q`
- `make test`

### Out of scope and follow-ups

- Loading validates structure and references, not answer truth.
- RCA row schema, metrics, the eval runner, and `/query` use of gold remain for later issues.

## Phase 10 — Reviewed Golden Q&A with Source Evidence

**Focus:** Task 2.2b — grow the golden spec Q&A set to 24 reviewed questions, each with evidence
another reader can find in the original documents.

**Status:** Complete — 2026-10-18

**PR:** https://github.com/tanzim10/clauseguard-ran/pull/43

### Delivered

In simple words: every golden question now has a review record saying who approved it, when,
what kind of question it is, and exactly where the answer sits in the original spec file
(document version, file hash, clause number and page). The set grew from 12 to 24 questions,
including harder wording, multi-clause answers and realistic "the corpus does not cover this"
cases.

### Technical implementation

- `eval/annotations.py` defines the annotation contract:
  - `SourceLocator`: `spec_id`, `version`, `sha256`, a bare `clause` number kept separate from the
    display `title`, and either `pdf_page` (1-based physical page in the file) or `docx_heading`
    (plus optional `docx_paragraph`). No chunk IDs, since those change on re-chunking.
  - `EvidenceGroup`: one required `claim` with one or more interchangeable `alternatives`.
  - `NoAnswerRecord`: `sources_checked` and `missing_evidence`.
  - `Annotation`: `status` (`draft` by default; `approved` needs `reviewer` and `reviewed_on`),
    overlapping `families` (`direct`, `multi_clause`, `difficult`, `no_answer`), free-form `tags`,
    and `corrections`. `no_answer` stands alone; `multi_clause` needs 2+ required groups.
- `load_annotations(path, manifest=None)` reports `file:line` errors (`AnnotationLoadError`
  subclasses `GoldenLoadError`) and, given manifest rows, checks every locator's spec, version,
  hash, file format (PDF page vs DOCX heading) and page range, plus no-answer sources.
- `check_against_golden(annotations, rows)` lists disagreements: unknown IDs, no-answer status
  mismatches, and expected citations whose clause is not in any evidence group.
- `data/golden/annotations.jsonl`: 24 approved records (reviewer `tanzim10`, 2026-10-14).
- `data/golden/spec_qa.jsonl`: seed rows 001–012 unchanged; added 013–024 in two batches of six.

### Coverage (24 unique approved IDs)

| Family | Count | IDs |
|---|---|---|
| direct | 10 | 001, 002, 004, 006–010, 014, 023 |
| multi_clause | 6 | 003, 005, 015, 017, 019, 021 |
| difficult | 6 | 013, 015, 016, 020, 021, 024 |
| no_answer | 4 | 011, 012, 018, 022 |

Families overlap (015 and 021 are both multi-clause and difficult). All 12 corpus documents are
cited by at least one answerable row.

### Verification

- `uv run pytest tests/test_golden_loader.py tests/test_golden_annotations.py -q`
- `make test`
- Real-data tests confirm annotations match the manifest and golden rows and that every golden
  row has an approved annotation.

### Out of scope and follow-ups

- Schema checks cannot prove a clause says what the claim says; that remains manual review.
- RCA cases, release membership, corpus expansion and metric calculation remain later work.

## Phase 11 — Golden RCA Contract and Release Manifest

**Focus:** Task 2.2c (sub-issue of #4): define golden RCA rows, give every golden row a
development/test membership, and pin the reviewed golden files as one checkable release.

**Status:** Partial, 2026-10-19. The RCA contract, membership and release manifest are done.
Approving the six RCA evaluation scenarios is still blocked, so the release reports itself
incomplete.

**PR:** https://github.com/tanzim10/clauseguard-ran/pull/50

**In simple words:** golden questions can now describe an incident, every reviewed row is labeled dev or test, and one command checks that the golden files still match a pinned release. The release stays incomplete until six RCA scenarios are approved.

**Technical details:** `RcaRow` separates measured KPI evidence, source labels, and mechanism citations. `load_rca()` does not change specification Q&A loading. `eval/release.py` pins file hashes, cited specification identities, and the scenario release, and `data/golden/rca.jsonl` is empty.

### Delivered

In simple words: there is now a fixed format for questions of the form "here is an incident, what caused it,
and which spec clauses back that up". It keeps three things apart: what the KPI data measured,
what the dataset label claims, and what a spec clause actually supports. Every golden question
is now labelled dev or test. One command checks that the golden files are consistent and
unchanged since the release was built. The release has 24 approved rows (19 dev, 5 test).
It reports itself incomplete until six RCA incidents are approved.

### Technical implementation

- `datasets/schema.py` adds `RcaRow` (separate from `GoldenEvalRow`, `type: "rca"`):
  - `scenario_id`, `origin` (`real`/`synthetic`), `inputs` (`kpi_only`/`multimodal`) and a
    symptom-only `question`.
  - `kpi_evidence`: for each channel, baseline and comparison intervals on the source timeline,
    both measured values, and a `direction` that must agree with those values.
  - `source_labels` (claims, not ground truth), `expected_conclusion`,
    `acceptable_conclusions`, `uncertainty`.
  - `expected_citations` with a `role` (`definition` or `mechanism`) and an `explanation`. At
    least one `mechanism` citation is required, because a measurement definition cannot support a cause.
  - The question may not contain any source label value, so it cannot reveal the diagnosis.
- `eval/golden.py` adds `load_rca(path, known_spec_ids=None)`, which shares the row loop with
  `load_golden`. Spec Q&A loading and its error messages are unchanged.
- `eval/annotations.py` adds an optional `membership` (`dev`/`test`). `check_against_golden`
  also accepts RCA rows.
- `eval/release.py`, run with `python -m clauseguard.cli.build_golden build|check [--strict]`:
  - Rejects missing or orphaned annotations, unapproved rows, missing membership, seed rows
    (001–012) outside dev, IDs repeated across files, and spec rows that cite a shared
    specification but sit in different splits.
  - RCA rows are checked against issue #3's scenario release (`data/scenarios/release`). The release is
    loaded through `ScenarioDataset`, so its identity, checksums and membership are validated
    by the same code that serves the baselines. Each referenced scenario must:
    - exist in a `scored` release;
    - not sit in `train` or `excluded`;
    - match the row's membership and origin;
    - have a usable KPI window.

    `multimodal` rows also need scenario text approved for the category task.
    The release is only required once `rca.jsonl` has rows.
  - `build` writes `data/golden/manifest.json`. It records:
    - file hashes and row counts;
    - the version and hash of every cited specification;
    - the scenario `release_id` and mode;
    - counts per split, a `release_id`, and coverage gaps.

    `check` also fails if any of those no longer match the files.
  - Fewer than 30 approved rows, fewer than 6 RCA rows, or a split without ordinary, difficult
    or RCA rows marks the release incomplete. That exits 0 unless `--strict` is passed.
- `data/golden/rca.jsonl` (empty) and `data/golden/manifest.json` are tracked.

### Membership

Split rule: spec rows that cite the same specification stay together, and seeds stay in dev.

| Split | Rows |
|---|---|
| dev (19) | 001–015, 019, 020, 021, 024 |
| test (5) | 016 (difficult), 017 (multi-clause), 018 and 022 (no-answer), 023 (direct) |

`annotations.jsonl` differs from Phase 10 only by the added `membership` key.

### Verification

- `uv run pytest tests/test_golden_rca.py tests/test_golden_release.py -q`. The tests cover:
  - RCA row validation and loading;
  - every release rejection above;
  - a check against a real scenario release built offline from the shared fixtures, plus a
    tampered-release case;
  - manifest reproducibility and staleness, and CLI exit codes;
  - a check that the repository's golden files match their manifest.
- `tests/test_golden_loader.py` and `tests/test_golden_annotations.py` are unchanged and pass.
- `make test` passes, and Ruff passes on the changed files.

### Out of scope and follow-ups

- **Blocked:** approving six eligible RCA incidents, reaching 30 approved rows, and RCA rows in both
  splits. No placeholder RCA rows or extra spec rows were added to fill these targets.
  The milestone stays incomplete until the six scenarios are resolved and reviewed.
- Scenario acquisition, RCA generation, training, metrics and evaluation runs are out of scope.

## Phase 12 — Golden Evaluation Runner

**Focus:** Task 2.2d (sub-issue of #23): one command that runs retrieval, optionally captures grounded answers, or replays a saved run, then scores and compares runs.

**Status:** Built, 2026-10-20. Final acceptance on an approved golden release still waits on the six RCA scenarios from Phase 11.

**PR:** TBD

**In simple words:** one command searches the golden questions, saves exactly what came back, and scores it. It can also capture grounded answers, replay a saved run offline, and compare two runs, including whether they are fair to compare.

**Technical details:** `eval/metrics.py` scores clause recall, group coverage, precision with duplicate rates, and MRR under a hashed policy. `eval/runner.py` reuses `SearchService` and `QueryService.answer()` so query mode searches once. Runs write `rows.jsonl`, `metrics.json`, `run.json`, and optional `comparison.json` under `artifacts/eval/`. `cli/run_eval.py` supports `retrieval`, `query`, `replay`, and `compare`.

### Out of scope and follow-ups

- BM25 or hybrid retrieval, reranking, corpus acquisition, full RCA generation, training, API/MCP endpoints, and automatic semantic grading.
- A live retrieval run against the indexed corpus is not included. Final acceptance waits for the approved golden release.

## Task 3.1 — Experiment Data

**Date:** 2026-10-18 · [Issue #29](https://github.com/tanzim10/clauseguard-ran/issues/29) · [PR #44](https://github.com/tanzim10/clauseguard-ran/pull/44)

Models now load checked batches from the saved split. Each window keeps its original 128 samples
and 16 numeric channels. Labels, source IDs and recording groups stay separate from observations.
The inspection CLI reports class counts, exclusions and whether the data is ready.

The local check validated all 32,000 records. Train/dev/test contain 19,089/6,590/6,321 windows;
category views contain 563/177/216. Text views are empty. Scored training remains blocked until
related source recordings are identified and kept together.

Files: `baselines/data.py`, `datasets/scenarios.py`, `cli/inspect_experiment.py`.
Verification: 285 tests passed at this step; Ruff and whitespace checks passed.

## Task 3.2 — Scoring and Saved Runs

**Date:** 2026-10-18 · [Issue #30](https://github.com/tanzim10/clauseguard-ran/issues/30) · [PR #44](https://github.com/tanzim10/clauseguard-ran/pull/44)

Runs save predictions, errors, model settings and data identity. The scorer covers detection,
fault category and combined diagnosis. Missing or failed predictions count as incorrect; reports
also show class counts and confusion matrices. Confidence intervals require at least ten
recording groups.

Saved runs can be checked, rescored and compared offline. Comparisons require identical evaluation
cases. The adapter interface lets later models use these functions. No model was trained.

Files: `baselines/contracts.py`, `scoring.py`, `artifacts.py`, `operations.py`,
`cli/baseline_runs.py`. Verification: 296 tests passed; Ruff and whitespace checks passed.

## Task 3.3 prerequisite — Overlap Audit

**Date:** 2026-10-18 · [Issue #31](https://github.com/tanzim10/clauseguard-ran/issues/31)

**PR:** [#45](https://github.com/tanzim10/clauseguard-ran/pull/45).

Added an audit command that saves proposed window selections without changing the source dataset.
Removing overlapping normal training windows reduces training from 19,089 to 5,381 while keeping
all 657 training faults. Dev/test remain 6,590/6,321; their final size is still being discussed.

A recording family means one original recording plus files derived from it, including synthetic
fault versions. The upstream paper and code do not provide the parent map needed to confirm these
families. Window reduction does not solve that missing information.

Files: `datasets/window_overlap.py`, `cli/audit_window_overlap.py`.
Verification: 301 tests passed; Ruff and whitespace checks passed. RF training comes next, with
shared modules used by both a notebook and CLI.

## Task 3.3 — Random Forest implementation

**Date:** 2026-10-18 · [Issue #31](https://github.com/tanzim10/clauseguard-ran/issues/31)

**PR:** [#46](https://github.com/tanzim10/clauseguard-ran/pull/46).

Added real-data plots and a notebook before training. Shared modules now provide 224 KPI features,
train-only median imputation, and separate detection/category Random Forest models. The runner
tries four 300-tree settings on dev, then saves the chosen settings for seeds 42/43/44.

The notebook and CLI use the same feature/model/runner code. Checkpoints save feature order and
medians; prediction reloads them without fitting. Runs record errors, settings, selected training
IDs, timings and process memory. Optional normal-window reduction leaves all faults and dev/test
cases intact. Tiny fixture runs cover both tasks; they are not TelecomTS performance results.

The notebook executed all 19 cells, including generated-data training and checkpoint reload.
Real scored training correctly remains blocked by the unresolved recording review. Verification:
325 tests passed, targeted Ruff and whitespace checks passed. Real development scores remain outstanding.

The RF CLI now has an explicit `--allow-exploratory` option when split coverage exists but recording
independence is unresolved. It records provisional status in the run metadata. Category runs remain
unavailable on the synthetic-training release because dev/test contain no category examples.

An exploratory detection run trained on 19,173 windows and selected the unlimited-depth, 1-minimum-leaf
model. Across three seeds, dev macro-F1 ranged from 0.414 to 0.419. On the held-out test set, macro-F1
was 0.436 and jammer recall was 0.935, but it raised 2,135 false alarms among 6,321 normal windows
(33.8% false-positive rate). The test has only four recording groups, so no confidence interval is
available. Synthetic faults occur only in training, so this run does not measure synthetic-fault
detection or category prediction. Treat all scores as provisional until recording independence is
resolved.

## Recording review audit

**Date:** 2026-10-18 · [Issue #31](https://github.com/tanzim10/clauseguard-ran/issues/31)

Added `scripts/audit_recording_families.py` to check acquired file hashes and scan all eligible
16-row numeric segments across source pairs. It distinguishes two channels matching the same
segment from weak matches at separate positions, then saves findings and unresolved review JSON.

The local scan covered 33 traces, 1,139,967 raw rows and 9,919,949 eligible channel segments.
It found no corroborated shared segments and one weak offset between normal files already in
the same split. Nine identical-description pairs are metadata hints only. Synthetic parent IDs
remain unknown; the release and approvals are unchanged. Six focused tests cover copied segments,
weak matches, repetitive values, missing values, integrity checks and review output.
Verification: 331 tests passed; targeted Ruff and whitespace checks passed.

The existing release command can also assign all synthetic source groups to training for an
exploratory detection view. It leaves category dev/test empty and recording status unresolved.
The new release contains 19,173/6,413/6,414 train/dev/test windows; detection label coverage is
present, while readiness stays blocked by recording review. The original release is unchanged.

## Task 3.4 — XGBoost baseline

**Date:** 2026-10-19 · [Issue #32](https://github.com/tanzim10/clauseguard-ran/issues/32)
**PR:** [#47](https://github.com/tanzim10/clauseguard-ran/pull/47)

Added CPU XGBoost for detection and all ten fault categories. The runner verifies its training
selection, development labels, 224 feature columns, fitted medians and dense feature matrices
against an existing Random Forest run. Both tree runners now share cohort-loading and training
selection helpers. The existing RF notebook work remains separate.

Training uses exact balanced weights `N / (K * class_count)` from training labels, with unweighted
development logloss/mlogloss for early stopping (30-round patience, at most 500 rounds). Four
depth/child-weight settings compete on dev macro-F1 at their loss-best checkpoints. The winner
is repeated with seeds 42/43/44; zero seed variation is accepted with full row/column sampling.

Native UBJSON checkpoints save selected rounds and feature names. Reload validates the checksum,
objective, classes and preprocessing, and preserves one-thread CPU inference. Runs save loss
histories, selected iterations, errors, software versions and resource measurements. CLI and
`notebooks/xgboost_baseline.ipynb` call the same modules. XGBoost 3.2.0 is locked as an optional
baseline dependency. Exploratory runs retain their status in the metadata.

Resource interpretation: XGBoost's feature-preparation time includes train and dev transformations;
RF's preparation field covers train only. Prediction timers include transformation for both.
Peak RSS is a process-wide high-water mark, including earlier fits and loaded reference models.
These fields are not isolated per-model memory or directly comparable feature-only timings.

The full-data run used the same 19,089 detection and 563 category training windows as RF. Both
tasks selected depth 6 / child weight 5, retaining 127 and 271 rounds respectively. All three
seeds produced identical development scores.

| Task | RF dev macro-F1 (mean) | XGBoost dev macro-F1 (mean) | RF test macro-F1 | XGBoost test macro-F1 |
|---|---:|---:|---:|---:|
| Detection | 0.661 | 0.882 | 0.663 | 0.925 |
| Category | 0.569 | 0.518 | 0.224 | 0.352 |

The seed-42 detection test caught 240/309 faults, with 69 misses and 11 false alarms, versus RF's
65/309 caught, 244 misses and one false alarm. XGBoost caught 90/93 jammer windows and 150/216
synthetic faults. Category test accuracy was 49.5%, versus RF's 31.0%. Native reload reproduced
every selected-model dev prediction for both tasks. Settings were chosen on dev; current test
results remain exploratory because recording independence is unresolved and test has already
been inspected in earlier work. Artifacts: `artifacts/experiments/xgboost-full-data-2026-10-19/`.

Verification: 353 tests passed (`make test`), including 22 new XGBoost model/runner checks.
Targeted Ruff and whitespace checks passed. The 14-cell XGBoost notebook executed successfully
and saves the data head, loss plots, RF comparison and source/category errors.

## Task 3.6 — Initial KPI model comparison

**Date:** 2026-10-20 · [Issue #34](https://github.com/tanzim10/clauseguard-ran/issues/34)
**PR:** [#49](https://github.com/tanzim10/clauseguard-ran/pull/49)

In simple words: one command now puts Random Forest, XGBoost and TimesNet side by side on the
development data, with detection and category kept in separate tables. Each row shows the
selected settings, the score for each of the three seeds, false alarms, missed faults, failed
predictions, training and prediction time, peak memory and model file size. It refuses to compare
models that were not trained and scored on exactly the same data, and it never looks at test.

Technical details: `baselines/comparison.py` reads each experiment's `selection.json`, reloads
every selected seed with `load_run` (checksum check, then rescoring from saved predictions) and
requires the run to be a dev run whose score equals the selection record. Experiments are
compared only when release ID, dev cohort ID, training-ID hash, evaluation status, scored cases
and recording groups all match. Standard deviation is the population value, as in the selection
files. `cli/compare_models.py` writes `comparison.json` and `comparison.md` to a new directory and
refuses to overwrite. It needs no model code, so it does not depend on the TimesNet branch.

Result on the full-data dev runs (exploratory release, so provisional):

| Task | Random Forest | XGBoost | TimesNet |
|---|---:|---:|---:|
| Detection dev macro-F1 | 0.661 | 0.882 | 0.616 |
| Category dev macro-F1 | 0.569 | 0.518 | 0.166 |

TimesNet raised 407-1,094 false alarms across seeds in detection (XGBoost 26), and its category
score is far below the others. Settings are therefore not frozen for the transformer stage yet.

Follow-up the same day: the 0.74 TimesNet figure quoted from earlier work was a different
experiment (trained on 5,381 windows, scored on the 1,894-window reduced dev set with repeated
normal windows removed, seeds 0.627 / 0.932 / 0.655, where 0.932 is a single lucky epoch). The
comparison now takes `--window-selection` and rescores every model from its saved predictions on
that same reduced set, and reports the seed median and range so an outlier is visible.

| Task (reduced dev set) | Random Forest | XGBoost | TimesNet (full-trained) |
|---|---:|---:|---:|
| Detection macro-F1, mean (median) | 0.669 (0.672) | 0.888 (0.888) | 0.729 (0.717) |
| Category macro-F1, mean (median) | 0.569 (0.567) | 0.518 (0.518) | 0.166 (0.153) |

The category cohort is identical in both views (all 177 windows are faults). No saved TimesNet
category run averages 0.32; the best single seed is 0.267 macro-F1 and 0.339 accuracy.

Not done here: batch-size-one latency (needs the TimesNet code from PR #48 and a model-loading
timing harness), the remaining GPU budget, and the written readiness review for the transformer
probes. Reported fit time and peak memory come from each run's own record; peak memory is a
process-wide high-water mark and fit times came from different devices (CPU trees, MPS TimesNet),
so they are not directly comparable.

Verification: 359 tests passed (`make test`), including 6 new comparison checks that cover
changed cohorts, test-split runs, tampered score records and corrupted runs. Ruff passes on the
new files. Artifacts: `artifacts/experiments/task-3-6-model-comparison-2026-10-20/`.
## Task 3.5 — TimesNet (PyTorch) implementation

**Date:** 2026-10-19 · [Issue #33](https://github.com/tanzim10/clauseguard-ran/issues/33)
**PR:** [#48](https://github.com/tanzim10/clauseguard-ran/pull/48)

**In simple words:** the TimesNet model, its training/scoring runner and a command line are
built and tested on tiny fixtures. No real training has been run; that is done manually next.

**Technical details:**
- `kpi/timesnet.py` adapts the classification path of thuml/Time-Series-Library at
  `4e938a17` (MIT, notice kept). Upstream's runner and loader are not used because they pick
  models on test and standardize each split separately. Parameters: 698,690 (detection),
  731,466 (category).
- `WindowScaler` fits medians, means and population SDs on train only; windows are
  `(128, 16)` in `NUMERIC_CHANNELS` order and the padding mask is all ones.
- Training uses balanced `N/(K·count)` weights (weighted-mean reduction), RAdam, gradient
  clip 4.0, batch 32, and stops on unweighted dev cross-entropy (patience 10, cap 100). The
  four-configuration grid is selected by dev macro-F1 at the loss-best epoch.
- Prediction runs in cohort order at batch size 32, recorded in metadata, because periods
  are chosen per batch. Each run reports batch-1 vs batch-32 agreement on 512 dev windows.
- `baselines/timesnet.py` checks release, training IDs, dev cohort and dev labels against the
  XGBoost seed-42 run and never loads test during training. Weights are saved as
  `model.pt` (`weights_only` reload, SHA-256). Runs stay `provisional_exploratory`.
- `cli/timesnet.py` adds `train` (`--max-epochs`, `--patience`, `--device`,
  `--skip-seed-repeats`) and `predict`. torch is an optional `timesnet` extra.
- Only CPU runs are bit-reproducible; MPS is recorded as non-deterministic.
- `tests/conftest.py` sets `OMP_NUM_THREADS=1` because torch and XGBoost OpenMP runtimes
  deadlock together on macOS.
- Measured on an M3 Pro: about 45 s per detection epoch on MPS, about 5 min on CPU
  (extrapolated); about 3 s per category epoch on MPS.

**Follow-up (2026-10-20): training diagnosis and switches.** The first capped runs reached
only 0.48–0.58 dev macro-F1 for detection and chance for category. A root-cause check found
that RSRP/UL_SNR levels act as a recording-location shortcut: one normal dev recording
(`d0fec5`, 41% of reduced dev) is classified correctly only 58% of the time, and
re-centring just those two channels fixes it (0.58 → 1.00). The same channels also carry the
real jamming signal (YouTube jammer 0.67 → 0.03 when re-centred), so per-window
normalization (`--instance-norm`) fixes the normal recording (dev macro-F1 0.67 → 0.71, three
seeds) but loses the jammer. XGBoost scores 0.888 on the same reduced dev windows. Category
memorizes (a boosted-tree model reaches 0.459 dev macro-F1 where TimesNet gets 0.165).
Opt-in switches, all off by default so earlier runs stay reproducible:
- `--window-selection`: apply an audited window selection to train, dev and test.
- `--recipe v2`: prior-corrected stopping, cosine LR decay, 3 inception kernels.
- `--head pooled`: average over time before the classifier (about 4,000 fewer weights per class).
- `--recording-weights`: every recording counts equally within its class.
- `--instance-norm`: per-window standardization.
Per-recording dev accuracy is logged every epoch. Results remain exploratory.

**Verification:** 24 TimesNet tests; full suite 377 passed; targeted Ruff and whitespace checks pass.

## Task 3.9 — Frozen MantisV2 baseline — 2026-10-28

### Delivered

The baseline now uses a frozen MantisV2 encoder for TelecomTS detection and category prediction.
It compares embeddings alone with embeddings plus 32 channel statistics. The runner checks both
feature sets against saved RF and XGBoost runs on the same release, windows and labels.

The full development run selected embeddings for detection and embeddings plus statistics for
categories. Detection macro-F1 was 0.766. Category macro-F1 was 0.351. These results are
provisional because the release still has unresolved recording reviews. No test scores were run.

## Task 4.1 — RCA input and output contract — 2026-11-03

### Delivered

Defined versioned records for RCA requests, evidence packets, model outputs, server responses, and evaluation references. The contract keeps detector predictions separate from final causes and keeps evaluation labels outside inference inputs.

### Technical implementation

- Added strict request and output schemas with bounded retrieval, evidence-linked claims, single or multiple causes, and explicit abstention rules.
- Added typed source, detector, KPI, text, specification, packet, and server response records, plus checks that resolve claim citations against the exact packet.
- Added the five-dataset source matrix, field rules, experiment projections, failure states, and all 16 acceptance checks in `data/golden/rca-contract.md`.
- Added an exploratory source example with its injected empty-retrieval condition and evidence limits. Recorded the owner's approval of the contract example only; it is not a golden RCA case.

### Verification

- Full test suite: 515 passed.
- Ruff and `git diff --check` passed.

### Technical implementation

- Pinned `mantis-tsfm` 1.1.0 to upstream commit
  `23ecfe493830db177546ef159947de7d79a573bd`.
- Pinned `paris-noah/MantisV2` revision
  `99fe0f548960e272fbfa4b82fd9b5b5956779dfd` and verified its weight checksum before loading.
- Kept 128 measurements and 16 ordered channels. The encoder uses 32 patches and returns 512
  features per channel, for 8,192 total. The second input adds 32 imputed-channel statistics.
- Fit missing-value medians and feature scaling on train only. Fit balanced logistic heads with
  exact `N / (K * class_count)` weights. Select each head by full-dev macro-F1.
- Saved both heads, predictions, settings, errors and feature caches. Cache keys include checkpoint,
  release, split, cohort, input and preprocessing identities. Cache files have checksums.
- Added the shared train/replay CLI and a notebook with data plots, comparisons and errors by origin.
  Cached dev replay does not need to load the encoder again.

### Development results

| Task | Model | Macro-F1 | Accuracy | Fault recall | False alarms | Missed faults |
|---|---|---:|---:|---:|---:|---:|
| Detection | MantisV2 | 0.766 | 0.963 | 0.561 | 128 | 118 |
| Detection | Random Forest | 0.663 | 0.958 | 0.275 | 83 | 195 |
| Detection | XGBoost | 0.882 | 0.983 | 0.691 | 26 | 83 |

MantisV2 category accuracy was 0.429. RF reached 0.672 and XGBoost reached 0.571.
The category dev set has 177 windows and only two recording groups. Group confidence intervals
are unavailable. MantisV2 did not beat either tree model on this task.

The CPU run encoded the detection train and dev sets in 189 seconds. The category sets took
5.5 seconds. Peak process memory was 3.3 GB. The feature cache used 869 MB.

### Verification

- Fifteen focused Mantis tests passed, including cache checks, checkpoint validation and replay.
- Ruff passed for the new Python files. The notebook ran all six code cells with cached features.
- The run used release `f3bd4d46bb04363adbecd0e3106767f179c560815b09fdf6309aaf013fc0b784`.
  Its status remains provisional. No test predictions were generated.

## Task 4.3 — Draft RCA golden cases and independent model review — 2026-11-04

**Focus:** dev #34 (public #58), first milestone: reviewed RCA cases for the golden release.

**Status:** Partial. Twelve evidence-grounded RCA cases are drafted and model-reviewed, but none
is approved yet and they stay outside `rca.jsonl`, because the only scored scenario release
excludes every window (all 33 recording reviews are unresolved).

### Delivered

In simple words: twelve "here is a KPI incident, what caused it, which spec clause explains it"
cases built from real TelecomTS measurements. Every number was measured from the raw source file
and re-computed by an independent script. Every citation quote was checked on its PDF page. A
model from a different family reviewed each case through a new CLI. A person still has to approve
each case before it can enter the golden release.

- `data/golden/rca_drafts/`: `rca.draft.jsonl` and `annotations.draft.jsonl` (12 rows, all
  `status: draft`), the selection rule with its amendments, the case inventory with every
  rejected candidate and its reason, the research notes, and the review packet.
- `clauseguard.cli.validate_rca`: reviews RCA rows with one named model on an OpenAI-compatible
  endpoint (`--provider gemini|groq|nvidia`) and saves one verdict file per row plus a summary.
- `scripts/rca_drafts/`: the scripts that rebuild the drafts from local data.

### Case set

| Split | Cases | Dataset labels |
|---|---|---|
| dev | 5 | Co-Channel Interference (Severe) ×2, Congestion Gradual Buildup, Congestion Sudden Spike ×2 |
| test | 7 | Co-Channel Interference (Mild, Severe ×2), Congestion Gradual Buildup ×2, Buffer Overflow ×2 |

The 12 cases come from 12 distinct anomaly events but only 4 synthetic base recordings (2 per
split). All are `kpi_only`: no scenario text is approved and the tickets reveal the label.
Rejected: real jamming (no in-window baseline; jammer-on and jammer-off rows are
indistinguishable), Antenna, RF filters and Doppler (no mechanism clause in the corpus), Handover
(its oscillation is not observable in the spans; one-gNB testbed), Resource Allocation (weak
mechanism), and events whose measured change did not show the category's primary symptom.

### Technical implementation

- Overlapping windows (stride 32) were merged into 41 dev/test synthetic events. Each case uses
  the event window with the most balanced pre-fault and fault samples; baseline and comparison
  come from the same window and were recomputed from `data/telecomts/raw/.../metrics.csv`
  (0 mismatches against the release; an independent re-computation matched all 192 values).
- A channel counts as changed only above 2x baseline std and 5 % of baseline (BLER also needs a
  1-point move). Each category's primary symptom must be present (amendment 5b).
- Conclusions state only measured changes. Where the data contradicts the dataset label (the
  buffer falls in the two "Buffer Overflow" windows; PRB utilization stays in low single-digit
  percent), the conclusion follows the data and the uncertainty says so.
- Citations come from 28.552, 32.425 and 38.300; quotes were verified on their physical PDF pages
  and locators match the corpus manifest. Mechanism support is partial: interference relies on an
  informative LTE annex (32.425 A.9); congestion and backlog rely on 38.300 scheduler and
  load-balancing text. These limits are written into each case's uncertainty.
- `eval/rca_validation.py` sends the row, its KPI evidence and the text of every cited page;
  validates the reply against a schema; keeps provider errors and malformed replies as failed
  records; resumes per provider, model and prompt hash. One model per run, no failover.

### Verification

- The drafts load through `load_rca`, `load_annotations` (against the corpus manifest) and
  `check_against_golden` with no problems.
- `uv run pytest tests/test_rca_validation.py` (11 passed); full suite 453 passed, 11 skipped.
- Independent review through `clauseguard.cli.validate_rca` (verdicts in
  `data/golden/rca_drafts/reviews/`), all models outside the drafting model's family:
  - gpt-oss-120b via OpenRouter, all 12 (final text): 1 valid, 10 valid with changes, 1 invalid (rca_001);
  - qwen3.8-27b via Groq, all 12 (final text): 7 valid, 5 valid with changes;
  - gemini-3.6-flash, 10 of 12 on an earlier draft (free daily quota): all valid.
  No reviewer found a question that leaks the answer. Recurring requests: an NR-specific mechanism
  clause for interference (none exists in the corpus), treating the dataset label as a definitive
  cause (it is kept only as `source_labels`), and the use of 28.552 A.78 (downlink SS-SINR) as the
  closest definition for UL_SNR. Verdicts differ between models and between runs of the same model,
  so they inform, but do not replace, the human review.

### Out of scope and follow-ups

- **Blocked:** approval and release entry. `build_golden check --strict` requires a scored scenario
  release; the current one excludes all windows. The PM must choose between reviewing the four
  source recordings, an exploratory-golden tier, or keeping the cases as drafts.
- A person must open each cited page and approve or reject each case (`review_packet.md`).
- NVIDIA NIM keys could not be created from Bangladesh, so the review used OpenRouter, Groq and Gemini (OpenAI-compatible providers in the same CLI).

## Task 4.3 — Apply GPT-OSS review edits — 2026-11-05

**Pull request:** [#66](https://github.com/tanzim10/clauseguard-ran/pull/66)

### Delivered

Populated the RCA file with 12 revised cases beside the 24 existing specification questions.
The project owner approved all 12 cases and confirmed human citation review on 2026-11-05.
The project owner waived recording-family verification for this release. Independence remains unverified.

### Technical implementation

- Added a repeatable script that applies bounded edits to the preserved drafts and saved GPT-OSS findings.
- Preserved source labels, measurements, scenario identities, and split membership.
- Removed downlink SS-SINR from required evidence for uplink SNR cases.
- Recorded approved annotations and rebuilt the golden manifest with 36 approved rows.
- Added a hashed release policy with per-case eligibility and the selected scenario membership snapshot.
- Allowed the explicit family waiver for this release while preserving source mode and scenario split checks.
- Kept draft research and scripts local and ignored, as requested.
- Added 12 reviewed question records in indented JSON, with descriptive and numeric variants.
- Preserved their KPI-derived origin, raw measurement links, development/test membership, and exclusion from SLM training.

### Verification

The loaders accepted 12 RCA rows and 36 annotations. Cross-reference checks found no problems.
The focused golden tests passed: 104 tests. Source-value checks confirmed unchanged measurements and the 24 original specification annotations.
The updated release passes strict checks under the approved waiver. The final focused suite passed 107 tests, and Ruff passed. Text-only RCA remains ineligible because approved incident text is absent.
The full suite passed 519 tests. The reviewed JSON contains 12 records; its JSONL duplicate was removed before PR preparation.

## Task 3.8 — Held-Out Evaluation of the KPI Baselines — 2026-11-05

**Focus:** [Issue #40](https://github.com/tanzim10/clauseguard-ran/issues/40) — one
reproducible test report for Random Forest, XGBoost and TimesNet, with settings frozen on dev.

**Status:** KPI-only report complete and provisional. The KPI/text comparison is blocked
because no text is approved (Task 3.7), so full Task 3 is not complete.

**PR:** [#67 — Freeze Task 3 KPI protocol and report held-out baseline results](https://github.com/tanzim10/clauseguard-ran/pull/67)

### Delivered

In simple words: one command writes down every choice made on development data (models,
settings, seeds, checkpoints, test cohorts, earlier test exposure, text status) and seals it
with a hash. A second command then runs the saved checkpoints on test, never refitting
anything, and builds CSV tables, a markdown report and a notebook with figures from the saved
per-case predictions.

| Task (test) | Random Forest | XGBoost | TimesNet | Reference |
|---|---:|---:|---:|---:|
| Detection macro-F1 (mean of 3 seeds) | 0.674 | **0.925** | 0.638 | 0.487 (all normal) |
| Fault recall | 0.19-0.28 | 0.777 | 0.64-0.72 | 0 |
| False alarms / 6,012 normal | 1 | 11 | 564-844 | 0 |
| Category macro-F1 (216 windows) | 0.229 | **0.352** | 0.138 | |
| Combined diagnosis macro-F1 | 0.202 | **0.322** | 0.211 | |

- XGBoost is the strongest detector; TimesNet catches many faults but raises hundreds of false
  alarms. Calling every window normal already gives 95.1% accuracy, so accuracy alone misleads.
- Category scores collapse from dev to test (RF 0.569 → 0.229). The category test set has only
  two recording groups, so this mostly measures recording shift, not model quality.
- Doppler Shift (Severe) and High Network Congestion (Sudden Spike) test windows are missed by
  every model. Random Forest misses 89 of 93 jamming windows; XGBoost misses 3.
- Group confidence intervals are unavailable (5 detection groups, 2 category groups; 10 needed).
- RF/XGBoost seed-42 numbers match the earlier Task 3.3/3.4 test figures, which confirms the replay.

### Technical implementation

- `baselines/final_evaluation.py`:
  - `freeze_protocol` reuses the Task 3.6 dev comparison (same release, dev cohort and training
    IDs required), records each selected seed's run and checkpoint hash, the dev comparator per
    task (detection XGBoost, category Random Forest), the test cohort IDs, every earlier test-split
    run found on disk, and the Task 3.7 text readiness. It refuses to overwrite a protocol, and
    any later edit breaks its hash.
  - `predict_test` checks the protocol hash, the test cohorts and every checkpoint, then calls each
    model's existing `predict_saved_model`. Completed runs are reused, so the report regenerates
    without model libraries.
  - `build_tables`/`write_report`: detection, category, combined diagnosis
    (`score_combined_diagnosis`, detector first, jamming only in detection), per-class, errors by
    origin, seed summary, cost, a normal-majority reference, and `confusion.json`.
    `matching_case_counts` (both correct / KPI only / text only / neither) is ready for when text exists.
- `cli/final_evaluation.py`: `freeze` and `evaluate`. `notebooks/task3_final_evaluation.ipynb`
  plots detection, per-class category F1 and the comparator's diagnosis confusion matrix.
- `baselines/artifacts.py` fix: `load_run` compared saved and recomputed metrics exactly. Python
  3.12 made float `sum()` compensated, so runs saved under 3.11 failed to reload under 3.12 by one
  unit in the last place. Floats now compare with `rel_tol=1e-12`; file checksums still catch edits.

Environment: test predictions ran on Python 3.11.15 with scikit-learn 1.9.1, XGBoost 3.2.0,
torch 2.14.1 and numpy 2.4.6, the versions the checkpoints were trained with (the lock resolves
XGBoost 3.4.1 on 3.12). TimesNet ran on CPU. A first attempt on Python 3.12 was stopped after it
wrote one Random Forest test run; that run was deleted unread and the protocol was frozen again
under 3.11 before any test score was viewed. On macOS without Homebrew `libomp`, XGBoost loads
with `DYLD_FALLBACK_LIBRARY_PATH` pointing at scikit-learn's bundled `sklearn/.dylibs`.

### Verification

- `uv run pytest tests/test_final_evaluation.py tests/test_baseline_artifacts.py -q`: hand-worked
  combined diagnosis (perfect, all-normal and all-alarm detectors), errors by origin, the four
  correctness groups, failure accounting, changed checkpoints, protocol tampering, reuse of runs.
- `make test` with all baseline extras: 512 passed, 1 skipped. Ruff passes on the new files.
- Artifacts: `artifacts/experiments/task-3-8-final-evaluation-2026-11-05/` (protocol `458c1e61`,
  18 test runs, `tables/*.csv`, `report.md`, `figures/`).

### Out of scope and follow-ups

- Prior test exposure: RF and XGBoost seed-42 test scores were seen in Tasks 3.3/3.4; this is not
  a fresh untouched evaluation. All numbers stay provisional until recording families are verified.
- The KPI/text matching-case analysis needs approved text (Task 3.7).
- Mantis (Task 3.9) and Toto (Task 3.10) are separate issues and not in this report.

## Task 3.10 — Frozen Toto 2.0 Detection Baseline — 2026-11-05

**Focus:** [Issue #53](https://github.com/tanzim10/clauseguard-ran/issues/53) — test whether
forecast errors from a frozen Toto 2.0 model add useful fault-detection evidence.

**Status:** Complete on dev; provisional because recording reviews are unresolved. No test scores were run.

**PR:** [#68 — Add frozen Toto 2.0 forecast-error detection baseline](https://github.com/tanzim10/clauseguard-ran/pull/68)

### Delivered

In simple words: Toto looks at the first 96 measurements of each window and predicts the last 32.
If the prediction is far from what really happened, the window is flagged. A simple
"repeat the last value" forecast is scored the same way as a reference.

| Detector (dev, 6,590 windows) | Macro-F1 | Fault recall | False alarms | Missed faults | Failed |
|---|---:|---:|---:|---:|---:|
| Toto 2.0 22m | 0.561 | 0.156 | 222 | 227 | 0 |
| Last value | 0.533 | 0.086 | 175 | 246 | 0 |
| XGBoost (seed 42) | 0.882 | 0.691 | 26 | 83 | 0 |

Toto beats the last-value reference but is far below XGBoost. It misses 88 of the 92 dev jamming
windows: the fault covers the whole window, so the context already contains it and the forecast
error stays small (the limitation the issue predicted). Of 269 dev faults, 222 have an injected
interval reaching the forecast segment; Toto catches 33 of those.

### Technical implementation

- `kpi/toto.py`: pins `toto-2==2.0.0`, `Datadog/Toto-2.0-22m` revision
  `685e4ae3e2be8d8998025e53dd98e7fdcb296a89` and `model.safetensors` SHA-256 `9cd503d8…`.
  The loader checks the hash, the 21,915,584 parameters and the nine quantile levels, then
  freezes every parameter.
  - Inputs are `(batch, 16, 96)` in `NUMERIC_CHANNELS` order, `series_ids` zeros `(batch, 16)`,
    `decode_block_size=None`, `horizon=32` (96 and 32 are whole 32-step patches; no padding).
    The median (index 4 of `(9, batch, 16, 32)`) is used, and the shape is checked.
  - Masked context values get a finite placeholder (last observation, else next, else 0); the
    future 32 steps never reach the model. `has_missing_values=True` for every batch, so a
    window's forecast never depends on whether a neighbour has gaps.
  - Scales: population SD of observed values in normal training windows, 1 for a constant
    channel. Score: mean over 16 channels of MAE on observed targets / scale. A channel with
    no observed context or targets, or a non-finite forecast, is a failed prediction.
  - Threshold: 101 quantiles of finite dev scores plus an all-normal endpoint (saved as
    `null`); best dev macro-F1, then fewer false alarms, then higher threshold. `score >= threshold`.
- `baselines/toto.py`: scales from 19,089 training windows, forecasts cached by checkpoint,
  release, cohort, context, masks, channel order and inference settings. It checks repeat
  predictions (identical) and batch isolation on 8 windows (max abs difference 0.125 on values
  in the hundreds of thousands, within `rtol=1e-4`), and checks that weights are unchanged after
  forecasting and threshold selection. Runs save scales, threshold, scores, observation masks, errors by
  origin and forecast-segment coverage, then are compared with XGBoost on identical dev IDs. `predict`
  replays a saved detector without refitting.
- `cli/toto.py` (`train`, `predict`), `notebooks/toto_baseline.ipynb` (comparison, score
  distributions with threshold, errors by origin, one forecast with the 10%-90% band for display).
- `pyproject.toml`: optional `toto` extra with a `python_version >= '3.12'` marker; the main
  package still supports 3.11. Install: `uv sync --python 3.12 --extra dev --extra baselines --extra toto`.

### Verification

- `uv run pytest tests/test_toto.py tests/test_toto_runner.py -q`: 10 passed with a fake
  forecaster (input split, scales, masks and placeholders, score formula and failure reasons,
  threshold rule and ties, forecast call arguments and shape check, cache reuse without the
  model, replay, tampered detector, failed forecasts kept, weight drift).
- Real run on CPU (M3 Pro): 70 s to forecast the 6,590 dev windows (119 s on a busier first
  run), peak process memory 1.27 GB, model 21.9M parameters (87.7 MB weights). Replaying both
  saved detectors on dev reproduced every prediction. The notebook ran all cells without errors.
- Artifacts: `artifacts/experiments/toto-detection-2026-11-05/`, cache `artifacts/cache/toto/`.

### Out of scope and follow-ups

- Test scoring belongs to the frozen final evaluation, after the threshold is fixed.
- Category prediction, fine-tuning, larger checkpoints and other horizons stay out of scope.

## Task 3.11 — PatchTST Baseline — 2026-11-06

**Focus:** [Issue #54](https://github.com/tanzim10/clauseguard-ran/issues/54) — test whether
a small supervised Transformer over KPI patches learns useful temporal patterns.

**Status:** Complete on dev; provisional because recording reviews are unresolved. No test scores were run.

**PR:** [#69 — Add supervised PatchTST baseline for detection and categories](https://github.com/tanzim10/clauseguard-ran/pull/69)

### Delivered

In simple words: each of the 16 KPI channels is cut into short overlapping pieces (patches).
A small Transformer reads the patches of one channel at a time, and a final linear layer combines
all 16 channels to label the window. Separate models are trained for detection and the ten
fault categories, with four settings tried and the best one repeated with two more seeds.

| Detection (dev, 6,590 windows) | Macro-F1 (3 seeds) | Accuracy | False alarms | Missed faults | Failed |
|---|---:|---:|---:|---:|---:|
| PatchTST | 0.807 ± 0.029 | 0.971–0.978 | 56 / 8 / 41 | 98 / 138 / 151 | 0 |
| XGBoost | 0.882 ± 0.000 | 0.983 | 26 | 83 | 0 |
| Random Forest | 0.661 ± 0.005 | 0.956–0.959 | 76–95 | 195–197 | 0 |
| TimesNet | 0.616 ± 0.032 | 0.822–0.916 | 407–1,094 | 65–146 | 0 |

PatchTST is second on detection: better than RF and TimesNet, below XGBoost. Its abnormal-class
F1 is 0.628 (XGBoost 0.773, RF 0.344, TimesNet 0.301). On categories
(177 dev windows) it reaches 0.159 ± 0.020 macro-F1, about the same as TimesNet (0.166) and far
below RF (0.569) and XGBoost (0.518). Every category trial had its best dev loss in the first or
second epoch, so the randomly initialized encoder overfits the 563 training windows, as the
issue's limitations section expected. Classes 3, 4, 5 and 7 are never predicted correctly.

### Technical implementation

- `kpi/patchtst.py` adapts the classification path of yuqinie98/PatchTST
  (`PatchTST_self_supervised`, commit `204c21ef…`, Apache 2.0; license in
  `licenses/PatchTST-LICENSE`, changes listed in the module docstring). Upstream defaults kept:
  residual attention, post-norm, learnable zero positional encoding, shared patch embedding, GELU.
- Architecture from the issue: 2 layers, width 32, 4 heads, feed-forward 64, LayerNorm,
  dropout 0.1, head dropout 0.1, attention dropout 0. Patch 8 / stride 4 gives 31 patches and
  patch 16 / stride 8 gives 15; `create_patches` refuses any setting that does not cover all
  128 steps. The head uses the last patch of each channel (16 × 32 = 512 features).
  Parameters: 19,140–19,396 (detection), 23,244–23,500 (category).
- Inputs reuse TimesNet's `WindowScaler` (train medians, means and population SDs, SD 1 for a
  constant channel, `NUMERIC_CHANNELS` order) with no per-window normalization.
- Training: AdamW (weight decay 1e-4), batch 32, gradient clip 1.0, balanced `N/(K·count)`
  weighted-mean cross-entropy; early stop on unweighted dev cross-entropy (patience 10, cap 100)
  and restore the best epoch. Grid patch {8, 16} × lr {1e-3, 3e-4} with seed 42, selected by
  full-dev macro-F1 (ties keep grid order), then seeds 43/44. Both tasks selected patch 16, lr 1e-3.
- `baselines/patchtst.py` checks release, training IDs, dev cohort and labels against the
  XGBoost seed-42 run (shared `_reference`), never loads test during training, and writes the
  same `selection.json` layout as TimesNet so `compare_experiments` works unchanged. Every
  saved model is reloaded before the run is written and must give the same argmax and logits
  within `rtol=1e-5, atol=1e-6`; the max difference is recorded (0.0 on all 12 runs).
- `cli/patchtst.py`: `train`, `predict` (replay a checkpoint without refitting) and `compare`
  (`--experiment NAME=DIR`). `notebooks/patchtst_baseline.ipynb` shows trials, training curves,
  the comparison, per-class F1 for both tasks, the reload check, and a CLI `predict` replay of
  the selected detection model on dev.
- Run on CPU (bit-reproducible) with Python 3.11.15 and torch 2.14.1 to match the earlier
  baselines: about 28 s per detection epoch at patch 16 (48 s at patch 8), under 1.5 s per
  category epoch; about 1 ms per window for prediction; 1.29 GB peak RSS.

### Verification

- 10 new tests: patch coverage and count, grid order, output shapes, no attention between
  channels, train-only scaling and channel order, weighted loss against a hand computation,
  deterministic CPU training, reload/checksum, runner (4 settings, 3 seeds, dev only, scaler fit
  on train), compare and CLI.
- Full suite after merging the latest main (all extras, Python 3.12): 538 passed. Ruff passes on
  all new files.
- Replaying `detection/config-2-seed-42` on dev with `predict` reproduced all 6,590 predictions
  and the same macro-F1 (0.8387).

### Out of scope and follow-ups

- Masked pretraining, external checkpoints, channel mixers, resampling, and wider searches (per the issue).
- Test scoring belongs to the frozen final evaluation (Task 3.8), not to this PR.


## Task 4.2 — Detector predictions for RCA and SLM data — 2026-11-09

**Focus:** [Issue #57](https://github.com/tanzim10/clauseguard-ran/issues/57) — select
a Task 3 detector, map its outputs into the Task 4.1 contract, and save reproducible predictions
for the RCA cases and for SLM training candidates.

**Status:** Complete; provisional because recording-family relationships are unresolved, so no
prediction is eligible for a scored release or the approved training release.

**PR:** [#70](https://github.com/tanzim10/clauseguard-ran/pull/70)

### Delivered

In simple words: the RCA pipeline now has a fixed detector. Detection runs on every KPI window;
when it says "abnormal", the category model names one of the ten synthetic faults. The choice
of models is sealed in a hashed manifest, the 12 golden RCA cases have saved predictions, and
every training window has a prediction from a model that never saw its recording.

| Task | Selected (dev mean macro-F1) | Other candidates (dev) |
|---|---|---|
| Detection | XGBoost 0.882 (depth 6, child weight 5, seed 42, 127 rounds) | PatchTST 0.807, MantisV2 0.766, RF 0.661, TimesNet 0.616 |
| Category | Random Forest 0.569 (depth 12, leaf 1, seed 42) | XGBoost 0.518, MantisV2 0.351, TimesNet 0.166, PatchTST 0.159 |

- **12 golden RCA cases:** detection flags 8 as abnormal, so category runs on those 8. Diagnosis
  predictions: Co-Channel Interference (Severe) 4, High Network Congestion (Gradual Buildup) 3,
  Antenna Failure 1, normal 4. All 12 detection and all 8 category predictions are identical to
  the saved Task 3 runs (5 dev cases against the selected dev runs, 7 test cases against the
  Task 3.8 test runs). Scored against the release labels, detection catches 8 of 12 faults and
  category agrees on 6 of 8.
- **19,089 SLM training candidates:** 5 family folds over the 22 training families. Out of fold,
  detection catches 354 of 563 synthetic faults and 0 of 94 jamming windows (the only jamming
  family is never in its own fold's training data), with 2,805 false alarms on 18,432 normal
  windows. These rates are far below the in-distribution dev scores, which is what an unseen
  recording looks like.
- **Text:** no case has text with RCA permission (`missing: no_approved_text` for all 19,101).

### Technical implementation

- `rca/detector.py`:
  - `freeze_selection` reads each Task 3 candidate's `selection.json`, takes the highest mean dev
    macro-F1 per task (ties keep candidate order), always uses the seed-42 run (the search seed,
    so no seed is chosen by score), and seals the manifest with SHA-256. It records model
    identity, checkpoint path and hash, preprocessing (channel and feature order, training
    medians, versions), data identity (source revision, release, cohorts, training-ID hash),
    training and development families, development use (XGBoost: early stopping and selection;
    RF: selection), label mapping, readiness, 23 earlier test-split runs found on disk, the
    five-dataset compatibility table and the text policy. A frozen selection is never overwritten.
  - `to_detector_prediction` maps `CasePrediction` into `DetectorPrediction` with the resolved
    label name, mapping version, run ID, KPI source modality and an uncalibrated confidence
    (`predict_proba` of the predicted class; source and calibration status are saved beside it).
    Failed calls stay `error`, unmappable outputs `invalid`, and absences keep a reason.
  - `run_pipeline`: detection for every window, category only when detection is abnormal,
    diagnosis `0` for normal and `category + 1` otherwise (the Task 3.8 mapping). Category
    predictions stay predictions; unsupported faults such as jamming are never mapped to a cause.
  - `load_head` rejects a different release, channel layout, label mapping or checkpoint hash.
    Prediction records (`CaseRecord`) hold no label, target or split field.
  - `rca_text_eligibility` requires explicit RCA permission; detection/category approval is not
    enough, and diagnosis-derived or label-revealing text is excluded.
- `rca/cross_fit.py`: families are ordered by origin (synthetic, real, normal) and family ID and
  dealt round-robin into folds. Dev and test families (and every RCA case family) are reserved
  and refused if they appear in training. Each fold refits the frozen configuration and its own
  channel medians on the other folds only; XGBoost uses `fit_fixed_rounds` with the recorded
  127 rounds and no validation data. A fold model that cannot be fitted is recorded as
  unavailable and its predictions are `absent` with the reason. `check_separation` verifies
  every prediction; `reproduce_cross_fit` reloads the saved fold checkpoints and compares.
- `kpi/xgboost.py`: the classifier settings moved into a shared `_classifier`; `fit_fixed_rounds`
  is new. A 127-round refit on the full training split reproduced the selected checkpoint's dev
  predictions and probabilities exactly (largest difference 0.0); the RF refit did the same.
- `rca/detector_report.py` and `cli/detector_predictions.py` (`freeze`, `predict`, `reproduce`)
  write `rca_cases.jsonl`, `cross_fit/` (fold checkpoints, `folds.json`, `predictions.jsonl`),
  `report.json` and `report.md`. Label agreement is computed only in the report's evaluation
  section and never written into prediction records.

Reproduction (Python 3.11 with scikit-learn 1.9.1 and XGBoost 3.2.0, the checkpoint versions):

```bash
uv run python -m clauseguard.cli.detector_predictions freeze --release-dir data/scenarios/release --allow-exploratory \
  --candidate random_forest=artifacts/experiments/rf-all-data-provisional-2026-10-19 \
  --candidate xgboost=artifacts/experiments/xgboost-full-data-2026-10-19 \
  --candidate timesnet=artifacts/experiments/timesnet-v3-full \
  --candidate patchtst=artifacts/experiments/patchtst-full-data-2026-11-05 \
  --candidate mantis=artifacts/experiments/mantis-full-data-provisional-2026-10-28 \
  --exposure-dir artifacts/experiments --frozen-on 2026-11-09 --output artifacts/rca/detector-selection-2026-11-09.json
uv run python -m clauseguard.cli.detector_predictions predict --release-dir data/scenarios/release --allow-exploratory \
  --selection artifacts/rca/detector-selection-2026-11-09.json --output-dir artifacts/rca/detector-predictions-2026-11-09
uv run python -m clauseguard.cli.detector_predictions reproduce --release-dir data/scenarios/release --allow-exploratory \
  --selection artifacts/rca/detector-selection-2026-11-09.json --output-dir artifacts/rca/detector-predictions-2026-11-09
```

### Dataset compatibility

| Dataset | Detector predictions |
|---|---|
| TelecomTS | Implemented against the checked release and its checkpoints |
| BubbleRAN | Absent: MAC KPIs at 1 Hz, cumulative counters and unverified units; no reviewed channel mapping |
| SpotLight | Absent: radio and platform streams with unverified alignment; no compatible window |
| TeleLogs | Absent: no compatible detector or KPI input; files not available locally |
| 5G core faults | Absent: text-only cases |

### Verification

- `uv run pytest tests/test_detector_predictions.py -q` (14 tests): label mappings for each task,
  adapter failures, category only after abnormal detection, error vs invalid vs absent,
  unavailable fold models, dataset table against the contract's five sources, no reference
  fields in records, RCA text permission, deterministic folds, dev-based selection and sealing,
  incompatible release/channels/mapping/checkpoint, saved-run agreement, family separation and
  fold medians, reproduction and tampering, and the CLI.
- Real data: selection `4f184db8…`; `reproduce` reloaded every checkpoint and matched all 12 RCA
  case records and all 19,089 out-of-fold records exactly; the separation check found no
  held-out or reserved family in any fold's training data.
- Artifacts (local, not committed): `artifacts/rca/detector-selection-2026-11-09.json` and
  `artifacts/rca/detector-predictions-2026-11-09/` (report, predictions, fold checkpoints).

### Out of scope and follow-ups

- The 5 dev golden cases come from the families XGBoost used for early stopping and both models
  used for selection; the report marks them. Earlier test exposure is listed in the manifest.
- New detector searches, evidence summaries, retrieval and SLM example approval stay in later tasks.

## Task 4.4 — Traceable evidence records — 2026-11-09

**Focus:** [Issue #59](https://github.com/tanzim10/clauseguard-ran/issues/59) — replace
the summarizer stub with deterministic, source-traceable evidence records for the RCA packet.


**PR:** https://github.com/tanzim10/clauseguard-ran/pull/71

**Status:** Complete for the available sources (TelecomTS, BubbleRAN, SpotLight); TeleLogs and
5G core faults are recorded as unavailable. Depends on Task 4.2 detector predictions.

### Delivered

In simple words: every number the RCA model will see is now computed the same way every time and
points back to its exact source (file, row, channel, revision and file hash). Anything that cannot
be measured safely is kept out with a written reason instead of being guessed.

- The 12 golden RCA windows produce 12 `EvidencePacket`s with 576 KPI records (16 channels ×
  mean, minimum, maximum) and the Task 4.2 diagnosis prediction. The 2 categorical channels per
  window are rejected with a reason (24 records). No unit, baseline or text is claimed: TelecomTS
  has no verified units, no baseline is verified, and no text has RCA permission.
- An independent recomputation from the raw `metrics.csv` files matched all 576 values (largest
  difference 2.7e-12; tolerance 1e-9, the release's raw-match bound). Rebuilding gave the same
  packet hash (`0840acc3…`).
- `data/golden/rca-evidence.example.json`: one real packet (rca_001), its audit records and
  rejections, plus BubbleRAN and SpotLight rejection examples (reasons only; their values are not
  published because reuse terms are unverified) and the reproduction commands.

### Technical implementation

- `rca/summarizer.py` (`kpi-stats-v1`): mean (`math.fsum`), minimum and maximum of observed values,
  each needing at least one finite value. Missing means absent (null or empty); zero is a value;
  non-finite numbers are excluded and counted. Extrema record the first position holding the value.
  Categorical channels, identifier fields, non-numeric values and empty intervals are rejected.
  Cumulative counters are rejected (`counter_semantics_unverified`) with their reset positions,
  because deltas or rates need verified semantics. `check_timestamps` rejects duplicate and
  out-of-order timestamps and reports gaps longer than 1.5 nominal periods.
- `rca/evidence.py`: `EvidenceRegistry` assigns `K`/`T`/`S` prompt IDs, maps each to one
  persistent ID, returns the same ID for a repeated fact and refuses duplicate IDs or a conflicting
  record for the same source. Each record has an `EvidenceAudit` (source revision, artifact hash,
  location, calculation settings, rules, extremum time, content hash); `check_audits` detects a
  changed record. `add_scalar` keeps scalar measurements without an invented sample range or time.
  `add_specification` stores the exact supplied text and, for a shortened excerpt, its span and
  the original chunk hash. `attach_baseline` links a baseline only when verified and compatible
  (same channel, statistic, unit and calculation; an earlier interval must use a fixed rule and end
  before the incident; verified normal needs a reviewer and evidence); `relative_change` is
  undefined for a zero baseline.
- `rca/sources.py`:
  - TelecomTS reads only `kpi_window`, provenance and the text review. Sample times follow
    `start_time + index / rate` (verified against raw files by the release), timezone unknown,
    source strings kept for the window bounds. Text enters only with RCA permission.
  - BubbleRAN: one UE on one gNB over consecutive 1 s rows, UTC Unix timestamps. `*_aggr_*`,
    HARQ and error counters are rejected as unverified counters; identifiers are excluded. The
    `persistent_anomaly` label column is excluded without reading its values.
  - SpotLight: one stream at a time (radio or platform; never combined without verified
    alignment). A repeated header row marks a capture boundary that a window may not cross. The
    `anomaly` label column is excluded. Timestamps keep their nanosecond source strings.
  - TeleLogs and 5G core faults return `unavailable` records.
- `cli/build_evidence.py`: `telecomts` (packets, audits, rejections and a hashed manifest, with
  optional Task 4.2 predictions), `verify-raw` (independent recomputation) and `table`
  (BubbleRAN or SpotLight window).

Reproduction:

```bash
uv run python -m clauseguard.cli.build_evidence telecomts --release-dir data/scenarios/release --allow-exploratory \
  --detector-predictions artifacts/rca/detector-predictions-2026-11-09/rca_cases.jsonl --output-dir artifacts/rca/evidence-2026-11-09
uv run python -m clauseguard.cli.build_evidence verify-raw --release-dir data/scenarios/release --allow-exploratory \
  --output-dir artifacts/rca/evidence-2026-11-09
uv run python -m clauseguard.cli.build_evidence table --dataset bubbleran --csv data/bubbleran/raw/anomalous_data.csv --start 100 --end 159
```

### Verification

- `uv run pytest tests/test_rca_evidence.py -q` (19 tests): statistics and extremum positions,
  zero vs missing vs non-finite, rejected channel kinds, counter resets, timestamp rules, ID
  assignment and conflicts, audit tampering, scalar records, specification spans, baseline rules
  and zero baselines, TelecomTS reference fields (labels, description, anomaly, QnA) absent from
  packets, eligible text, BubbleRAN and SpotLight label exclusion, capture boundaries and time
  gaps, the CLI with a raw-file check, and the published example against the contract.
- Real files: BubbleRAN windows from both CSVs gave 231 KPI records each, with 20 counters and the
  label column rejected; a SpotLight radio window gave 36 records and a window across the repeated
  header was refused.

### Out of scope and follow-ups

- Retrieval execution and specification excerpts per case belong to Task 4.5.
- Verified units, counter semantics, BubbleRAN clean-data baselines and SpotLight stream
  alignment need source review before they can be used.

## Task 4.5 — Frozen retrieval and specification snapshots — 2026-11-09

**Focus:** [Issue #60](https://github.com/tanzim10/clauseguard-ran/issues/60) — build queries from permitted inputs, freeze retrieval settings, save each case's exact passages, and replay them so both generators receive identical specification evidence.

**PR:** https://github.com/tanzim10/clauseguard-ran/pull/72

**Status:** Code, manifest and replay complete. The live snapshots for the 12 RCA cases still need
an NVIDIA embedding key and the Qdrant index: on this machine the key is not set, so the run saved
12 snapshots in the `failed` state at the `configuration` stage. Depends on Task 4.4 evidence packets.

### Delivered

In simple words: for each RCA case, one search query is built from the case's own evidence (the
question, the measured KPI statistics and the detector's prediction, never the answer). The search
runs once with sealed settings and the exact passages it returns are saved. Later runs reuse those
saved passages, so the base LLM and the trained SLM read the same specification text in the same
order, and a changed index cannot silently change a comparison.

- Frozen manifest `fe854ad2…` over 10,715 chunks (inventory `9f9eb03a…`) from the 12 corpus
  documents, with top-k 8, the NVIDIA embedding model and the Qdrant collection.
- 12 real snapshots, all `failed` at the `configuration` stage (`NVIDIA_API_KEY is required for
  embeddings`); replay attached them to the Task 4.4 packets with `retrieval_status: failed`.
- `data/golden/rca-retrieval.example.json`: the manifest summary, the real rca_001 snapshot, and a
  clearly marked FIXTURE snapshot (toy corpus, hand-made ranking) showing a completed run and its replay.

### Technical implementation

- `rca/retrieval.py`:
  - `build_query` (`rca-query-v1`) reads only the `EvidencePacket`, which has no field for
    reference labels, expected conclusions, golden citations or review decisions. It uses the
    question, one line per channel (mean, minimum, maximum at 4 significant digits), approved
    text, and the detector label only when its status is `ok`. It records the contributing
    evidence IDs, modality states and whether the prediction was used; a missing prediction is
    never replaced by anything.
  - `freeze_manifest` seals corpus identity (manifest hash, document versions and hashes), the
    chunk inventory rebuilt with the indexer's `chunk_document` (per-chunk hashes), index
    collection and point-ID rule, embedding model and dimensions (revision recorded as
    unavailable), query rules, search settings (top-k, no filter, no threshold, approximate
    HNSW), ordering, duplicate and excerpt rules, and known limits. It is never overwritten.
  - `snapshot_case` calls the shared `SearchService` once. Each hit must resolve to a manifest
    document, match its file name, have the point ID `uuid5(chunk_id)` and the frozen chunk text;
    otherwise the snapshot fails at `source_identity`. A repeated chunk keeps its first rank and is
    listed as filtered. States: `completed`, `empty` (search ran, nothing found), `not_run`
    (empty query, retriever not called), `failed` with stage `configuration`, `dependency` or
    `source_identity`. Scores are ranking values only.
  - Snapshots are content-addressed (`snapshot_id` is a hash) and never overwritten.
    `load_snapshot` checks the hash, the manifest and every passage hash; `replay` also checks the
    input packet hash, assigns the saved `S` IDs through the Task 4.4 registry and records replay
    time separately from the original retrieval time. `specification_for_configurations` gives
    all four configurations the same passages.
- `cli/retrieval_snapshots.py`: `freeze`, `run` (refuses settings that differ from the manifest
  and an existing run directory, so a change becomes a new snapshot set) and `replay` (no live
  call; checks that all configurations receive identical passages).
- Shared retrieval limit (written into the manifest): all four configurations reuse passages
  retrieved with the full packet, so they can carry information from an omitted modality;
  modality-specific queries are a separate experiment.

To produce the real snapshots where the key and index exist (Python 3.11 or 3.12):

```bash
uv run python -m clauseguard.cli.index_corpus            # if the collection is not built yet
uv run python -m clauseguard.cli.retrieval_snapshots freeze --frozen-on <date> --output artifacts/rca/retrieval-manifest-<date>.json
uv run python -m clauseguard.cli.retrieval_snapshots run --manifest artifacts/rca/retrieval-manifest-<date>.json \
  --packets artifacts/rca/evidence-2026-11-09/packets.jsonl --run-dir artifacts/rca/retrieval-<date>
uv run python -m clauseguard.cli.retrieval_snapshots replay --manifest artifacts/rca/retrieval-manifest-<date>.json \
  --packets artifacts/rca/evidence-2026-11-09/packets.jsonl --run-dir artifacts/rca/retrieval-<date> \
  --output artifacts/rca/retrieval-<date>/packets.replayed.jsonl
```

### Verification

- `uv run pytest tests/test_rca_retrieval.py -q` (7 tests): deterministic query and its
  contributions, absent prediction left out, reference fields rejected by the packet contract,
  sealed manifest and conflicting document versions, completed snapshot with ranks, versions and
  duplicate filtering, empty vs not run vs configuration vs dependency vs source-identity failures,
  idempotent and refused overwrites, altered snapshot, different manifest, changed packet, replay
  without a live call and identical passages for all configurations, CLI including a settings
  change, and the published example.
- Real data: manifest frozen; 12 snapshots saved as failed with their reason; replay verified all
  12 and attached them.
- Local note: one parsed corpus file in this checkout is named with `_` where the manifest has `;`
  (`…Principles; satisfies…`). The manifest was frozen through links with the expected names. The
  chunk-text check rejects any hit whose text differs from the frozen inventory, so a mismatch
  between this inventory and the live index cannot pass silently.

### Out of scope and follow-ups

- Running the live snapshots needs `NVIDIA_API_KEY` and the Qdrant index.
- Retrieval tuning, rerankers, prompt projections per configuration and generation stay in later tasks.

## Task 4.6 — SLM candidate pool, manifest and gap report — 2026-11-17

**Focus:** [Issue #61](https://github.com/tanzim10/clauseguard-ran/issues/61) — inventory the five
planned sources, reserve evaluation data, and build traceable SLM training candidates with separate
references, explicit eligibility, a versioned manifest, an initial batch and a gap report.

**Status:** Complete for the available sources. Every admissible candidate is `exploratory`, not
`eligible_pilot`, because all TelecomTS recording families are unresolved. BubbleRAN candidates are
built but `ineligible` until its CC BY-SA 4.0 terms are cleared for training, and SpotLight candidates
are built from the local sample but `ineligible` because it has no licence. TeleLogs and 5G core
faults are inventory entries with blockers (not local). No target answers are created (Task 4.9).

**PR:** [#77 — Build the SLM candidate pool with inventory and verified manifest](https://github.com/tanzim10/clauseguard-ran/pull/77)

### Delivered

In simple words: every training window now has a record that says where it came from, what the
model would see, which detector output goes with it, which recording and fault event it belongs to,
and whether it may be used, with the reasons. The labels and existing answers sit in a separate file
that never reaches a prompt. Golden cases, their recordings and every golden question, paraphrase and
answer are reserved before any candidate is admitted.

Real run (`artifacts/rca/candidates-2026-11-17-r3/`, release `aa2e758a…`, local and gitignored):

| Source | Rows | Families | Events | Status |
|---|---|---|---|---|
| TelecomTS (train split) | 19,089 | 22 | 78 (60 synthetic, 2 jamming, 16 normal) | `exploratory` (families unresolved) |
| BubbleRAN | 86 grid windows | 1 | 76 (5 interference episodes) | `ineligible` (terms pending) |
| SpotLight (sample) | 109 grid windows (77 radio, 32 platform) | 1 | 109 | `ineligible` (no licence) |
| TeleLogs, 5G core faults | 0 | — | — | blocked in the inventory (not local) |

- Labels: 672 `labelled_fault` rows (67 events), 18,503 `labelled_normal`, 109 `ambiguous`
  (SpotLight: anomaly type not stated, label-time alignment unverified); every cause list has at
  most one cause. All 19,284 candidates are KPI-only; there is no text-only, combined or settings
  candidate. Detector predictions: 19,089 available (diagnosis head), 195 recorded absences.
- Grouped split over admissible TelecomTS families: 1 of 5 synthetic, 0 of 1 jamming and 3 of 16
  normal families go to validation (12,667 train and 6,422 validation rows).
- Reservations: 11 development and test families, 12 golden scenarios, 120 golden texts (questions,
  answers, conclusions and the 24 paraphrases) and two declared TeleLogs test reservations. No
  candidate is in a reserved family or scenario. No candidate carries text, so the text check ran on
  the question templates only; they match no golden text.
- Duplicates: no exact observation duplicates and no quarantined candidates. 13 groups of events
  share raw rows across different labels; they stay separate events inside one family.
- Batch: 25 of at most 30 TelecomTS candidates from 9 families, one per event: 2 per synthetic
  cause (20), 2 jamming and 3 normal windows (21 train, 4 validation). The default share asks for
  half detector corrections per stratum; 17 of 25 are corrections because several strata have no
  agreeing event (the out-of-fold detector catches no jamming window). BubbleRAN's quota stays empty
  while its terms are pending; SpotLight has no quota because it is ineligible.
- Source fields, units, observation/reference split, taxonomy and grouping per source:
  `data/candidates/source-fields.md` (TeleLogs and 5G core faults marked card-based and unverified).

### Technical implementation

- `rca/candidates.py` (`rca-candidate-v1`, Pydantic, `extra="forbid"`): `SourceInventory`,
  `CandidateRecord` (identity, modality states including a candidate-only `settings` state,
  observation link with packet and observation hashes, prediction reference with all three detector
  heads, grouping with `event_id`, taxonomy, eligibility, versions), `CandidateReference` (source
  labels as lists, a cause list, `label_status` ∈ labelled_fault / labelled_normal / unlabelled /
  ambiguous so an empty cause list is never ambiguous, annotation, source response and its hash),
  `Eligibility` (`eligible_pilot`, `exploratory`, `ineligible`, `quarantined`, with
  `{check_id, version, status, reasons}` checks) and `CandidateManifest`. A candidate starts
  `ineligible` until its checks run.
- `rca/candidate_pool.py`:
  - `build_reservations` checks the golden files against `data/golden/manifest.json`, reserves the
    golden scenarios, `reserved_families()` (dev and test families plus golden-case families) and
    normalized golden texts, matched exactly or by character 5-gram Jaccard ≥ 0.8 (not tuned).
  - `build_inventory` records identifier, revision, licence, access, redistribution, training use,
    local file hashes (TelecomTS from its acquisition record) and blockers for all five sources.
  - `CandidateConfig` holds every open decision with a documented default: packet head `diagnosis`
    (as in the Task 4.4 packets), one fixed question per modality (KPI uses the `rca_policy.json`
    minimal question), 60-row BubbleRAN grid at stride 60, pending terms → `ineligible`, 20%
    validation families, batch quotas and seeds.
  - `telecomts_candidates` builds train-split windows only, through `telecomts_evidence` and
    `EvidenceRegistry.packet` with `retrieval_status: not_run`, the out-of-fold Task 4.2 prediction
    (file hash checked against `folds.json`) and no baseline. Labels, anomaly bounds and the names of
    reference text fields go into `CandidateReference`; the reference text itself is not copied.
  - `bubbleran_candidates` cuts label-blind grid windows inside one capture segment (a segment ends
    at a time gap, timestamp reset or UE/gNB change) and builds them through `bubbleran_evidence`.
    The detector prediction is the documented absence. `persistent_anomaly` is read only into the
    reference; a window with any anomalous row is `labelled_fault` with its row count.
  - `spotlight_candidates` cuts label-blind 100-row windows inside one capture of one stream (radio
    or platform, never joined; a repeated header starts a capture) through `spotlight_evidence`.
    The in-file `anomaly` column and the overlap with `label` intervals (computed assuming the naive
    timestamps are UTC) go to references; every window is `ambiguous`. One family covers the sample
    experiment. Both table sources share `_table_packet` and `_table_candidate`.
- `rca/candidate_release.py`: events (overlapping windows of one source file with the same labels),
  exact and near-duplicate checks with quarantine for unresolved cross-group matches, eligibility
  checks (`not_reserved`, `terms_cleared`, `observations_built`, `family_resolved`, `event_grouped`,
  `duplicates_resolved`, `prediction_reference`; the most restrictive outcome wins), the grouped
  family split per (source, origin) stratum, the seeded batch (`batch_correction_share`, default 0.5,
  mixes detector corrections with agreements in each stratum, filling a short side from the other),
  counts (rows, families and events
  reported separately by source, family, modality, packet prediction, cause count, label status,
  eligibility and split),
  the gap report, export and `verify_release`. Export rebuilds every batch packet and refuses one
  whose hash differs from its candidate; packets are written only for the batch.
- `cli/build_candidates.py`: `inventory`, `reserve`, `build` (writes the whole release) and `verify`
  (file hashes, release ID, reservation hash, one-to-one references, taxonomy agreement, no reserved
  data admitted, no family in both splits, one window per event in the batch, batch packets valid as
  `EvidencePacket` and matching their candidates).

Reproduction:

```bash
uv run python -m clauseguard.cli.build_candidates inventory
uv run python -m clauseguard.cli.build_candidates build --release-dir data/scenarios/release --allow-exploratory \
  --predictions artifacts/rca/detector-predictions-2026-11-09/cross_fit/predictions.jsonl \
  --created-on <date> --output-dir artifacts/rca/candidates-<date>
uv run python -m clauseguard.cli.build_candidates verify --output-dir artifacts/rca/candidates-<date>
```

`--acquisition`, `--bubbleran-dir` and `--spotlight-dir` default to `data/telecomts/acquisition.json`,
`data/bubbleran/raw` and `data/spotlight/raw`. Pass `--config <json>` to change any `CandidateConfig`
default; the resolved configuration is saved in `config.json` and the manifest. Release folders
`artifacts/rca/candidates-*/` are gitignored.

### Contract updates needed in #56

`data/candidates/contract-updates.md` lists eight changes with reasons and code evidence: cause lists
in golden references, the output for normal windows, `label_status` kept as reference metadata,
a settings evidence type, the packet prediction head, a taxonomy for `final_fault_categories`,
filling `reuse_terms`, and a baseline rule for training packets.

### Verification

- `uv run pytest tests/test_rca_candidates.py -q` (34 tests): schema rules (unknown fields, hashes,
  label status vs causes, eligibility states, prediction absence), references refused by
  `EvidencePacket`, reservations and golden-file tampering, the inventory, TelecomTS candidates
  (train only, rebuilt packet hash, no reference fields in packets, configurable head,
  prediction-file hash), events and duplicate quarantine, eligibility under family, terms, reserved
  and quarantined conditions, deterministic grouped splits, the BubbleRAN and SpotLight grids
  (label-blind, never across a gap or a repeated header), multiple-cause lists, source taxonomies
  and the four label states surviving export (synthetic fixtures), the correction/agreement batch
  mix, export and tampering, and the CLI.
- Real data: every TelecomTS and BubbleRAN candidate and reference line of the earlier build
  reappears byte-identical in the rebuild, as do the 17 batch packets both batches share; `verify` passed on `candidates-2026-11-17-r3` (19,284 candidates, 25 batch
  packets, 10 files).

### Out of scope and follow-ups

- Target answers, teacher and judge runs and the training export stay in Task 4.9.
- Retrieval snapshots for candidates need `NVIDIA_API_KEY` and the Qdrant index (Task 4.5).
- TeleLogs and 5G core faults adapters wait for access and identity decisions; SpotLight candidates
  wait for a licence.
- Resolving or waiving TelecomTS recording families for training is the lead's decision; until then
  no candidate can be `eligible_pilot`.


## Task 4.7 — Base LLM pipeline for cited RCA — 2026-11-17

**Focus:** [Issue #62](https://github.com/tanzim10/clauseguard-ran/issues/62) — the shared RCA core: from a saved case to cited candidate causes or an explicit abstention, with runtime checks, pending human review and a full run record.

**Status:** Core, prompt, checks, run records and CLI complete and tested with fakes and FIXTURE
recordings, and run live (exploratory) on the 12 golden cases; see "Live results" below. The 12 Task
4.5 snapshots saved on 2026-11-09 are still `failed` at `configuration`; the live run used new snapshots
from a re-frozen manifest. The multiple-cause contract amendment is
drafted for the lead and #32's owner; until it is approved, the extra #63 rules run as pipeline
checks and the approved `rca-v1` schema is unchanged.

**PR:** [#78 — Add the cited base LLM pipeline for RCA](https://github.com/tanzim10/clauseguard-ran/pull/78)

### Delivered

In simple words: the pipeline takes one saved case (measured KPI statistics, approved text, the
detector's guess and the saved specification passages), checks that nothing was altered, shows the
model only the inputs the experiment allows, and asks it for causes where every claim cites
evidence IDs. The answer is then checked strictly. Only claims that pass are turned into the
explanation. Empty retrieval abstains without calling the model; a broken dependency or a malformed
answer is reported as a failure at its stage, never as an abstention. Every result says that human
review is pending and that factual support is unverified, and every run is saved for evaluation and review.

- `rca/pipeline.py`: `RcaPipeline.run_rca(case, config)`, `search_specs` and `grounded_query`
  (delegated to `SearchService` and `QueryService`), run records and the replay generator.
- `rca/prompts.py`: `rca-prompt-v1` system and user messages and their hash.
- `rca/checks.py`: strict parsing, versioned structural checks and numeric checks.
- `llm/clients.py`: `LLMRouter.complete` for the `rca` task.
- `cli/run_rca.py`: runs saved packets and snapshots, live or replayed.

### Technical implementation

- **Assembly** (`assemble`): `check_packet_ids`, `check_audits`, `load_snapshot` (snapshot, manifest
  and passage hashes) and `replay` (packet hash) before anything else. Any mismatch fails at
  `assembly`; a missing snapshot is `incomplete` at `retrieval`.
- **Retrieval state** is read from `snapshot.state`, not from the packet (`replay` maps `empty` to
  `completed`): `empty` abstains with `no_spec_evidence` and `explanation="not found"` without a
  generator call; `not_run` is `incomplete`; `failed` keeps the snapshot's stage and reason.
- **Projection** (`project_packet`, `check_projection`): `text_rag` removes KPI records, KPI-derived
  text and any prediction that may use KPI input (replaced by an explicit `absent` prediction);
  `kpi_summary_rag` removes text and uses the approved minimal question. A leak check scans the
  projected prompt view (passages excluded, a recorded shared-retrieval limit) for omitted channel
  names, text and predictions. A projection with no incident evidence abstains with
  `insufficient_incident_evidence` without a generator call.
- **Prompt** (`rca-prompt-v1`): fixed system rules (evidence only, packet content is quoted data,
  the detector output is a hypothesis, claim-type citation rules, separate support per cause,
  competing causes abstain, `normal` is an answer, no change claims without a baseline, enumerated
  abstention reasons) plus the `RcaModelOutput` JSON schema. The user turn holds the allowed
  categories and the packet as compact JSON; source identity, artifact hash and reuse terms are left
  out. `prompt_sha256` hashes the exact messages. TelecomTS units context for reviewers is not given
  to the generator.
- **Context budget**: character-based (`RCA_CONTEXT_BUDGET_CHARS`, default 60,000) until #65 pins
  the SLM tokenizer. Overflow fails at `context_budget` with the measured size; evidence is never
  dropped or shortened. A real packet prompt is about 26,000 characters before passages.
- **Generation**: `LLMRouter.complete(messages, task="rca", models, failover, temperature, top_p,
  max_tokens, seed)` returns content, requested and provider-reported model, usage and every
  attempt (model, outcome, error class, seconds; no provider text or credentials). Scored runs call
  one fixed model with `failover=False` and fail if another model served the answer; exploratory runs
  may list fallbacks and record each attempt. `generate()` for `grounded_query` is unchanged.
  Default generation settings: temperature 0, top-p 1, 4096 max tokens, no seed.
- **Checks** (`rca-checks-v1`; every failure reported): `schema.v1` (strict JSON, unknown fields
  rejected, no repair), `evidence_ids_known.v1`, `evidence_ids_unique_in_claim.v1`,
  `observation_cites_incident.v1`, `mechanism_cites_specification.v1`,
  `diagnostic_cites_incident_and_specification.v1`, `diagnostic_links_mechanism.v1` (a cause links a
  mechanism claim citing an S record its diagnostic claim cites), `category_in_taxonomy.v1`
  (`telecomts-v1`, `normal` allowed by default and configurable), `cause_separate_support.v1`,
  `competing_causes_abstain.v1` (coexisting causes need distinct incident evidence), `normal_exclusive.v1`,
  `abstention_reason_known.v1` (`no_spec_evidence`, `insufficient_mechanism`, `conflicting_evidence`,
  `insufficient_incident_evidence`) and `packet_contract.v1` (the existing
  `validate_output_for_packet` as a drift guard). `numeric.v1` checks each number in observation and
  diagnostic claims against the cited KPI fields or text at the precision shown; specification
  numbers are ignored and percentages match only a missing fraction or a verified baseline change.
  Scored runs fail on a numeric mismatch; exploratory runs may record it instead.
- **Result**: any failed check fails at `validation` with `result=None`; the parsed output and the
  check results stay in the run record. A passing output is rendered deterministically: per cause,
  observation, mechanism and diagnostic claims with their evidence IDs. Uncertainty reasons stay in
  the result and are not rendered. `RcaPipelineResponse` keeps the original detector prediction and
  the measured evidence apart from the result, with `human_review_status`, `semantic_review_status`
  `pending`, `factual_support="unverified"`, `golden_eligible` and `training_target_eligible` false.
  `to_server_response()` gives the approved `RcaServerResponse` for completed runs.
- **Run record** (`rca-run-v1`): configuration and its hash, packet contract, checks and prompt
  versions, taxonomy, source packet, audits, manifest, snapshot and projected packet hashes, the exact
  messages and projected packet, budget, generator kind, models, attempts, usage, raw and parsed
  output, response, stage timings, total latency and peak process RSS. `result_sha256` covers what
  must reproduce; `run_id` also covers time and timings. Records are written once
  (`save_run_record`) and hold no credentials.
- **Replay generator**: returns a recorded raw output only for the exact `prompt_sha256` and model,
  so a changed prompt cannot be answered by an old output. Replayed runs are labelled
  `generator_kind: "replay"`.

Reproduce (Python 3.11):

```bash
# FIXTURE demonstration: toy corpus and hand-written recorded outputs, not a model response
uv run python -m clauseguard.cli.run_rca --manifest tests/fixtures/rca_pipeline/retrieval-manifest.json \
  --packets tests/fixtures/rca_pipeline/packets.jsonl --audits tests/fixtures/rca_pipeline/audits.jsonl \
  --run-dir tests/fixtures/rca_pipeline/snapshots --replay tests/fixtures/rca_pipeline/recordings \
  --output <new-dir>

# Real cases, live base LLM (needs NVIDIA_API_KEY and completed Task 4.5 snapshots)
uv run python -m clauseguard.cli.run_rca --manifest artifacts/rca/retrieval-manifest-<date>.json \
  --packets artifacts/rca/evidence-2026-11-09/packets.jsonl --audits artifacts/rca/evidence-2026-11-09/audits.jsonl \
  --run-dir artifacts/rca/retrieval-<date> --model <fixed-model> [--run-kind scored] \
  --output artifacts/rca/rca-runs-<date>
```

### Verification

- `uv run pytest tests/test_llm_clients.py tests/test_rca_prompts.py tests/test_rca_checks.py
  tests/test_rca_pipeline.py tests/test_rca_cli.py -q`: fixed model without substitution, attempts
  and served model, settings passed through, `grounded_query` unchanged; deterministic prompt hash,
  no identity or reference fields, injected text kept inside JSON; every check failure reported,
  taxonomy and `normal`, shared and competing causes, enumerated reasons, numeric matches and
  mismatches; altered audits, snapshot or packet, projections and leak checks, empty retrieval with
  zero generator calls, failed and not-run retrieval, malformed output, unknown fields, unknown IDs,
  model abstention, a wrong detector prediction kept apart, missing modalities and absent
  predictions, context overflow, generation failures, run records without credentials,
  reproducible `result_sha256`, delegation, no FastAPI or MCP import, and the CLI.
- Full suite: 592 passed, 12 skipped.
- FIXTURE replay over 4 cases: one answered incident (cited cause, pending review, the detector's
  `normal` kept apart), one abstention after empty retrieval (no generator call), one dependency
  failure at `retrieval` (`configuration`), and one recorded model abstention (`conflicting_evidence`).
  Live mode without a key fails at `generation`, not as an abstention.
- Real data: the 12 Task 4.4 packets with the 12 Task 4.5 snapshots all fail at `retrieval`
  (`configuration`), with no generator call and the detector prediction preserved.

### Update 2026-11-18: lead decisions and live results

- **numeric.v2.** A number written in the packet question is an accepted source (recorded as field
  `question`). Invented, truncated and mis-rounded numbers still fail. Scored runs still fail on a mismatch.
- **`rca-prompt-v2`.** The raw JSON schema is replaced by a plain field list and one format-only example
  with placeholder values and an instruction not to copy it or output `$defs`/`properties`. In the first
  live run 5 of 12 cases echoed the schema; in the v2 run none did. v1 stays selectable
  (`--prompt-version rca-prompt-v1`); each version's system prompt hash is pinned in a test, and the
  prompt version is recorded in every run record. The FIXTURE recordings were re-keyed to the v2 hash.
- **RCA timeout and tokens.** `RCA_TIMEOUT_SECONDS` (default 240, sent per request on the rca path only;
  `grounded_query` keeps `NVIDIA_TIMEOUT_SECONDS`=30) and `RCA_MAX_TOKENS` (default 16384, `--max-tokens`).
  The selected model is a reasoning model: with 4096 tokens 9 of 12 cases returned empty content.
- **Embedding default.** `nvidia_embedding_model` is now `nvidia/nemotron-3-embed-1b` (2048 dimensions),
  because `llama-nemotron-embed-vl-1b-v2` was retired on 2026-11-16 (HTTP 410). The retrieval manifest has
  to be re-frozen; the 2026-11-09 manifest records the retired model. Historical records still name it.
- **Live results** (exploratory, `nvidia/nemotron-3.5-lightning-30b-a3b`, no fallbacks, re-frozen manifest
  `e90e3985…`, 12 completed snapshots, `rca-prompt-v2`): 7 of 12 answered with every check passing
  (rca_002, 004, 005, 007, 009, 010, 011); 5 failed at `validation` (rca_001, 006, 012:
  `diagnostic_links_mechanism.v1`; rca_003: numeric `8.02`; rca_008: mechanism link and numbers invented
  at full precision). None failed at generation. Generation took 11-47 s and 2.8k-11.7k completion tokens
  per case; prompts were 42-55k characters. The checks pass structure only: for example rca_004, 005 and
  007 answered a fault although the detector said `normal`, which human review has to judge.
- **Live abstention.** On snapshots built from the real index with a deliberately off-topic query (E2AP
  clauses, a constructed input) the model itself abstained with `insufficient_mechanism` in 2 of 3 cases
  (rca_001, rca_009) and answered in the third (rca_003, which then failed a check).

### Limits and follow-ups

- Runtime checks show structure, citation types and copied numbers. They cannot show that a
  diagnosis is correct, that a citation supports its claim, or physical causation. Human review stays pending.
- The extra #63 rules are pipeline checks, not schema rules; the contract amendment (cause-claim
  links, separate support, competing causes, enumerated reasons, declared taxonomy, `normal`) awaits
  approval. `competing_causes_abstain.v1` is a proxy: causes over the same incident evidence compete.
- The numeric check reads numbers from claim text; numbers from the question are accepted since `numeric.v2`, and rounding to the shown precision accepts coarse values such as "about 21".
- The context budget counts characters, not tokens; the limit is a placeholder until #65 (live prompts were 42-55k of 60k).
- The scored base model, whether `normal` is allowed, and the numeric failure policy are lead decisions.
- Live generation, live snapshots and scored runs need the key, the index, approved evaluation cases
  and the evaluation policy. FastAPI, MCP, review tooling and evaluation metrics stay out of scope.

## Task 4.8 — Evaluation of base RCA results — 2026-11-18

**Focus:** [Issue #63](https://github.com/tanzim10/clauseguard-ran/issues/63) — score saved base LLM runs with automatic metrics and a separate judge for provisional citation faithfulness, without waiting for human review.

**Status:** Evaluator, judge, CLI and tests complete. Real saved runs were scored (exploratory, provisional). Scored identification is blocked: no golden RCA case has an approved reference.

**PR:** TBD

### Delivered

In simple words: the evaluator reads the run records saved by Task 4.7 and reports four things separately: whether the returned causes match an approved answer, whether every claim and cause links to real evidence, whether a second, separate model finds each claim supported by the evidence it cites, and how many runs answered, abstained or failed. Every ratio shows its counts; a ratio with nothing to divide by is "unavailable", never 0 or 1. Judge results are provisional and human review stays pending.

- `eval/rca_runs.py`: reads `rca-run-v1` records into a small view; merges reruns.
- `eval/rca_eval.py`: the versioned policy (`rca-eval-v1`), references, identification, link coverage, support metrics, status, resources, error tags and the report.
- `eval/rca_judge.py`: the separate judge, structured judgments, resume support.
- `cli/eval_rca.py`: `python -m clauseguard.cli.eval_rca`.

### Technical implementation

- **References stay outside generator inputs.** They are loaded only by the evaluator. A case is approved only when `expected_causes` is non-empty. An optional overlay adds alternative acceptable sets; an optional abstention file holds unanswerable fixtures apart from the golden cases.
- **Identification** compares category sets after declared mappings (no text similarity). A prediction is exact when it equals any one acceptable set; sets are never merged. Precision, recall and F1 pool counts over eligible cases, using the best-overlap set. Failed runs and false abstentions predict the empty set. `normal` is an ordinary category; an empty set is never a valid reference. Scored and exploratory cases are separate sections; each reports `with_failures` and `successful_only`. Missing approved references return `blocked` with the reason and do not block the rest.
- **Structure**: citation coverage (every claim cites at least one ID, no repeats, all IDs in the packet the generator saw) and cause-link coverage (a returned cause links to a diagnostic claim with valid links). Abstentions add nothing.
- **Judge** (`rca-judge-v1`, default `nvidia/nemotron-3.5-super-vl-120b-a12b`, refused when it equals a generator model): one request per complete run, each claim next to only its own cited records, evidence inside delimited blocks the judge is told to treat as data. The prompt holds no generator name, detector prediction or expected diagnosis. Five decisions; only `fully_supported` counts as faithful. Malformed replies and judge failures are separate visible statuses. Judgments are reused on rerun only when `judged`, so failures are retried.
- **Metrics**: provisional faithfulness (fully supported over cited claims with a completed decision), provisional supported coverage (only when judging is complete), judgment coverage.
- **Resources**: pipeline timings by stage and generator kind (replay loading reported apart), client process memory with its scope, judge latency and tokens separate; hosted model memory and GPU memory are marked unavailable.
- **Error tags** with the evidence behind each: retrieval miss (an annotated expected clause is absent from the packet), corpus gap (needs a declared gap), generation miss (approved reference, expected evidence supplied, no acceptable answer), citation mismatch (invalid links; judge-based ones are provisional), execution failure, undetermined.
- **Spec-QA retrieval** is read from an existing `run_eval` directory (`--retrieval-eval`); Recall@k and MRR are not recomputed.
- **Outputs** go to a new directory under `artifacts/rca_eval/` (ignored): `policy.json` (policy hashes), `judgments.jsonl`, `metrics.json` (deterministic), `summary.md`, `eval.json` (hardware, run IDs, command, time).

### Reproduce

```bash
export DYLD_LIBRARY_PATH=$PWD/.venv/lib/python3.11/site-packages/sklearn/.dylibs
.venv/bin/python -m pytest -q tests/test_rca_runs.py tests/test_rca_judge.py tests/test_rca_eval.py tests/test_rca_eval_cli.py
# structure and identification only
uv run python -m clauseguard.cli.eval_rca --runs <run_rca output dir> --output artifacts/rca_eval/<name> --no-judge
# with the judge (needs NVIDIA_API_KEY), spec_qa retrieval and a rerun directory merged in
uv run python -m clauseguard.cli.eval_rca --runs <dir> --runs <rerun dir> --output artifacts/rca_eval/<name> \
  --retrieval-eval artifacts/eval/<retrieval run> [--reuse-judgments artifacts/rca_eval/<earlier>/judgments.jsonl]
```

### Results on saved development runs (exploratory, provisional)

All 12 golden RCA cases, `rca-prompt-v2`, judge `nemotron-3.5-super-120b`. The 12 cases come from 4 recording families, so they are not independent.

| Generator | Answered / abstained / failed | Judged claims | Fully supported | Provisional faithfulness |
|---|---|---|---|---|
| Kimi K3.1 Pro | 3 / 7 / 2 (timeouts) | 12 | 10 | 0.833 |
| Nemotron 3 Ultra | 6 / 6 / 0 | 33 | 19 | 0.576 |
| Gemma 4 31B | 12 / 0 / 0 | 41 | 21 | 0.512 |
| Nemotron 3.5 Lightning | 7 / 0 / 5 (validation) | 27 | 16 | 0.593 |

Citation and cause-link coverage are 1.000 for every generator (the pipeline checks already enforce valid links). Spec-QA retrieval (dev split, 22 answerable rows): MRR 0.523, clause recall@5 0.504, @10 0.606.

### Limits and follow-ups

- **Scored identification is blocked.** Every golden RCA case has an empty `expected_causes`, and all are exploratory. Until references (and alternative sets) are approved, cause-set accuracy, precision, recall and F1 are unavailable. A source-label agreement block is reported, labelled as a dataset claim and not an accuracy.
- **Prior test exposure.** 7 of the 12 golden cases are in the test split and were run and read while Task 4.7's prompt and checks were developed. These results are development evidence, not a frozen test comparison.
- **Retrieval miss on every case.** Expected annotated clauses are absent from the live packets for all 12 cases, which caps citation-based and generation-miss results.
- **Judge limits.** One judge model, one pass; it shares biases with the Nemotron generators and scores by reading single-window aggregates strictly. It is a screen, not ground truth.
- **Correct abstention is unavailable** until an abstention fixture file is approved.
- Final four-configuration comparison, human review and SLM targets remain out of scope.


## Task 4.9 — Cited SLM training examples — 2026-11-25

**Focus:** [Issue #75](https://github.com/tanzim10/clauseguard-ran/issues/75) — build checked, cited RCA training examples from the exact evidence the SLM will receive, and export versioned train and validation JSONL for Task 4.10.

**Status:** Release `release-final` built: 50 rows (40 train, 10 validation), TelecomTS only, status `sample_approved`. Ten sampled rows were reviewed and approved on 2026-11-25. BubbleRAN, SpotLight, TeleLogs and 5G core faults are not in this version.

**PR:** TBD

### Delivered

In simple words: a large teacher model reads the same prompt the small model will get and writes a cited answer. The teacher also sees the true cause as a hint that never enters the training messages. Code checks then confirm the cause, the citations and the numbers; a judge from another model family reads every claim next to the evidence it cites. Only answers that pass everything are exported. The training files hold only `system`, `user` and `assistant` messages; labels, reviews and provenance stay in separate files.

The release directory stays outside the repository.

| File | Content |
|---|---|
| `train.jsonl` / `validation.jsonl` | 40 / 10 rows, `messages` only (sha256 `8c4d19e0…` / `d18f6a36…`) |
| `index.jsonl`, `canonical.jsonl` | row links; 65 full records with input, target, provenance, review, exclusion reasons |
| `manifest.json` | versions, hashes, judge, acceptance rule, counts, 26 exclusions with reasons, `release_status` |
| `qwen_check.json`, `human_review.json`, `sources.json` | template/length/mask check per row; the 10 reviewed rows; source coverage |

New code: `rca/training_inputs.py`, `rca/training_baselines.py`, `rca/training_checks.py`, `rca/training_release.py`, `cli/build_training_inputs.py`, `cli/judge_training.py`, `cli/build_training_release.py`; extensions to `run_rca`, `eval/rca_judge.py`, the indexer and the NVIDIA client.

### Technical implementation

- **Decisions as data** (`data/candidates/decisions.json`): the TelecomTS and BubbleRAN family waivers and the academic-terms decision of 2026-11-24 are applied by `build_candidates --decisions`.
- **Selection:** one window per eligible event; every fault event once, normal events at a 0.12 share, seed 42. Result: 76 events (TelecomTS 62 fault + 4 normal, BubbleRAN 5 + 5).
- **Label-blind baselines:** TelecomTS uses a train-split normal window from the same zone and application; BubbleRAN uses the equal-length window just before. 70 of 76 windows have one. The rules read only window location, never labels.
- **Retrieval:** re-indexed with `nvidia/nemotron-3.5-embed-1b` (the previous embedder returned HTTP 410) and chunker v2 (4 of 20 expected dev passages found, against 2 of 20 for v1); collection `oran_specs_nemotron35_2048_chunker_v2`, 6,892 points. `rca-query-v3` adds the six largest KPI changes to the query; snapshots frozen at top_k 16.
- **Teacher:** `nvidia/nemotron-3-ultra-550b-a55b`, `rca-prompt-v2`, temperature 0.0, 90,000-character budget (largest measured prompt 84,819). A teacher-only hint (`rca-teacher-hint-v2`) names the source label and asks for an evidence-only cited explanation, abstaining if the KPIs contradict it. One label-free revision is allowed after a failed check (used by 26 of 66 runs).
- **NVIDIA key fallback:** after a 429, 502–504, timeout or connection error the same model is retried on the next key; the model is never swapped.
- **Checks before the judge:** the existing structural and numeric checks plus `label.v1` (category equals the source label) and `kpi_support.v1` (each diagnostic claim cites KPI evidence).
- **Judge and acceptance:** `minimaxai/minimax-m3.1-flash` with `rca-judge-v2` (v1 unchanged). Rule `rca-acceptance-v2`: complete run, both checks pass, observation and mechanism claims fully supported, diagnostic claims at least partially supported, nothing unsupported or contradicted.
- **Release builder:** chat-only files, no group in both splits, duplicate prompts excluded, a reason for every exclusion, deterministic bytes. Each row is rendered with the `Qwen/Qwen3.5-4B` template; over-length rows are rejected and the assistant mask must end at `<|im_end|>`.

### Results

| Stage | Count |
|---|---|
| Selected events | 76 (66 TelecomTS, 10 BubbleRAN) |
| Complete TelecomTS runs | 65 |
| Machine accepted / rejected | 50 / 15 |
| Exported | 50 (40 train, 10 validation) |
| Excluded | 26 = 15 rejected + 1 incomplete TelecomTS run + 10 BubbleRAN without a run |

- 28 judge calls first failed with provider errors; a second pass with the same judge and rule accepted 23 and rejected 5.
- Exported rows: 47 fault, 3 normal, all `answered`, covering all 11 TelecomTS causes. 9 groups (7 train, 2 validation), none in both.
- Qwen check: 50 of 50 rows pass at 40,960 tokens; rows are 16,766–34,948 tokens.
- Sources: TelecomTS included; BubbleRAN excluded (prompts about 202k characters against the 90k budget); SpotLight excluded (no licence); TeleLogs and 5G core faults deferred.

### Limits and follow-ups

- **One source, KPI-only, single cause, no abstentions.** All 50 targets are TelecomTS `answered` rows.
- **Label-guided targets.** The teacher knew the label, so targets explain a known cause rather than show a diagnosis.
- **The mechanism link is only partly proven.** The frozen corpus has no passage naming the mechanism of most TelecomTS categories, so rule v2 accepts partially supported diagnostic claims.
- **Sampled review** covers 10 of 50 rows.
- **Model.** Issue #75 names Qwen3-8B. This release is checked with the `Qwen/Qwen3.5-4B` tokenizer.
- BubbleRAN needs smaller evidence and a new review; TeleLogs and 5G core faults need adapters. Training is Task 4.10.


## Adding a New Phase

Append a focused phase using this structure:

```markdown
## Phase N — Feature Name

**Focus:** Issue or PR reference — one-sentence purpose.

**Status:** Complete — YYYY-MM-DD

### Delivered
What users or downstream features can do now.

### Technical implementation
- Files/modules and important design decisions.

### Verification
- Commands run and meaningful results.

### Out of scope and follow-ups
- Explicitly deferred work.
```
