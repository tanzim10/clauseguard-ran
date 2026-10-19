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
- [Task 3.1 — Validated Experiment Data](#task-31--validated-experiment-data)
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

## Task 3.1 — Validated Experiment Data

**Focus:** [#29](https://github.com/tanzim10/clauseguard-ran/issues/29) — expose consistent experiment
cohorts and unchanged observations before model-specific preprocessing.

**Date:** 2026-10-18

**Status:** Input layer implemented; real scored-data acceptance still needs recording review.

### Delivered

Models can share one checked data boundary: saved task membership, explicit exclusions, bounded
observation batches and a readiness report. A valid dataset and a dataset ready for scored
experiments are reported separately. The source release is neither resplit nor transformed.

### Technical implementation

- `baselines/data.py` reuses `ScenarioDataset`, task/modality selection and approved input
  projection. Cohorts carry release/source/split identities and a hash of ordered IDs and exact
  observation payloads. They can be serialized and checked against the release before iteration.
- Batches retain the existing numeric/text/paired payloads, with targets, scenario IDs and family
  IDs outside predictor inputs. Native values, nulls and all 128 time steps are preserved.
- Combined-diagnosis membership includes normal and the ten supported synthetic categories,
  excluding real jamming, without reconstructing source targets.
- The loader exposes validated manifest metadata and a public source-change check. Its existing
  readiness gates continue to govern batching, including explicit exploratory opt-in.
- `cli/inspect_experiment.py` writes cohorts and class/group/exclusion diagnostics to a fresh
  directory; unready releases retain reports with exit 3. Generated experiment reports are ignored.

### Verification

- Fixture tests cover cohort identities, invalid membership/targets, shuffled channel keys,
  unchanged observations, metadata isolation, both tasks, approved text/paired layers, source
  changes and unresolved recording gates. CLI fixtures check ready/unready output and overwrites.
- `make test` — 285 passed. Targeted Ruff and `git diff --check` passed.
- Full local inspection validated all 32,000 source records. Exploratory detection counts were
  19,089 / 6,590 / 6,321 for train/dev/test; category counts were 563 / 177 / 216. Both tasks
  retained all required labels, but every cohort remained unready because recording independence
  is unresolved. Detection test has five unverified source groups; category test has two.
- Text and paired cohorts remained empty. Inspecting the scored release retained 32,000 exclusions
  with `recording_family_unresolved`. Both real inspections correctly exited 3 and saved reports
  under ignored `artifacts/experiments/task-3.1-2026-10-18-{exploratory,scored}/`.

### Out of scope and follow-ups

Task 3.2 now owns scoring and saved model-run results. Model-specific feature extraction,
shortening, train-fitted imputation/scaling and model implementations remain in later baseline
issues. This work does not establish missing recording relationships or create approved text.
The real release remains unready for scored model runs until recording-family review is complete.

## Task 3.2 — Baseline Scoring and Saved Runs

**Focus:** [#30](https://github.com/tanzim10/clauseguard-ran/issues/30) — provide comparable,
replayable scores over fixed experiment cohorts.

**Date:** 2026-10-18

**Status:** Scoring and artifact foundation implemented; model adapters are intentionally supplied
by later baseline issues.

### Delivered

Every cohort case contributes to headline metrics. Failed, invalid and missing outputs remain
visible; saved runs can be checked, replayed and compared offline. No baseline model was trained.

### Technical implementation

- `baselines/contracts.py` defines per-case outputs and run metadata, including cohort identity,
  transformations, software versions, seed, configuration, checkpoint and resource fields.
- `baselines/scoring.py` calculates detection, category and combined diagnosis results, including
  class support, confusion matrices, successful-only summaries and group-bootstrap intervals.
- `baselines/artifacts.py` saves write-once run directories with per-case predictions, targets,
  groups, scoring settings, metrics and file checksums. Replay recomputes metrics; comparison rejects
  mismatched cohorts or case sets.
- `baselines/operations.py` provides an injected adapter seam. Training receives labeled batches;
  prediction receives observation payloads only and retains inference errors per case.
- `cli/baseline_runs.py` provides offline `score` and `compare` commands. The golden retrieval
  evaluator remains separate.

### Verification

- Fixture tests cover hand-calculated misses, false alarms, wrong categories, invalid and missing
  predictions, combined diagnosis, repeatable group bootstrap, artifact replay/integrity/overwrite,
  comparison compatibility, CLI output and the adapter input boundary.
- `make test` — 296 passed. Targeted Ruff and `git diff --check` passed.

### Out of scope and follow-ups

No model implementation or model-specific preprocessing is included. Later issues add adapters and
must record their fitted transformations and checkpoint identity. The validated real scored release
is still blocked on recording-family review under #29.

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
