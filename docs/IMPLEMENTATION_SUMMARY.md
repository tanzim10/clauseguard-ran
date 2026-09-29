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
