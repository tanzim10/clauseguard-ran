# Implementation Summary

This document records completed feature work by implementation phase. Each phase is focused on
the corresponding PR or issue: what it delivered, how it was built, how it was verified, and what
remains outside its scope.

## Table of Contents

- [Phase 1 — Corpus Manifest Validation](#phase-1--corpus-manifest-validation)
- [Phase 2 — Clause-Aware Corpus Chunking](#phase-2--clause-aware-corpus-chunking)
- [Phase 3 — Qdrant Corpus Indexing](#phase-3--qdrant-corpus-indexing)
- [Phase 4 — POST /search Retrieval API](#phase-4--post-search-retrieval-api)
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
