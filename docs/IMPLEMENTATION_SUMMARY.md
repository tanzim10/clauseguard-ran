# Implementation Summary

Running log of what shipped in each pull request, explained twice: once in plain language, once with technical detail. A new section is appended each time a PR is opened.

<!-- Template for new entries:

## PR #<number>: <title> (<YYYY-MM-DD>)
**Link:** <PR URL>

### In simple words
What changed and why, in a sentence or two anyone can follow.

### Technical details
- Files/modules touched
- Approach and key decisions
- Notable tradeoffs or follow-ups

-->

## PR #TBD: Implement Qdrant corpus indexing (2026-09-20)
**Link:** TBD

### In simple words
The project can now turn existing parsed O-RAN/3GPP text into batched OpenAI embeddings and
citation-ready Qdrant points through `make index`, with deterministic IDs and clear failures for
missing parsed inputs or incompatible vector collections.

### Technical details
- Added explicit OpenAI/Qdrant dependencies and retrieval settings for model, dimensions, batch
  sizes, distance, collection, and parsed corpus location.
- Implemented injectable, batched embedding requests with response ordering and dimension
  validation in `retrieval/embeddings.py`.
- Implemented idempotent Qdrant collection validation, deterministic UUID point IDs, batched
  upserts, citation text/provenance payloads, and the low-level `query_points` search wrapper.
- Implemented the parsed-text indexer and `index_corpus` CLI. Parsed files use the manifest
  filename with a `.txt` suffix; PDF parsing remains a separate prerequisite.
- Added fixture-only tests for settings, embedding batches, Qdrant schemas/upserts/search, the
  indexer, and CLI success/failure paths.
- Verified with `make test`: 37 passed, 1 existing deprecation warning.
- Deferred `/search`, PDF parsing, BM25/hybrid retrieval, stale-point deletion, and dual-index
  work to later issues.

## PR #TBD: Implement corpus manifest validation (2026-09-10)
**Link:** TBD

### In simple words
The starter O-RAN/3GPP corpus is now represented by a real manifest, and the code can verify
that each listed file exists and matches its recorded checksum.

### Technical details
- Implemented `load_manifest` and `validate_manifest_row` in `src/clauseguard/corpus/manifest.py`.
- Added `data/corpus/manifest.csv` with 12 starter corpus rows copied from the local corpus pack.
- Copied the 12 starter files into ignored local storage under `data/corpus/raw/`.
- Added fixture-based manifest tests for required fields, doc types, checksum mismatches,
  documented blockers, `has_text_layer`, and multi-row CSV loading.
- Verified with `ruff`, full `pytest`, and a manifest validation pass against local raw files.

---

## PR #9: Implement clause-aware corpus chunking (2026-09-16)
**Link:** https://github.com/tanzim10/clauseguard-ran/pull/9

### In simple words
Specification text can now be split into useful, repeatable chunks that keep numbered clauses
and paragraphs together when possible while still honoring the configured size limit.

### Technical details
- Implemented numbered-heading and paragraph-aware chunking in `src/clauseguard/corpus/chunker.py`.
- Added sentence- and word-level overflow splitting with deterministic IDs and retrieval metadata.
- Added nine contract tests covering section boundaries, fallback behavior, metadata, stable IDs,
  and token budgets.
- Verified with `uv run pytest tests/test_corpus_chunker.py` — 9 passed.
- Kept PDF parsing, embeddings, Qdrant indexing, retrieval, and end-to-end RCA out of scope.
