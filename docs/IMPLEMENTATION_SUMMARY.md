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
