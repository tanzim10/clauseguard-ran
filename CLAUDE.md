# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repository.

## What this project is

**ClauseGuardRAN** (ORAN-Sage AI): citation-grounded multimodal root-cause analysis (RCA) for
O-RAN troubleshooting. It fuses KPI time series, alarms/logs/NL context, and retrieved
O-RAN/3GPP specification passages into evidence-backed explanations — never acting on a live
network, only suggesting to a human. See `README.md` for the full pipeline diagram and
deliverables table.

**Current status: scaffold.** Package layout, Docker Compose (Qdrant + API), and stubs exist;
most routes return `501`/`not_implemented` and CLI commands print TODO and exit. Don't assume
end-to-end RCA behavior works until the relevant module is implemented — check the status table
in `README.md` first.

## Repository layout

```
src/clauseguard/
├── corpus/       # manifest, PDF parser, clause-boundary chunker
├── retrieval/    # embeddings, Qdrant store, indexer
├── datasets/     # TelecomTS ingestion, scenario schema
├── kpi/          # KPI baselines (classical + deep)
├── text/         # text-only fault classifier
├── rca/          # sequential multimodal RCA pipeline (shared core)
├── llm/          # OpenAI client + LoRA hooks
├── eval/         # Recall@k, MRR, eval runner
├── api/          # FastAPI routes (/search, /query, /rca, /evaluate, /health)
├── mcp/          # MCP server — thin wrapper over the same core, not a second retrieval copy
└── cli/          # operational commands (parse_corpus, index_corpus, ingest_telecomts, run_eval, run_pipeline)
```

`data/`, `artifacts/`, and `models/` hold corpus/eval/model outputs; only example files and
`.gitkeep` placeholders are tracked (see `.gitignore`).

## Working conventions

- **Package manager: `uv`.** This repo already has `pyproject.toml` + `src/clauseguard` — use
  `uv sync --extra dev`, never `uv init` (that scaffolds a new project).
- **Run locally:** `make install`, `make api`, `make test` (wrappers around `uv sync`,
  `uv run uvicorn ...`, `uv run pytest`). Docker: `make up` / `make down` / `make logs` for
  Qdrant + API.
- **One pipeline, two interfaces.** FastAPI and the MCP server must call the same
  `RcaPipeline` / retrieval / eval code — never duplicate retrieval or RCA logic between them.
- **Tests use fixtures/mocks, not live calls.** No live OpenAI or Qdrant calls in CI — fake
  embedding clients and mocked Qdrant clients, per the existing issue conventions (see `#6`,
  `#7` for examples of expected test scope).
- **Corpus and eval integrity:** don't auto-generate golden citations with an LLM; golden
  `spec_refs_golden` are hand-written. Retrieval that comes up empty must abstain
  (`"not found"`), not fall back to pretraining.
- **Follow the phase plan.** Work is organized into weekly phases (corpus → RAG → dataset →
  baselines → citation-grounded RCA + LoRA → API/MCP/UI → demo). Each week has an explicit
  "Done when" bar and an explicit out-of-scope list — don't pull forward later-week work (e.g.
  HyDE, BM25 hybrid, hyperparameter sweeps) into an earlier issue.

## Issue tracking

GitHub issues are the source of truth for in-flight work, organized as parent/epic issues with
sub-issues (e.g. `#1` "O-RAN/3GPP Corpus Acquisition & Manifest" → `#2`, `#3`; `#4` "RAG
Pipeline: Chunking, Embedding & Qdrant Indexing" → `#5`, `#6`, `#7`). Each issue defines Problem
Statement, Solution, Success Criteria, Implementation Decisions, Testing Decisions, and Out of
Scope — treat those as the spec; don't expand scope beyond what an issue states without calling
it out. Run `gh issue list --state all` to see current status before starting related work.

## Local-only files (do not commit)

- `.claude/` — personal Claude Code settings/hooks, gitignored.
- `.context/` — local planning notes (12-week plan, task merge notes), gitignored.

## Pull requests

Every PR should update `docs/IMPLEMENTATION_SUMMARY.md` with a new dated section covering what
shipped, in both plain language and technical detail (template is in that file).
