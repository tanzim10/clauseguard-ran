# AGENTS.md

Guidance for Codex and other coding agents working in this repository.

## What this project is

**ClauseGuardRAN** (ORAN-Sage AI): citation-grounded multimodal root-cause analysis (RCA) for
O-RAN troubleshooting. It fuses KPI time series, alarms/logs/NL context, and retrieved
O-RAN/3GPP specification passages into evidence-backed explanations. It never acts on a live
network; it only suggests next steps to a human. See `README.md` for the full pipeline diagram
and deliverables table.

**Current status: scaffold.** Package layout, Docker Compose (Qdrant + API), and stubs exist.
Most routes return `501`/`not_implemented`, and CLI commands print TODO and exit. Do not assume
end-to-end RCA behavior works until the relevant module is implemented. Check the status table
in `README.md` first.

## Repository layout

```text
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
├── mcp/          # MCP server; thin wrapper over the same core, not a second retrieval copy
└── cli/          # operational commands
```

`data/`, `artifacts/`, and `models/` hold corpus/eval/model outputs. Only example files and
`.gitkeep` placeholders are tracked.

## Working conventions

- **Package manager: `uv`.** This repo already has `pyproject.toml` and `src/clauseguard`; use
  `uv sync --extra dev`, never `uv init`.
- **Run locally:** `make install`, `make api`, `make test`. Docker: `make up`, `make down`,
  and `make logs` for Qdrant + API.
- **One pipeline, two interfaces.** FastAPI and the MCP server must call the same
  `RcaPipeline` / retrieval / eval code. Do not duplicate retrieval or RCA logic between them.
- **Tests use fixtures/mocks, not live calls.** No live OpenAI or Qdrant calls in CI. Use fake
  embedding clients and mocked Qdrant clients.
- **Corpus and eval integrity:** do not auto-generate golden citations with an LLM. Golden
  `spec_refs_golden` values are hand-written. Retrieval that comes up empty must abstain
  (`"not found"`), not fall back to pretraining.
- **Follow the phase plan.** Work is organized into weekly phases: corpus, RAG, dataset,
  baselines, citation-grounded RCA + LoRA, API/MCP/UI, demo. Do not pull later-week work into an
  earlier issue without calling out the scope change.

## Issue tracking

GitHub issues are the source of truth for in-flight work. Each issue defines Problem Statement,
Solution, Success Criteria, Implementation Decisions, Testing Decisions, and Out of Scope. Treat
those as the spec. Run `gh issue list --state all` before starting related work.

## Codex workflow

- Inspect the repo before editing, especially `README.md`, `CLAUDE.md`, this file, and any
  issue-specific context.
- Use focused edits that match existing patterns.
- Run the narrowest meaningful verification for the change. Prefer `make test` when behavior is
  touched.
- Do not revert unrelated uncommitted changes. Work with the current tree.
- Before opening a PR, update `docs/IMPLEMENTATION_SUMMARY.md` with a dated section explaining
  what shipped in plain language and technical detail.

## Local-only files

- `.codex/` - personal Codex workflows/settings, gitignored.
- `.claude/` - personal Claude Code settings/hooks, gitignored.
- `.context/` - local planning notes, gitignored.

## Pull requests

Every PR should update `docs/IMPLEMENTATION_SUMMARY.md` with a new dated section covering what
shipped, in both plain language and technical detail. The template is in that file.
