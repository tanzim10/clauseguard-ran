.PHONY: up down logs parse index ingest-telecomts eval e2e mcp install test api

TELECOMTS_REVISION ?= 1b3a88a440f4922edfcd1e230e42c003c0e13bdf

up:
	docker compose up -d qdrant api

down:
	docker compose down

logs:
	docker compose logs -f api

parse:
	docker compose run --rm api python -m clauseguard.cli.parse_corpus

index:
	docker compose run --rm api python -m clauseguard.cli.index_corpus

ingest-telecomts:
	docker compose run --rm api python -m clauseguard.cli.ingest_telecomts --revision $(TELECOMTS_REVISION)

eval:
	docker compose run --rm api python -m clauseguard.cli.run_eval

e2e:
	docker compose run --rm api python -m clauseguard.cli.run_pipeline

mcp:
	docker compose --profile mcp up mcp

install:
	uv sync --extra dev

api:
	uv run uvicorn clauseguard.api.main:app --reload --port 8000

test:
	uv run pytest tests/ -q
