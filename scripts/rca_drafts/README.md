# RCA draft-case scripts (Task 4.3)

These scripts rebuild `data/golden/rca_drafts/` from the local TelecomTS data. They need the
exploratory scenario release in `data/scenarios/release`, `data/scenarios/scenarios.jsonl` and the
raw traces in `data/telecomts/raw`. Run them from the repository root, in this order:

```bash
python3 -I scripts/rca_drafts/events.py data/scenarios/release data/scenarios/scenarios.jsonl /tmp/events.json
python3 -I scripts/rca_drafts/measure.py /tmp/events.json data/telecomts/raw /tmp/measured.json
python3 -I scripts/rca_drafts/select_cases.py /tmp/measured.json /tmp/selection.json
python3 -I scripts/rca_drafts/draft.py /tmp/measured.json /tmp/selection.json data/corpus/manifest.csv \
  data/golden/rca_drafts/rca.draft.jsonl data/golden/rca_drafts/annotations.draft.jsonl
```

| Script | What it does |
|---|---|
| `events.py` | Merges overlapping dev/test synthetic fault windows into distinct anomaly events (41). |
| `measure.py` | Picks each event's representative window and measures baseline vs fault means from the raw `metrics.csv`, checking they equal the release window values. |
| `select_cases.py` | Applies the rules in `data/golden/rca_drafts/selection_rule.md` and picks 12 events. |
| `draft.py` | Writes the draft `RcaRow` and `Annotation` records (`status: draft`). |

Then review the drafts with a model from another family:

```bash
uv run python -m clauseguard.cli.validate_rca --provider openrouter --model openai/gpt-oss-120b \
  --rows data/golden/rca_drafts/rca.draft.jsonl --annotations data/golden/rca_drafts/annotations.draft.jsonl \
  --out-dir artifacts/eval/rca_review_gptoss --delay 3 --max-tokens 8000
# Groq free tier: --provider groq --delay 65 --max-tokens 3000 (8,000 tokens/minute limit)
```

Model verdicts are an audit; only a person sets an annotation to `approved`.

Render the human review packet from the saved verdicts:

```bash
uv run python -I scripts/rca_drafts/review_packet.py data/golden/rca_drafts/rca.draft.jsonl \
  data/golden/rca_drafts/annotations.draft.jsonl data/golden/rca_drafts/measured_events.json \
  data/golden/rca_drafts/review_packet.md "gpt-oss-120b=data/golden/rca_drafts/reviews/gpt-oss-120b-openrouter" "qwen3.8-27b=data/golden/rca_drafts/reviews/qwen3.8-27b-groq"
```
