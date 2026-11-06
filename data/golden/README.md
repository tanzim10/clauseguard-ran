# Golden release

The release contains 24 specification questions and 12 approved RCA cases.
The project owner confirmed human citation review for all revised RCA cases on 2026-11-05.

`manifest.json` pins the golden files and `rca_policy.json` by content hash.
The policy records configuration eligibility, selected scenario membership, and the approved recording-family waiver.
It preserves the source release's exploratory mode. It does not establish recording independence.

Each RCA case supports KPI evidence with the base model, the adapted SLM, and the KPI summary configuration.
The full-evidence configurations record text as missing. They must not be reported as complete multimodal cases.
Text-only RCA is ineligible because no approved incident text exists.
The common cohort for the four-configuration comparison is therefore empty.

`rca_questions.candidate.json` contains two question variants for each of the 12 existing cases.
One variant describes measurement directions; the other includes recomputed numeric means.
Each record links to its raw CSV, content hash, source rows, and existing comparison intervals.
The project owner approved the question wording and measured observations on 2026-11-05.
These questions remain KPI-derived text. They are not independent alarm or log evidence.
The indented JSON file contains each question record and its approval details.
These supplemental questions are outside the frozen manifest and do not replace the golden evaluation questions.
They do not change configuration eligibility or add new evaluation incidents.
The cases retain development/test membership and must stay outside SLM training.

The KPI summary configuration uses only the generic question recorded in the policy as minimal text.
It excludes alarm text, context descriptions, and the original question's embedded numeric summary.
These are input eligibility decisions. They do not show that generator execution or model availability has passed review.

Release checks validate local scenario artifacts when available. Otherwise, they use the selected membership snapshot pinned in the policy.
Unknown scenario IDs, training membership, excluded membership, and changed splits remain invalid.
Default release behavior still requires scored scenario data when no approved waiver policy exists.

Run the release check with:

```bash
uv run python -m clauseguard.cli.build_golden check --strict
```
