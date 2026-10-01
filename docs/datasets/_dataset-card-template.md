# Dataset card: <name>

**Version:** <version tag, e.g. 0.1.0> | **Content hash:** <content_sha256 from dataset.json>
**Status:** draft | released

## Purpose and intended use
What the dataset measures and what it must not be used for.

## Language
- Languages present, as ISO 639-1 codes (`en`, `hi`, `kn`).
- Script per case, as ISO 15924 (`Latn`, `Deva`, `Knda`). Romanized Hindi or Kannada is `hi`/`kn` with script `Latn`.
- `code_mixed` flag per case.
- Allowed combinations (enforced by the loader): `en` with `Latn`; `hi` with `Deva` or `Latn`; `kn` with `Knda` or `Latn`.
- All text is UTF-8 in Unicode NFC. The loader rejects non-NFC text instead of normalising it. Normalisation applied at scoring time is documented in the metric docs, not applied to stored cases.

The case schema carries `language`, `script` and `code_mixed` fields from day one so Hindi and Kannada can be added without a schema change. The language and script sets are closed (`en`, `hi`, `kn`) until another is needed.

## Source and origin
Per case: `origin` (`original` or `public`), source name and URL if public, author if original, creation date.

## Licence and attribution
Licence per case or per source, and required attribution. Public data is added only after its licence is recorded here.

## Construction and annotation
How cases were written or selected, the annotation guideline, number of annotators, who they were and their language background, inter-annotator agreement on the double-annotated subset, and how ambiguous cases were handled.

## Splits and case IDs
Split names (`dev`, `test`, optional `holdout`) and sizes. Case IDs are stable and never reused. State how the holdout is protected and who has seen it.

## Known biases and limitations
Domain coverage, translation artifacts, single-annotator limits, size and the resulting confidence-interval width.

## Contamination and leakage
Which items come from public sources that may be in model training data. Canary strings used, if any. Statement of what is unknown.

## Files and identity
A dataset is a directory with `dataset.json` (schema version, name, version tag, task, description, `content_sha256`, and `output_schema` for extraction) and `cases.jsonl` (one case per line; file order is the dataset order). Case fields: `id`, `split`, `language`, `script`, `code_mixed`, `origin`, `source`, `license`, plus `question` and `answers` (short-answer QA) or `text` and `expected` (JSON extraction).

`content_sha256` is computed over the parsed cases, task, output schema and hash/schema versions, so formatting does not matter but case order, values and JSON types do (avoid floats in gold data). Name, version and description are not hashed. Keep `dataset.json` pretty-printed with the `content_sha256` pin on its own line: the secret scanner skips only such lines, so a pin inside compact JSON is flagged (see `SECURITY.md`). The exact payload is in [ADR 0003](../adr/0003-dataset-identity-and-run-persistence.md).

## Version and changelog
Workflow when a case must change: edit `cases.jsonl`; run the loader, which fails and prints the new computed hash; update `content_sha256`, bump `version`, and add a row below explaining the change and why. The loader cannot detect a content change that reuses an old version number, so the bump is a rule for the maintainer. Never change cases to improve a model's score.
| Version | Date | Change | Reason |
|---|---|---|---|

Cases are immutable. A label fix creates a new version with an entry here.
