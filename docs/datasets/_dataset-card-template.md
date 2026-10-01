# Dataset card: <name>

**Version:** <tag + content hash>
**Status:** draft | released

## Purpose and intended use
What the dataset measures and what it must not be used for.

## Language
- Languages present, as ISO 639-1 codes (`en`, `hi`, `kn`).
- Script per case, as ISO 15924 (`Latn`, `Deva`, `Knda`). Romanized Hindi or Kannada is `hi`/`kn` with script `Latn`.
- Code-mixed flag per case, with the languages mixed.
- All text is UTF-8, stored in Unicode NFC. Normalization applied at scoring time is documented in the metric docs, not applied to stored cases.

The case schema carries `language`, `script` and `code_mixed` fields from day one so Hindi and Kannada can be added without a schema change.

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

## Version and changelog
| Version | Date | Change | Reason |
|---|---|---|---|

Cases are immutable. A label fix creates a new version with an entry here.
