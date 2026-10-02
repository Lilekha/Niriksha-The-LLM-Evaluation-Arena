# Metric: normalized_exact_match

**Version:** 0.1.0
**Status:** implemented; unit and integration tested; not yet validated against human labels
**Direction:** higher is better (`higher_is_better`; 1.0 is the desired outcome)

## Purpose
Whether the stored answer, after a fixed normalisation, is identical to one of the accepted answers of a short-answer QA case. It measures string identity under that normalisation, nothing else. It does not measure whether an answer is semantically correct, well reasoned or useful.

## Applicable task
`short_answer_qa` datasets. It is not computed for `json_extraction` datasets.

## Formula
For a case with accepted answers `a_0 … a_n` and a stored output `o`, let `N` be the normalisation below.

- `value = 1.0` if `N(o) == N(a_i)` for some `i`; `matched_answer_index` is the first such `i`.
- `value = 0.0` otherwise (`matched_answer_index` is `None`).

**Normalisation `N`**, applied identically to the output and to each accepted answer, in this order:

1. Unicode NFC. NFC only, never NFKC.
2. `str.casefold()` (Python's locale-independent full case folding).
3. Unicode NFC again.
4. Collapse every run of whitespace to one space and strip both ends. "Whitespace" is what `str.split()` treats as whitespace, which includes the no-break space.

**Not applied:** punctuation removal, accent removal, digit or width folding (NFKC), removal of zero-width characters, transliteration, stemming, article removal, number-word conversion, or any language-specific rule.

## Required inputs
- The stored `output_text` of a successful generation.
- The case's `answers` (at least one accepted answer).

Output record (`ScoreRecord`): `metric="normalized_exact_match"`, `metric_version="0.1.0"`, `status`, `value`, `reason`, and `details` with `matched_answer_index` (int or null) and `empty_output` (bool).

## Failed generations and invalid output
- **Failed generation** (any `GenerationFailure`): `status="not_scored"`, `value=None`, `reason="generation failed: <kind>"`, `details={"failure_kind": "<kind>"}`. It is never counted as an incorrect answer and never dropped.
- **Empty or whitespace-only output** from a successful generation is a scored answer: `value=0.0` with `empty_output=true`.
- **Prose, extra words or a truncated answer** are scored `0.0`; nothing is extracted or repaired.

## Unicode behaviour
- Canonically equivalent text compares equal. Devanagari U+0958 equals U+0915 U+093C (the NFC form is the two-code-point sequence); Kannada U+0C95 U+0CCA equals U+0C95 U+0CC6 U+0CC2.
- Zero-width joiner and non-joiner (U+200D, U+200C) and zero-width space (U+200B) are kept, because they change the meaning of Indic text: `क्ष` and `क्‍ष` (with a joiner) are different answers.
- Full case folding changes some Latin characters: German sharp s becomes `ss`, and the `fi` ligature U+FB01 becomes `fi`. Fullwidth letters are lowercased but stay fullwidth, so `ＰＡＲＩＳ` does not match `Paris`.
- Scripts without case (Devanagari, Kannada) are unaffected by case folding.

## Assumptions
- The accepted answers are complete for the question; a correct answer missing from the list scores 0.
- The model was asked for the bare answer. Any wrapper text counts against it.

## Limitations
- Paraphrases, synonyms, units, spelled-out numbers and transliterations (the same word in Latin script instead of Devanagari) score 0.
- `Paris.` scores 0 against `Paris`: punctuation is never removed.
- A longer correct answer ("The capital is Paris") scores 0.
- A truncated generation is scored like any other output; `finish_reason` is not consulted.
- Per-case value; aggregation is per artifact (M2.2); confidence intervals are not implemented.

## Examples
- **Positive:** answers `["Paris"]`, stored output `"  PARIS \n"` gives `1.0`, `matched_answer_index=0`.
- **Positive (Hindi, equivalence):** output U+0958 against an accepted answer U+0915 U+093C gives `1.0`.
- **Negative:** answers `["Paris"]`, output `"Paris."` gives `0.0`.
- **Negative:** output `"The capital is Paris"` gives `0.0`.
- **Not scored:** the generation timed out: `status="not_scored"`, `reason="generation failed: timeout"`.

## Validation method
`tests/test_scorers.py` covers accepted equalities, adversarial near-misses (punctuation, fullwidth characters, zero-width characters, combining marks, accents), the `fi` ligature and sharp s under case folding, empty output, Hindi and Kannada fixture cases, canonical-equivalence pairs, a guard that the Unicode test constants really are the intended code points, and every failure kind. `tests/test_score_run.py` scores stored runs end to end. There is no agreement study against human labels yet; this metric makes no claim beyond its definition.
