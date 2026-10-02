# Metric: field_exact_match

**Version:** 0.1.0
**Status:** implemented; unit and integration tested; not yet validated against human labels
**Direction:** higher is better (`higher_is_better`; 1.0 is the desired outcome)

## Purpose
The fraction of the expected top-level fields that the stored output reproduces exactly, under strict, type-sensitive comparison. It measures value identity per field. It is **not** JSON Schema validation (see `json_schema_validity`) and does not measure semantic equivalence.

## Applicable task
`json_extraction` datasets. The expected data is the case's `expected` object.

## Formula
Let `E` be the expected object with `n` fields, and `O` the output parsed with the same strict policy as `json_parse_validity`.

- If `O` is not a usable JSON object (output is invalid JSON under the policy of `json_parse_validity`, including a number literal that overflows or underflows, a non-object value, or an object whose keys collide after NFC), then `value = 0.0`, `output_is_json_object=false`, and all `n` fields count as missing.
- Otherwise, for each expected field, the key is looked up in `O` (keys compared after Unicode NFC, case-sensitive). The field is **matched** if the key is present and the values are equal under the rules below, **missing** if the key is absent, and **mismatched** if present but unequal.
- `value = matched / n`.
- If `n = 0`: `value = 1.0` when `O` is an empty object, else `0.0`.
- A key that is absent is **missing** even when its expected value is `null`; `null` in the output matches an expected `null`, an empty string does not.
- Keys in `O` that are not in `E` are **extra**. They do not change `value`; they are reported in `details`.

**Value equality** (`values_equal`, recursive, no coercion between JSON types):

- Booleans equal only booleans: `true` is not `1`, and `false` is not `0`. `null` equals only `null`.
- Numbers equal numbers by value; JSON has one number type, so `30` equals `30.0`, and `-0.0` equals `0`. A number never equals a string: `30` and `"30"` differ. Integers compare exactly; a comparison involving a float uses Python's int/float equality on IEEE-754 doubles. A number literal that cannot be a finite double (`1e999`, or a non-zero `1e-400` that would read as 0.0) makes the whole output unusable instead of being converted, so no output number is silently replaced by 0.0 or infinity.
- Strings equal strings after Unicode NFC, with runs of whitespace collapsed and both ends stripped. Case and punctuation matter, and no case folding is applied. Zero-width characters are kept.
- Arrays are equal only with the same length and equal elements in the same order.
- Objects are equal only with the same key set (keys after NFC) and equal values. A nested object is compared in full, so an extra nested key makes it unequal. There is no partial credit inside a nested value.

## Required inputs
The stored `output_text` of a successful generation and the case's `expected` object.

Output record: `metric="field_exact_match"`, `metric_version="0.1.0"`, `status`, `value`, `reason`, and `details` with `output_is_json_object`, `expected_fields`, `matched_fields`, `missing_fields`, `mismatched_fields`, `extra_fields` and `exact_object_match` (true only when every expected field matched and there are no extra keys).

## Failed generations and invalid output
- **Failed generation:** `status="not_scored"`, `value=None`, `reason="generation failed: <kind>"`, `details={"failure_kind": "<kind>"}`. Never dropped, never a zero.
- **Unusable output from a successful generation** is scored `0.0` with `output_is_json_object=false`. That is a score for a wrong answer, distinct from a failed generation.

## Assumptions
- `expected` lists exactly the fields the answer must contain; fields the dataset author did not list are not rewarded.
- Gold values avoid floating point where possible, since `0.1 + 0.2` style differences are compared exactly.

## Limitations
- Extra fields are not penalised in `value`. A model that returns many extra keys can still score `1.0`; check `exact_object_match` or `extra_fields`.
- No semantic equivalence: `"2026-10-01"` and `"1 October 2026"` differ, as do `"Asha"` and `"asha"`.
- No partial credit within a nested object or array.
- The value is per case; fields are weighted equally, and cases with different field counts are not weighted. Aggregation is per artifact (M2.2); there are no confidence intervals.
- A truncated generation is scored like any other output.

## Examples
Expected `{"name": "Asha", "age": 30}`.

- **Positive:** `{"name": "Asha", "age": 30}` gives `1.0`, `exact_object_match=true`.
- **Positive:** `{"age": 30.0, "name": " Asha "}` gives `1.0` (number equality, whitespace stripped, key order irrelevant).
- **Partial:** `{"name": "Asha", "age": "30"}` gives `0.5`: the age is a string, not a number.
- **Partial:** `{"name": "Asha"}` gives `0.5`, `missing_fields=1`.
- **Extra key:** `{"name": "Asha", "age": 30, "city": "Pune"}` gives `1.0`, `extra_fields=1`, `exact_object_match=false`.
- **Negative:** `Sure! {"name": "Asha", "age": 30}` gives `0.0`, `output_is_json_object=false`.
- **Not scored:** the generation timed out: `status="not_scored"`.

## Validation method
`tests/test_scorers.py` covers exact, partial, missing, mismatched and extra fields; equality of every JSON type pair (booleans, null, numbers, arrays, nested objects, strings, case, punctuation, whitespace, a precision edge); null versus an absent key; unrepresentable numbers and real zeros; NFC key and value equivalence for Devanagari and Kannada; zero-width joiners; unusable output of many kinds; colliding keys; an empty expected object; and the Hindi extraction fixture. `tests/test_score_run.py` scores stored runs end to end. No human-label study exists.
