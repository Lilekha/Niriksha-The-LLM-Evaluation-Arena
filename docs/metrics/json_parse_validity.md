# Metric: json_parse_validity

**Version:** 0.1.0
**Status:** implemented; unit and integration tested; not yet validated against human labels
**Direction:** higher is better (`higher_is_better`; 1.0 is the desired outcome)

## Purpose
Whether the stored output parses as JSON under the policy below. It measures parse validity only. It is **not** JSON Schema compliance and says nothing about whether the content is right: `{"unrelated": true}` is valid.

## Applicable task
`json_extraction` datasets. The parsing function is task-independent, but this metric is not computed for QA datasets.

## Formula
`value = 1.0` if the output parses under the policy, else `0.0`.

**Policy** (implemented by `parse_json_strict`, applied to the output exactly as stored, nothing stripped, extracted or repaired):

- Valid means Python's `json.loads` accepts the entire string. Whitespace around the value is allowed.
- Empty or whitespace-only output is invalid (`empty_output`).
- Surrounding prose, markdown code fences, trailing commas, single quotes, unquoted keys, truncated text, two concatenated documents and a leading byte-order mark are invalid (`invalid_json`).
- `NaN`, `Infinity` and `-Infinity` are invalid (`non_finite_number`).
- A number literal that overflows to infinity (`1e999`) or whose non-zero value underflows to zero (`1e-400`) is invalid too (`number_out_of_range`), because it would otherwise be silently replaced by a different number. A literal whose digits are all zero (`0.0`, `-0.0`, `0e-400`) is a real zero and is valid.
- An object with a repeated key, at any depth, is invalid (`duplicate_key`).
- Nesting deeper than the interpreter allows is invalid (`too_deeply_nested`); an integer beyond Python's digit limit is invalid (`invalid_json`).
- Any JSON value is valid at the top level (object, array, string, number, boolean, null). An escaped lone surrogate inside a string is syntactically valid and is accepted.

## Required inputs
The stored `output_text` of a successful generation. The expected data of the case is not read.

Output record: `metric="json_parse_validity"`, `metric_version="0.1.0"`, `status`, `value`, `reason`, and `details` with `top_level_type` (`object`, `array`, `string`, `number`, `boolean`, `null`, or null when invalid) and `failure` (one of the codes above, or null when valid).

## Failed generations and invalid output
- **Failed generation:** `status="not_scored"`, `value=None`, `reason="generation failed: <kind>"`, `details={"failure_kind": "<kind>"}`. Never dropped, never a zero.
- **Unparsable output from a successful generation** is a scored answer: `value=0.0` with the failure code in `details`. It is not `not_scored`.

## Assumptions
- The model was asked to reply with JSON only. Output that wraps JSON in prose or a code fence is treated as not valid; no repair is attempted, because repair would hide a real failure of the model to follow the format.

## Limitations
- A valid document of the wrong shape or with wrong values scores 1.0. Use `field_exact_match` for values; use `json_schema_validity` for shape.
- Python's parser decides edge cases. Behaviour on integers beyond the digit limit and on extreme nesting follows the interpreter (its integer digit limit and recursion limit), not the JSON specification. The tests therefore check the real inputs only under the default limits and check the mapping to the named failure codes with injected errors under any limits.
- A truncated generation is scored like any other output; `finish_reason` is not consulted.
- Per-case value; aggregation is per artifact (M2.2); there are no confidence intervals.

## Examples
- **Positive:** `{"name": "Asha", "age": 30}` gives `1.0`, `top_level_type="object"`.
- **Positive (not a check of content):** `{"unrelated": true}` gives `1.0`.
- **Negative:** `Sure! {"name": "Asha"}` gives `0.0`, `failure="invalid_json"`.
- **Negative:** `{"a": 1, "a": 2}` gives `0.0`, `failure="duplicate_key"`.
- **Negative:** `{"a": NaN}` gives `0.0`, `failure="non_finite_number"`.
- **Negative:** `{"a": 1e-400}` gives `0.0`, `failure="number_out_of_range"` (it would otherwise read as 0.0).
- **Not scored:** the generation hit a rate limit: `status="not_scored"`, `reason="generation failed: rate_limit"`.

## Validation method
`tests/test_scorers.py` covers valid forms (every top-level type, padded whitespace, escapes, zeros and boundary numbers) and invalid forms (every failure code, prose, fences, trailing commas, byte-order mark, truncation, overflow and underflow, over-deep nesting, oversized integers), plus injected-error tests for the interpreter-dependent codes. `tests/test_score_run.py` scores stored runs end to end. No human-label study exists.
