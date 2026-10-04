# Metric: json_schema_validity

**Version:** 0.1.0
**Status:** implemented; unit and workflow tested; not validated against human labels
**Direction:** higher is better (`higher_is_better`; 1.0 is the desired outcome)

## Purpose
Whether the stored output is valid JSON **and** satisfies the dataset's `output_schema`. It measures structural conformance only. It does not say the values are correct: `{"name": "Zed", "age": 1}` conforms to the fixture schema whatever the case expects. Use `field_exact_match` for values.

## Applicable task
`json_extraction` datasets. The schema is the dataset-level `output_schema` in `dataset.json`, shared by every case. It is covered by the dataset's `content_sha256`, so an artifact is bound to the exact schema it was scored with. No schema is read from anywhere else and none is substituted.

## Formula
`value = 1.0` if the output parses under the `json_parse_validity` policy and the parsed value satisfies the schema, else `0.0`.

| Output of a successful generation | Result |
|---|---|
| parses and satisfies the schema | scored, `1.0` |
| parses but violates the schema, including a wrong top-level type | scored, `0.0`, `failure="schema_violation"` |
| does not parse (empty, prose, fences, duplicate keys, `NaN`, out-of-range numbers, over-deep) | scored, `0.0`, `failure` is the parse code from `json_parse_validity.md` |
| the generation failed | `not_scored`, `reason="generation failed: <kind>"` |

Nothing is repaired, extracted, coerced or defaulted. `"30"` does not satisfy `integer`.

## Dialect and supported behaviour
- **Draft 2020-12 only**, checked by the `jsonschema` library (declared range `>=4.21,<5`). A `$schema` other than `https://json-schema.org/draft/2020-12/schema` is refused; an absent `$schema` means 2020-12. Full dialect conformance is **not** claimed. The behaviour below is what the tests exercise: `type`, `required`, `properties`, `enum`, `minimum`, `maxLength`, `pattern`, `additionalProperties`, `items`, `$defs` with a local `$ref`.
- **`format` is not checked.** It is an annotation only; `{"format": "email"}` accepts any string. Checking it needs optional libraries and would make scores depend on what is installed.
- **`pattern` and `patternProperties` use Python's `re`**, not ECMA-262. Some patterns differ in meaning (for example `\d` matches any Unicode digit in Python). Patterns are searched, not anchored.
- **`minLength`/`maxLength` count Unicode code points**, not grapheme clusters: one Kannada conjunct such as `ಕ್ಷ` counts as 3.
- **Numbers:** `30.0` satisfies `integer`; `30.5` does not; booleans are never numbers or integers.
- **References:** only local ones (`#...`). A `$ref` or `$dynamicRef` that does not start with `#`, or any `$id`, is refused. Nothing is ever fetched; validation does not touch the network.
- **Textual checks are conservative:** a key named `$id`, `$ref` or `$dynamicRef` is inspected wherever it appears in the schema, so a property literally named `$ref` is refused too.

## Invalid schemas
A schema that is not a JSON object, is not a valid 2020-12 schema, declares another dialect, uses `$id` or a non-local reference, contains an invalid regular expression, recurses without consuming the output, or has a local reference that cannot be resolved raises `ScoreArtifactError` **before any artifact is written**. It is a dataset defect, not a model failure, so it never becomes a per-case score. Most are found before any case is scored; an unresolvable local reference that the trivial probe instances do not reach is found when a case first reaches it.

## Details
`schema_valid`, `failure`, `error_count`, `first_error_keyword`, `first_error_schema_path`. The last three are `null` unless the output violated the schema. The "first" error is the one with the smallest schema path (then keyword), not the library's iteration order. Details contain only text derived from the schema (keywords, property names the schema declares). They never contain output text or paths into the output.

## Failed generations and invalid output
Same distinction as the other metrics: a failed generation is never dropped or zeroed; unparsable output from a successful generation is a scored `0.0`.

## Assumptions
- The model was asked for JSON only. Output wrapped in prose or a fence is invalid, not repaired.
- The dataset's `output_schema` states what a correct answer must look like. A test checks that the extraction fixtures' gold `expected` data conforms to it; the loader does not.

## Limitations
- Conformance is not correctness. A permissive schema scores everything 1.0.
- A catastrophic-backtracking `pattern` combined with adversarial output can make validation run for a very long time. There is no in-process timeout.
- A schema whose recursion needs no output depth to overflow the stack (for example `{"$ref": "#"}`, or a `$ref` cycle) is detected before scoring by validating a null, boolean, number, string, empty array and empty object, and raises `ScoreArtifactError`. A defect that only a particular output shape reaches cannot be told apart from an over-deep output: that case is reported as `failure="too_deeply_nested"` (scored `0.0`), the same code used for an over-deep output. A legitimate recursive schema (for example a tree) works; only an output deep enough to exhaust the stack (about 160 levels for a simple tree schema, fewer with more nested keywords) gets `too_deeply_nested`.
- Results depend on the installed `jsonschema` version, which is not recorded in the artifact (the artifact format is unchanged). Verification recomputes and compares, so a behaviour change in the library shows up as a mismatch; a deliberate rule change must ship as a new metric version.
- Per-case value; aggregation is per artifact and two runs can be compared descriptively (M2.4); there is no confidence interval.

## Examples (fixture schema: `name` string, `age` integer, both required)
- **Positive:** `{"name": "Asha", "age": 30}` gives `1.0`.
- **Positive:** `{"name": "Asha", "age": 30.0, "extra": 1}` gives `1.0` (extras are allowed unless the schema forbids them).
- **Negative:** `{"name": "Asha"}` gives `0.0`, `first_error_keyword="required"`.
- **Negative:** `{"name": "Asha", "age": "30"}` gives `0.0`, `first_error_schema_path="properties/age/type"`.
- **Negative:** `[]` gives `0.0` (valid JSON, wrong top-level type), not an unparsable-output score.
- **Negative:** `Sure! {...}` gives `0.0`, `failure="invalid_json"`.
- **Not scored:** a rate-limited generation gives `status="not_scored"`.

## Validation method
`tests/test_json_schema_scorer.py` (valid output, each violation kind, wrong top-level types, every parse failure, every generation failure kind, Hindi and Kannada, numbers, `format`, references, invalid schemas, fixture gold data) and `tests/test_json_schema_workflow.py` (persist, reload, verify, report, reuse, tamper detection, invalid schemas write nothing, no provider or socket activity, byte identity across hash seeds). No human-label study exists.
