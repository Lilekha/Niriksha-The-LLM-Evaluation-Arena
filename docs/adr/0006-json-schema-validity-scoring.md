# ADR 0006: JSON Schema validity scoring

Status: accepted. Date: 2026-10-02.

Covers M2.3: a deterministic `json_schema_validity` metric (version 0.1.0) for extraction runs, integrated with the M2.2 artifact, verify and report workflow. Still offline, with no provider, CLI, confidence interval or comparison across runs. Metric definition: [json_schema_validity](../metrics/json_schema_validity.md).

## Context
`field_exact_match` compares values; `json_parse_validity` only checks syntax. Neither says whether an answer has the declared shape. The only declared shape is `DatasetMeta.output_schema`: dataset-level, required for extraction datasets, and part of the dataset hash.

## Decisions
1. **Source of the schema.** Only the dataset's `output_schema`. Because it is hashed into `content_sha256`, every artifact is already bound to it. No dataset, run or artifact format or hash changed.
2. **Library.** `jsonschema>=4.21,<5` as a runtime dependency. 4.18 is the first release that resolves references through `referencing` with no implicit remote fetch; 4.21 is the floor and is tested together with the latest release in isolated environments. A hand-written validator was rejected (conformance surface); `fastjsonschema` was rejected because it generates and executes code from the schema.
3. **Dialect.** Draft 2020-12 only; another `$schema` is refused. `format` is not asserted.
4. **Offline.** Only local references; `$id` and non-local `$ref`/`$dynamicRef` are refused before validation, and the validator gets no registry beyond the library's built-in metaschemas.
5. **Scoring contract.** Conforming `1.0`; valid JSON that violates the schema `0.0`; unparsable output `0.0` with the existing parse failure codes; failed generation `not_scored`. No repair or coercion; strict parsing is shared with `json_parse_validity`.
6. **Invalid schema.** A typed `ScoreArtifactError` before any file is written, never a per-case score.
7. **Details** carry schema-derived text only, never output text or output-derived paths. The first error is chosen by sorting, not by library order.
8. **Workflow hook.** The scorer needs run-level data that `score_case` does not receive. A scorer module may define `prepare(run)`; `score_records` calls it once and passes the result as an extra argument to `score_case`. The three existing scorers do not define it and are unchanged.
9. **Registration.** In the M2.2 `REGISTRY` only. `score_run` and `SCORERS_BY_TASK` (M2.1) are unchanged, so their output is unchanged. `available_metrics("json_extraction")` now lists three metrics.

## Consequences and limits
- Scores depend on the installed `jsonschema` version, which is not recorded; a rule change needs a new metric version.
- `pattern` uses Python regex semantics; a pathological pattern can hang scoring (no timeout).
- Native transitive dependency: `rpds-py` (a compiled extension; wheels exist for common platforms).
- The loader still validates only that `output_schema` is a JSON object; schema usability is checked at scoring time.
