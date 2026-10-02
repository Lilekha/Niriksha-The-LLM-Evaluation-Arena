# ADR 0005: Score artifacts and reports

Status: accepted. Date: 2026-10-02. Partly supersedes ADR 0004 item 9 (scores were in-memory only in M2.1).

Note (M2.4): the statements below that say comparison across runs does not exist describe M2.2; a descriptive two-run comparison arrived in M2.4 ([ADR 0007](0007-run-comparison.md)). Ranking, confidence intervals and significance are still absent.

Covers M2.2: persisting scores apart from the immutable run, verifying them, aggregating them and rendering reports. Everything is offline and read-only with respect to runs. There is still no real provider, JSON Schema validity (added in M2.3, ADR 0006), confidence interval, ranking or comparison across runs.

## Context
Generation and scoring are separate (ADR 0004). To make scoring reproducible, its results need to be stored, tied to the exact run they were computed from, checkable without any provider, and summarised without ambiguity about failed generations.

## Decisions

### Artifact scope and location
1. One artifact is one metric applied to one run. A QA run has one artifact; an extraction run has two. Rescoring the same outputs with another metric or version is another artifact, never a regeneration.
2. Artifacts live in a separate `scores/` tree (gitignored), never inside `runs/<run_id>/`. A run directory keeps holding exactly `manifest.json` and `results.jsonl`, and `scores_dir` inside a run directory is refused. The file is `<scores_dir>/<run_id>--<metric>--<version>.json`, built only from validated parts, with a length cap for Windows paths.

### Format (`niriksha.core.scorestore`)
3. One pretty-printed JSON document, strict and closed like the other models: `artifact_version` (1), `artifact_id` (must equal the file name), `source`, `metric`, `records_sha256`, `records`. No timestamp is stored, so an artifact is a pure function of the run files, the dataset and the scorer version, and rewriting it gives identical bytes.
4. `source` holds the run ID, the SHA-256 of the exact bytes of the run's `manifest.json` and `results.jsonl`, the dataset identity, the selection (case count and case-ID hash, the existing selection hash), the prompt hash, the provider name and the requested model. `records` are the existing `ScoreRecord` objects in selection order. No scorer configuration exists in 0.1.0: a rule change must ship as a new metric version.
5. Aggregates are not stored. They are computed from the records, so they cannot disagree with them.

### `records_sha256` (exact canonical bytes)
6. SHA-256, lowercase hex, of the UTF-8 bytes of
   `json.dumps([record.model_dump(mode="json") for record in records], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`.
   That is the JSON array of the records in artifact order; each record an object with the keys `details`, `metric`, `metric_version`, `reason`, `request_id`, `status`, `value` in sorted order; no whitespace; non-ASCII text as raw UTF-8; `null` for absent values; floats in Python's shortest round-trip form (`1.0` is written `1.0`, never `1`). It is the canonical form of ADR 0003, applied to the record array.
7. `tests/test_score_artifact.py` pins it independently, as the M1.4 hash tests do: it recomputes the hash with only `hashlib` and `json`, asserts the exact canonical text of a small example (including `null`, `1.0`, a non-ASCII reason and `0.3333333333333333`), and compares golden digests with both its own calculation and the production helper. The golden digests are literals with a narrow per-line scanner allowlist.
8. **Integrity, not authenticity.** `records_sha256` and the file hashes detect accidental corruption and drift. Anyone who can write the files can rewrite an artifact and all its hashes consistently. Nothing here proves who produced a score.

### Reading, verifying, reusing
9. `read_artifact` is strict and needs neither the run nor a provider: valid UTF-8, no byte-order mark, one JSON object with no trailing data, duplicate keys, `NaN` or `Infinity`; a supported `artifact_version`; no unknown or missing field; valid score records; unique case IDs; the record count equal to the selection count; the case-ID hash equal to the selection hash; every record on the artifact's metric and version; `records_sha256` matching; and a file name equal to `<artifact_id>.json`. Any failure raises `ScoreArtifactError`; error text never echoes stored values.
10. `verify_artifact(artifact, run_dir, dataset_dir)` reloads the run with `load_run` (any failure becomes `ScoreArtifactMismatchError` with the loader error as cause), recomputes the artifact with the scorer registered for the artifact's exact `(metric, version)`, and compares it with the stored one. An unregistered version raises `ScoreArtifactError`: it can still be read and aggregated, but it cannot be verified. There are no automatic version upgrades.
11. `score_run_to_artifact` never overwrites. It creates the file exclusively (flush, fsync). Cleanup on failure removes only a file this call created: if the exclusive create itself fails, whatever is at the path, including an existing artifact, is left alone. A `scores_dir` that is a regular file or cannot be created, and a directory occupying the artifact path, raise `ScoreArtifactError`. If the file exists, it is read strictly, compared with what the current code computes from the verified run, and reused only if equal, which checks in one step its structure, its source (including the run file hashes) and the exact scorer implementation. A corrupt file raises `ScoreArtifactError`; a well-formed but different or unverifiable one raises `ScoreArtifactMismatchError`; neither is touched. To score again under different rules, ship a new metric version.

### Aggregation and reports (`niriksha.core.report`)
12. `aggregate` returns `total_cases`, `scored`, `not_scored` (with `scored + not_scored == total_cases`), `mean` over scored records only, and `not_scored_by_reason` sorted by reason. The mean is `math.fsum(values) / len(values)`, independent of order. A real 0.0 is a score and counts. A not-scored record has no value and is never treated as 0.0. With no scored record the mean is `None` with `mean_unavailable_reason` `no_cases` or `no_scored_cases`; it is never reported as 0.
13. `build_report` adds the artifact ID and `records_sha256`, the run, dataset, provider and requested model, the metric and version, and one row per case (`request_id`, `status`, `value`, `reason`) in artifact order. `render_report_json` is the exact, lossless form; `render_report_markdown` is a presentation (six decimals, `n/a` for not scored, pipes and line breaks in cells neutralised). Neither contains a timestamp, so both are deterministic. Reports are regenerated from the artifact on demand and are not persisted.

## Limitations
- The metrics measure only what `docs/metrics/` defines and are not validated against human labels; a mean of per-case values is a summary, not a claim of quality.
- Integrity hashes are not authentication (decision 8). The source file hashes are taken when scoring, so a file changed between loading and hashing in that window would be bound as it was at hashing time.
- A process killed mid-write can leave a truncated artifact. Readers refuse it, and it blocks its name until it is removed by hand.
- Run file hashes are raw-byte hashes; a run directory rewritten byte for byte differently (for example, different line endings or whitespace) no longer matches its artifacts even if the data is the same.
- No ranking, confidence interval, significance statement or comparison across runs or artifacts exists; those need a defined protocol (later milestones).
- The request-hash dependence on the request schema from ADR 0003 still applies to `load_run`, and therefore to verification.
- The Markdown rendering rounds values to six decimals; use the JSON report for exact values.

## Consequences
- A completed run can be scored, persisted, verified, aggregated and reported with no provider and no network, and the run directory is never modified.
- Adding a scorer or a metric version means a module, a registry entry and a metric document; stored artifacts of older versions stay readable.
