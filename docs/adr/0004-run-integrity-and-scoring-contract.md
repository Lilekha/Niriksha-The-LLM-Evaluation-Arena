# ADR 0004: Run integrity and the scoring contract

Status: accepted. Date: 2026-10-02. Item 9 and the matching limitations are superseded in part by [ADR 0005](0005-score-artifacts-and-reports.md): since M2.2 scores are persisted as separate artifacts and aggregated per artifact.

Note (M2.4): the statements below that say comparison across runs does not exist describe M2.1; a descriptive two-run comparison arrived in M2.4 ([ADR 0007](0007-run-comparison.md)).

Covers M2.1: an offline run reader, a score-record contract and three deterministic scorers. Everything is offline and read-only. There is still no real provider, score persistence, aggregation, JSON Schema validation or benchmark dataset.

## Context
Generation and scoring are separate: raw outputs are stored first, so scoring can be repeated, and corrected, with no new inference calls. A scorer must be able to trust that the outputs it reads belong to the dataset and configuration the manifest says.

## Decisions

### Run reader (`niriksha.core.runload`)
1. `load_run(run_dir, dataset_dir)` is read-only. It never calls a provider, uses the network, repairs, truncates or writes. A torn final line is refused (resume repairs it). It takes a dataset *directory*, so the dataset always goes through `load_dataset` and its hash pin; a hand-built `Dataset` cannot bypass it.
2. It returns a `LoadedRun` only if all checks pass, otherwise it raises `RunIntegrityError` with the original exception as `__cause__`:
   - manifest and results parse strictly (no corrupt or blank line, no duplicate request ID, no unknown field);
   - the manifest `run_id` equals the run directory name;
   - the dataset loads, matches its pin, and equals the identity in the manifest (name, version, task, schema version, content hash);
   - the stored prompt text hashes to the stored prompt hash;
   - the rebuilt selection has the recorded case count and case-ID hash;
   - the requests rebuilt from the manifest, dataset and prompt hash to the stored request hash of every result;
   - the results are complete, contain no ID outside the selection, and are in selection order;
   - each result has the manifest's provider and requested model and its own request ID.
3. The configuration is rebuilt from the manifest alone, so no caller-supplied configuration can disagree with the run.
4. Trust model: these checks use the production hash helpers. They detect drift and tampering between a run's files and its dataset, prompt and configuration. They cannot detect a defect in the helpers (tests pin the algorithms independently, ADR 0003), and they do not authenticate the files: a deliberate, internally consistent rewrite of both files would pass.

### Score record (`niriksha.core.scoring`)
5. `ScoreRecord` has `request_id`, `metric`, `metric_version`, `status`, `value`, `reason` and flat `details`, and is frozen, strict and closed like the other schemas.
6. Statuses are `scored` (a finite `value` in [0, 1]) and `not_scored` (`value` is `None`, a `reason` is required). The validators enforce this, so a number cannot be invented for a result that was not scored.
7. A failed generation is `not_scored` with `reason="generation failed: <kind>"`. It is never dropped and never counted as an incorrect answer. Output that exists but is wrong or unparsable (empty, prose, invalid JSON) is a `scored` 0.0, which keeps "the model failed to answer" and "the model answered wrongly" separate.
8. Every applicable metric returns exactly one record per selected request, so scored plus not-scored equals the applicable requests for each metric.
9. Scores are in memory only in M2.1. `details` holds counts, flags and short enumerations, never raw model output.

### Scorers (`niriksha.scorers`)
10. Scorers are pure and deterministic: stored outputs and dataset values in, a record out. They import no provider and no network library (a boundary test checks this), and `niriksha.core` imports no scorer.
11. Metrics: `normalized_exact_match` (QA), `json_parse_validity` and `field_exact_match` (extraction), each at version 0.1.0 and defined in `docs/metrics/`. A change to a rule bumps the version.
12. Text rules use Unicode NFC (never NFKC), optional full case folding, and whitespace collapsing, and keep punctuation and zero-width joiners, which matter in Devanagari and Kannada. Comparison of extracted values does no type coercion, and a number literal that cannot be a finite double (overflow, or a non-zero literal that underflows to zero) makes the output unusable instead of being converted.
13. `score_run(loaded_run)` applies the metrics that match the dataset task, in a fixed order.

## Limitations
- The metrics measure what their documents define (string identity after normalisation, parse validity, per-field value identity), not semantic correctness. They have not been validated against human labels.
- Per-artifact aggregation arrived in M2.2 (ADR 0005). Confidence intervals and comparison across runs are later milestones with a defined protocol; JSON Schema validity arrived in M2.3 (ADR 0006).
- Extra fields in an extraction answer are reported but do not lower `field_exact_match`.
- In M2.1 scores were not persisted; since M2.2 they are (ADR 0005).
- `LoadedRun` is a plain frozen dataclass: constructing one by hand bypasses the checks. Build it only through `load_run`.
- The request-hash dependence on the request schema noted in ADR 0003 still applies to the reader: a schema change makes older runs fail request-hash verification.

## Consequences
- A scorer never sees outputs that do not belong to the dataset and configuration of their manifest.
- Adding a scorer means adding a module, a version, a metric document and an entry in `SCORERS_BY_TASK`; the core does not change.
