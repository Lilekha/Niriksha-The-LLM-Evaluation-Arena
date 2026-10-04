# ADR 0007: Run comparison

Status: accepted. Date: 2026-10-02.

Note (M2.5): the statements below that say there is no confidence interval describe M2.4; the paired summary gained a bootstrap interval in M2.5 ([ADR 0008](0008-paired-bootstrap-interval.md)). There is still no significance test or winner.

Covers M2.4: a descriptive, deterministic comparison of two completed runs on one metric, built from their verified score artifacts. Offline and read-only. There is still no real provider, CLI, UI, confidence interval or significance test. Usage: [scoring guide](../scoring-guide.md).

## Context
M2.2 made scores persistent and verifiable per run; M2.3 added a third extraction metric. To understand how different models do on the same workload the scores of two runs have to be set side by side, without comparing things that are not comparable and without turning missing scores into zeros. The roadmap places paired statistics and confidence intervals later (M5); this milestone stays descriptive.

## Decisions
1. **Exactly two runs, one metric.** A report has a `baseline` and a `candidate`. Several metrics need several reports. There is no combined score across metrics, and no ranking.
2. **Comparable means:** the same metric name, version and task; the same dataset identity (name, version, task, schema version, content hash); the same selection (case count and case-ID hash, so the same cases in the same order); and two distinct runs (different run IDs; the manifest hash covers the run ID, so one rule suffices). Otherwise `IncompatibleRunsError` names every mismatching field and never echoes values. There are no automatic metric version upgrades.
3. **Verified only.** Each side is loaded with `load_run`, its artifact is read with `read_artifact` and checked with `verify_artifact` against that run and dataset. A missing artifact is never created: the error says to score the run first. Existing typed errors are re-raised with the side (`baseline` or `candidate`) prefixed. Errors and reports contain no filesystem paths.
4. **What may vary is informational.** Provider, requested model, prompt hash and generation parameters may differ and are recorded. Parameters are compared as structured values (field by field, so `0` equals `0.0` and `None` differs from `0`), and the report lists the names of the differing parameters. If the prompt or any parameter differs the report carries a warning that a score difference cannot necessarily be attributed to the model.
5. **Missing is not zero.** A not-scored case has no value and no delta. Per case the outcome is one of: both scored and candidate higher, baseline higher or equal; only the baseline scored; only the candidate scored; neither scored. "Equal" means exactly equal floats. A difference is `candidate - baseline`; "higher" means a larger number only.
6. **Two kinds of difference.** Each run's own mean uses its own scored cases, so a difference of the two means is meaningful only if both scored the same cases (`same_scored_cases`). The overall difference is given only then; otherwise it is null with the reason `different_scored_cases` and is never shown as a number. The paired summary uses only the cases scored in both runs and is always reported; when the runs scored different cases it is shown first. With no paired case it is unavailable (`no_paired_scored_cases`).
7. **Direction is a documented metric property.** Every scorer module declares `DIRECTION` (`higher_is_better` or `lower_is_better`), a test requires it of every registered scorer, and the report records it. Wording stays neutral: no run is called better and there is no winner. The four current metrics are `higher_is_better`. The direction is read from the registry for the artifact's exact metric version; artifact formats do not change.
8. **On demand, not persisted.** A comparison is a pure function of two verified artifacts and the run manifests, so it is generated when needed and not stored. This avoids a second store with its own naming, reuse and verification rules. The report records the artifact IDs and records hashes, the run file hashes, the dataset hash, the case-ID hash, the prompt hash and the parameters, so any saved copy can be checked by regenerating it from the same inputs and comparing bytes. Persisting comparisons is possible later if a need appears.
9. **Layering.** `niriksha.core.compare` holds the models, rules and renderers and imports no scorers, providers or network libraries (the existing boundary test covers it). `niriksha.scorers.comparison.compare_runs` does the loading and verification using existing public interfaces.
10. **Determinism.** No timestamps and no paths. JSON is the exact rendering; Markdown shows six decimals.

## Limitations
- A difference on a handful of cases is descriptive. It is not a significance test, has no confidence interval, and does not show that a model is better in general.
- Comparing runs with different prompts or parameters is allowed and flagged, not blocked.
- Only the requested model recorded in the manifest is compared, not the model identifier a provider returned.
- The hashes detect drift and corruption, not authorship.
- Each run is loaded twice (once for its manifest, once inside `verify_artifact`). That is cheap and avoids private helpers.
- Two runs only; a ranking of several models needs a later decision.

## Consequences
- Artifact, run and dataset formats and hashes are unchanged; one constant was added to each scorer module and the test helper `make_config` gained an optional `model` argument.
- Paired statistics and confidence intervals remain for a later milestone and will need their own protocol.
