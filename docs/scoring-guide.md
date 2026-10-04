# Scoring and reports: usage guide

Everything here works on a completed run and its dataset, offline. No provider is called, nothing is written under `runs/`, and there is no command-line tool yet; the examples use the Python API. Design and limits: [ADR 0005](adr/0005-score-artifacts-and-reports.md). What each metric means: [docs/metrics/](metrics/).

## What you need
- A completed run directory, for example `runs/my-run`, produced by `execute_run` and, if it was interrupted, finished with `resume_run`.
- The dataset directory the run used, for example `datasets/my-dataset`.
- A directory for the scores, for example `scores` (gitignored; it must not be inside the run directory).

## Score a run

```python
from niriksha.scorers.artifacts import available_metrics, score_run_to_artifact

print(available_metrics("short_answer_qa"))  # (('normalized_exact_match', '0.1.0'),)

outcome = score_run_to_artifact(
    "runs/my-run", "datasets/my-dataset", "scores", "normalized_exact_match"
)
print(outcome.path)  # scores/my-run--normalized_exact_match--0.1.0.json
print(outcome.created)  # True the first time; False when a verified artifact was reused
```

The run is loaded and verified first (`RunIntegrityError` if it is incomplete or inconsistent). A QA run has one applicable metric. An extraction run has three, `json_parse_validity`, `field_exact_match` and `json_schema_validity` (checks the output against the dataset's `output_schema`, Draft 2020-12; see [its definition](metrics/json_schema_validity.md)), and each gets its own artifact. Scoring the same outputs with another metric never regenerates anything:

```python
score_run_to_artifact("runs/ex-run", "datasets/ex", "scores", "json_parse_validity")
score_run_to_artifact("runs/ex-run", "datasets/ex", "scores", "field_exact_match")
score_run_to_artifact("runs/ex-run", "datasets/ex", "scores", "json_schema_validity")
```

A metric that does not apply to the run's task, an unknown metric, an unregistered version, and (for `json_schema_validity`) a dataset schema that cannot be used offline all raise `ScoreArtifactError` before anything is written. Versions are never upgraded automatically.

## Scoring again
Scoring a run with a metric that already has an artifact never overwrites it. The existing file is read strictly and compared with what the current code computes from the verified run (its structure, its source including the hashes of the run files, and the exact scorer version). If everything matches it is reused and nothing is written. If it is corrupt, `ScoreArtifactError` is raised; if it is well-formed but differs from the run or from the scorer's output, `ScoreArtifactMismatchError` is raised. In both cases the file is left untouched. To score under changed rules, release a new metric version.

## Read and verify an artifact

```python
from niriksha.core.scorestore import read_artifact
from niriksha.scorers.artifacts import verify_artifact

artifact = read_artifact("scores/my-run--normalized_exact_match--0.1.0.json")  # needs no run
verify_artifact(artifact, "runs/my-run", "datasets/my-dataset")  # raises if it does not match
```

`read_artifact` checks the file on its own (structure, scores, case IDs, `records_sha256`, file name). `verify_artifact` also reloads the run and dataset and recomputes the scores with the scorer registered for the artifact's exact `(metric, version)`. An artifact of an unregistered version can be read and aggregated but not verified.

## Aggregate and report

```python
from niriksha.core.report import (
    aggregate,
    build_report,
    render_report_json,
    render_report_markdown,
)

agg = aggregate(artifact)
print(agg.total_cases, agg.scored, agg.not_scored, agg.mean)

report = build_report(artifact)
open("report.json", "w", encoding="utf-8", newline="\n").write(render_report_json(report))
print(render_report_markdown(report))
```

Reports are built from the artifact alone: you can delete the run and the dataset and still regenerate them. Both renderings are deterministic (no timestamps), so the same artifact always gives the same text. They are not saved automatically.

### Aggregation semantics
- `total_cases` is the number of records; `scored + not_scored == total_cases`.
- `mean` is the mean of the values of scored cases only (`math.fsum(values) / len(values)`, so it does not depend on order).
- A score of `0.0` is a score and is included. A not-scored case has no value and is never counted as 0.0.
- A generation that failed is not scored (reason `generation failed: <kind>`). Output that exists but is wrong or unparsable is scored 0.0.
- With no scored case, `mean` is `None` and `mean_unavailable_reason` is `no_cases` or `no_scored_cases`.
- `not_scored_by_reason` counts not-scored cases by reason, sorted by reason.
- Per-case rows keep the artifact's order (the run's selection order).

## Compare two runs

```python
from niriksha.core.compare import render_comparison_json, render_comparison_markdown
from niriksha.scorers.comparison import RunInput, compare_runs

baseline = RunInput("runs/run-a", "datasets/my-dataset", "scores")
candidate = RunInput("runs/run-b", "datasets/my-dataset", "scores")
comparison = compare_runs(baseline, candidate, "normalized_exact_match")
print(render_comparison_markdown(comparison))
open("comparison.json", "w", encoding="utf-8", newline="\n").write(
    render_comparison_json(comparison)
)
```

Both runs must already be scored with that metric (`score_run_to_artifact`); comparing never creates or changes a file. Each side is loaded, its artifact is read and verified against its run and dataset, and a problem stops the comparison with the existing typed error prefixed `baseline:` or `candidate:`. Reports contain no paths and no timestamps, so the same inputs always give the same bytes. Nothing is saved automatically.

What counts as comparable: the same metric name, version and task; the same dataset (name, version, task, schema version, content hash); the same selection (case count and case-ID hash); and two different runs. Otherwise `IncompatibleRunsError` lists the mismatching fields.

What may differ: provider, requested model, prompt and generation parameters. The report records which do. If the prompt or any parameter differs it warns that a score difference cannot necessarily be attributed to the model.

How to read it:
- A difference is always `candidate - baseline`. Higher means a larger number; the report records the metric's declared direction but never names a winner and makes no significance claim.
- A case that failed or was not scored has no value and is never counted as 0.0. Per case the outcome is one of `both_scored_candidate_higher`, `both_scored_baseline_higher`, `both_scored_equal`, `only_baseline_scored`, `only_candidate_scored` or `neither_scored`; "equal" means exactly equal.
- Each run's mean uses its own scored cases, so the difference of the two means is given only if both runs scored the same cases (`same_scored_cases`); otherwise it is null with the reason `different_scored_cases`. The paired summary uses only the cases both scored and is shown first when the runs scored different cases. With no such case it is unavailable (`no_paired_scored_cases`).
- The paired summary also carries a 95% bootstrap interval for the mean paired difference: `summary.paired.confidence_interval` with `method`, `confidence_level`, `n_resamples`, `seed`, `minimum_paired_cases`, `lower`, `upper` and `unavailable_reason`. It resamples the paired cases (the per-case deltas, never an imputed score) 10,000 times with seed `0` and reports the 2.5th to 97.5th percentile of the resample means. With no paired case, or fewer than 30, the endpoints are null with the reason `no_paired_scored_cases` or `too_few_paired_cases`.
- How to read the interval: it describes how much the mean difference could vary if the evaluated cases were resampled. It does not capture variability across repeated generations, other prompts or cases that were not evaluated, it assumes the cases are independent, and reaching 30 paired cases does not guarantee that it has its nominal coverage (it can be too narrow, especially when most differences are ties). It is an uncertainty estimate, not a significance test, and the report never says whether it contains zero. If the paired differences are all identical the interval has zero width; that means no variation in the sample, not certainty.
- Reports are version 2 (`comparison_version`); a version 1 comparison JSON does not load.
- One metric per comparison; there is no overall score across metrics.

## Artifact file format (version 1)
One JSON document at `<scores_dir>/<run_id>--<metric>--<version>.json`:

| Field | Meaning |
|---|---|
| `artifact_version` | `1`. Other versions are refused. |
| `artifact_id` | `<run_id>--<metric>--<version>`; must equal the file name. |
| `source` | `run_id`; SHA-256 of the exact bytes of the run's `manifest.json` and `results.jsonl`; the dataset's name, version, task, schema version and content hash; the selection's case count and case-ID hash; the prompt hash; the provider name; the requested model. |
| `metric` | `name`, `version`, `task`. |
| `records_sha256` | SHA-256 of the canonical JSON of `records` (the exact bytes are defined in ADR 0005). |
| `records` | One `ScoreRecord` per case, in selection order: `request_id`, `metric`, `metric_version`, `status`, `value`, `reason`, `details`. |

There is no timestamp, so rewriting an artifact gives identical bytes.

## Errors
| Error | Meaning |
|---|---|
| `RunIntegrityError` | The run (or its dataset) failed `load_run`. Raised unchanged. |
| `ScoreArtifactError` | The artifact is missing, malformed, corrupt or of an unsupported version; the metric or version is unknown; the metric does not fit the run's task; the version cannot be verified; `scores_dir` is a regular file or cannot be created; or a directory occupies the artifact path. |
| `IncompatibleRunsError` | (Comparison only.) The two runs cannot be compared; `.differences` names the mismatching fields. |
| `ScoreArtifactMismatchError` | The artifact is well-formed but does not match the run, the dataset or the scorer (including a source run that cannot be verified). A subclass of `ScoreArtifactError`. |

## Known limitations
- The metrics measure string or value identity under rules written down in `docs/metrics/`. They do not measure semantic correctness, and they are not validated against human labels. A mean is a summary of per-case values, not a statement of model quality.
- The hashes detect accidental corruption and drift. They do not authenticate: anyone who can write the files can rewrite an artifact and its hashes consistently.
- A process killed while writing can leave a truncated artifact; it is refused on reading and blocks its name until it is removed by hand.
- There is no ranking or significance test; two runs can be compared descriptively, with a bootstrap interval for the paired mean difference (see "Compare two runs"). Also, `json_schema_validity` does not check `format` (see its definition for the dialect and limits).
