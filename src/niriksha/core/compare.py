"""Comparison of two verified score artifacts. Pure, offline, deterministic.

This module only compares artifacts it is given. The caller must have verified them against their
runs and datasets first (``niriksha.scorers.comparison.compare_runs`` does). It reads no file, runs
no scorer and calls no provider, and rendering the same comparison twice gives identical text
(no timestamps, no paths).

Comparable means the same metric (name, version, task), the same dataset identity and the same
selection (case count and case-ID hash, so the same cases in the same order), and two distinct
runs. Anything else raises ``IncompatibleRunsError`` naming every mismatching field, never values.
Provider, requested model, prompt and generation parameters may differ; the report records which
do, and warns when the prompt or parameters differ because a score difference then cannot
necessarily be attributed to the model.

Semantics:

- A difference is always ``candidate - baseline``. "Higher" means a larger number only. The metric's
  ``direction`` is recorded but no run is called better or a winner, and nothing here is a
  significance test.
- Two values are equal only if they are exactly equal floats. There is no tolerance.
- A not-scored case has no value. It is never treated as 0.0 and has no per-case delta.
- The overall means of the two runs average different cases unless both scored the same cases
  (``same_scored_cases``), so the overall difference is given only then; otherwise it is null
  with the reason ``different_scored_cases``. Each run's own mean is always reported. The paired
  summary uses only the cases scored in both runs and is always reported; with no such case it
  is unavailable, with a reason.
- The paired summary carries a 95% paired percentile bootstrap interval for the mean of the paired
  differences (``niriksha.core.bootstrap`` defines the method, seed and resample count). It uses
  only the existing per-case deltas, never an imputed score, and is unavailable with fewer than
  ``MINIMUM_PAIRED_CASES`` paired cases. It reflects resampling of the evaluated cases only, not
  repeated-generation variability or performance on unseen cases, and the minimum does not
  guarantee its nominal coverage. A zero-width interval means the sample showed no variation, not
  certainty.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from niriksha.core.bootstrap import (
    CONFIDENCE_PERCENT,
    METHOD,
    MINIMUM_PAIRED_CASES,
    N_RESAMPLES,
    SEED,
    paired_bootstrap_interval,
)
from niriksha.core.dataset import Task
from niriksha.core.generation import GenerationParams, NonBlank
from niriksha.core.report import Aggregate, aggregate_records
from niriksha.core.runstore import ManifestDataset
from niriksha.core.scorestore import ArtifactSelection, ScoreArtifact
from niriksha.core.scoring import MetricName, MetricVersion, ScoreRecord, ScoreStatus

COMPARISON_VERSION = 2
NO_PAIRED_SCORED_CASES = "no_paired_scored_cases"
TOO_FEW_PAIRED_CASES = "too_few_paired_cases"
MEAN_UNAVAILABLE = "a_run_has_no_mean"
DIFFERENT_SCORED_CASES = "different_scored_cases"


def _cell(text: str) -> str:
    """Markdown table cells cannot contain pipes or line breaks; this is presentation only."""
    return " ".join(text.replace("|", "/").split())


class IncompatibleRunsError(Exception):
    """Two runs cannot be compared. ``differences`` names the fields that do not match."""

    def __init__(self, differences: list[str]):
        self.differences = tuple(differences)
        super().__init__("the runs are not comparable; mismatching: " + ", ".join(differences))


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


@dataclass(frozen=True)
class ComparisonInput:
    """A verified artifact and the generation parameters of the run it was scored from."""

    artifact: ScoreArtifact
    params: GenerationParams


class Outcome(StrEnum):
    CANDIDATE_HIGHER = "both_scored_candidate_higher"
    BASELINE_HIGHER = "both_scored_baseline_higher"
    EQUAL = "both_scored_equal"
    ONLY_BASELINE = "only_baseline_scored"
    ONLY_CANDIDATE = "only_candidate_scored"
    NEITHER = "neither_scored"


class MetricInfo(_Model):
    name: MetricName
    version: MetricVersion
    task: Task
    direction: Literal["higher_is_better", "lower_is_better"]


class ComparedRun(_Model):
    run_id: NonBlank
    provider: NonBlank
    requested_model: NonBlank
    prompt_sha256: NonBlank
    params: GenerationParams
    artifact_id: NonBlank
    artifact_records_sha256: NonBlank
    run_manifest_sha256: NonBlank
    run_results_sha256: NonBlank
    aggregate: Aggregate


class Varying(_Model):
    provider: bool
    requested_model: bool
    prompt: bool
    params: tuple[str, ...]  # names of the generation parameters whose values differ, sorted


class Side(_Model):
    status: ScoreStatus
    value: float | None
    reason: str | None


class CaseComparison(_Model):
    request_id: NonBlank
    outcome: Outcome
    baseline: Side
    candidate: Side
    delta: float | None  # candidate - baseline, only when both are scored

    @model_validator(mode="after")
    def _consistent(self):
        both = self.baseline.status is self.candidate.status is ScoreStatus.SCORED
        if both != (self.delta is not None):
            raise ValueError("a delta exists exactly when both sides are scored")
        return self


class ConfidenceInterval(_Model):
    """Interval for the mean paired difference. Endpoints are null with a reason if unavailable."""

    method: Literal["paired_percentile_bootstrap_v1"]
    confidence_level: float = Field(allow_inf_nan=False)
    n_resamples: int
    seed: int
    minimum_paired_cases: int
    lower: float | None = Field(allow_inf_nan=False)
    upper: float | None = Field(allow_inf_nan=False)
    unavailable_reason: Literal["no_paired_scored_cases", "too_few_paired_cases"] | None

    @model_validator(mode="after")
    def _consistent(self):
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper are both given or both null")
        if (self.lower is None) != (self.unavailable_reason is not None):
            raise ValueError("a reason is given exactly when there is no interval")
        if self.lower is not None and self.upper is not None:
            if not -1.0 <= self.lower <= self.upper <= 1.0:
                raise ValueError("an interval for differences of values in [0, 1] lies in [-1, 1]")
        return self


class Paired(_Model):
    paired_cases: int
    baseline_mean: float | None
    candidate_mean: float | None
    mean_difference: float | None
    unavailable_reason: str | None
    confidence_interval: ConfidenceInterval

    @model_validator(mode="after")
    def _consistent(self):
        available = self.mean_difference is not None
        if available != (self.unavailable_reason is None):
            raise ValueError("unavailable_reason is given exactly when there is no difference")
        return self


class Summary(_Model):
    same_scored_cases: bool
    paired: Paired
    mean_difference: float | None  # overall means; None unless same_scored_cases
    mean_difference_unavailable_reason: str | None
    outcomes: dict[str, int]

    @model_validator(mode="after")
    def _consistent(self):
        if (self.mean_difference is None) != (self.mean_difference_unavailable_reason is not None):
            raise ValueError("a reason is given exactly when there is no mean difference")
        if not self.same_scored_cases and self.mean_difference is not None:
            raise ValueError("overall means of different scored cases are not comparable")
        return self


class Comparison(_Model):
    comparison_version: int
    metric: MetricInfo
    dataset: ManifestDataset
    selection: ArtifactSelection
    baseline: ComparedRun
    candidate: ComparedRun
    varying: Varying
    confounded: bool  # the prompt or the parameters differ
    summary: Summary
    rows: tuple[CaseComparison, ...]


# -- compatibility --------------------------------------------------------------------------------


def _differences(baseline: ScoreArtifact, candidate: ScoreArtifact) -> list[str]:
    found = [
        f"metric.{name}"
        for name in type(baseline.metric).model_fields
        if getattr(baseline.metric, name) != getattr(candidate.metric, name)
    ]
    found += [
        f"dataset.{name}"
        for name in type(baseline.source.dataset).model_fields
        if getattr(baseline.source.dataset, name) != getattr(candidate.source.dataset, name)
    ]
    found += [
        f"selection.{name}"
        for name in type(baseline.source.selection).model_fields
        if getattr(baseline.source.selection, name) != getattr(candidate.source.selection, name)
    ]
    if baseline.source.run_id == candidate.source.run_id:
        found.append("run_id (the same run)")
    return found


def check_compatible(baseline: ScoreArtifact, candidate: ScoreArtifact) -> None:
    found = _differences(baseline, candidate)
    if found:
        raise IncompatibleRunsError(found)


# -- building -------------------------------------------------------------------------------------


def _side(record: ScoreRecord) -> Side:
    return Side(status=record.status, value=record.value, reason=record.reason)


def _case(baseline: ScoreRecord, candidate: ScoreRecord) -> CaseComparison:
    b, c = baseline.value, candidate.value
    scored_b, scored_c = (
        baseline.status is ScoreStatus.SCORED,
        candidate.status is ScoreStatus.SCORED,
    )
    delta = None
    if scored_b and scored_c:
        assert b is not None and c is not None
        delta = c - b
        outcome = (
            Outcome.EQUAL
            if c == b
            else Outcome.CANDIDATE_HIGHER
            if c > b
            else Outcome.BASELINE_HIGHER
        )
    elif scored_b:
        outcome = Outcome.ONLY_BASELINE
    elif scored_c:
        outcome = Outcome.ONLY_CANDIDATE
    else:
        outcome = Outcome.NEITHER
    return CaseComparison(
        request_id=baseline.request_id,
        outcome=outcome,
        baseline=_side(baseline),
        candidate=_side(candidate),
        delta=delta,
    )


def _compared_run(item: ComparisonInput) -> ComparedRun:
    source = item.artifact.source
    return ComparedRun(
        run_id=source.run_id,
        provider=source.provider,
        requested_model=source.requested_model,
        prompt_sha256=source.prompt_sha256,
        params=item.params,
        artifact_id=item.artifact.artifact_id,
        artifact_records_sha256=item.artifact.records_sha256,
        run_manifest_sha256=source.run_manifest_sha256,
        run_results_sha256=source.run_results_sha256,
        aggregate=aggregate_records(item.artifact.records),
    )


def _difference(baseline: float | None, candidate: float | None) -> float | None:
    return None if baseline is None or candidate is None else candidate - baseline


def _interval(deltas: list[float]) -> ConfidenceInterval:
    """The interval over the paired deltas (selection order), or why there is none."""
    reason = (
        NO_PAIRED_SCORED_CASES
        if not deltas
        else TOO_FEW_PAIRED_CASES
        if len(deltas) < MINIMUM_PAIRED_CASES
        else None
    )
    lower, upper = (None, None) if reason else paired_bootstrap_interval(deltas)
    return ConfidenceInterval(
        method=METHOD,
        confidence_level=CONFIDENCE_PERCENT / 100,
        n_resamples=N_RESAMPLES,
        seed=SEED,
        minimum_paired_cases=MINIMUM_PAIRED_CASES,
        lower=lower,
        upper=upper,
        unavailable_reason=reason,
    )


def build_comparison(
    baseline: ComparisonInput, candidate: ComparisonInput, direction: str
) -> Comparison:
    """Compare two verified artifacts. Raises ``IncompatibleRunsError`` if they are not comparable.

    ``direction`` is the metric's declared direction (``higher_is_better`` or ``lower_is_better``).
    """
    check_compatible(baseline.artifact, candidate.artifact)
    a, c = baseline.artifact, candidate.artifact
    rows = tuple(_case(b, k) for b, k in zip(a.records, c.records, strict=True))
    paired = [
        (b, k)
        for b, k in zip(a.records, c.records, strict=True)
        if b.status is ScoreStatus.SCORED and k.status is ScoreStatus.SCORED
    ]
    paired_b = aggregate_records([b for b, _ in paired]).mean
    paired_c = aggregate_records([k for _, k in paired]).mean
    paired_diff = _difference(paired_b, paired_c)
    run_b, run_c = _compared_run(baseline), _compared_run(candidate)
    same_cases = all(
        (b.status is ScoreStatus.SCORED) == (k.status is ScoreStatus.SCORED)
        for b, k in zip(a.records, c.records, strict=True)
    )
    # The overall means average different cases when coverage differs, so no difference is given.
    overall = _difference(run_b.aggregate.mean, run_c.aggregate.mean) if same_cases else None
    params_b = baseline.params.model_dump(mode="json")
    params_c = candidate.params.model_dump(mode="json")
    varying = Varying(
        provider=a.source.provider != c.source.provider,
        requested_model=a.source.requested_model != c.source.requested_model,
        prompt=a.source.prompt_sha256 != c.source.prompt_sha256,
        params=tuple(sorted(name for name in params_b if params_b[name] != params_c[name])),
    )
    summary = Summary(
        same_scored_cases=same_cases,
        paired=Paired(
            paired_cases=len(paired),
            baseline_mean=paired_b,
            candidate_mean=paired_c,
            mean_difference=paired_diff,
            unavailable_reason=None if paired_diff is not None else NO_PAIRED_SCORED_CASES,
            confidence_interval=_interval([row.delta for row in rows if row.delta is not None]),
        ),
        mean_difference=overall,
        mean_difference_unavailable_reason=(
            None
            if overall is not None
            else MEAN_UNAVAILABLE
            if same_cases
            else DIFFERENT_SCORED_CASES
        ),
        outcomes={o.value: sum(1 for row in rows if row.outcome is o) for o in Outcome},
    )
    for value in (paired_diff, overall, *(row.delta for row in rows)):
        if value is not None and not (math.isfinite(value) and -1.0 <= value <= 1.0):
            raise ValueError("a difference of two values in [0, 1] must lie in [-1, 1]")
    return Comparison(
        comparison_version=COMPARISON_VERSION,
        metric=MetricInfo(
            name=a.metric.name, version=a.metric.version, task=a.metric.task, direction=direction
        ),
        dataset=a.source.dataset,
        selection=a.source.selection,
        baseline=run_b,
        candidate=run_c,
        varying=varying,
        confounded=varying.prompt or bool(varying.params),
        summary=summary,
        rows=rows,
    )


# -- rendering ------------------------------------------------------------------------------------


def render_comparison_json(comparison: Comparison) -> str:
    """The exact, lossless comparison: indented JSON in field order with a final newline."""
    return comparison.model_dump_json(indent=2) + "\n"


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "unavailable"
    return f"{value:+.6f}" if signed else f"{value:.6f}"


def _mean_cell(run: ComparedRun) -> str:
    agg = run.aggregate
    return f"unavailable ({agg.mean_unavailable_reason})" if agg.mean is None else _number(agg.mean)


def _varying_text(v: Varying) -> str:
    parts = []
    if v.provider:
        parts.append("provider")
    if v.requested_model:
        parts.append("model")
    if v.prompt:
        parts.append("prompt")
    if v.params:
        parts.append("generation parameters (" + ", ".join(v.params) + ")")
    return ", ".join(parts)


def _interval_lines(paired: Paired) -> list[str]:
    ci = paired.confidence_interval
    label = f"- {round(ci.confidence_level * 100)}% bootstrap interval for the mean difference: "
    if ci.lower is None or ci.upper is None:
        reason = (
            f"{ci.unavailable_reason}: fewer than {ci.minimum_paired_cases} paired cases"
            if ci.unavailable_reason == TOO_FEW_PAIRED_CASES
            else ci.unavailable_reason
        )
        return [label + f"unavailable ({reason})"]
    lines = [
        label + f"[{_number(ci.lower, signed=True)}, {_number(ci.upper, signed=True)}] "
        f"({ci.n_resamples:,} resamples of the {paired.paired_cases} paired cases, "
        f"seed {ci.seed}, method `{ci.method}`)"
    ]
    if ci.lower == ci.upper:
        lines.append(
            "- The interval has zero width: the paired differences did not vary in this sample. "
            "That does not imply certainty."
        )
    return lines


def render_comparison_markdown(comparison: Comparison) -> str:
    """A readable rendering. Values show six decimals; the JSON comparison is exact."""
    m, b, c, s = comparison.metric, comparison.baseline, comparison.candidate, comparison.summary
    n = comparison.selection.case_count
    direction = m.direction.replace("_", " ")
    lines = [
        f"# Comparison: {m.name} {m.version}",
        "",
        "Descriptive only: no run is declared the winner, and this is not a significance test.",
        "",
        f"- Metric direction: {direction}. Below, higher means a larger number only.",
        f"- Dataset: `{comparison.dataset.name}` {comparison.dataset.version} "
        f"(`{comparison.dataset.content_sha256}`), {n} cases",
    ]
    for label, run in (("Baseline", b), ("Candidate", c)):
        lines.append(
            f"- {label}: run `{run.run_id}`; provider `{_cell(run.provider)}`, requested model "
            f"`{_cell(run.requested_model)}`; artifact `{run.artifact_id}` "
            f"(records_sha256 `{run.artifact_records_sha256}`)"
        )
    lines += ["", "## Conditions", ""]
    varying = _varying_text(comparison.varying)
    lines.append(
        f"- Varying between the runs: {varying}."
        if varying
        else "- Nothing varies between the runs (same provider, model, prompt and parameters)."
    )
    if comparison.confounded:
        lines.append(
            "- **Warning:** the prompt or the generation parameters differ, so a difference in "
            "scores cannot necessarily be attributed to the model."
        )
    lines += ["", "## Aggregates", "", "| | Baseline | Candidate |", "|---|---|---|"]
    lines += [
        f"| Total cases | {b.aggregate.total_cases} | {c.aggregate.total_cases} |",
        f"| Scored | {b.aggregate.scored} | {c.aggregate.scored} |",
        f"| Not scored | {b.aggregate.not_scored} | {c.aggregate.not_scored} |",
        f"| Mean over scored cases | {_mean_cell(b)} | {_mean_cell(c)} |",
    ]
    for label, run in (("Baseline", b), ("Candidate", c)):
        if run.aggregate.not_scored_by_reason:
            lines += ["", f"{label}, not scored by reason:", ""]
            lines += [
                f"- {_cell(reason)}: {count}"
                for reason, count in run.aggregate.not_scored_by_reason.items()
            ]
    paired = [
        "",
        f"Cases scored in both runs: {s.paired.paired_cases}",
        f"- Baseline mean: {_number(s.paired.baseline_mean)}",
        f"- Candidate mean: {_number(s.paired.candidate_mean)}",
        "- Difference (candidate - baseline): "
        + (
            f"unavailable ({s.paired.unavailable_reason})"
            if s.paired.mean_difference is None
            else _number(s.paired.mean_difference, signed=True)
        ),
        *_interval_lines(s.paired),
    ]
    overall = "- Difference of the overall means (candidate - baseline): " + (
        f"unavailable ({s.mean_difference_unavailable_reason})"
        if s.mean_difference is None
        else _number(s.mean_difference, signed=True)
    )
    if s.same_scored_cases:
        lines += ["", "## Difference", "", overall, *paired]
    else:
        lines += [
            "",
            "## Paired comparison (use this one)",
            "",
            f"**The runs scored different cases (baseline {b.aggregate.scored}, candidate "
            f"{c.aggregate.scored} of {n}), so their overall means are not directly "
            "comparable and no overall difference is given. Compare them on the cases both "
            "scored:**",
            *paired,
            "",
            overall,
        ]
    lines += ["", "## Per-case outcomes", ""]
    lines += [f"- {name}: {count}" for name, count in s.outcomes.items()]
    lines += [
        "",
        "## Cases",
        "",
        "| request_id | outcome | baseline | candidate | delta |",
        "|---|---|---|---|---|",
    ]
    for row in comparison.rows:
        delta = "n/a" if row.delta is None else _number(row.delta, signed=True)
        lines.append(
            f"| {_cell(row.request_id)} | {row.outcome.value} | {_side_cell(row.baseline)} "
            f"| {_side_cell(row.candidate)} | {delta} |"
        )
    ci = s.paired.confidence_interval
    lines += [
        "",
        f"Interval method: `{ci.method}`, a percentile bootstrap of the mean paired difference "
        f"({ci.n_resamples:,} resamples, seed {ci.seed}, computed only with at least "
        f"{ci.minimum_paired_cases} paired cases). It resamples the evaluated cases only: it does "
        "not capture variability across repeated generations, other prompts or cases that were "
        "not evaluated, and the minimum does not guarantee its nominal coverage. It is an "
        "uncertainty estimate, not a significance test.",
        "",
        f"This is a descriptive difference on {n} cases. It makes no claim of statistical "
        "significance and does not show that either model is better in general.",
    ]
    return "\n".join(lines) + "\n"


def _side_cell(side: Side) -> str:
    if side.value is None:
        return f"not scored ({_cell(side.reason or '')})"
    return _number(side.value)
