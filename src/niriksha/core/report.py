"""Aggregation and reports over a verified score artifact. Pure, offline, deterministic.

Nothing here reads a run, a dataset or a provider: a report is regenerated from the persisted
artifact alone, and rendering the same artifact twice gives identical text (no timestamps).

Aggregate semantics:

- ``total_cases`` is the number of records, ``scored`` those with status ``scored`` and
  ``not_scored`` the rest, so ``scored + not_scored == total_cases``.
- ``mean`` is the mean of the values of the scored records only. It is computed as
  ``math.fsum(values) / len(values)``, so it does not depend on summation order.
- A score of 0.0 is a score and counts in the mean. A not-scored record has no value and is never
  treated as 0.0.
- With no scored record, ``mean`` is ``None`` and ``mean_unavailable_reason`` says why
  (``no_cases`` or ``no_scored_cases``). The mean is never reported as 0.
- ``not_scored_by_reason`` counts the not-scored records by their reason, sorted by reason.

This is a per-artifact summary. There is no ranking, confidence interval or significance claim;
two runs are compared descriptively in niriksha.core.compare. The metrics measure only what
docs/metrics/ defines.
"""

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, model_validator

from niriksha.core.generation import NonBlank
from niriksha.core.runstore import ManifestDataset
from niriksha.core.scorestore import ScoreArtifact
from niriksha.core.scoring import MetricName, MetricVersion, ScoreRecord, ScoreStatus

REPORT_VERSION = 1
NO_CASES = "no_cases"
NO_SCORED_CASES = "no_scored_cases"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Aggregate(_Model):
    total_cases: int
    scored: int
    not_scored: int
    mean: float | None
    mean_unavailable_reason: str | None
    not_scored_by_reason: dict[str, int]

    @model_validator(mode="after")
    def _consistent(self):
        if self.scored + self.not_scored != self.total_cases:
            raise ValueError("scored plus not_scored must equal total_cases")
        if (self.mean is None) != (self.mean_unavailable_reason is not None):
            raise ValueError("mean is None exactly when an unavailable reason is given")
        if self.mean is not None and not (math.isfinite(self.mean) and 0.0 <= self.mean <= 1.0):
            raise ValueError("mean must be a finite number in [0, 1]")
        if sum(self.not_scored_by_reason.values()) != self.not_scored:
            raise ValueError("not_scored_by_reason must add up to not_scored")
        return self


class ReportRow(_Model):
    request_id: NonBlank
    status: ScoreStatus
    value: float | None
    reason: str | None


class ReportSource(_Model):
    run_id: NonBlank
    dataset: ManifestDataset
    provider: NonBlank
    requested_model: NonBlank


class Report(_Model):
    report_version: int
    artifact_id: NonBlank
    artifact_records_sha256: NonBlank
    source: ReportSource
    metric: MetricName
    metric_version: MetricVersion
    aggregate: Aggregate
    rows: tuple[ReportRow, ...]


def aggregate_records(records: Sequence[ScoreRecord]) -> Aggregate:
    """Aggregate any sequence of records. An empty sequence is allowed and has no mean."""
    values = [r.value for r in records if r.status is ScoreStatus.SCORED and r.value is not None]
    reasons: dict[str, int] = {}
    for record in records:
        if record.status is ScoreStatus.NOT_SCORED:
            key = record.reason or ""
            reasons[key] = reasons.get(key, 0) + 1
    if values:
        mean, unavailable = math.fsum(values) / len(values), None
    else:
        mean, unavailable = None, (NO_CASES if not records else NO_SCORED_CASES)
    return Aggregate(
        total_cases=len(records),
        scored=len(values),
        not_scored=len(records) - len(values),
        mean=mean,
        mean_unavailable_reason=unavailable,
        not_scored_by_reason=dict(sorted(reasons.items())),
    )


def aggregate(artifact: ScoreArtifact) -> Aggregate:
    return aggregate_records(artifact.records)


def build_report(artifact: ScoreArtifact) -> Report:
    source = artifact.source
    return Report(
        report_version=REPORT_VERSION,
        artifact_id=artifact.artifact_id,
        artifact_records_sha256=artifact.records_sha256,
        source=ReportSource(
            run_id=source.run_id,
            dataset=source.dataset,
            provider=source.provider,
            requested_model=source.requested_model,
        ),
        metric=artifact.metric.name,
        metric_version=artifact.metric.version,
        aggregate=aggregate(artifact),
        rows=tuple(
            ReportRow(
                request_id=record.request_id,
                status=record.status,
                value=record.value,
                reason=record.reason,
            )
            for record in artifact.records
        ),
    )


def render_report_json(report: Report) -> str:
    """The exact, lossless report: indented JSON in field order with a final newline."""
    return report.model_dump_json(indent=2) + "\n"


def _cell(text: str) -> str:
    """Markdown table cells cannot contain pipes or line breaks; this is presentation only."""
    return " ".join(text.replace("|", "/").split())


def render_report_markdown(report: Report) -> str:
    """A readable rendering. Values are shown with six decimals; the JSON report is exact."""
    agg, src = report.aggregate, report.source
    mean = (
        "unavailable (" + str(agg.mean_unavailable_reason) + ")"
        if agg.mean is None
        else (f"{agg.mean:.6f}")
    )
    lines = [
        f"# Score report: {report.metric} {report.metric_version}",
        "",
        f"- Artifact: `{report.artifact_id}` (records_sha256 `{report.artifact_records_sha256}`)",
        f"- Run: `{src.run_id}`; provider `{_cell(src.provider)}`, requested model "
        f"`{_cell(src.requested_model)}`",
        f"- Dataset: `{src.dataset.name}` {src.dataset.version} (`{src.dataset.content_sha256}`)",
        "",
        "## Aggregate",
        "",
        f"- Total cases: {agg.total_cases}",
        f"- Scored: {agg.scored}",
        f"- Not scored: {agg.not_scored}",
        f"- Mean over scored cases: {mean}",
    ]
    if agg.not_scored_by_reason:
        lines += ["", "Not scored, by reason:", ""]
        lines += [
            f"- {_cell(reason)}: {count}" for reason, count in agg.not_scored_by_reason.items()
        ]
    lines += ["", "## Cases", "", "| request_id | status | value | reason |", "|---|---|---|---|"]
    for row in report.rows:
        value = "n/a" if row.value is None else f"{row.value:.6f}"
        reason = _cell(row.reason or "")
        lines.append(f"| {_cell(row.request_id)} | {row.status.value} | {value} | {reason} |")
    return "\n".join(lines) + "\n"
