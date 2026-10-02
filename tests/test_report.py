"""Aggregation and reports: pure functions of a persisted artifact."""

import itertools
import json
import math
import shutil

import pytest
from pydantic import ValidationError

from niriksha.core.report import (
    NO_CASES,
    NO_SCORED_CASES,
    Aggregate,
    Report,
    aggregate,
    aggregate_records,
    build_report,
    render_report_json,
    render_report_markdown,
)
from niriksha.core.runload import load_run
from niriksha.core.scorestore import build_artifact, read_artifact, source_from_run, write_artifact
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from score_helpers import finished_run

NAME, VERSION = "normalized_exact_match", "0.1.0"


def scored(request_id, value):
    return ScoreRecord(
        request_id=request_id,
        metric=NAME,
        metric_version=VERSION,
        status=ScoreStatus.SCORED,
        value=value,
    )


def not_scored(request_id, reason="generation failed: timeout"):
    return ScoreRecord(
        request_id=request_id,
        metric=NAME,
        metric_version=VERSION,
        status=ScoreStatus.NOT_SCORED,
        reason=reason,
    )


def artifact_with(tmp_path, spec):
    """An artifact over the five tiny_qa cases. ``spec`` is a list of floats or reason strings."""
    run = finished_run(tmp_path)
    loaded = load_run(run.run_dir, run.dataset_dir)
    records = tuple(
        not_scored(case.id, item) if isinstance(item, str) else scored(case.id, item)
        for case, item in zip(loaded.cases, spec, strict=True)
    )
    artifact = build_artifact(source_from_run(loaded), NAME, VERSION, "short_answer_qa", records)
    return run, artifact


def all_keys(value):
    """Every object key at any depth (to show that a report carries no timestamp field)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from all_keys(item)


# -- aggregation ----------------------------------------------------------------------------------


def test_mean_is_over_scored_cases_only_and_not_scored_is_never_zero():
    agg = aggregate_records([scored("a", 1.0), scored("b", 0.0), not_scored("c")])
    assert (agg.total_cases, agg.scored, agg.not_scored) == (3, 2, 1)
    assert agg.mean == 0.5  # (1.0 + 0.0) / 2, not / 3
    assert agg.mean_unavailable_reason is None
    assert agg.not_scored_by_reason == {"generation failed: timeout": 1}


def test_a_genuine_score_of_zero_is_a_score_not_a_missing_value():
    agg = aggregate_records([scored("a", 0.0), scored("b", 0.0)])
    assert agg.mean == 0.0 and agg.mean is not None and agg.scored == 2
    assert agg.mean_unavailable_reason is None


def test_a_perfect_score_has_a_mean_of_one():
    assert aggregate_records([scored("a", 1.0)]).mean == 1.0


@pytest.mark.parametrize(
    ("records", "reason"),
    [
        ([], NO_CASES),
        ([not_scored("a")], NO_SCORED_CASES),
        ([not_scored("a"), not_scored("b", "generation failed: rate_limit")], NO_SCORED_CASES),
    ],
)
def test_without_a_scored_case_the_mean_is_unavailable_not_zero(records, reason):
    agg = aggregate_records(records)
    assert agg.mean is None and agg.mean_unavailable_reason == reason
    assert agg.scored == 0 and agg.total_cases == len(records)


def test_reason_counts_are_complete_and_sorted_by_reason():
    records = [
        not_scored("a", "generation failed: timeout"),
        not_scored("b", "generation failed: rate_limit"),
        not_scored("c", "generation failed: timeout"),
        scored("d", 1.0),
    ]
    agg = aggregate_records(records)
    assert list(agg.not_scored_by_reason.items()) == [
        ("generation failed: rate_limit", 1),
        ("generation failed: timeout", 2),
    ]
    assert agg.scored + agg.not_scored == agg.total_cases == 4


def test_the_mean_does_not_depend_on_the_order_of_the_records():
    values = [0.1, 0.2, 0.3, 0.7, 0.9, 1.0 / 3.0, 0.0, 0.05, 0.15, 0.85]
    means = {
        aggregate_records([scored(f"c{i}", v) for i, v in enumerate(order)]).mean
        for order in itertools.islice(itertools.permutations(values), 0, 200_000, 997)
    }
    assert len(means) == 1  # math.fsum: exactly rounded, so every order gives the same float
    assert means.pop() == pytest.approx(math.fsum(values) / len(values), abs=0.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"scored": 3},  # scored + not_scored != total
        {"mean": None},  # a mean cannot vanish without a reason
        {"mean_unavailable_reason": "no_cases"},  # a mean and a reason together
        {"mean": 1.5},
        {"mean": math.nan},
        {"not_scored_by_reason": {"x": 5}},  # reasons must add up to not_scored
    ],
)
def test_an_inconsistent_aggregate_is_rejected(fields):
    base = {
        "total_cases": 3,
        "scored": 2,
        "not_scored": 1,
        "mean": 0.5,
        "mean_unavailable_reason": None,
        "not_scored_by_reason": {"generation failed: timeout": 1},
    }
    with pytest.raises(ValidationError):
        Aggregate(**{**base, **fields})


# -- reports --------------------------------------------------------------------------------------


def test_the_report_identifies_its_sources_and_keeps_artifact_order(tmp_path):
    run, artifact = artifact_with(tmp_path, [1.0, 0.0, "generation failed: timeout", 0.5, 1.0])
    report = build_report(artifact)
    assert report.artifact_id == artifact.artifact_id
    assert report.artifact_records_sha256 == artifact.records_sha256
    assert report.source.run_id == run.config.run_id
    assert report.source.dataset == artifact.source.dataset
    assert (report.metric, report.metric_version) == (NAME, VERSION)
    assert (report.source.provider, report.source.requested_model) == ("fake", "m")
    assert [r.request_id for r in report.rows] == [r.request_id for r in artifact.records]
    assert [(r.status, r.value, r.reason) for r in report.rows][:3] == [
        (ScoreStatus.SCORED, 1.0, None),
        (ScoreStatus.SCORED, 0.0, None),
        (ScoreStatus.NOT_SCORED, None, "generation failed: timeout"),
    ]
    assert report.aggregate == aggregate(artifact)
    assert report.aggregate.mean == pytest.approx(0.625)


def test_json_and_markdown_reports_are_deterministic(tmp_path):
    _, artifact = artifact_with(tmp_path, [1.0, 0.0, "generation failed: timeout", 0.5, 1.0])
    first, second = build_report(artifact), build_report(artifact)
    assert render_report_json(first) == render_report_json(second)
    assert render_report_markdown(first) == render_report_markdown(second)
    text = render_report_json(first)
    assert text.endswith("\n")
    keys = list(all_keys(json.loads(text)))
    assert not {"created_at", "generated_at", "timestamp", "recorded_at"} & set(keys)
    assert Report.model_validate_json(text) == first  # the JSON report is lossless


def test_markdown_shows_a_zero_score_and_a_missing_one_differently(tmp_path):
    _, artifact = artifact_with(tmp_path, [0.0, 1.0, "generation failed: timeout", 1.0, 1.0])
    markdown = render_report_markdown(build_report(artifact))
    rows = [line for line in markdown.splitlines() if line.startswith("| qa-")]
    assert "| scored | 0.000000 |" in rows[0]
    assert "| not_scored | n/a | generation failed: timeout |" in rows[2]
    assert "- Total cases: 5" in markdown and "- Scored: 4" in markdown
    assert "- Not scored: 1" in markdown and "- Mean over scored cases: 0.750000" in markdown
    assert "- generation failed: timeout: 1" in markdown
    assert f"`{artifact.artifact_id}`" in markdown and artifact.records_sha256 in markdown


def test_markdown_for_an_all_failed_artifact_says_the_mean_is_unavailable(tmp_path):
    _, artifact = artifact_with(tmp_path, ["generation failed: timeout"] * 5)
    markdown = render_report_markdown(build_report(artifact))
    assert "- Mean over scored cases: unavailable (no_scored_cases)" in markdown
    assert "0.000000" not in markdown  # not-scored cases are never shown as zero


def test_markdown_cells_cannot_be_broken_by_pipes_or_line_breaks(tmp_path):
    _, artifact = artifact_with(tmp_path, [1.0, "a | b\nsecond line", 1.0, 1.0, 1.0])
    markdown = render_report_markdown(build_report(artifact))
    assert "| not_scored | n/a | a / b second line |" in markdown
    assert (
        build_report(artifact).rows[1].reason == "a | b\nsecond line"
    )  # the JSON keeps it exactly


def test_a_report_is_reproducible_from_the_persisted_artifact_alone(tmp_path):
    run, artifact = artifact_with(tmp_path, [1.0, 0.0, "generation failed: timeout", 0.5, 1.0])
    path = write_artifact(tmp_path / "scores", artifact)
    expected = render_report_json(build_report(artifact))

    # Without the run, the dataset or any provider: delete them, then read and render.
    shutil.rmtree(run.run_dir)
    shutil.rmtree(run.dataset_dir)
    reloaded = read_artifact(path)
    assert render_report_json(build_report(reloaded)) == expected
    assert render_report_markdown(build_report(reloaded)) == render_report_markdown(
        build_report(artifact)
    )
