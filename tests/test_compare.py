"""niriksha.core.compare on hand-built artifacts: compatibility, semantics, rendering."""

import json

import pytest

from niriksha.core.compare import (
    ComparisonInput,
    IncompatibleRunsError,
    Outcome,
    Paired,
    Summary,
    build_comparison,
    check_compatible,
    render_comparison_json,
    render_comparison_markdown,
)
from niriksha.core.generation import GenerationParams
from niriksha.core.runstore import ManifestDataset, ids_sha256
from niriksha.core.scorestore import ArtifactSelection, ArtifactSource, build_artifact
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.scorers.artifacts import REGISTRY

METRIC, VERSION, TASK = "normalized_exact_match", "0.1.0", "short_answer_qa"
IDS = ["c1", "c2", "c3", "c4"]
NS = "not_scored"


def dataset(**changes):
    base = {
        "name": "ds",
        "version": "1.0.0",
        "task": TASK,
        "schema_version": 1,
        "content_sha256": "d" * 64,
    }
    return ManifestDataset(**{**base, **changes})


def record(request_id, value, metric=METRIC, version=VERSION):
    if value is None:
        return ScoreRecord(
            request_id=request_id,
            metric=metric,
            metric_version=version,
            status=ScoreStatus.NOT_SCORED,
            reason="generation failed: timeout",
        )
    return ScoreRecord(
        request_id=request_id,
        metric=metric,
        metric_version=version,
        status=ScoreStatus.SCORED,
        value=value,
    )


def artifact(
    run_id, values, *, ids=IDS, ds=None, metric=METRIC, version=VERSION, task=TASK, **source
):
    fields = {
        "run_id": run_id,
        "run_manifest_sha256": (run_id[0] * 64),
        "run_results_sha256": "e" * 64,
        "dataset": ds or dataset(),
        "selection": ArtifactSelection(case_count=len(ids), case_ids_sha256=ids_sha256(ids)),
        "prompt_sha256": "f" * 64,
        "provider": "fake",
        "requested_model": f"model-{run_id}",
    }
    fields.update(source)
    records = tuple(record(i, v, metric, version) for i, v in zip(ids, values, strict=True))
    return build_artifact(ArtifactSource(**fields), metric, version, task, records)


def compare(
    a_values,
    b_values,
    a_params=None,
    b_params=None,
    direction="higher_is_better",
    b_source=None,
    **kw,
):
    a = ComparisonInput(artifact("aa", a_values, **kw), a_params or GenerationParams())
    b = ComparisonInput(
        artifact("bb", b_values, **kw, **(b_source or {})), b_params or GenerationParams()
    )
    return build_comparison(a, b, direction)


# -- compatibility --------------------------------------------------------------------------------


def mismatch(**b_changes):
    a = artifact("aa", [1.0, 0.0, 1.0, 1.0])
    b = artifact("bb", [1.0, 0.0, 1.0, 1.0], **b_changes)
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, b)
    return caught.value


@pytest.mark.parametrize(
    ("changes", "field"),
    [
        ({"ds": dataset(content_sha256="1" * 64)}, "dataset.content_sha256"),
        ({"ds": dataset(version="2.0.0")}, "dataset.version"),
        ({"ds": dataset(name="other")}, "dataset.name"),
        ({"ds": dataset(schema_version=2)}, "dataset.schema_version"),
        ({"ids": ["c1", "c2", "c3"]}, "selection.case_count"),
        ({"ids": ["c1", "c2", "c3", "c5"]}, "selection.case_ids_sha256"),
        ({"ids": ["c4", "c3", "c2", "c1"]}, "selection.case_ids_sha256"),
    ],
)
def test_different_dataset_or_selection_is_refused(changes, field):
    n = len(changes.get("ids", IDS))
    a = artifact("aa", [1.0] * 4)
    b = artifact("bb", [1.0] * n, **changes)
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, b)
    assert field in caught.value.differences


def test_a_different_metric_version_or_task_is_refused():
    a = artifact("aa", [1.0] * 4)
    b = artifact("bb", [1.0] * 4, version="0.2.0")
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, b)
    assert caught.value.differences == ("metric.version",)
    c = artifact("cc", [1.0] * 4, metric="json_parse_validity", task="json_extraction")
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, c)
    assert {"metric.name", "metric.task"} <= set(caught.value.differences)


def test_every_mismatch_is_listed_together_and_values_are_not_echoed():
    err = mismatch(ds=dataset(version="9.9.9", content_sha256="1" * 64))
    assert set(err.differences) == {"dataset.version", "dataset.content_sha256"}
    assert "9.9.9" not in str(err) and "1" * 64 not in str(err)


def test_the_same_run_twice_is_refused():
    a = artifact("aa", [1.0] * 4)
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, a)
    assert any("same run" in d for d in caught.value.differences)


def test_runs_with_the_same_id_but_different_files_are_still_refused():
    a = artifact("aa", [1.0] * 4)
    b = artifact("aa", [1.0] * 4, run_manifest_sha256="9" * 64)
    with pytest.raises(IncompatibleRunsError) as caught:
        check_compatible(a, b)
    assert caught.value.differences == ("run_id (the same run)",)


def test_an_unknown_direction_is_refused():
    with pytest.raises(ValueError, match="direction"):
        compare([1.0] * 4, [1.0] * 4, direction="bigger")


# -- semantics ------------------------------------------------------------------------------------


def test_all_six_outcomes_and_deltas_without_imputing_zeros():
    ids = ["c1", "c2", "c3", "c4", "c5", "c6"]
    result = compare(
        [0.0, 1.0, 1.0, 1.0, None, None],
        [1.0, 0.0, 1.0, None, 1.0, None],
        ids=ids,
    )
    assert [r.outcome for r in result.rows] == [
        Outcome.CANDIDATE_HIGHER,
        Outcome.BASELINE_HIGHER,
        Outcome.EQUAL,
        Outcome.ONLY_BASELINE,
        Outcome.ONLY_CANDIDATE,
        Outcome.NEITHER,
    ]
    assert [r.delta for r in result.rows] == [1.0, -1.0, 0.0, None, None, None]
    assert result.summary.outcomes == {o.value: 1 for o in Outcome}
    assert result.rows[3].candidate.value is None and result.rows[3].candidate.reason
    assert not result.summary.same_scored_cases
    # paired: c1..c3 only. baseline 0,1,1; candidate 1,0,1
    paired = result.summary.paired
    assert paired.paired_cases == 3
    assert paired.baseline_mean == pytest.approx(2 / 3)
    assert paired.candidate_mean == pytest.approx(2 / 3)
    assert paired.mean_difference == pytest.approx(0.0)
    # each run still reports its own mean, but no overall difference is given for unequal coverage
    assert result.baseline.aggregate.mean == pytest.approx(0.75)
    assert result.candidate.aggregate.mean == pytest.approx(0.75)
    assert result.summary.mean_difference is None
    assert result.summary.mean_difference_unavailable_reason == "different_scored_cases"


def test_the_difference_is_candidate_minus_baseline_with_exact_floats():
    result = compare([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
    assert result.summary.mean_difference == -0.25
    assert result.summary.paired.mean_difference == -0.25
    assert result.summary.same_scored_cases
    assert result.rows[1].delta == -1.0
    swapped = compare([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert swapped.summary.mean_difference == 0.25


def test_no_paired_scored_cases_makes_the_paired_result_unavailable_with_a_reason():
    result = compare([1.0, 1.0, None, None], [None, None, 0.0, 1.0])
    paired = result.summary.paired
    assert paired.paired_cases == 0
    assert paired.baseline_mean is None and paired.candidate_mean is None
    assert paired.mean_difference is None and paired.unavailable_reason == "no_paired_scored_cases"
    assert result.baseline.aggregate.mean == 1.0 and result.candidate.aggregate.mean == 0.5
    assert result.summary.mean_difference is None  # the runs scored different cases
    assert result.summary.mean_difference_unavailable_reason == "different_scored_cases"
    text = render_comparison_markdown(result)
    assert "Paired comparison (use this one)" in text
    assert "unavailable (no_paired_scored_cases)" in text
    summary = json.loads(render_comparison_json(result))["summary"]
    assert summary["same_scored_cases"] is False
    assert summary["mean_difference"] is None
    assert summary["mean_difference_unavailable_reason"] == "different_scored_cases"
    assert summary["paired"] == {
        "paired_cases": 0,
        "baseline_mean": None,
        "candidate_mean": None,
        "mean_difference": None,
        "unavailable_reason": "no_paired_scored_cases",
    }


def test_a_run_with_nothing_scored_has_no_mean_and_no_difference():
    result = compare([None] * 4, [1.0, 1.0, 0.0, 0.0])
    assert result.baseline.aggregate.mean is None
    assert result.summary.mean_difference is None
    assert result.summary.mean_difference_unavailable_reason == "different_scored_cases"
    assert result.summary.paired.paired_cases == 0
    both = compare([None] * 4, [None] * 4)
    assert both.summary.same_scored_cases and both.summary.outcomes[Outcome.NEITHER] == 4
    assert both.summary.mean_difference is None
    assert both.summary.mean_difference_unavailable_reason == "a_run_has_no_mean"


def test_a_missing_score_is_never_a_zero():
    zero = compare([0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    missing = compare([None, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
    assert zero.rows[0].outcome is Outcome.CANDIDATE_HIGHER and zero.rows[0].delta == 1.0
    assert missing.rows[0].outcome is Outcome.ONLY_CANDIDATE and missing.rows[0].delta is None
    assert missing.summary.paired.paired_cases == 3
    assert missing.summary.paired.mean_difference == 0.0  # the missing case is not in the pairing


def test_unequal_coverage_gives_no_overall_difference_but_keeps_the_paired_one():
    result = compare([None, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0])
    summary = json.loads(render_comparison_json(result))["summary"]
    assert summary["same_scored_cases"] is False
    assert summary["mean_difference"] is None  # never a number that looks comparable
    assert summary["mean_difference_unavailable_reason"] == "different_scored_cases"
    paired = summary["paired"]
    assert paired["paired_cases"] == 3 and paired["unavailable_reason"] is None
    assert paired["baseline_mean"] == 1.0
    assert paired["candidate_mean"] == pytest.approx(2 / 3)
    assert paired["mean_difference"] == pytest.approx(-1 / 3)
    # each run still reports its own mean over the cases it scored
    assert (result.baseline.aggregate.mean, result.candidate.aggregate.mean) == (
        1.0,
        pytest.approx(0.75),
    )


def test_a_summary_cannot_pair_unequal_coverage_with_a_numeric_overall_difference():
    paired = Paired(
        paired_cases=1,
        baseline_mean=1.0,
        candidate_mean=0.0,
        mean_difference=-1.0,
        unavailable_reason=None,
    )
    with pytest.raises(ValueError, match="not comparable"):
        Summary(
            same_scored_cases=False,
            paired=paired,
            mean_difference=-0.5,
            mean_difference_unavailable_reason=None,
            outcomes={},
        )


def test_different_scored_cases_make_the_paired_comparison_prominent():
    text = render_comparison_markdown(compare([None, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]))
    assert text.index("## Paired comparison (use this one)") < text.index("## Per-case outcomes")
    assert "scored different cases" in text and "no overall difference is given" in text
    assert "overall means (candidate - baseline): unavailable (different_scored_cases)" in text
    same = render_comparison_markdown(compare([1.0] * 4, [1.0, 1.0, 1.0, 0.0]))
    assert "Paired comparison (use this one)" not in same and "## Difference" in same


# -- what varies ----------------------------------------------------------------------------------


def test_what_varies_is_recorded_and_the_prompt_or_params_trigger_the_warning():
    base = compare([1.0] * 4, [1.0] * 4)
    assert base.varying.requested_model and not base.varying.provider and not base.varying.prompt
    assert base.varying.params == () and not base.confounded
    assert "cannot necessarily be attributed" not in render_comparison_markdown(base)

    prompt = compare([1.0] * 4, [1.0] * 4, b_source={"prompt_sha256": "1" * 64})
    assert prompt.varying.prompt and prompt.confounded
    assert "cannot necessarily be attributed to the model" in render_comparison_markdown(prompt)

    params = compare(
        [1.0] * 4,
        [1.0] * 4,
        GenerationParams(temperature=0.0, stop=("a",)),
        GenerationParams(temperature=0.7, stop=("a",)),
    )
    assert params.varying.params == ("temperature",) and params.confounded

    unset = compare([1.0] * 4, [1.0] * 4, GenerationParams(), GenerationParams(seed=0))
    assert unset.varying.params == ("seed",)  # not sent versus 0 is a difference


def test_nothing_varying_is_stated():
    a = ComparisonInput(artifact("aa", [1.0] * 4, requested_model="m"), GenerationParams())
    b = ComparisonInput(artifact("bb", [1.0] * 4, requested_model="m"), GenerationParams())
    result = build_comparison(a, b, "higher_is_better")
    assert "Nothing varies" in render_comparison_markdown(result)


def test_parameters_are_compared_as_structured_values_not_text():
    a = GenerationParams(temperature=0.0, stop=("x", "y"))
    b = GenerationParams(temperature=0, stop=("x", "y"))  # an int zero is the same value
    assert compare([1.0] * 4, [1.0] * 4, a, b).varying.params == ()
    reordered = GenerationParams(temperature=0.0, stop=("y", "x"))
    assert compare([1.0] * 4, [1.0] * 4, a, reordered).varying.params == ("stop",)


# -- rendering ------------------------------------------------------------------------------------


def test_the_json_is_lossless_and_stable_and_the_markdown_is_deterministic():
    result = compare([1.0, 0.0, None, 1.0], [1.0, 1.0, 0.0, None])
    first, second = render_comparison_json(result), render_comparison_json(result)
    assert first == second and first.endswith("\n")
    document = json.loads(first)
    assert document["comparison_version"] == 1
    assert document["metric"]["direction"] == "higher_is_better"
    assert list(document["summary"]["outcomes"]) == [o.value for o in Outcome]
    assert type(result).model_validate_json(first) == result
    assert render_comparison_markdown(result) == render_comparison_markdown(result)


def test_the_report_makes_no_winner_or_significance_claim():
    text = render_comparison_markdown(compare([1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0])).lower()
    assert "winner" in text and "not a significance test" in text
    for banned in ("better than", "outperform", "significantly", "p-value"):
        assert banned not in text


def test_unicode_and_markdown_special_characters_are_rendered_safely():
    ids = ["ಕ|1", "हिं\n2", "c3", "c4"]
    result = compare([1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, None], ids=ids)
    text = render_comparison_markdown(result)
    assert "ಕ/1" in text and "हिं 2" in text
    assert json.loads(render_comparison_json(result))["rows"][0]["request_id"] == "ಕ|1"


def test_lower_is_better_is_recorded_without_changing_the_wording_or_numbers():
    low = compare([1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], direction="lower_is_better")
    high = compare([1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0])
    assert low.metric.direction == "lower_is_better"
    assert low.summary == high.summary and low.rows == high.rows
    assert "lower is better" in render_comparison_markdown(low)


# -- every registered scorer declares a direction -------------------------------------------------


def test_every_registered_scorer_declares_a_valid_direction():
    assert REGISTRY
    for key, module in REGISTRY.items():
        assert module.DIRECTION in ("higher_is_better", "lower_is_better"), key
