"""Scoring a stored run end to end: load_run -> score_run. Offline, deterministic, no provider."""

import pytest

import niriksha.core.execution as execution_module
import niriksha.core.runner as runner_module
from ds_helpers import FIXTURES, qa, write_dataset
from e2e_support import CrashingProvider, make_config
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run
from niriksha.core.generation import FailureKind
from niriksha.core.runload import load_run
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.providers.fake import FakeProvider
from niriksha.scorers.run import SCORERS_BY_TASK, score_run
from score_helpers import deterministic, finished_run, snapshot

QA_IDS = ["qa-en-001", "qa-en-002", "qa-hi-001", "qa-kn-001", "qa-hi-002"]


def qa_script():
    """Outputs for the five tiny_qa cases, derived from the dataset's own accepted answers."""
    cases = {c.id: c for c in load_dataset(FIXTURES / "tiny_qa").cases}
    return [
        "  paris  ",  # English: right after case and whitespace normalisation
        "366 days",  # English: a near miss
        cases["qa-hi-001"].answers[1],  # Hindi: the second accepted answer
        cases["qa-kn-001"].answers[0],  # Kannada
        FailureKind.TIMEOUT,  # romanised Hindi: the generation failed
    ]


def extraction_script():
    return [
        '{"name": "Asha", "age": 30}',
        'Sure! {"name": "Ravi", "age": 25}',  # prose around the JSON: not valid output
        FailureKind.RATE_LIMIT,
    ]


def by_metric(records):
    grouped = {}
    for record in records:
        grouped.setdefault(record.metric, []).append(record)
    return grouped


def loaded(run):
    return load_run(run.run_dir, run.dataset_dir)


def test_qa_run_scores_english_hindi_and_kannada_and_marks_the_failure_not_scored(tmp_path):
    records = score_run(loaded(finished_run(tmp_path, script=qa_script())))
    assert [r.request_id for r in records] == QA_IDS
    assert {r.metric for r in records} == {"normalized_exact_match"}
    assert [r.value for r in records] == [1.0, 0.0, 1.0, 1.0, None]
    assert [r.status for r in records] == [ScoreStatus.SCORED] * 4 + [ScoreStatus.NOT_SCORED]
    assert records[2].details["matched_answer_index"] == 1  # Hindi matched its second answer
    assert records[4].reason == "generation failed: timeout"


def test_extraction_run_scores_both_metrics_per_case_in_a_fixed_order(tmp_path):
    run = finished_run(tmp_path, fixture="tiny_extraction", script=extraction_script())
    records = score_run(loaded(run))
    assert [(r.request_id, r.metric) for r in records] == [
        ("ex-en-001", "json_parse_validity"),
        ("ex-en-001", "field_exact_match"),
        ("ex-en-002", "json_parse_validity"),
        ("ex-en-002", "field_exact_match"),
        ("ex-hi-001", "json_parse_validity"),
        ("ex-hi-001", "field_exact_match"),
    ]
    assert [r.value for r in records] == [1.0, 1.0, 0.0, 0.0, None, None]
    assert records[2].details == {"top_level_type": None, "failure": "invalid_json"}
    assert records[4].reason == "generation failed: rate_limit"


def test_devanagari_extraction_is_scored_field_by_field(tmp_path):
    cases = {c.id: c for c in load_dataset(FIXTURES / "tiny_extraction").cases}
    name = cases["ex-hi-001"].expected["name"]
    script = [
        '{"name": "Asha", "age": 30}',
        '{"name": "Ravi", "age": 25}',
        f'{{"name": "{name}", "age": "30"}}',  # the right name; the age as a string, not a number
    ]
    records = by_metric(
        score_run(loaded(finished_run(tmp_path, fixture="tiny_extraction", script=script)))
    )
    assert [r.value for r in records["json_parse_validity"]] == [1.0, 1.0, 1.0]
    assert [r.value for r in records["field_exact_match"]] == [1.0, 1.0, 0.5]


def test_scoring_the_same_run_twice_gives_identical_records(tmp_path):
    run = finished_run(tmp_path, script=qa_script())
    first = score_run(loaded(run))
    assert first == score_run(loaded(run))
    assert first == score_run(loaded(run))


def test_scoring_makes_no_provider_call_and_reads_only_stored_data(tmp_path, monkeypatch):
    run = finished_run(tmp_path, script=qa_script())
    before = snapshot(run.run_dir, run.dataset_dir)

    def explode(*args, **kwargs):
        raise AssertionError("a provider was called while scoring")

    # Every route to a provider is rigged to fail: the fake provider, the crash double, and the
    # runner entry points that would call one. tests/conftest.py already blocks the network.
    monkeypatch.setattr(FakeProvider, "generate", explode)
    monkeypatch.setattr(CrashingProvider, "generate", explode)
    monkeypatch.setattr(runner_module, "run_one", explode)
    monkeypatch.setattr(runner_module, "run_requests", explode)
    monkeypatch.setattr(execution_module, "run_one", explode)

    assert len(score_run(loaded(run))) == 5
    assert snapshot(run.run_dir, run.dataset_dir) == before  # scoring wrote nothing


@pytest.mark.parametrize("fixture", ["tiny_qa", "tiny_extraction"])
def test_every_request_gets_one_record_per_applicable_metric(tmp_path, fixture):
    run_view = loaded(finished_run(tmp_path, fixture=fixture))
    records = score_run(run_view)
    metrics = [s.METRIC for s in SCORERS_BY_TASK[run_view.dataset.meta.task]]
    assert len(records) == len(run_view.cases) * len(metrics)
    for metric in metrics:
        assert [r.request_id for r in by_metric(records)[metric]] == [c.id for c in run_view.cases]


def test_scored_plus_not_scored_equals_the_applicable_requests_for_each_metric(tmp_path):
    kinds = list(FailureKind)
    qa_run = finished_run(tmp_path / "qa", script=["Paris", kinds[0], "366", kinds[3], "New Delhi"])
    qa_view = loaded(qa_run)
    for metric, rows in by_metric(score_run(qa_view)).items():
        scored = [r for r in rows if r.status is ScoreStatus.SCORED]
        unscored = [r for r in rows if r.status is ScoreStatus.NOT_SCORED]
        assert len(scored) + len(unscored) == len(qa_view.cases) == 5, metric
        assert len(unscored) == 2, metric

    script = ['{"name": "Asha", "age": 30}', kinds[1], "not json at all"]
    ex_view = loaded(finished_run(tmp_path / "ex", fixture="tiny_extraction", script=script))
    for metric, rows in by_metric(score_run(ex_view)).items():
        scored = [r for r in rows if r.status is ScoreStatus.SCORED]
        unscored = [r for r in rows if r.status is ScoreStatus.NOT_SCORED]
        assert len(scored) + len(unscored) == len(ex_view.cases) == 3, metric
        assert len(unscored) == 1, metric  # only the failed generation; "not json" scores 0.0


def test_every_failure_kind_becomes_a_not_scored_record(tmp_path):
    kinds = list(FailureKind)
    directory = write_dataset(
        tmp_path, [qa(id=f"f{i}") for i in range(len(kinds))], dirname="failures"
    )
    config = make_config("failures")
    execute_run(
        config,
        load_dataset(directory),
        CrashingProvider(kinds),
        tmp_path / "runs",
        **deterministic(),
    )
    records = score_run(load_run(tmp_path / "runs" / "failures", directory))
    assert [r.reason for r in records] == [f"generation failed: {k.value}" for k in kinds]
    assert all(r.value is None and r.status is ScoreStatus.NOT_SCORED for r in records)


def test_a_resumed_run_scores_like_one_that_never_crashed(tmp_path):
    script = ["Paris", "366", "x", "y", "z"]
    whole = finished_run(tmp_path / "a", script=script)
    crashed = finished_run(tmp_path / "b", script=script, crash_after=2)
    assert score_run(loaded(whole)) == score_run(loaded(crashed))


def test_records_are_serialisable_and_carry_the_pinned_metric_identity(tmp_path):
    run = finished_run(tmp_path, fixture="tiny_extraction", script=extraction_script())
    records = score_run(loaded(run))
    assert {(r.metric, r.metric_version) for r in records} == {
        ("json_parse_validity", "0.1.0"),
        ("field_exact_match", "0.1.0"),
    }
    for record in records:
        assert ScoreRecord.model_validate_json(record.model_dump_json()) == record
