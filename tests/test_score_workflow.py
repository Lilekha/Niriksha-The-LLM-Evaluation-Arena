"""The scoring workflow: score a run, persist, reuse, verify, rescore. Offline and read-only."""

import hashlib
import json
import shutil

import pytest

import niriksha.core.execution as execution_module
import niriksha.core.runner as runner_module
from ds_helpers import FIXTURES
from e2e_support import CrashingProvider
from niriksha.core.dataset import load_dataset
from niriksha.core.generation import FailureKind
from niriksha.core.report import aggregate, build_report, render_report_json, render_report_markdown
from niriksha.core.runload import RunIntegrityError, load_run
from niriksha.core.scorestore import (
    ScoreArtifactError,
    ScoreArtifactMismatchError,
    build_artifact,
    read_artifact,
    records_hash,
    source_from_run,
    write_artifact,
)
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.providers.fake import FakeProvider
from niriksha.scorers.artifacts import (
    available_metrics,
    resolve_scorer,
    score_run_to_artifact,
    verify_artifact,
)
from niriksha.scorers.run import score_run
from score_helpers import finished_run, snapshot

QA, JSON_VALID, FIELD = "normalized_exact_match", "json_parse_validity", "field_exact_match"


def qa_script():
    cases = {c.id: c for c in load_dataset(FIXTURES / "tiny_qa").cases}
    return [
        "  paris  ",  # right after normalisation
        "366 days",  # a near miss: scored 0.0
        cases["qa-hi-001"].answers[1],  # Hindi
        cases["qa-kn-001"].answers[0],  # Kannada
        FailureKind.TIMEOUT,  # generation failed: not scored
    ]


def extraction_script():
    return ['{"name": "Asha", "age": 30}', "not json at all", FailureKind.RATE_LIMIT]


@pytest.fixture
def qa_run(tmp_path):
    return finished_run(tmp_path, script=qa_script())


@pytest.fixture
def extraction_run(tmp_path):
    return finished_run(tmp_path, fixture="tiny_extraction", script=extraction_script())


def score(run, tmp_path, metric=QA, **kwargs):
    return score_run_to_artifact(
        run.run_dir, run.dataset_dir, tmp_path / "scores", metric, **kwargs
    )


def rewrite_records(path, change):
    """Edit the records of an artifact and recompute records_sha256 with the stdlib, so the file
    stays structurally valid and only the comparison with the source can catch it."""
    document = json.loads(path.read_bytes())
    change(document["records"])
    text = json.dumps(
        document["records"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    document["records_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.write_bytes(json.dumps(document, indent=2).encode() + b"\n")


def rig_providers(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("a provider was called")

    monkeypatch.setattr(FakeProvider, "generate", explode)
    monkeypatch.setattr(CrashingProvider, "generate", explode)
    monkeypatch.setattr(runner_module, "run_one", explode)
    monkeypatch.setattr(runner_module, "run_requests", explode)
    monkeypatch.setattr(execution_module, "run_one", explode)


# -- the main workflow ----------------------------------------------------------------------------


def test_score_persist_reload_verify_aggregate_and_report(qa_run, tmp_path):
    outcome = score(qa_run, tmp_path)
    assert outcome.created and outcome.path.name == f"r1--{QA}--0.1.0.json"
    artifact = read_artifact(outcome.path)
    assert artifact == outcome.artifact
    assert [r.value for r in artifact.records] == [1.0, 0.0, 1.0, 1.0, None]
    assert [r.status for r in artifact.records][-1] is ScoreStatus.NOT_SCORED

    verify_artifact(artifact, qa_run.run_dir, qa_run.dataset_dir)  # no exception
    agg = aggregate(artifact)
    assert (agg.total_cases, agg.scored, agg.not_scored, agg.mean) == (5, 4, 1, 0.75)
    assert agg.not_scored_by_reason == {"generation failed: timeout": 1}
    assert "Mean over scored cases: 0.750000" in render_report_markdown(build_report(artifact))


def test_the_artifact_holds_exactly_what_the_m2_1_scorers_produce(qa_run, tmp_path):
    artifact = score(qa_run, tmp_path).artifact
    assert artifact.records == score_run(load_run(qa_run.run_dir, qa_run.dataset_dir))
    assert artifact.records_sha256 == records_hash(artifact.records)


def test_a_run_directory_is_never_touched_and_keeps_exactly_two_files(qa_run, tmp_path):
    before = snapshot(qa_run.run_dir, qa_run.dataset_dir)
    outcome = score(qa_run, tmp_path)
    assert snapshot(qa_run.run_dir, qa_run.dataset_dir) == before
    assert sorted(p.name for p in qa_run.run_dir.iterdir()) == ["manifest.json", "results.jsonl"]
    assert outcome.path.parent == tmp_path / "scores"


def test_scoring_is_deterministic_across_scores_directories(qa_run, tmp_path):
    first = score_run_to_artifact(qa_run.run_dir, qa_run.dataset_dir, tmp_path / "a", QA)
    second = score_run_to_artifact(qa_run.run_dir, qa_run.dataset_dir, tmp_path / "b", QA)
    assert first.path.read_bytes() == second.path.read_bytes()


def test_a_resumed_run_scores_like_one_that_never_crashed(tmp_path):
    script = ["Paris", "366", "x", "y", "z"]
    whole = finished_run(tmp_path / "w", script=script)
    crashed = finished_run(tmp_path / "c", script=script, crash_after=2)
    a = score_run_to_artifact(whole.run_dir, whole.dataset_dir, tmp_path / "s1", QA).artifact
    b = score_run_to_artifact(crashed.run_dir, crashed.dataset_dir, tmp_path / "s2", QA).artifact
    assert a.records == b.records


# -- failed generations versus invalid output -----------------------------------------------------


def test_a_failed_generation_is_not_scored_but_invalid_output_scores_zero(extraction_run, tmp_path):
    for metric in (JSON_VALID, FIELD):
        records = score(extraction_run, tmp_path, metric).artifact.records
        assert [r.status for r in records] == [
            ScoreStatus.SCORED,
            ScoreStatus.SCORED,  # "not json at all": generated, but wrong, so scored 0.0
            ScoreStatus.NOT_SCORED,  # the generation itself failed
        ]
        assert [r.value for r in records] == [1.0, 0.0, None], metric
        assert records[2].reason == "generation failed: rate_limit"
        agg = aggregate(read_artifact(tmp_path / "scores" / f"r1--{metric}--0.1.0.json"))
        assert (agg.scored, agg.not_scored, agg.mean) == (2, 1, 0.5)


def test_an_artifact_where_every_generation_failed_has_no_mean(tmp_path):
    run = finished_run(tmp_path, script=[FailureKind.TIMEOUT] * 5)
    artifact = score(run, tmp_path).artifact
    assert all(r.status is ScoreStatus.NOT_SCORED and r.value is None for r in artifact.records)
    agg = aggregate(artifact)
    assert agg.mean is None and agg.mean_unavailable_reason == "no_scored_cases"


# -- rescoring the same outputs -------------------------------------------------------------------


def test_the_same_outputs_are_rescored_with_other_metrics_without_regenerating(
    extraction_run, tmp_path, monkeypatch
):
    rig_providers(monkeypatch)  # any provider call would raise
    before = snapshot(extraction_run.run_dir)
    first = score(extraction_run, tmp_path, JSON_VALID)
    second = score(extraction_run, tmp_path, FIELD)
    assert first.path != second.path and first.created and second.created
    assert sorted(p.name for p in (tmp_path / "scores").iterdir()) == [
        f"r1--{FIELD}--0.1.0.json",
        f"r1--{JSON_VALID}--0.1.0.json",
    ]
    assert snapshot(extraction_run.run_dir) == before
    for outcome in (first, second):
        verify_artifact(outcome.artifact, extraction_run.run_dir, extraction_run.dataset_dir)


def test_there_are_no_automatic_version_upgrades():
    assert available_metrics("json_extraction") == (
        (FIELD, "0.1.0"),
        (JSON_VALID, "0.1.0"),
        ("json_schema_validity", "0.1.0"),
    )
    assert resolve_scorer(QA).VERSION == "0.1.0"  # the single registered version
    assert resolve_scorer(QA, "0.1.0").METRIC == QA
    with pytest.raises(ScoreArtifactError, match="no scorer implements"):
        resolve_scorer(QA, "0.2.0")
    with pytest.raises(ScoreArtifactError, match="unknown metric"):
        resolve_scorer("bleu")


# -- an artifact that already exists --------------------------------------------------------------


def test_scoring_again_reuses_the_verified_artifact_and_writes_nothing(qa_run, tmp_path):
    first = score(qa_run, tmp_path)
    before = first.path.read_bytes()
    stamp = first.path.stat().st_mtime_ns
    second = score(qa_run, tmp_path)
    assert not second.created and second.artifact == first.artifact
    assert first.path.read_bytes() == before and first.path.stat().st_mtime_ns == stamp


def test_a_corrupt_existing_artifact_is_refused_and_left_alone(qa_run, tmp_path):
    path = score(qa_run, tmp_path).path
    path.write_bytes(path.read_bytes()[:150])  # a truncated file, e.g. a crash mid-write
    before = path.read_bytes()
    with pytest.raises(ScoreArtifactError, match="invalid JSON"):
        score(qa_run, tmp_path)
    assert path.read_bytes() == before


def test_a_well_formed_but_different_artifact_is_refused_and_left_alone(qa_run, tmp_path):
    path = score(qa_run, tmp_path).path
    rewrite_records(path, lambda records: records[0].update(value=0.0))  # valid file, wrong score
    before = path.read_bytes()
    with pytest.raises(ScoreArtifactMismatchError, match="records") as info:
        score(qa_run, tmp_path)
    assert "not overwritten" in str(info.value)
    assert path.read_bytes() == before


def test_an_artifact_for_a_changed_run_file_is_refused_even_if_the_run_still_loads(
    qa_run, tmp_path
):
    path = score(qa_run, tmp_path).path
    manifest = qa_run.run_dir / "manifest.json"
    manifest.write_bytes(
        json.dumps(json.loads(manifest.read_bytes())).encode()
    )  # same data, new bytes
    load_run(qa_run.run_dir, qa_run.dataset_dir)  # the run itself is still valid
    before = path.read_bytes()
    with pytest.raises(ScoreArtifactMismatchError, match=r"source\.run_manifest_sha256"):
        score(qa_run, tmp_path)
    assert path.read_bytes() == before


def test_an_artifact_that_claims_another_name_is_refused(qa_run, tmp_path):
    path = score(qa_run, tmp_path).path
    path.rename(path.with_name(f"r1--{QA}--0.1.1.json"))
    with pytest.raises(ScoreArtifactError, match="file name"):
        read_artifact(path.with_name(f"r1--{QA}--0.1.1.json"))


def test_scoring_into_a_regular_file_is_a_typed_error_not_a_missing_artifact(qa_run, tmp_path):
    target = tmp_path / "scores"
    target.write_bytes(b"precious")
    with pytest.raises(ScoreArtifactError, match="scores_dir") as info:
        score(qa_run, tmp_path)
    assert "not found" not in str(info.value)
    assert target.read_bytes() == b"precious"


def test_scoring_where_a_directory_occupies_the_artifact_name_is_a_typed_error(qa_run, tmp_path):
    occupied = tmp_path / "scores" / f"r1--{QA}--0.1.0.json"
    occupied.mkdir(parents=True)
    with pytest.raises(ScoreArtifactError, match="directory occupies"):
        score(qa_run, tmp_path)
    assert occupied.is_dir()


# -- verify_artifact ------------------------------------------------------------------------------


def test_an_artifact_verified_against_a_different_run_is_a_mismatch(tmp_path):
    one = finished_run(tmp_path / "one", script=qa_script())
    two = finished_run(tmp_path / "two", script=["Paris", "366", "x", "y", "z"])
    artifact = score_run_to_artifact(one.run_dir, one.dataset_dir, tmp_path / "s", QA).artifact
    with pytest.raises(ScoreArtifactMismatchError, match="source.run_results_sha256"):
        verify_artifact(artifact, two.run_dir, two.dataset_dir)


@pytest.mark.parametrize(
    "damage",
    ["delete run", "truncate results", "tamper request hash", "edit dataset", "delete dataset"],
)
def test_a_missing_or_corrupted_source_means_the_artifact_cannot_be_verified(
    qa_run, tmp_path, damage
):
    artifact = score(qa_run, tmp_path).artifact
    results = qa_run.run_dir / "results.jsonl"
    if damage == "delete run":
        shutil.rmtree(qa_run.run_dir)
    elif damage == "truncate results":
        results.write_bytes(results.read_bytes()[:-40])
    elif damage == "tamper request hash":
        first = json.loads(results.read_bytes().split(b"\n")[0])["request_sha256"].encode()
        results.write_bytes(results.read_bytes().replace(first, b"0" * 64, 1))
    elif damage == "edit dataset":
        cases = qa_run.dataset_dir / "cases.jsonl"
        cases.write_bytes(cases.read_bytes().replace(b"capital", b"CAPITAL", 1))
    else:
        shutil.rmtree(qa_run.dataset_dir)
    with pytest.raises(ScoreArtifactMismatchError, match="source run cannot be verified") as info:
        verify_artifact(artifact, qa_run.run_dir, qa_run.dataset_dir)
    assert isinstance(info.value.__cause__, RunIntegrityError)


def test_an_artifact_for_an_unregistered_metric_version_is_readable_but_unverifiable(
    qa_run, tmp_path
):
    loaded = load_run(qa_run.run_dir, qa_run.dataset_dir)
    records = tuple(
        ScoreRecord(
            request_id=case.id,
            metric=QA,
            metric_version="0.0.9",
            status=ScoreStatus.SCORED,
            value=1.0,
        )
        for case in loaded.cases
    )
    old = build_artifact(source_from_run(loaded), QA, "0.0.9", "short_answer_qa", records)
    path = write_artifact(tmp_path / "scores", old)
    reread = read_artifact(path)  # can be read and aggregated ...
    assert aggregate(reread).mean == 1.0
    with pytest.raises(ScoreArtifactError, match="no scorer implements"):  # ... not verified
        verify_artifact(reread, qa_run.run_dir, qa_run.dataset_dir)


# -- refusals before anything is written ----------------------------------------------------------


def test_a_metric_for_the_wrong_task_is_refused_and_nothing_is_created(qa_run, tmp_path):
    with pytest.raises(ScoreArtifactError, match="applies to 'json_extraction'"):
        score(qa_run, tmp_path, JSON_VALID)
    assert not (tmp_path / "scores").exists()


def test_an_incomplete_run_is_refused_by_the_loader_and_no_artifact_is_written(qa_run, tmp_path):
    results = qa_run.run_dir / "results.jsonl"
    results.write_bytes(b"\n".join(results.read_bytes().split(b"\n")[:2]) + b"\n")
    with pytest.raises(RunIntegrityError, match="incomplete"):
        score(qa_run, tmp_path)
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("where", ["inside", "same"])
def test_scores_may_not_be_written_inside_the_run_directory(qa_run, where):
    target = qa_run.run_dir / "scores" if where == "inside" else qa_run.run_dir
    before = snapshot(qa_run.run_dir)
    with pytest.raises(ScoreArtifactError, match="inside the run directory"):
        score_run_to_artifact(qa_run.run_dir, qa_run.dataset_dir, target, QA)
    assert snapshot(qa_run.run_dir) == before
    assert sorted(p.name for p in qa_run.run_dir.iterdir()) == ["manifest.json", "results.jsonl"]


# -- no provider, no network ----------------------------------------------------------------------


def test_the_whole_workflow_runs_with_no_provider_and_no_network(qa_run, tmp_path, monkeypatch):
    rig_providers(monkeypatch)  # tests/conftest.py already blocks the network
    before = snapshot(qa_run.run_dir, qa_run.dataset_dir)
    outcome = score(qa_run, tmp_path)
    artifact = read_artifact(outcome.path)
    verify_artifact(artifact, qa_run.run_dir, qa_run.dataset_dir)
    assert score(qa_run, tmp_path).created is False  # the reuse path too
    report = build_report(artifact)
    assert render_report_json(report) and render_report_markdown(report)
    assert snapshot(qa_run.run_dir, qa_run.dataset_dir) == before
