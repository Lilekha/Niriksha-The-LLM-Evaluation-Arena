"""json_schema_validity through the M2.2 workflow: persist, reload, verify, report. Offline."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import niriksha.core.execution as execution_module
import niriksha.core.runner as runner_module
from e2e_support import CrashingProvider, make_config
from niriksha.core.dataset import compute_content_sha256, load_dataset
from niriksha.core.execution import execute_run
from niriksha.core.generation import FailureKind
from niriksha.core.report import aggregate, build_report, render_report_json, render_report_markdown
from niriksha.core.runload import RunIntegrityError
from niriksha.core.scorestore import ScoreArtifactError, ScoreArtifactMismatchError, read_artifact
from niriksha.core.scoring import ScoreStatus
from niriksha.providers.fake import FakeProvider
from niriksha.scorers.artifacts import resolve_scorer, score_run_to_artifact, verify_artifact
from score_helpers import SPLITS, copy_fixture, deterministic, finished_run, snapshot

METRIC = "json_schema_validity"
REPO = Path(__file__).resolve().parents[1]
SCRIPT = ['{"name": "Asha", "age": 30}', '{"name": "Ravi", "age": "25"}', FailureKind.RATE_LIMIT]


def score(run, tmp_path, metric=METRIC):
    return score_run_to_artifact(run.run_dir, run.dataset_dir, tmp_path / "scores", metric)


def run_with_schema(tmp_path, schema, script=SCRIPT):
    """An extraction run on a copy of the fixture whose output_schema (and pin) is replaced."""
    dataset_dir = copy_fixture(tmp_path, "tiny_extraction")
    set_schema(dataset_dir, schema)
    config = make_config("r1", SPLITS["tiny_extraction"])
    execute_run(
        config,
        load_dataset(dataset_dir),
        CrashingProvider(script),
        tmp_path / "runs",
        **deterministic(),
    )
    return type("Run", (), {"run_dir": tmp_path / "runs" / "r1", "dataset_dir": dataset_dir})


def set_schema(dataset_dir, schema):
    meta_path = dataset_dir / "dataset.json"
    meta = json.loads(meta_path.read_bytes())
    raw = [
        json.loads(line) for line in (dataset_dir / "cases.jsonl").read_text("utf-8").splitlines()
    ]
    meta["output_schema"] = schema
    meta["content_sha256"] = compute_content_sha256(meta["task"], schema, raw)
    meta_path.write_bytes(json.dumps(meta, ensure_ascii=False).encode("utf-8"))


@pytest.fixture
def run(tmp_path):
    return finished_run(tmp_path, fixture="tiny_extraction", script=SCRIPT)


def test_score_persist_reload_verify_aggregate_and_report(run, tmp_path):
    outcome = score(run, tmp_path)
    assert outcome.created and outcome.path.name == f"r1--{METRIC}--0.1.0.json"
    artifact = read_artifact(outcome.path)
    assert artifact == outcome.artifact
    assert [r.status for r in artifact.records] == [
        ScoreStatus.SCORED,
        ScoreStatus.SCORED,
        ScoreStatus.NOT_SCORED,
    ]
    assert [r.value for r in artifact.records] == [1.0, 0.0, None]
    assert artifact.records[1].details["first_error_schema_path"] == "properties/age/type"
    assert artifact.records[2].reason == "generation failed: rate_limit"

    verify_artifact(artifact, run.run_dir, run.dataset_dir)  # no exception
    agg = aggregate(artifact)
    assert (agg.total_cases, agg.scored, agg.not_scored, agg.mean) == (3, 2, 1, 0.5)
    report = build_report(artifact)
    assert "Mean over scored cases: 0.500000" in render_report_markdown(report)
    assert render_report_json(report) == render_report_json(
        build_report(read_artifact(outcome.path))
    )


def test_scoring_again_reuses_the_verified_artifact(run, tmp_path):
    first = score(run, tmp_path)
    before = first.path.read_bytes()
    second = score(run, tmp_path)
    assert not second.created and second.path.read_bytes() == before


def test_unicode_output_persists_and_reloads_intact(tmp_path):
    script = [
        '{"name": "आशा", "age": 30}',
        '{"name": "ಆಶಾ", "age": "x"}',
        '{"name": "n", "age": 1}',
    ]
    run = finished_run(tmp_path, fixture="tiny_extraction", script=script)
    artifact = read_artifact(score(run, tmp_path).path)
    assert [r.value for r in artifact.records] == [1.0, 0.0, 1.0]
    verify_artifact(artifact, run.run_dir, run.dataset_dir)


def test_the_dataset_schema_is_the_one_used(tmp_path):
    schema = {
        "type": "object",
        "properties": {"name": {}},
        "required": ["name"],
        "additionalProperties": False,
    }
    run = run_with_schema(
        tmp_path, schema, ['{"name": "A"}', '{"name": "A", "age": 1}', '{"name": "A"}']
    )
    assert [r.value for r in score(run, tmp_path).artifact.records] == [1.0, 0.0, 1.0]


def test_the_same_outputs_are_rescored_with_other_metrics_without_regenerating(
    run, tmp_path, monkeypatch
):
    def explode(*args, **kwargs):
        raise AssertionError("a provider was called")

    for target, name in [
        (FakeProvider, "generate"),
        (CrashingProvider, "generate"),
        (runner_module, "run_one"),
        (runner_module, "run_requests"),
        (execution_module, "run_one"),
    ]:
        monkeypatch.setattr(target, name, explode)
    before = snapshot(run.run_dir, run.dataset_dir)
    for metric in ("json_parse_validity", "field_exact_match", METRIC):
        verify_artifact(score(run, tmp_path, metric).artifact, run.run_dir, run.dataset_dir)
    assert snapshot(run.run_dir, run.dataset_dir) == before
    assert sorted(p.name for p in run.run_dir.iterdir()) == ["manifest.json", "results.jsonl"]


def test_scoring_makes_no_network_calls(run, tmp_path):
    events = []

    def hook(event, args):
        if event.startswith("socket."):
            events.append(event)

    sys.addaudithook(hook)  # cannot be removed; it only records socket activity
    mark = len(events)
    verify_artifact(score(run, tmp_path).artifact, run.run_dir, run.dataset_dir)
    assert events[mark:] == []


# -- an invalid dataset schema is a typed error before any write ----------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "nonsense"},
        {"$schema": "http://json-schema.org/draft-07/schema#"},
        {"$ref": "https://example.com/s.json"},
        {"$id": "https://example.com/s"},
        {"pattern": "("},
    ],
)
def test_an_unusable_schema_raises_before_any_artifact_is_written(tmp_path, schema):
    run = run_with_schema(tmp_path, schema)
    before = snapshot(run.run_dir, run.dataset_dir)
    with pytest.raises(ScoreArtifactError):
        score(run, tmp_path)
    assert not (tmp_path / "scores").exists()
    assert snapshot(run.run_dir, run.dataset_dir) == before


def test_a_self_recursive_schema_raises_before_any_artifact_is_written(tmp_path):
    run = run_with_schema(tmp_path, {"$ref": "#"})
    with pytest.raises(ScoreArtifactError, match="recurses"):
        score(run, tmp_path)
    assert not (tmp_path / "scores").exists()


def test_an_unresolvable_reference_discovered_while_scoring_writes_nothing(tmp_path):
    schema = {"properties": {"name": {"$ref": "#/$defs/missing"}}}
    run = run_with_schema(tmp_path, schema)
    with pytest.raises(ScoreArtifactError, match="unresolvable"):
        score(run, tmp_path)
    assert not (tmp_path / "scores").exists()


# -- tampering is detected ------------------------------------------------------------------------


def test_a_changed_dataset_schema_is_detected_by_run_verification(run, tmp_path):
    artifact = score(run, tmp_path).artifact
    set_schema(run.dataset_dir, {"type": "object"})  # a different schema, correctly re-pinned
    with pytest.raises(ScoreArtifactMismatchError, match="cannot be verified"):
        verify_artifact(artifact, run.run_dir, run.dataset_dir)
    with pytest.raises(RunIntegrityError):
        score(run, tmp_path)


def test_an_edited_score_in_the_artifact_is_detected(run, tmp_path):
    outcome = score(run, tmp_path)
    document = json.loads(outcome.path.read_bytes())
    document["records"][1]["value"] = 1.0
    text = json.dumps(
        document["records"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    document["records_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    outcome.path.write_bytes(json.dumps(document, indent=2).encode() + b"\n")
    with pytest.raises(ScoreArtifactMismatchError):
        verify_artifact(read_artifact(outcome.path), run.run_dir, run.dataset_dir)
    with pytest.raises(ScoreArtifactMismatchError):
        score(run, tmp_path)


def test_the_metric_is_registered_for_extraction_runs_only(tmp_path):
    assert resolve_scorer(METRIC).TASK == "json_extraction"
    qa = finished_run(tmp_path, script=["a"] * 5)
    with pytest.raises(ScoreArtifactError, match="applies to"):
        score(qa, tmp_path)
    assert not (tmp_path / "scores").exists()


# -- determinism ----------------------------------------------------------------------------------

CHILD = """
import hashlib, sys, tempfile
from pathlib import Path
from score_helpers import finished_run
from niriksha.core.report import build_report, render_report_json, render_report_markdown
from niriksha.core.scorestore import read_artifact
from niriksha.scorers.artifacts import score_run_to_artifact
script = ['{"name": "\\u0906\\u0936\\u093e", "age": 30}', '{"name": "x", "age": "1"}', 'nope']
tmp = Path(tempfile.mkdtemp())
run = finished_run(tmp, fixture="tiny_extraction", script=script)
out = score_run_to_artifact(run.run_dir, run.dataset_dir, tmp / "s", "json_schema_validity")
report = build_report(read_artifact(out.path))
parts = [out.path.read_bytes(), render_report_json(report).encode()]
parts.append(render_report_markdown(report).encode())
print(hashlib.sha256(b"".join(parts)).hexdigest())
"""


def test_artifact_and_reports_are_byte_identical_across_hash_seeds_and_directories():
    digests = set()
    for seed in ("0", "1", "12345"):
        env = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONPATH": os.pathsep.join([str(REPO / "tests"), str(REPO / "src")]),
            "PYTHONUTF8": "1",
        }
        done = subprocess.run(
            [sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, check=True
        )
        digests.add(done.stdout.strip())
    assert len(digests) == 1 and len(next(iter(digests))) == 64
