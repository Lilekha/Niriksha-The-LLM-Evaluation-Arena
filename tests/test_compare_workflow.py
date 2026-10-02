"""compare_runs end to end: verified artifacts, refusals, tamper detection, safety, determinism."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import niriksha.core.execution as execution_module
import niriksha.core.runner as runner_module
from ds_helpers import FIXTURES, qa, write_dataset
from e2e_support import CrashingProvider, make_config
from niriksha.core.compare import (
    IncompatibleRunsError,
    Outcome,
    render_comparison_json,
    render_comparison_markdown,
)
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run
from niriksha.core.generation import FailureKind, GenerationParams
from niriksha.core.report import aggregate
from niriksha.core.runload import RunIntegrityError
from niriksha.core.runstore import PromptTemplate
from niriksha.core.scorestore import ScoreArtifactError, ScoreArtifactMismatchError, read_artifact
from niriksha.providers.fake import FakeProvider
from niriksha.scorers.artifacts import score_run_to_artifact
from niriksha.scorers.comparison import RunInput, compare_runs
from score_helpers import SPLITS, copy_fixture, deterministic, edit_manifest, snapshot

METRIC = "normalized_exact_match"
REPO = Path(__file__).resolve().parents[1]
ALL = SPLITS["tiny_qa"]
CASES = load_dataset(FIXTURES / "tiny_qa").cases
RIGHT = [case.answers[0] for case in CASES]
WRONG = "definitely wrong"
T = FailureKind.TIMEOUT


class World:
    """A shared dataset copy, a runs directory and a scores directory under one tmp_path."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.dataset_dir = copy_fixture(tmp_path, "tiny_qa")
        self.runs = tmp_path / "runs"
        self.scores = tmp_path / "scores"

    def run(self, run_id, script, *, model="m", splits=ALL, dataset_dir=None, score=True, **update):
        dataset_dir = dataset_dir or self.dataset_dir
        config = make_config(run_id, splits, model=model).model_copy(update=update)
        execute_run(
            config,
            load_dataset(dataset_dir),
            CrashingProvider(script),
            self.runs,
            **deterministic(),
        )
        item = RunInput(self.runs / run_id, dataset_dir, self.scores)
        if score:
            score_run_to_artifact(item.run_dir, item.dataset_dir, self.scores, METRIC)
        return item


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def pair(world):
    baseline = world.run("run-a", [RIGHT[0], WRONG, RIGHT[2], RIGHT[3], T], model="model-a")
    candidate = world.run("run-b", [RIGHT[0], RIGHT[1], WRONG, RIGHT[3], RIGHT[4]], model="model-b")
    return baseline, candidate


def message(call):
    with pytest.raises(Exception) as caught:
        call()
    return caught.value


def no_paths(world, text):
    for fragment in (str(world.tmp_path), world.tmp_path.name, "\\", "/runs"):
        assert fragment not in text, fragment


# -- the main workflow ----------------------------------------------------------------------------


def test_two_verified_runs_are_compared_with_aggregates_pairing_and_cases(world, pair):
    result = compare_runs(*pair, METRIC)
    assert result.metric.name == METRIC and result.metric.direction == "higher_is_better"
    assert (result.baseline.run_id, result.candidate.run_id) == ("run-a", "run-b")
    assert (result.baseline.requested_model, result.candidate.requested_model) == (
        "model-a",
        "model-b",
    )
    assert result.varying.requested_model and not result.confounded

    # aggregates are exactly what the M2.2 aggregation gives for each artifact
    for side, run_id in ((result.baseline, "run-a"), (result.candidate, "run-b")):
        stored = read_artifact(world.scores / f"{run_id}--{METRIC}--0.1.0.json")
        assert side.aggregate == aggregate(stored)
        assert side.artifact_records_sha256 == stored.records_sha256
        assert side.run_manifest_sha256 == stored.source.run_manifest_sha256
        assert side.run_results_sha256 == stored.source.run_results_sha256
    assert result.baseline.aggregate.mean == 0.75 and result.candidate.aggregate.mean == 0.8

    assert [r.outcome for r in result.rows] == [
        Outcome.EQUAL,
        Outcome.CANDIDATE_HIGHER,
        Outcome.BASELINE_HIGHER,
        Outcome.EQUAL,
        Outcome.ONLY_CANDIDATE,
    ]
    assert result.rows[4].baseline.value is None
    assert result.rows[4].baseline.reason == "generation failed: timeout"
    assert [r.request_id for r in result.rows] == [c.id for c in CASES]
    s = result.summary
    assert not s.same_scored_cases and s.paired.paired_cases == 4
    assert (s.paired.baseline_mean, s.paired.candidate_mean, s.paired.mean_difference) == (
        0.75,
        0.75,
        0.0,
    )
    assert s.mean_difference is None
    assert s.mean_difference_unavailable_reason == "different_scored_cases"
    assert json.loads(render_comparison_json(result))["summary"]["mean_difference"] is None
    text = render_comparison_markdown(result)
    assert "Paired comparison (use this one)" in text and "generation failed: timeout" in text
    no_paths(world, text + render_comparison_json(result))


def test_the_comparison_is_mirrored_when_the_roles_are_swapped(pair):
    forward = compare_runs(pair[0], pair[1], METRIC)
    backward = compare_runs(pair[1], pair[0], METRIC)
    assert backward.baseline == forward.candidate and backward.candidate == forward.baseline
    assert backward.summary.paired.mean_difference == -forward.summary.paired.mean_difference
    assert [r.delta for r in backward.rows][1] == -[r.delta for r in forward.rows][1]


def test_no_paired_scored_cases_is_reported_as_unavailable(world):
    a = world.run("run-a", [RIGHT[0], RIGHT[1], T, T, T])
    b = world.run("run-b", [T, T, RIGHT[2], WRONG, RIGHT[4]])
    result = compare_runs(a, b, METRIC)
    assert result.summary.paired.paired_cases == 0
    assert result.summary.paired.unavailable_reason == "no_paired_scored_cases"
    assert "unavailable (no_paired_scored_cases)" in render_comparison_markdown(result)
    summary = json.loads(render_comparison_json(result))["summary"]
    assert summary["paired"]["mean_difference"] is None
    assert summary["paired"]["unavailable_reason"] == "no_paired_scored_cases"
    assert summary["mean_difference"] is None  # different scored cases: no overall difference


def test_runs_where_everything_failed_have_no_mean_and_no_difference(world):
    a = world.run("run-a", [T] * 5)
    b = world.run("run-b", RIGHT)
    result = compare_runs(a, b, METRIC)
    assert result.baseline.aggregate.mean is None and result.summary.mean_difference is None
    assert result.summary.outcomes[Outcome.ONLY_CANDIDATE] == 5


def test_what_varies_between_runs_is_reported_from_the_manifests(world):
    base = world.run("run-a", RIGHT)
    prompt = world.run("run-b", RIGHT, prompt=PromptTemplate(user="Q: {input}"))
    params = world.run("run-c", RIGHT, params=GenerationParams(temperature=0.7))
    same = world.run("run-d", RIGHT)
    result = compare_runs(base, prompt, METRIC)
    assert result.varying.prompt and result.confounded and not result.varying.requested_model
    assert "cannot necessarily be attributed to the model" in render_comparison_markdown(result)
    result = compare_runs(base, params, METRIC)
    assert result.varying.params == ("temperature",) and result.confounded
    result = compare_runs(base, same, METRIC)
    assert not result.confounded and "Nothing varies" in render_comparison_markdown(result)


# -- refusals -------------------------------------------------------------------------------------


def test_a_different_dataset_is_refused(world):
    other = write_dataset(world.tmp_path / "other", [qa(id=f"q{i}") for i in range(3)], dirname="o")
    a = world.run("run-a", RIGHT)
    b = world.run("run-b", ["x", "y", "z"], splits=("dev",), dataset_dir=other)
    err = message(lambda: compare_runs(a, b, METRIC))
    assert isinstance(err, IncompatibleRunsError)
    assert "dataset.content_sha256" in err.differences and "selection.case_count" in err.differences
    no_paths(world, str(err))


def test_a_different_selection_of_the_same_dataset_is_refused(world):
    a = world.run("run-a", RIGHT)
    b = world.run("run-b", RIGHT[:2], splits=("dev",))
    err = message(lambda: compare_runs(a, b, METRIC))
    assert isinstance(err, IncompatibleRunsError)
    assert set(err.differences) >= {"selection.case_count", "selection.case_ids_sha256"}


def test_the_same_run_twice_is_refused(world):
    a = world.run("run-a", RIGHT)
    err = message(lambda: compare_runs(a, a, METRIC))
    assert isinstance(err, IncompatibleRunsError) and "same run" in str(err)


def test_an_unknown_metric_or_version_is_refused(pair):
    for kwargs in ({"metric": "bleu"}, {"metric": METRIC, "version": "0.2.0"}):
        with pytest.raises(ScoreArtifactError):
            compare_runs(*pair, **kwargs)


@pytest.mark.parametrize("missing", ["baseline", "candidate"])
def test_a_missing_artifact_is_never_created_and_names_the_side(world, missing):
    a = world.run("run-a", RIGHT, score=missing != "baseline")
    b = world.run("run-b", RIGHT, score=missing != "candidate")
    before = snapshot(world.scores)
    err = message(lambda: compare_runs(a, b, METRIC))
    assert isinstance(err, ScoreArtifactError) and str(err).startswith(f"{missing}: ")
    assert "score the run first" in str(err)
    assert snapshot(world.scores) == before  # nothing was created
    no_paths(world, str(err))


# -- tampering ------------------------------------------------------------------------------------


def test_an_edited_artifact_is_detected_even_with_a_recomputed_hash(world, pair):
    path = world.scores / f"run-b--{METRIC}--0.1.0.json"
    document = json.loads(path.read_bytes())
    document["records"][0]["value"] = 0.0
    text = json.dumps(
        document["records"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    document["records_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    path.write_bytes(json.dumps(document, indent=2).encode() + b"\n")
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, ScoreArtifactMismatchError) and str(err).startswith("candidate: ")
    no_paths(world, str(err))


def test_a_corrupt_artifact_and_a_renamed_artifact_are_refused(world, pair):
    path = world.scores / f"run-a--{METRIC}--0.1.0.json"
    original = path.read_bytes()
    path.write_bytes(original + b"junk")
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, ScoreArtifactError) and str(err).startswith("baseline: ")
    path.write_bytes(original)
    other = world.scores / f"run-b--{METRIC}--0.1.0.json"
    other.write_bytes(original)  # run-a's artifact under run-b's name
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, ScoreArtifactError) and str(err).startswith("candidate: ")
    no_paths(world, str(err))


def test_a_tampered_run_is_refused_with_the_side_named(world, pair):
    edit_manifest(pair[0].run_dir, lambda m: m.update(requested_model="someone-else"))
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, RunIntegrityError) and str(err).startswith("baseline: ")
    no_paths(world, str(err))


def test_a_modified_dataset_is_refused(world, pair):
    cases = world.dataset_dir / "cases.jsonl"
    cases.write_bytes(cases.read_bytes().replace(b"Paris", b"Pariz"))
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, RunIntegrityError)
    no_paths(world, str(err))


def test_a_modified_results_file_is_refused(world, pair):
    results = pair[1].run_dir / "results.jsonl"
    results.write_bytes(results.read_bytes().replace(b"model-b", b"model-x"))
    err = message(lambda: compare_runs(*pair, METRIC))
    assert isinstance(err, RunIntegrityError) and str(err).startswith("candidate: ")


# -- safety ---------------------------------------------------------------------------------------


def test_comparing_writes_nothing_and_calls_no_provider_or_network(world, pair, monkeypatch):
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
    events = []
    sys.addaudithook(
        lambda event, args: events.append(event) if event.startswith("socket.") else None
    )
    mark = len(events)
    before = snapshot(world.runs, world.dataset_dir, world.scores)
    first = compare_runs(*pair, METRIC)
    second = compare_runs(*pair, METRIC)
    assert first == second
    assert snapshot(world.runs, world.dataset_dir, world.scores) == before
    assert sorted(p.name for p in (world.runs / "run-a").iterdir()) == [
        "manifest.json",
        "results.jsonl",
    ]
    assert sorted(p.name for p in world.scores.iterdir()) == [
        f"run-a--{METRIC}--0.1.0.json",
        f"run-b--{METRIC}--0.1.0.json",
    ]
    assert events[mark:] == []


# -- determinism ----------------------------------------------------------------------------------

CHILD = """
import hashlib, tempfile
from pathlib import Path
from test_compare_workflow import METRIC, RIGHT, WRONG, T, World
from niriksha.core.compare import render_comparison_json, render_comparison_markdown
from niriksha.scorers.comparison import compare_runs
world = World(Path(tempfile.mkdtemp()))
a = world.run("run-a", [RIGHT[0], WRONG, RIGHT[2], RIGHT[3], T], model="model-a")
b = world.run("run-b", [RIGHT[0], RIGHT[1], WRONG, RIGHT[3], RIGHT[4]], model="model-b")
result = compare_runs(a, b, METRIC)
data = render_comparison_json(result).encode() + render_comparison_markdown(result).encode()
print(hashlib.sha256(data).hexdigest())
"""


def test_the_comparison_is_byte_identical_across_hash_seeds_and_directories():
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
