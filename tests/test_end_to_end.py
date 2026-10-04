"""End-to-end tests through the public APIs: dataset on disk -> run -> persisted files -> resume.

Only the fake provider and temporary directories are used. Unit-level behaviour (validation rules,
single functions) is covered in the per-module test files and is not repeated here.
"""

import itertools
import json
import math
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import niriksha
from ds_helpers import FIXTURES, qa, write_dataset
from e2e_support import (
    CHILD_ENV_FLAG,
    CRASH_EXIT_CODE,
    GUARD_MISSING_EXIT_CODE,
    SKIP_GUARD_FLAG,
    Crash,
    CrashingProvider,
    ViolatingProvider,
    audit_run,
    make_config,
    network_guard_is_active,
)
from niriksha.core.dataset import DatasetError, compute_content_sha256, load_dataset
from niriksha.core.execution import execute_run, resume_run
from niriksha.core.generation import FailureKind, GenerationFailure
from niriksha.core.provenance import collect_software_info
from niriksha.core.runner import ProviderContractViolation
from niriksha.core.runstore import (
    CorruptRunError,
    ResumeError,
    read_manifest,
    read_results,
)
from niriksha.providers.fake import FakeProvider

TESTS_DIR = Path(__file__).parent
NOW = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
FIXED_SOFTWARE = collect_software_info(lambda: (None, None))  # deterministic across sessions
HINDI = "भारत की राजधानी?"
KANNADA = "ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ?"


def ticking_clock():
    """A fresh clock for each session: every provider call measures exactly 0.5 s."""
    ticks = itertools.count()
    return lambda: next(ticks) * 0.5


def deterministic(**extra):
    return {"clock": ticking_clock(), "now": lambda: NOW, "software": FIXED_SOFTWARE, **extra}


@pytest.fixture
def ten_dir(tmp_path):
    """Ten dev cases, two of them non-ASCII (so the bytes on disk are not plain ASCII)."""
    questions = [f"Question {i}?" for i in range(10)]
    questions[3], questions[7] = HINDI, KANNADA
    cases = [qa(id=f"c{i:02d}", question=q) for i, q in enumerate(questions)]
    return write_dataset(tmp_path, cases, dirname="ten")


def snapshot(run_dir: Path) -> dict:
    return {path.name: path.read_bytes() for path in sorted(run_dir.iterdir())}


def partial_run(tmp_path, dataset_dir, *, completed=3, run_id="r1"):
    """A run interrupted after ``completed`` results; returns (run_dir, config, dataset)."""
    config, dataset = make_config(run_id), load_dataset(dataset_dir)
    with pytest.raises(Crash):
        execute_run(config, dataset, CrashingProvider(crash_after=completed), tmp_path / "runs")
    return tmp_path / "runs" / run_id, config, dataset


# -- an interrupted run ends identical to an uninterrupted one -----------------------------------


@pytest.mark.parametrize("completed_before_crash", [0, 4, 9])
def test_resumed_run_is_identical_to_an_uninterrupted_run(
    tmp_path, ten_dir, completed_before_crash
):
    runs, dataset = tmp_path / "runs", load_dataset(ten_dir)
    whole = execute_run(make_config("whole"), dataset, CrashingProvider(), runs, **deterministic())

    split_config = make_config("split")
    with pytest.raises(Crash):
        execute_run(
            split_config,
            dataset,
            CrashingProvider(crash_after=completed_before_crash),
            runs,
            **deterministic(),
        )
    resumed = resume_run(split_config, dataset, CrashingProvider(), runs, **deterministic())

    assert (resumed.already_recorded, resumed.executed) == (
        completed_before_crash,
        10 - completed_before_crash,
    )
    assert (runs / "split" / "results.jsonl").read_bytes() == whole.run_dir.joinpath(
        "results.jsonl"
    ).read_bytes()
    manifests = [read_manifest(runs / name).model_dump(mode="json") for name in ("whole", "split")]
    assert {k for k in manifests[0] if manifests[0][k] != manifests[1][k]} == {"run_id"}
    audit_run(runs / "whole", make_config("whole"), ten_dir)
    audit_run(runs / "split", split_config, ten_dir)


# -- the default paths: no injected clock, time or software ---------------------------------------


def test_default_execute_and_resume_paths(tmp_path, ten_dir):
    runs, dataset, config = tmp_path / "runs", load_dataset(ten_dir), make_config("defaults")
    started = datetime.now(UTC)
    with pytest.raises(Crash):
        execute_run(config, dataset, CrashingProvider(crash_after=4), runs)  # no injection at all
    summary = resume_run(config, dataset, CrashingProvider(), runs)  # real software info again
    assert (summary.already_recorded, summary.executed) == (4, 6)

    manifest = read_manifest(summary.run_dir)
    assert manifest.created_at.utcoffset() == timedelta(0)
    assert abs(manifest.created_at - started) < timedelta(minutes=5)
    software = manifest.software
    assert software.niriksha_version == niriksha.__version__
    assert software.python_version == collect_software_info().python_version
    assert software.git_commit is None or len(software.git_commit) == 40
    for line in read_results(summary.run_dir).lines:
        assert line.recorded_at.utcoffset() == timedelta(0)
        assert math.isfinite(line.execution.elapsed_s) and line.execution.elapsed_s >= 0
    audit_run(summary.run_dir, config, ten_dir)
    assert resume_run(config, dataset, CrashingProvider(), runs).executed == 0


# -- the audit itself must be able to fail --------------------------------------------------------


def _drop_last_result(run_dir):
    path = run_dir / "results.jsonl"
    path.write_bytes(b"\n".join(path.read_bytes().split(b"\n")[:-2]) + b"\n")


def _swap_first_two_results(run_dir):
    path = run_dir / "results.jsonl"
    lines = path.read_bytes().split(b"\n")
    lines[0], lines[1] = lines[1], lines[0]
    path.write_bytes(b"\n".join(lines))


def _corrupt_request_hash(run_dir):
    path = run_dir / "results.jsonl"
    first_hash = read_results(run_dir).lines[0].request_sha256.encode()
    path.write_bytes(path.read_bytes().replace(first_hash, b"0" * 64, 1))


def _stray_file(run_dir):
    (run_dir / "results.jsonl.tmp").write_bytes(b"leftover")


def _wrong_dataset_hash_in_manifest(run_dir):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest["dataset"]["content_sha256"] = "0" * 64
    path.write_bytes(json.dumps(manifest).encode())


@pytest.mark.parametrize(
    ("tamper", "complaint"),
    [
        (_drop_last_result, "result count or order"),
        (_swap_first_two_results, "result count or order"),
        (_corrupt_request_hash, "request hash differs"),
        (_stray_file, "persisted files"),
        (_wrong_dataset_hash_in_manifest, "dataset hash disagrees"),
    ],
)
def test_the_audit_detects_tampering_with_persisted_artifacts(tmp_path, ten_dir, tamper, complaint):
    config = make_config("audited")
    summary = execute_run(config, load_dataset(ten_dir), CrashingProvider(), tmp_path / "runs")
    audit_run(summary.run_dir, config, ten_dir)  # clean run passes
    tamper(summary.run_dir)
    with pytest.raises(AssertionError, match=complaint):
        audit_run(summary.run_dir, config, ten_dir)


# -- a real process crash, with and without a torn final record -----------------------------------


def run_child(dataset_dir, runs_dir, run_id, *, crash_after, torn, splits):
    env = {**os.environ, "PYTHONPATH": str(TESTS_DIR), CHILD_ENV_FLAG: "1"}
    code = "import sys, e2e_support; sys.exit(e2e_support.child_main(sys.argv[1:]))"
    args = [dataset_dir, runs_dir, run_id, str(crash_after), "1" if torn else "0", splits]
    return subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("torn", [False, True], ids=["killed-between-writes", "torn-final-record"])
def test_process_crash_then_resume_matches_an_uninterrupted_run(tmp_path, torn):
    dataset_dir, splits = FIXTURES / "tiny_qa", ("dev", "test", "holdout")  # 5 cases, hi + kn
    runs = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket.create_connection(("127.0.0.1", 9))  # the parent's guard is still active

    child = run_child(
        dataset_dir, runs, "crashed", crash_after=2, torn=torn, splits=",".join(splits)
    )
    assert child.returncode == CRASH_EXIT_CODE, child.stderr  # died by os._exit, not by an error

    run_dir = runs / "crashed"
    assert [r.request_id for r in read_results(run_dir, allow_torn_tail=True).lines] == [
        "qa-en-001",
        "qa-en-002",
    ]
    if torn:
        assert read_results(run_dir, allow_torn_tail=True).torn_tail_bytes > 0
        with pytest.raises(CorruptRunError, match="torn"):
            read_results(run_dir)  # a strict reader refuses it until resume repairs it

    config, dataset = make_config("crashed", splits), load_dataset(dataset_dir)
    provider = CrashingProvider()
    summary = resume_run(config, dataset, provider, runs)  # the default path, in this process
    assert (summary.already_recorded, summary.executed) == (2, 3)
    assert [r.request_id for r in provider.calls] == ["qa-hi-001", "qa-kn-001", "qa-hi-002"]
    audit_run(run_dir, config, dataset_dir)

    whole_config = make_config("whole", splits)
    whole = execute_run(whole_config, dataset, CrashingProvider(), runs)

    def comparable(directory):
        lines = read_results(directory).lines
        return [
            (
                line.request_id,
                line.request_sha256,
                line.execution.result.model_dump(mode="json"),
            )
            for line in lines
        ]

    assert comparable(run_dir) == comparable(whole.run_dir)
    crashed_manifest = read_manifest(run_dir).model_dump(mode="json")
    whole_manifest = read_manifest(whole.run_dir).model_dump(mode="json")
    for manifest in (crashed_manifest, whole_manifest):
        for name in ("run_id", "created_at"):
            del manifest[name]
        del manifest["software"]["git_dirty"]  # the working tree may change between the runs
    assert crashed_manifest == whole_manifest


# -- invalid or missing manifests are rejected before any provider call --------------------------


def _set_manifest(run_dir, change):
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_bytes())
    change(manifest)
    path.write_bytes(json.dumps(manifest).encode())


MANIFEST_DAMAGE = {
    "missing file": lambda run_dir: (run_dir / "manifest.json").unlink(),
    "empty file": lambda run_dir: (run_dir / "manifest.json").write_bytes(b""),
    "not json": lambda run_dir: (run_dir / "manifest.json").write_bytes(b"{not json"),
    "empty object": lambda run_dir: (run_dir / "manifest.json").write_bytes(b"{}"),
    "unsupported version": lambda run_dir: _set_manifest(
        run_dir, lambda m: m.update(manifest_version=2)
    ),
    "unknown field": lambda run_dir: _set_manifest(run_dir, lambda m: m.update(api_key="x")),
    "naive timestamp": lambda run_dir: _set_manifest(
        run_dir, lambda m: m.update(created_at="2026-10-01T09:30:00")
    ),
    "wrong type": lambda run_dir: _set_manifest(
        run_dir, lambda m: m.update(dataset="not an object")
    ),
}


@pytest.mark.parametrize("damage", list(MANIFEST_DAMAGE))
def test_damaged_manifest_is_rejected_before_any_provider_call_or_file_change(
    tmp_path, ten_dir, damage
):
    run_dir, config, dataset = partial_run(tmp_path, ten_dir)
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b'{"request_id": "c03", "requ')  # a torn tail that resume would truncate
    MANIFEST_DAMAGE[damage](run_dir)
    before, provider = snapshot(run_dir), CrashingProvider()
    with pytest.raises(CorruptRunError):
        resume_run(config, dataset, provider, tmp_path / "runs")
    assert provider.calls == () and snapshot(run_dir) == before  # nothing run, nothing truncated


# -- editing the dataset after a run has started --------------------------------------------------


def _edit_question(lines):
    return [lines[0] | {"question": "Edited question?"}, *lines[1:]]


def _reorder(lines):
    return [lines[1], lines[0], *lines[2:]]


def _drop_a_case(lines):
    return lines[:-1]


def _repin(directory):
    """What a maintainer does after an intended edit: recompute and store the new pin."""
    lines = (directory / "cases.jsonl").read_text(encoding="utf-8").split("\n")
    meta = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in lines if line]
    meta["content_sha256"] = compute_content_sha256(meta["task"], None, cases)
    (directory / "dataset.json").write_bytes(json.dumps(meta, indent=2).encode() + b"\n")


@pytest.mark.parametrize(
    ("edit", "differences"),
    [
        (_edit_question, {"dataset"}),
        (_reorder, {"dataset", "selection"}),
        (_drop_a_case, {"dataset", "selection"}),
    ],
)
def test_dataset_edit_after_run_start_blocks_loading_then_resume(
    tmp_path, ten_dir, edit, differences
):
    run_dir, config, _ = partial_run(tmp_path, ten_dir)
    before = snapshot(run_dir)

    cases_path = ten_dir / "cases.jsonl"
    lines = [
        json.loads(line) for line in cases_path.read_text(encoding="utf-8").split("\n") if line
    ]
    cases_path.write_bytes(
        b"".join(json.dumps(case, ensure_ascii=False).encode() + b"\n" for case in edit(lines))
    )
    with pytest.raises(DatasetError, match="does not match"):  # the pin is checked on load
        load_dataset(ten_dir)

    _repin(ten_dir)  # the edit is now an accepted new dataset version
    edited = load_dataset(ten_dir)
    provider = CrashingProvider()
    with pytest.raises(ResumeError) as info:
        resume_run(config, edited, provider, tmp_path / "runs")
    for name in differences:
        assert name in str(info.value)
    assert provider.calls == () and snapshot(run_dir) == before


# -- contract violations stop the run -------------------------------------------------------------


@pytest.mark.parametrize("kind", ["request_id", "requested_model", "provider", "not_a_result"])
def test_contract_violation_stops_the_run_and_later_cases_are_never_called(tmp_path, ten_dir, kind):
    config, dataset, runs = make_config("bad"), load_dataset(ten_dir), tmp_path / "runs"
    provider = ViolatingProvider(violate_on="c02", kind=kind)
    with pytest.raises(ProviderContractViolation):
        execute_run(config, dataset, provider, runs)
    assert [r.request_id for r in provider.calls] == ["c00", "c01", "c02"]  # nothing after c02
    assert [r.request_id for r in read_results(runs / "bad").lines] == ["c00", "c01"]

    before = snapshot(runs / "bad")
    again = ViolatingProvider(violate_on="c02", kind=kind)  # same implementation, same bug
    with pytest.raises(ProviderContractViolation):
        resume_run(config, dataset, again, runs)
    assert [r.request_id for r in again.calls] == ["c02"]  # stopped at the first bad case again
    assert snapshot(runs / "bad") == before  # and still no record for it


# -- failures and model identifiers persist faithfully --------------------------------------------


def test_all_nine_failure_kinds_persist_and_read_back(tmp_path):
    kinds = list(FailureKind)
    assert len(kinds) == 9
    cases = [qa(id=f"f{i}") for i in range(len(kinds) + 1)]
    directory = write_dataset(tmp_path, cases, dirname="failures")
    config, dataset = make_config("failures"), load_dataset(directory)
    provider = FakeProvider([*kinds, "and one success"])
    summary = execute_run(config, dataset, provider, tmp_path / "runs", **deterministic())

    lines = read_results(summary.run_dir).lines
    assert len(provider.calls) == 10  # one call per case: no failure was retried
    for line, kind in zip(lines, kinds, strict=False):
        result = line.execution.result
        assert isinstance(result, GenerationFailure)
        assert result.kind is kind and result.request_id == line.request_id
        assert result.retryable is (
            kind in {"timeout", "rate_limit", "server_error", "connection_error"}
        )
        assert line.execution.elapsed_s == 0.5
    assert lines[-1].execution.result.status == "ok"
    audit_run(summary.run_dir, config, directory)

    again = FakeProvider()
    assert resume_run(config, dataset, again, tmp_path / "runs", **deterministic()).executed == 0
    assert again.calls == ()  # recorded failures are completed attempts, never retried


def test_a_different_returned_model_is_recorded_not_rejected(tmp_path, ten_dir):
    config = make_config("alias")
    provider = FakeProvider(returned_model="fake-snapshot-2")
    summary = execute_run(config, load_dataset(ten_dir), provider, tmp_path / "runs")
    results = [line.execution.result for line in read_results(summary.run_dir).lines]
    assert {r.requested_model for r in results} == {"m"}
    assert {r.returned_model for r in results} == {"fake-snapshot-2"}
    audit_run(summary.run_dir, config, ten_dir)


# -- the child process installs its own network guard before it evaluates anything ----------------
# tests/conftest.py patches only the pytest process. The child's provider probes for the guard on
# every call (with invalid arguments, so nothing is ever sent), from inside execute_run.


def test_the_parent_process_guard_is_active_and_detected_by_the_probe():
    assert network_guard_is_active()  # tests/conftest.py is untouched and still in force


def test_child_installs_the_network_guard_before_evaluating(tmp_path):
    runs = tmp_path / "runs"
    child = run_child(
        FIXTURES / "tiny_qa", runs, "guarded", crash_after=99, torn=False, splits="dev"
    )
    assert child.returncode == 0, child.stderr
    # Both dev cases ran; the probe found the guard before each provider call.
    assert [r.request_id for r in read_results(runs / "guarded").lines] == [
        "qa-en-001",
        "qa-en-002",
    ]


def test_the_probe_fails_a_child_that_has_no_guard(tmp_path, monkeypatch):
    # Negative control: this is what a removed, or too late, block_network() would look like. The
    # switch is honoured only by child_main and only inside the child process.
    monkeypatch.setenv(SKIP_GUARD_FLAG, "1")
    runs = tmp_path / "runs"
    child = run_child(
        FIXTURES / "tiny_qa", runs, "unguarded", crash_after=99, torn=False, splits="dev"
    )
    assert child.returncode == GUARD_MISSING_EXIT_CODE, child.stderr
    assert read_results(runs / "unguarded").lines == ()  # it stopped at the first provider call
