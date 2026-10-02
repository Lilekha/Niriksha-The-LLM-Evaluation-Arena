"""The offline run reader: what it accepts, and every way it must refuse."""

import json
from pathlib import Path

import pytest

from e2e_support import CrashingProvider, make_config
from niriksha.core.dataset import DatasetError, compute_content_sha256, load_dataset
from niriksha.core.execution import execute_run
from niriksha.core.generation import FailureKind
from niriksha.core.runload import LoadedRun, RunIntegrityError, load_run
from niriksha.core.runstore import CorruptRunError
from score_helpers import (
    SPLITS,
    copy_fixture,
    deterministic,
    edit_manifest,
    finished_run,
    interrupted_run,
    result_lines,
    snapshot,
    write_result_lines,
)

# -- valid runs are accepted ----------------------------------------------------------------------


def test_a_complete_qa_run_loads_with_aligned_cases_requests_and_results(tmp_path):
    run = finished_run(tmp_path)
    loaded = load_run(run.run_dir, run.dataset_dir)
    ids = ["qa-en-001", "qa-en-002", "qa-hi-001", "qa-kn-001", "qa-hi-002"]
    assert [c.id for c in loaded.cases] == ids
    assert [r.request_id for r in loaded.requests] == ids
    assert [r.request_id for r in loaded.results] == ids
    assert loaded.config == run.config  # rebuilt from the manifest alone
    assert loaded.manifest.dataset.content_sha256 == loaded.dataset.content_sha256
    assert loaded.run_dir == run.run_dir


def test_an_extraction_run_loads(tmp_path):
    run = finished_run(tmp_path, fixture="tiny_extraction")
    loaded = load_run(run.run_dir, run.dataset_dir)
    assert [c.id for c in loaded.cases] == ["ex-en-001", "ex-en-002", "ex-hi-001"]


def test_a_run_that_crashed_and_was_resumed_loads(tmp_path):
    run = finished_run(tmp_path, crash_after=2)
    assert len(load_run(run.run_dir, run.dataset_dir).results) == 5


def test_runs_containing_failures_load_because_failures_are_valid_results(tmp_path):
    script = list(FailureKind)[:5]
    run = finished_run(tmp_path, script=script)
    assert len(load_run(run.run_dir, run.dataset_dir).results) == 5


def test_loading_is_read_only_and_repeatable(tmp_path):
    run = finished_run(tmp_path)
    before = snapshot(run.run_dir, run.dataset_dir)
    first = load_run(run.run_dir, str(run.dataset_dir))  # str and Path are both accepted
    second = load_run(str(run.run_dir), run.dataset_dir)
    assert first == second
    assert snapshot(run.run_dir, run.dataset_dir) == before


def test_a_hand_built_loaded_run_with_misaligned_parts_is_refused(tmp_path):
    run = finished_run(tmp_path)
    loaded = load_run(run.run_dir, run.dataset_dir)
    with pytest.raises(ValueError, match="aligned"):
        LoadedRun(
            run_dir=loaded.run_dir,
            manifest=loaded.manifest,
            dataset=loaded.dataset,
            config=loaded.config,
            cases=loaded.cases,
            requests=loaded.requests[:-1],
            results=loaded.results,
        )


# -- every refusal --------------------------------------------------------------------------------


def _delete_results(run):
    (run.run_dir / "results.jsonl").unlink()


def _torn_tail(run):
    with open(run.run_dir / "results.jsonl", "ab") as handle:
        handle.write(b'{"request_id": "qa-hi-002", "requ')


def _corrupt_middle(run):
    lines = result_lines(run.run_dir)
    lines[1] = b"garbage"
    write_result_lines(run.run_dir, lines)


def _blank_line(run):
    path = run.run_dir / "results.jsonl"
    lines = path.read_bytes().split(b"\n")
    path.write_bytes(b"\n".join([lines[0], b"", *lines[1:]]))


def _duplicate_result(run):
    lines = result_lines(run.run_dir)
    write_result_lines(run.run_dir, [*lines, lines[0]])


def _tamper_request_hash(run):
    lines = result_lines(run.run_dir)
    record = json.loads(lines[0])
    record["request_sha256"] = "0" * 64
    lines[0] = json.dumps(record).encode()
    write_result_lines(run.run_dir, lines)


def _swap_order(run):
    lines = result_lines(run.run_dir)
    lines[0], lines[1] = lines[1], lines[0]
    write_result_lines(run.run_dir, lines)


def _foreign_id_replacing_a_result(run):
    lines = result_lines(run.run_dir)
    record = json.loads(lines[2])
    record["request_id"] = "not-in-the-selection"
    record["execution"]["result"]["request_id"] = "not-in-the-selection"
    lines[2] = json.dumps(record).encode()
    write_result_lines(run.run_dir, lines)


def _extra_foreign_result(run):
    lines = result_lines(run.run_dir)
    record = json.loads(lines[0])
    record["request_id"] = "an-extra-case"
    record["execution"]["result"]["request_id"] = "an-extra-case"
    write_result_lines(run.run_dir, [*lines, json.dumps(record).encode()])


def _edit_line(run, index, change):
    lines = result_lines(run.run_dir)
    record = json.loads(lines[index])
    change(record)
    lines[index] = json.dumps(record).encode()
    write_result_lines(run.run_dir, lines)


def _result_from_another_provider(run):
    _edit_line(run, 1, lambda r: r["execution"]["result"].update(provider="impostor"))


def _result_from_another_model(run):
    _edit_line(run, 1, lambda r: r["execution"]["result"].update(requested_model="another-model"))


def _result_with_another_identity(run):
    _edit_line(run, 1, lambda r: r["execution"]["result"].update(request_id="someone-else"))


def _edit_case(run):
    path = run.dataset_dir / "cases.jsonl"
    lines = path.read_text(encoding="utf-8").split("\n")
    first = json.loads(lines[0])
    first["question"] = "An edited question?"
    lines[0] = json.dumps(first, ensure_ascii=False)
    path.write_bytes("\n".join(lines).encode("utf-8"))


def _repin(directory: Path):
    meta = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    raw = [
        json.loads(line)
        for line in (directory / "cases.jsonl").read_text(encoding="utf-8").split("\n")
        if line
    ]
    meta["content_sha256"] = compute_content_sha256(meta["task"], meta.get("output_schema"), raw)
    (directory / "dataset.json").write_bytes(json.dumps(meta, indent=2).encode() + b"\n")


def _edit_case_and_repin(run):
    _edit_case(run)
    _repin(run.dataset_dir)


def _bump_dataset_version(run):
    path = run.dataset_dir / "dataset.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["version"] = "0.2.0"
    path.write_bytes(json.dumps(meta, indent=2).encode() + b"\n")


def _manifest_missing(run):
    (run.run_dir / "manifest.json").unlink()


def _manifest_garbage(run):
    (run.run_dir / "manifest.json").write_bytes(b"{not json")


def _set(path, value):
    def change(manifest):
        target = manifest
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


REFUSALS = {
    "results file missing": (_delete_results, "incomplete: 0 of 5", None),
    "torn final line": (_torn_tail, "results are unusable", CorruptRunError),
    "corrupt middle line": (_corrupt_middle, "results are unusable", CorruptRunError),
    "blank line": (_blank_line, "results are unusable", CorruptRunError),
    "duplicate request id": (_duplicate_result, "results are unusable", CorruptRunError),
    "tampered request hash": (_tamper_request_hash, "request hash", None),
    "results out of order": (_swap_order, "selection order", None),
    "foreign id replaces a result": (_foreign_id_replacing_a_result, "outside the selection", None),
    "extra foreign result": (_extra_foreign_result, "outside the selection", None),
    "result from another provider": (_result_from_another_provider, "different provider", None),
    "result from another model": (_result_from_another_model, "different provider", None),
    "result with another identity": (
        _result_with_another_identity,
        "results are unusable",
        CorruptRunError,
    ),
    "dataset edited in place": (_edit_case, "no longer matches its pin", DatasetError),
    "dataset edited and re-pinned": (_edit_case_and_repin, "content_sha256", None),
    "dataset version bumped": (_bump_dataset_version, "version", None),
    "manifest missing": (_manifest_missing, "manifest is unusable", CorruptRunError),
    "manifest not json": (_manifest_garbage, "manifest is unusable", CorruptRunError),
    "manifest unknown version": (
        lambda run: edit_manifest(run.run_dir, _set(["manifest_version"], 2)),
        "manifest is unusable",
        CorruptRunError,
    ),
    "manifest unknown field": (
        lambda run: edit_manifest(run.run_dir, _set(["api_key"], "x")),
        "manifest is unusable",
        CorruptRunError,
    ),
    "manifest wrong type": (
        lambda run: edit_manifest(run.run_dir, _set(["dataset"], "not an object")),
        "manifest is unusable",
        CorruptRunError,
    ),
    "manifest prompt text changed": (
        lambda run: edit_manifest(run.run_dir, _set(["prompt", "user"], "Changed: {input}")),
        "prompt hash",
        None,
    ),
    "manifest prompt without placeholder": (
        lambda run: edit_manifest(run.run_dir, _set(["prompt", "user"], "no placeholder")),
        "prompt template",
        Exception,
    ),
    "manifest selection hash changed": (
        lambda run: edit_manifest(run.run_dir, _set(["selection", "case_ids_sha256"], "0" * 64)),
        "selection hash",
        None,
    ),
    "manifest case count changed": (
        lambda run: edit_manifest(run.run_dir, _set(["selection", "case_count"], 4)),
        "records 4",
        None,
    ),
    "manifest splits changed": (
        lambda run: edit_manifest(run.run_dir, _set(["selection", "splits"], ["dev"])),
        "selection has 2 cases",
        None,
    ),
    "manifest model changed": (
        lambda run: edit_manifest(run.run_dir, _set(["requested_model"], "another-model")),
        "request hash",
        None,
    ),
    "manifest parameters changed": (
        lambda run: edit_manifest(run.run_dir, _set(["params", "temperature"], 0.5)),
        "request hash",
        None,
    ),
    "manifest provider changed": (
        lambda run: edit_manifest(run.run_dir, _set(["provider", "name"], "impostor")),
        "different provider",
        None,
    ),
}  # fmt: skip


@pytest.mark.parametrize("name", list(REFUSALS))
def test_load_run_refuses_and_never_modifies_files(tmp_path, name):
    tamper, message, cause = REFUSALS[name]
    run = finished_run(tmp_path)
    tamper(run)
    before = snapshot(run.run_dir, run.dataset_dir)
    with pytest.raises(RunIntegrityError, match=message) as info:
        load_run(run.run_dir, run.dataset_dir)
    if cause is not None:
        assert isinstance(info.value.__cause__, cause)
    assert (
        snapshot(run.run_dir, run.dataset_dir) == before
    )  # nothing truncated, repaired or written


def test_an_interrupted_run_is_incomplete_and_a_torn_tail_is_not_repaired(tmp_path):
    run = interrupted_run(tmp_path, completed=2)
    with pytest.raises(RunIntegrityError, match="incomplete: 2 of 5"):
        load_run(run.run_dir, run.dataset_dir)
    _torn_tail(run)
    before = snapshot(run.run_dir)
    with pytest.raises(RunIntegrityError, match="results are unusable"):
        load_run(run.run_dir, run.dataset_dir)
    assert snapshot(run.run_dir) == before  # resuming repairs a torn tail; the reader never does


def test_a_run_moved_under_another_directory_name_is_refused(tmp_path):
    run = finished_run(tmp_path)
    moved = run.run_dir.rename(run.run_dir.parent / "renamed")
    with pytest.raises(RunIntegrityError, match="run directory name"):
        load_run(moved, run.dataset_dir)


def test_a_different_dataset_directory_is_refused(tmp_path):
    run = finished_run(tmp_path)
    other = copy_fixture(tmp_path, "tiny_extraction")
    with pytest.raises(RunIntegrityError, match="differs from the one the run used"):
        load_run(run.run_dir, other)


def test_a_missing_dataset_directory_is_refused_as_an_integrity_error(tmp_path):
    run = finished_run(tmp_path)
    with pytest.raises(RunIntegrityError) as info:
        load_run(run.run_dir, tmp_path / "does-not-exist")
    assert isinstance(info.value.__cause__, DatasetError)


def test_the_original_dataset_still_loads_after_a_refused_edit(tmp_path):
    run = finished_run(tmp_path)
    _edit_case(run)
    with pytest.raises(RunIntegrityError):
        load_run(run.run_dir, run.dataset_dir)
    with pytest.raises(DatasetError):  # the loader refuses it too; nothing was "fixed" for us
        load_dataset(run.dataset_dir)


def test_a_missing_run_directory_is_refused(tmp_path):
    run = finished_run(tmp_path)
    with pytest.raises(RunIntegrityError, match="manifest is unusable"):
        load_run(Path(tmp_path) / "no-such-run", run.dataset_dir)


def test_the_returned_model_is_preserved_and_never_replaced_by_the_requested_one(tmp_path):
    dataset_dir = copy_fixture(tmp_path, "tiny_qa")
    config = make_config("alias", SPLITS["tiny_qa"])
    provider = CrashingProvider(returned_model="fake-snapshot-2")
    execute_run(config, load_dataset(dataset_dir), provider, tmp_path / "runs", **deterministic())

    loaded = load_run(tmp_path / "runs" / "alias", dataset_dir)  # recorded, not flagged
    results = [line.execution.result for line in loaded.results]
    assert {r.requested_model for r in results} == {"m"} == {loaded.manifest.requested_model}
    assert {r.returned_model for r in results} == {"fake-snapshot-2"}
    assert {request.model for request in loaded.requests} == {"m"}  # requests keep the request


def test_a_failure_result_keeps_its_absent_returned_model(tmp_path):
    run = finished_run(tmp_path, script=[FailureKind.TIMEOUT] * 5)
    results = [line.execution.result for line in load_run(run.run_dir, run.dataset_dir).results]
    assert {r.returned_model for r in results} == {None}  # nothing is filled in for a failure
