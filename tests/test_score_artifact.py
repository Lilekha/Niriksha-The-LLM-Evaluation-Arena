"""Score artifacts: schema, persistence, strict reading, and the records_sha256 definition."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from niriksha.core.runload import load_run
from niriksha.core.runstore import MANIFEST_FILE, RESULTS_FILE, ids_sha256
from niriksha.core.scorestore import (
    ARTIFACT_VERSION,
    ScoreArtifact,
    ScoreArtifactError,
    artifact_id,
    artifact_path,
    build_artifact,
    read_artifact,
    records_hash,
    serialize_artifact,
    source_from_run,
    write_artifact,
)
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from score_helpers import finished_run, snapshot

NAME, VERSION, TASK = "normalized_exact_match", "0.1.0", "short_answer_qa"
VALUES = [1.0, 0.0, 1.0, 0.5, 0.0]


def make_records(loaded, values=None, *, failed=()):
    """Hand-built records for a QA run's cases; ``failed`` lists the indexes not scored."""
    records = []
    for index, (case, value) in enumerate(zip(loaded.cases, values or VALUES, strict=True)):
        if index in failed:
            records.append(
                ScoreRecord(
                    request_id=case.id,
                    metric=NAME,
                    metric_version=VERSION,
                    status=ScoreStatus.NOT_SCORED,
                    reason="generation failed: timeout",
                    details={"failure_kind": "timeout"},
                )
            )
        else:
            records.append(
                ScoreRecord(
                    request_id=case.id,
                    metric=NAME,
                    metric_version=VERSION,
                    status=ScoreStatus.SCORED,
                    value=value,
                )
            )
    return tuple(records)


@pytest.fixture
def made(tmp_path):
    run = finished_run(tmp_path)
    loaded = load_run(run.run_dir, run.dataset_dir)
    artifact = build_artifact(source_from_run(loaded), NAME, VERSION, TASK, make_records(loaded))
    return run, loaded, artifact, tmp_path / "scores"


# -- round trip and determinism -------------------------------------------------------------------


def test_write_read_round_trip_and_location(made):
    run, _, artifact, scores = made
    run_files_before = snapshot(run.run_dir)
    path = write_artifact(scores, artifact)
    assert path == scores / f"{run.config.run_id}--{NAME}--{VERSION}.json"
    assert path == artifact_path(scores, artifact.artifact_id)
    assert read_artifact(path) == artifact
    assert snapshot(run.run_dir) == run_files_before  # the run is untouched
    assert sorted(p.name for p in run.run_dir.iterdir()) == [MANIFEST_FILE, RESULTS_FILE]


def test_the_same_artifact_serialises_to_identical_bytes_anywhere(made, tmp_path):
    _, _, artifact, scores = made
    first = write_artifact(scores, artifact).read_bytes()
    second = write_artifact(tmp_path / "elsewhere", artifact).read_bytes()
    assert first == second == serialize_artifact(artifact)
    assert first.endswith(b"\n") and b"\r" not in first and b'"created_at"' not in first


def test_write_never_overwrites(made):
    _, _, artifact, scores = made
    path = write_artifact(scores, artifact)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_artifact(scores, artifact)
    assert path.read_bytes() == before


def test_a_failed_write_removes_its_partial_file(made, monkeypatch):
    import niriksha.core.scorestore as scorestore

    _, _, artifact, scores = made

    def broken(fd):
        raise OSError("disk full")

    monkeypatch.setattr(scorestore.os, "fsync", broken)
    with pytest.raises(OSError, match="disk full"):
        write_artifact(scores, artifact)
    monkeypatch.undo()
    assert list(scores.iterdir()) == []  # nothing left behind to block the name
    assert write_artifact(scores, artifact).is_file()


def _fake_open(target, exc=None, wrap=None):
    """A stand-in for ``open`` inside scorestore: refuse the exclusive create of ``target`` with
    ``exc``, or wrap the handle it returns so that a later write fails."""
    real_open = open

    def fake_open(file, mode="r", *args, **kwargs):
        if Path(file) == target and "x" in mode:
            if exc is not None:
                raise exc
            return wrap(real_open(file, mode, *args, **kwargs))
        return real_open(file, mode, *args, **kwargs)

    return fake_open


@pytest.mark.parametrize(
    "exc", [PermissionError("denied"), OSError("i/o error"), KeyboardInterrupt()], ids=type
)
def test_a_failed_exclusive_create_never_deletes_an_existing_artifact(made, monkeypatch, exc):
    import niriksha.core.scorestore as scorestore

    _, _, artifact, scores = made
    path = write_artifact(scores, artifact)
    original = path.read_bytes()
    monkeypatch.setattr(scorestore, "open", _fake_open(path, exc=exc), raising=False)
    with pytest.raises(type(exc)):
        write_artifact(scores, artifact)  # the create fails, but not with "file exists"
    monkeypatch.undo()
    assert path.exists() and path.read_bytes() == original  # the artifact is not ours to delete
    assert read_artifact(path) == artifact


class _FailsAfterCreate:
    """Wraps the handle of a just-created file: half of the data is written, then it fails."""

    def __init__(self, handle, exc):
        self._handle, self._exc = handle, exc

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._handle.close()

    def write(self, data):
        self._handle.write(data[: len(data) // 2])
        self._handle.flush()
        raise self._exc

    def __getattr__(self, name):
        return getattr(self._handle, name)


@pytest.mark.parametrize("exc", [OSError("disk full"), KeyboardInterrupt()], ids=type)
def test_a_failed_write_after_creation_removes_only_its_own_file(made, monkeypatch, exc):
    import niriksha.core.scorestore as scorestore

    _, _, artifact, scores = made
    scores.mkdir()
    bystander = scores / "unrelated.json"
    bystander.write_bytes(b"keep me")
    target = artifact_path(scores, artifact.artifact_id)
    wrap = lambda handle: _FailsAfterCreate(handle, exc)  # noqa: E731
    monkeypatch.setattr(scorestore, "open", _fake_open(target, wrap=wrap), raising=False)
    with pytest.raises(type(exc)):
        write_artifact(scores, artifact)
    monkeypatch.undo()
    assert sorted(p.name for p in scores.iterdir()) == ["unrelated.json"]  # partial file removed
    assert bystander.read_bytes() == b"keep me"
    assert write_artifact(scores, artifact).is_file()  # and the name is not left blocked


def test_a_regular_file_used_as_scores_dir_is_a_typed_error_and_left_alone(made, tmp_path):
    _, _, artifact, _ = made
    not_a_directory = tmp_path / "not_a_directory"
    not_a_directory.write_bytes(b"precious")
    for target in (not_a_directory, not_a_directory / "below"):
        with pytest.raises(ScoreArtifactError, match="scores_dir"):
            write_artifact(target, artifact)
    assert not_a_directory.read_bytes() == b"precious"


def test_a_directory_at_the_artifact_path_is_a_typed_error_and_left_alone(made):
    _, _, artifact, scores = made
    occupied = artifact_path(scores, artifact.artifact_id)
    occupied.mkdir(parents=True)
    with pytest.raises(ScoreArtifactError, match="directory occupies"):
        write_artifact(scores, artifact)
    with pytest.raises(ScoreArtifactError, match="is a directory"):
        read_artifact(occupied)
    assert occupied.is_dir()


def test_the_source_identity_holds_the_run_file_hashes_and_run_identity(made):
    run, loaded, artifact, _ = made
    source = artifact.source
    assert (
        source.run_manifest_sha256
        == hashlib.sha256((run.run_dir / MANIFEST_FILE).read_bytes()).hexdigest()
    )
    assert (
        source.run_results_sha256
        == hashlib.sha256((run.run_dir / RESULTS_FILE).read_bytes()).hexdigest()
    )
    assert source.run_id == loaded.manifest.run_id == run.config.run_id
    assert source.dataset == loaded.manifest.dataset
    assert source.selection.case_ids_sha256 == loaded.manifest.selection.case_ids_sha256
    assert source.prompt_sha256 == loaded.manifest.prompt.sha256
    assert (source.provider, source.requested_model) == ("fake", "m")


def test_artifact_ids_are_bounded_and_derived_from_validated_parts():
    assert artifact_id("r1", NAME, VERSION) == f"r1--{NAME}--{VERSION}"
    with pytest.raises(ValueError, match="longer than"):
        artifact_id("r" * 64, "m" * 64, "99999999.99999999.99999999")


# -- strict reading: every way an artifact can be wrong -------------------------------------------


def rewrite(path: Path, change, *, rehash=False):
    """Edit the parsed document; optionally recompute records_sha256 with the stdlib so that the
    edit trips a specific validator instead of the hash check."""
    document = json.loads(path.read_bytes())
    change(document)
    if rehash:
        document["records_sha256"] = independent_hash(document["records"])
    path.write_bytes(json.dumps(document, indent=2).encode() + b"\n")


def independent_hash(records) -> str:
    text = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _set(*keys, value):
    def change(document):
        target = document
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return change


def _swap_records(document):
    records = document["records"]
    records[0], records[1] = records[1], records[0]


CORRUPTIONS = {
    "empty file": (lambda p: p.write_bytes(b""), "invalid JSON", False),
    "whitespace only": (lambda p: p.write_bytes(b"  \n\n"), "invalid JSON", False),
    "truncated": (lambda p: p.write_bytes(p.read_bytes()[:200]), "invalid JSON", False),
    "invalid utf-8": (lambda p: p.write_bytes(b"\xff\xfe{}"), "invalid UTF-8", False),
    "byte-order mark": (
        lambda p: p.write_bytes(bytes([0xEF, 0xBB, 0xBF]) + p.read_bytes()),
        "byte-order mark",
        False,
    ),
    "not json": (lambda p: p.write_bytes(b"not json"), "invalid JSON", False),
    "trailing data": (lambda p: p.write_bytes(p.read_bytes() + b"x"), "invalid JSON", False),
    "second document": (
        lambda p: p.write_bytes(p.read_bytes() + p.read_bytes()),
        "invalid JSON",
        False,
    ),
    "top-level array": (lambda p: p.write_bytes(b"[1, 2]"), "JSON object", False),
    "duplicate key": (
        lambda p: p.write_bytes(
            p.read_bytes().replace(
                b'"artifact_version": 1,', b'"artifact_version": 1, "artifact_version": 1,', 1
            )
        ),
        "duplicate JSON key",
        False,
    ),
    "NaN literal": (
        lambda p: p.write_bytes(p.read_bytes().replace(b'"value": 0.5', b'"value": NaN', 1)),
        "invalid JSON",
        False,
    ),
    "unsupported version": (
        lambda p: rewrite(p, _set("artifact_version", value=2)),
        "unsupported artifact_version",
        False,
    ),
    "missing version": (
        lambda p: rewrite(p, lambda d: d.pop("artifact_version")),
        "unsupported artifact_version",
        False,
    ),
    "unknown field": (lambda p: rewrite(p, _set("api_key", value="x")), "invalid artifact", False),
    "missing field": (lambda p: rewrite(p, lambda d: d.pop("source")), "invalid artifact", False),
    "wrong artifact_id": (
        lambda p: rewrite(p, _set("artifact_id", value="other--x--0.1.0")),
        "artifact_id",
        False,
    ),
    "wrong records_sha256": (
        lambda p: rewrite(p, _set("records_sha256", value="0" * 64)),
        "records_sha256",
        False,
    ),
    "value changed": (
        lambda p: rewrite(p, _set("records", 0, "value", value=0.0)),
        "records_sha256",
        False,
    ),
    "records swapped": (lambda p: rewrite(p, _swap_records, rehash=True), "selection hash", False),
    "record dropped": (
        lambda p: rewrite(p, lambda d: d["records"].pop(), rehash=True),
        "number of records",
        False,
    ),
    "record duplicated": (
        lambda p: rewrite(p, lambda d: d["records"].append(dict(d["records"][0])), rehash=True),
        "duplicate request_id",
        False,
    ),
    "unexpected case id": (
        lambda p: rewrite(p, _set("records", 2, "request_id", value="not-a-case"), rehash=True),
        "selection hash",
        False,
    ),
    "score above one": (
        lambda p: rewrite(p, _set("records", 0, "value", value=1.5), rehash=True),
        "invalid artifact",
        False,
    ),
    "negative score": (
        lambda p: rewrite(p, _set("records", 0, "value", value=-0.1), rehash=True),
        "invalid artifact",
        False,
    ),
    "scored without value": (
        lambda p: rewrite(p, _set("records", 0, "value", value=None), rehash=True),
        "invalid artifact",
        False,
    ),
    "not_scored with value": (
        lambda p: rewrite(
            p, lambda d: d["records"][0].update(status="not_scored", reason="x"), rehash=True
        ),
        "invalid artifact",
        False,
    ),
    "unknown status": (
        lambda p: rewrite(p, _set("records", 0, "status", value="maybe"), rehash=True),
        "invalid artifact",
        False,
    ),
    "invalid metric name": (
        lambda p: rewrite(p, _set("metric", "name", value="Bad Metric")),
        "invalid artifact",
        False,
    ),
    "invalid metric version": (
        lambda p: rewrite(p, _set("metric", "version", value="1.0")),
        "invalid artifact",
        False,
    ),
    "record of another metric": (
        lambda p: rewrite(p, _set("records", 0, "metric", value="field_exact_match"), rehash=True),
        "different metric",
        False,
    ),
    "record of another version": (
        lambda p: rewrite(p, _set("records", 0, "metric_version", value="9.9.9"), rehash=True),
        "different metric",
        False,
    ),
    "case count changed": (
        lambda p: rewrite(p, _set("source", "selection", "case_count", value=4)),
        "number of records",
        False,
    ),
    "bad run file hash": (
        lambda p: rewrite(p, _set("source", "run_results_sha256", value="xyz")),
        "invalid artifact",
        False,
    ),
}


@pytest.mark.parametrize("name", list(CORRUPTIONS))
def test_read_artifact_refuses_a_damaged_file_and_does_not_modify_it(made, name):
    _, _, artifact, scores = made
    path = write_artifact(scores, artifact)
    corrupt, message, _ = CORRUPTIONS[name]
    corrupt(path)
    before = snapshot(scores)
    with pytest.raises(ScoreArtifactError, match=message):
        read_artifact(path)
    assert snapshot(scores) == before


def test_the_model_itself_rejects_an_artifact_id_that_does_not_match_its_source(made):
    _, _, artifact, _ = made
    document = json.loads(serialize_artifact(artifact))
    document["artifact_id"] = "other--x--0.1.0"
    with pytest.raises(ValidationError, match="artifact_id does not match"):
        ScoreArtifact.model_validate_json(json.dumps(document))


def test_a_missing_artifact_is_a_typed_error(tmp_path):
    with pytest.raises(ScoreArtifactError, match="not found"):
        read_artifact(tmp_path / "no-such--normalized_exact_match--0.1.0.json")


def test_a_file_whose_name_is_not_its_artifact_id_is_refused(made):
    _, _, artifact, scores = made
    path = write_artifact(scores, artifact)
    renamed = path.rename(path.with_name("renamed.json"))
    with pytest.raises(ScoreArtifactError, match="file name"):
        read_artifact(renamed)


def test_refusal_messages_do_not_echo_stored_values(made):
    _, _, artifact, scores = made
    path = write_artifact(scores, artifact)
    rewrite(path, _set("records", 0, "value", value="SENTINEL-TEXT"))
    with pytest.raises(ScoreArtifactError) as info:
        read_artifact(path)
    assert "SENTINEL-TEXT" not in str(info.value)


def test_all_not_scored_and_mixed_artifacts_are_valid(made):
    _, loaded, _, scores = made
    mixed = build_artifact(
        source_from_run(loaded), NAME, VERSION, TASK, make_records(loaded, failed=(1, 4))
    )
    assert read_artifact(write_artifact(scores, mixed)) == mixed
    everything = build_artifact(
        source_from_run(loaded), NAME, VERSION, TASK, make_records(loaded, failed=range(5))
    )
    assert all(r.status is ScoreStatus.NOT_SCORED for r in everything.records)
    assert ARTIFACT_VERSION == 1


# -- records_sha256: the canonical bytes, pinned independently of the production helper ----------
# Expected values come from the stdlib code in this file, written from the definition in the
# scorestore module docstring. The digests are golden values with a narrow per-line scanner
# allowlist, since a SHA-256 digest looks like a high-entropy secret to detect-secrets.

E_ACUTE = chr(0xE9)
SHORT = [
    {
        "details": {"empty_output": False, "matched_answer_index": 0},
        "metric": NAME,
        "metric_version": VERSION,
        "reason": None,
        "request_id": "q1",
        "status": "scored",
        "value": 1.0,
    },
    {
        "details": {"failure_kind": "timeout"},
        "metric": NAME,
        "metric_version": VERSION,
        "reason": "generation failed: timeout",
        "request_id": "q2",
        "status": "not_scored",
        "value": None,
    },
]
FULL = [
    *SHORT,
    {
        "details": {"matched_fields": 1, "output_is_json_object": True},
        "metric": NAME,
        "metric_version": VERSION,
        "reason": None,
        "request_id": "q3",
        "status": "scored",
        "value": 1 / 3,
    },
    {
        "details": {},
        "metric": NAME,
        "metric_version": VERSION,
        "reason": "echec " + E_ACUTE,
        "request_id": "q4",
        "status": "not_scored",
        "value": None,
    },
]
GOLDEN = {
    "short": "19b8dd9569fa642a656b05187310faceef97a262e39e18b4be1b920733336716",  # pragma: allowlist secret  # noqa: E501
    "full": "07aadfe3d43dbb53dfe096d8f4f4ba78dbb490453055486f945c229f7d206558",  # pragma: allowlist secret  # noqa: E501
}


def as_records(documents):
    return tuple(
        ScoreRecord(
            request_id=d["request_id"],
            metric=d["metric"],
            metric_version=d["metric_version"],
            status=ScoreStatus(d["status"]),
            value=d["value"],
            reason=d["reason"],
            details=d["details"],
        )
        for d in documents
    )


def test_records_sha256_is_the_sha256_of_the_exact_canonical_record_array():
    canonical = json.dumps(SHORT, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert canonical == (
        '[{"details":{"empty_output":false,"matched_answer_index":0},"metric":"normalized_exact_match",'
        '"metric_version":"0.1.0","reason":null,"request_id":"q1","status":"scored","value":1.0},'
        '{"details":{"failure_kind":"timeout"},"metric":"normalized_exact_match","metric_version":"0.1.0",'
        '"reason":"generation failed: timeout","request_id":"q2",'
        '"status":"not_scored","value":null}]'
    )  # sorted keys, no whitespace, null for absent values, the float 1.0 written as 1.0
    assert independent_hash(SHORT) == GOLDEN["short"] == records_hash(as_records(SHORT))


def test_the_canonical_form_keeps_raw_utf8_and_the_shortest_float_repr():
    canonical = json.dumps(FULL, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert "echec " + E_ACUTE in canonical and "\\u00e9" not in canonical  # raw UTF-8, no escapes
    assert '"value":0.3333333333333333' in canonical
    assert independent_hash(FULL) == GOLDEN["full"] == records_hash(as_records(FULL))


def test_records_sha256_depends_on_order_values_and_types():
    base = records_hash(as_records(FULL))
    assert records_hash(as_records(FULL[::-1])) != base  # order matters
    changed = [dict(d) for d in FULL]
    changed[0]["value"] = 0.0
    assert records_hash(as_records(changed)) != base
    assert records_hash(as_records(FULL[:3])) != base


def test_selection_ids_hash_in_an_artifact_is_the_existing_run_selection_hash(made):
    _, loaded, artifact, _ = made
    ids = [c.id for c in loaded.cases]
    assert artifact.source.selection.case_ids_sha256 == ids_sha256(ids)
    assert isinstance(artifact, ScoreArtifact)
