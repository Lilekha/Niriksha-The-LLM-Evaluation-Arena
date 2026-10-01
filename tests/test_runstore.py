import hashlib
import json
import math
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

import niriksha.core.runstore as runstore
from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationParams,
    GenerationRequest,
    Message,
)
from niriksha.core.runstore import (
    CorruptRunError,
    PersistenceError,
    PromptTemplate,
    RunConfig,
    RunManifest,
    RunResultLine,
    append_result,
    create_run,
    ids_sha256,
    read_manifest,
    read_results,
    request_sha256,
    truncate_torn_tail,
    validate_run_id,
)
from run_helpers import (
    HASH_A,
    NOW,
    make_line,
    make_manifest,
    make_request,
    make_success,
    manifest_data,
)

# -- run ids and configuration --------------------------------------------------------------------


@pytest.mark.parametrize("run_id", ["run1", "a-b_c", "0abc", "x" * 64, "a"])
def test_valid_run_ids(run_id):
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "", "Run1", "RUN", "-a", "_a", "a.b", "a b", "a/b", "a\\b", "..", ".", "x" * 65, " run",
        "run\n", "con", "nul", "aux", "prn", "com1", "com9", "lpt1", "lpt9", "ünï",
    ],
)  # fmt: skip
def test_invalid_run_ids(run_id):
    with pytest.raises(ValueError):
        validate_run_id(run_id)


def test_prompt_template_needs_exactly_one_placeholder():
    PromptTemplate(user="Q: {input}")
    for bad in ("no placeholder", "{input} and {input}", " ", ""):
        with pytest.raises(ValidationError):
            PromptTemplate(user=bad)


def test_prompt_rendering_uses_replace_not_format():
    template = PromptTemplate(system="Be brief {not_a_field}", user='Return {"answer": "{input}"}')
    system, user = template.render("what is {x}? {input}")
    assert system.role == "system" and system.content == "Be brief {not_a_field}"
    assert user.content == 'Return {"answer": "what is {x}? {input}"}'


def test_prompt_without_system_renders_one_message():
    assert len(PromptTemplate(user="{input}").render("hi")) == 1


def test_prompt_hash_is_stable_and_sensitive():
    base = PromptTemplate(user="Q: {input}")
    assert base.sha256 == PromptTemplate(user="Q: {input}").sha256
    assert base.sha256 != PromptTemplate(user="Q:  {input}").sha256
    assert base.sha256 != PromptTemplate(system="s", user="Q: {input}").sha256
    assert len(base.sha256) == 64


def test_run_config_validation():
    prompt = PromptTemplate(user="{input}")
    RunConfig(run_id="run1", splits=("dev", "test"), model="m", prompt=prompt)
    for bad in (
        {"splits": ()},
        {"splits": ("dev", "dev")},
        {"splits": ("train",)},
        {"run_id": "Run1"},
        {"model": " "},
        {"unknown": 1},
    ):
        fields = {"run_id": "run1", "splits": ("dev",), "model": "m", "prompt": prompt}
        with pytest.raises(ValidationError):
            RunConfig(**{**fields, **bad})


# -- manifest -------------------------------------------------------------------------------------


def test_manifest_round_trips_through_json():
    manifest = make_manifest()
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(api_key="x"),
        lambda d: d["provider"].update(base_url="http://u:p@host"),
        lambda d: d["software"].update(hostname="box"),
        lambda d: d.update(manifest_version=2),
        lambda d: d.update(created_at="2026-10-01T09:30:00"),  # naive
        lambda d: d["dataset"].update(content_sha256="abc"),
        lambda d: d["dataset"].update(task="summarisation"),
        lambda d: d["selection"].update(case_count=0),
        lambda d: d["selection"].update(splits=["train"]),
        lambda d: d["prompt"].update(sha256="G" * 64),
        lambda d: d["params"].update(temperature=-1),
        lambda d: d["software"].update(git_dirty=None),
        lambda d: d.update(run_id="Bad Id"),
        lambda d: d.pop("software"),
    ],
)
def test_invalid_manifests_are_rejected(mutate):
    data = manifest_data()
    mutate(data)
    with pytest.raises(ValidationError):
        RunManifest.model_validate_json(json.dumps(data))


def test_manifest_contains_no_paths_or_machine_identity(tmp_path):
    dumped = make_manifest().model_dump_json()
    assert str(tmp_path) not in dumped and str(Path.home()) not in dumped
    assert "Users" not in dumped and "hostname" not in dumped


def test_unknown_git_is_recorded_as_null():
    data = manifest_data()
    data["software"].update(git_commit=None, git_dirty=None)
    manifest = RunManifest.model_validate_json(json.dumps(data))
    assert manifest.software.git_commit is None


def test_request_and_id_hashes():
    assert request_sha256(make_request("a")) == request_sha256(make_request("a"))
    assert request_sha256(make_request("a")) != request_sha256(make_request("b"))
    assert request_sha256(make_request("a")) != request_sha256(make_request("a", content="other"))
    assert ids_sha256(["a", "b"]) != ids_sha256(["b", "a"])


def test_result_line_checks_that_ids_agree():
    with pytest.raises(ValidationError, match="request_id"):
        RunResultLine(
            request_id="other",
            request_sha256=HASH_A,
            recorded_at=NOW,
            execution=make_line("r1").execution,
        )


# -- creating runs --------------------------------------------------------------------------------


def test_create_run_writes_manifest_once(tmp_path):
    run_dir = create_run(tmp_path / "runs", make_manifest("run1"))
    assert run_dir == tmp_path / "runs" / "run1"
    assert read_manifest(run_dir) == make_manifest("run1")
    raw = (run_dir / "manifest.json").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw


def test_duplicate_run_id_is_refused_and_leaves_the_run_untouched(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    append_result(run_dir, make_line("r1"))
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(FileExistsError):
        create_run(tmp_path, make_manifest("run1", requested_model="different"))
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def _filesystem_is_case_insensitive(root: Path) -> bool:
    (root / "CaseProbe").mkdir()
    return (root / "caseprobe").exists()


def test_case_variant_run_id_collides_on_case_insensitive_filesystems(tmp_path):
    if not _filesystem_is_case_insensitive(tmp_path):
        pytest.skip("case-sensitive filesystem")
    (tmp_path / "RUN1").mkdir()
    with pytest.raises(FileExistsError):
        create_run(tmp_path, make_manifest("run1"))


def test_failed_manifest_write_cleans_up_so_the_id_is_reusable(tmp_path, monkeypatch):
    def broken(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(runstore, "_write_new_file", broken)
    with pytest.raises(OSError, match="disk full"):
        create_run(tmp_path, make_manifest("run1"))
    assert not (tmp_path / "run1").exists()
    monkeypatch.undo()
    assert create_run(tmp_path, make_manifest("run1")).is_dir()


def test_writes_are_fsynced(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(runstore.os, "fsync", lambda fd: (calls.append(fd), real(fd)))
    run_dir = create_run(tmp_path, make_manifest("run1"))
    assert len(calls) == 1
    append_result(run_dir, make_line("r1"))
    assert len(calls) == 2


# -- appending and reading results ----------------------------------------------------------------


def test_results_round_trip_in_order(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    lines = [make_line("r1", "नमस्ते"), make_line("r2", "ಕನ್ನಡ"), make_line("r3", 'quote " and \\')]
    for line in lines:
        append_result(run_dir, line)
    result = read_results(run_dir)
    assert result.lines == tuple(lines) and result.torn_tail_bytes == 0


def test_failures_round_trip_too(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    failure = GenerationFailure(
        request_id="r1",
        provider="fake",
        requested_model="m",
        kind=FailureKind.RATE_LIMIT,
        message="slow down",
        http_status=429,
    )
    append_result(run_dir, make_line("r1", result=failure))
    assert read_results(run_dir).lines[0].execution.result == failure


def test_results_file_is_utf8_with_lf_only(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    append_result(run_dir, make_line("r1", "नमस्ते"))
    append_result(run_dir, make_line("r2"))
    raw = (run_dir / "results.jsonl").read_bytes()
    assert b"\r" not in raw and raw.count(b"\n") == 2 and raw.endswith(b"\n")
    assert "नमस्ते" in raw.decode("utf-8")


def test_missing_or_empty_results_file_reads_as_empty(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    assert read_results(run_dir).lines == ()
    (run_dir / "results.jsonl").write_bytes(b"")
    assert read_results(run_dir) == read_results(run_dir)
    assert read_results(run_dir).lines == ()


# -- persistence boundary -------------------------------------------------------------------------


def _results_bytes(run_dir):
    path = run_dir / "results.jsonl"
    return path.read_bytes() if path.exists() else None


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("authorization", "Bearer abc-sentinel"),
        ("X_API_KEY", "abc-sentinel"),
        ("nested", ["abc-sentinel"]),
        ("nested", {"k": "abc-sentinel"}),
        ("number", math.nan),
        ("number", math.inf),
    ],
)
@pytest.mark.filterwarnings("error")  # serializer warnings would echo the values
def test_metadata_mutated_after_construction_is_refused_at_the_boundary(tmp_path, key, value):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    line = make_line("r1", result=make_success("r1", provider_metadata={"fine": 1}))
    line.execution.result.provider_metadata[key] = value  # frozen model, mutable dict
    with pytest.raises(PersistenceError) as info:
        append_result(run_dir, line)
    assert "abc-sentinel" not in str(info.value)
    assert _results_bytes(run_dir) is None


def test_unmutated_metadata_is_written(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    append_result(run_dir, make_line("r1", result=make_success("r1", provider_metadata={"a": 1})))
    assert read_results(run_dir).lines[0].execution.result.provider_metadata == {"a": 1}


GUARD = "FAKE-GUARD-VALUE-0001"


@pytest.mark.parametrize(
    "result",
    [
        make_success("r1", f"the key is {GUARD}"),
        make_success("r1", "ok", provider_metadata={"note": f"x{GUARD}y"}),
        GenerationFailure(
            request_id="r1",
            provider="fake",
            requested_model="m",
            kind=FailureKind.INVALID_REQUEST,
            message=f"bad key {GUARD}",
        ),
    ],
)
def test_secret_value_guard_refuses_and_never_echoes_the_value(tmp_path, result):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    with pytest.raises(PersistenceError) as info:
        append_result(run_dir, make_line("r1", result=result), secret_values=(GUARD,))
    assert GUARD not in str(info.value) and "FAKE-GUARD" not in str(info.value)
    assert _results_bytes(run_dir) is None


@pytest.mark.parametrize("guard", ['has"quote-and-slash\\-1', "गुप्त-कुंजी-12345", "tab\there-12345"])
def test_secret_guard_matches_json_escaped_forms(tmp_path, guard):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    line = make_line("r1", result=make_success("r1", f"prefix {guard} suffix"))
    with pytest.raises(PersistenceError):
        append_result(run_dir, line, secret_values=(guard,))
    assert _results_bytes(run_dir) is None


def test_secret_guard_lets_clean_results_through(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    append_result(run_dir, make_line("r1"), secret_values=(GUARD,))
    assert len(read_results(run_dir).lines) == 1


@pytest.mark.parametrize("short", ["", "a", "1234567"])
def test_too_short_guard_values_are_rejected_outright(tmp_path, short):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    with pytest.raises(ValueError, match="at least 8"):
        append_result(run_dir, make_line("r1"), secret_values=(short,))
    assert _results_bytes(run_dir) is None


# -- corruption and torn writes -------------------------------------------------------------------


def _run_with(tmp_path, *request_ids):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    for request_id in request_ids:
        append_result(run_dir, make_line(request_id))
    return run_dir


def test_torn_tail_is_an_error_unless_allowed(tmp_path):
    run_dir = _run_with(tmp_path, "r1", "r2")
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b'{"request_id": "r3", "requ')
    with pytest.raises(CorruptRunError, match="line 3.*torn"):
        read_results(run_dir)
    result = read_results(run_dir, allow_torn_tail=True)
    assert [line.request_id for line in result.lines] == ["r1", "r2"]
    assert result.torn_tail_bytes == len(b'{"request_id": "r3", "requ')


def test_truncate_removes_only_the_torn_tail(tmp_path):
    run_dir = _run_with(tmp_path, "r1", "r2")
    good = (run_dir / "results.jsonl").read_bytes()
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b"half a line")
    torn = read_results(run_dir, allow_torn_tail=True).torn_tail_bytes
    truncate_torn_tail(run_dir, torn)
    assert (run_dir / "results.jsonl").read_bytes() == good
    assert read_results(run_dir).torn_tail_bytes == 0
    truncate_torn_tail(run_dir, 0)
    assert (run_dir / "results.jsonl").read_bytes() == good


def test_a_complete_json_line_without_newline_still_counts_as_torn(tmp_path):
    run_dir = _run_with(tmp_path, "r1")
    path = run_dir / "results.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    assert read_results(run_dir, allow_torn_tail=True).lines == ()


@pytest.mark.parametrize(
    ("bad_line", "message"),
    [
        (b"not json", "line 2"),
        (b"", "line 2.*blank"),
        (b"   ", "line 2.*blank"),
        (b'{"request_id": "r9"}', "line 2"),
        (b"[1, 2]", "line 2"),
    ],
)
def test_corruption_in_the_middle_is_an_error_not_skipped(tmp_path, bad_line, message):
    run_dir = _run_with(tmp_path, "r1")
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(bad_line + b"\n")
    append_result(run_dir, make_line("r3"))
    with pytest.raises(CorruptRunError, match=message):
        read_results(run_dir, allow_torn_tail=True)


def test_extra_fields_and_mismatched_ids_in_stored_lines_are_corruption(tmp_path):
    run_dir = _run_with(tmp_path, "r1")
    path = run_dir / "results.jsonl"
    data = json.loads(path.read_bytes())
    path.write_bytes(json.dumps({**data, "api_key": "x"}).encode() + b"\n")
    with pytest.raises(CorruptRunError, match="line 1"):
        read_results(run_dir)
    path.write_bytes(json.dumps({**data, "request_id": "other"}).encode() + b"\n")
    with pytest.raises(CorruptRunError, match="line 1"):
        read_results(run_dir)


def test_duplicate_request_ids_are_corruption_with_both_line_numbers(tmp_path):
    run_dir = _run_with(tmp_path, "r1", "r2", "r1")  # append_result does not dedupe
    with pytest.raises(CorruptRunError, match=r"line 3.*duplicate.*first on line 1"):
        read_results(run_dir)


def test_corruption_messages_do_not_echo_stored_values(tmp_path):
    run_dir = _run_with(tmp_path, "r1")
    path = run_dir / "results.jsonl"
    data = json.loads(path.read_bytes())
    data["execution"]["elapsed_s"] = "SENTINEL-TEXT"
    path.write_bytes(json.dumps(data).encode() + b"\n")
    with pytest.raises(CorruptRunError) as info:
        read_results(run_dir)
    assert "SENTINEL-TEXT" not in str(info.value)


def test_reading_never_modifies_files(tmp_path):
    run_dir = _run_with(tmp_path, "r1", "r2")
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b"torn")
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    read_results(run_dir, allow_torn_tail=True)
    with pytest.raises(CorruptRunError):
        read_results(run_dir)
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_corrupt_or_missing_manifest_is_reported(tmp_path):
    run_dir = create_run(tmp_path, make_manifest("run1"))
    (run_dir / "manifest.json").write_bytes(b"{}")
    with pytest.raises(CorruptRunError, match="manifest.json"):
        read_manifest(run_dir)
    (run_dir / "manifest.json").unlink()
    with pytest.raises(CorruptRunError, match="not found"):
        read_manifest(run_dir)


def test_new_files_are_written_with_exclusive_create(tmp_path):
    path = tmp_path / "once.json"
    runstore._write_new_file(path, b"first")
    with pytest.raises(FileExistsError):
        runstore._write_new_file(path, b"second")
    assert path.read_bytes() == b"first"


# -- regression: failed appends, torn tails, manifest guard (pre-commit audit) --------------------


def test_append_refuses_when_the_file_ends_with_an_incomplete_line(tmp_path):
    run_dir = _run_with(tmp_path, "r1")
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b'{"request_id": "r2", "req')  # torn tail left by a crash
    before = (run_dir / "results.jsonl").read_bytes()
    with pytest.raises(PersistenceError, match="incomplete line"):
        append_result(run_dir, make_line("r3"))
    assert (run_dir / "results.jsonl").read_bytes() == before  # not merged into a corrupt line
    truncate_torn_tail(run_dir, read_results(run_dir, allow_torn_tail=True).torn_tail_bytes)
    append_result(run_dir, make_line("r3"))
    assert [line.request_id for line in read_results(run_dir).lines] == ["r1", "r3"]


class _FailingWrite:
    """File wrapper whose write() stores half the bytes, then fails (a full disk)."""

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


@pytest.mark.parametrize("exc", [OSError("disk full"), KeyboardInterrupt()])
def test_failed_write_leaves_the_file_exactly_as_it_was(tmp_path, monkeypatch, exc):
    run_dir = _run_with(tmp_path, "r1")
    before = (run_dir / "results.jsonl").read_bytes()
    real_open = open
    monkeypatch.setattr(
        runstore,
        "open",
        lambda path, mode: _FailingWrite(real_open(path, mode), exc),
        raising=False,
    )
    with pytest.raises(type(exc)):
        append_result(run_dir, make_line("r2"))
    monkeypatch.undo()
    assert (run_dir / "results.jsonl").read_bytes() == before
    append_result(run_dir, make_line("r2"))  # the run is still appendable and consistent
    assert [line.request_id for line in read_results(run_dir).lines] == ["r1", "r2"]


def test_failed_fsync_does_not_leave_an_unconfirmed_line_behind(tmp_path, monkeypatch):
    run_dir = _run_with(tmp_path, "r1")
    before = (run_dir / "results.jsonl").read_bytes()

    def broken(fd):
        raise OSError("fsync failed")

    monkeypatch.setattr(runstore.os, "fsync", broken)
    with pytest.raises(OSError, match="fsync failed"):
        append_result(run_dir, make_line("r2"))
    monkeypatch.undo()
    assert (run_dir / "results.jsonl").read_bytes() == before


def test_refusals_do_not_chain_the_offending_value(tmp_path):
    run_dir = _run_with(tmp_path)
    line = make_line("r1", result=make_success("r1", provider_metadata={"fine": 1}))
    line.execution.result.provider_metadata["authorization"] = "Bearer chained-sentinel"
    with pytest.raises(PersistenceError) as info:
        append_result(run_dir, line)
    assert info.value.__cause__ is None and info.value.__context__ is None


def test_secret_guard_covers_the_manifest_and_nothing_is_created(tmp_path):
    prompt = {"system": None, "user": f"Use key {GUARD} for {{input}}", "sha256": HASH_A}
    manifest = make_manifest("run1", prompt=prompt)
    with pytest.raises(PersistenceError) as info:
        create_run(tmp_path / "runs", manifest, secret_values=(GUARD,))
    assert GUARD not in str(info.value) and "manifest" in str(info.value)
    assert not (tmp_path / "runs").exists()
    with pytest.raises(ValueError, match="at least 8"):
        create_run(tmp_path / "runs", make_manifest("run1"), secret_values=("short",))
    assert not (tmp_path / "runs").exists()
    assert create_run(tmp_path / "runs", make_manifest("run1"), secret_values=(GUARD,)).is_dir()


# -- the selection, prompt and request hashes, pinned independently of the production helpers ----
# The expected values come from the stdlib implementation below, written from the payloads
# documented in ADR 0003 ("Hash definitions"), never from the code under test. The digests are
# golden values: if the implementation and a payload were both changed, these tests would still
# fail until the digest was changed deliberately (a hash change invalidates stored runs). Each
# digest line carries a narrow per-line scanner allowlist, because a SHA-256 digest looks like a
# high-entropy secret to detect-secrets.

KANNADA_WORD = "ಉತ್ತರ"
HINDI_WORD = "भारत"


def independent_canonical(value) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def independent_sha256(value) -> str:
    return hashlib.sha256(independent_canonical(value).encode("utf-8")).hexdigest()


GOLDEN = {
    "ids-three": "05f7becd110a8af9c99ca7fc34915b314750b74d4670fade57e25dcaac734c00",  # pragma: allowlist secret  # noqa: E501
    "ids-two": "a6b819ad4164f4fad4379c6093b54ef78da2481934dfc63e27d69f7679fdd863",  # pragma: allowlist secret  # noqa: E501
    "prompt-plain": "47689fbe87d25659afa5c27d7e7e91acabd8bb5ba5513168343cc502a95270a3",  # pragma: allowlist secret  # noqa: E501
    "prompt-system-unicode": "e7ea642344acf226e3fa4f9c0543033af79e5abf7b054b80189d27e38c302846",  # pragma: allowlist secret  # noqa: E501
    "request-unset-params": "0519307d01aa1bcb827e18d149c39674088cb6e0fefd905b13b6245d9ea22a2e",  # pragma: allowlist secret  # noqa: E501
    "request-all-params-unicode": "795d593767b9192a450d041c3b1f054c81dc2bf67c48ef904d5a51b3b3d0d6d5",  # pragma: allowlist secret  # noqa: E501
}


@pytest.mark.parametrize(
    ("name", "ids", "canonical"),
    [
        ("ids-three", ["c00", "c01", "c02"], '["c00","c01","c02"]'),
        ("ids-two", ["qa-en-001", "qa-hi-001"], '["qa-en-001","qa-hi-001"]'),
    ],
)
def test_selection_hash_is_sha256_of_the_canonical_id_array_in_dataset_order(name, ids, canonical):
    assert independent_canonical(ids) == canonical  # the exact serialisation, in dataset order
    assert independent_sha256(ids) == GOLDEN[name]
    assert ids_sha256(ids) == GOLDEN[name]
    assert ids_sha256(ids[::-1]) != GOLDEN[name]  # order is part of the identity


def test_prompt_hash_is_sha256_of_the_canonical_system_and_user_object():
    plain = PromptTemplate(user="Answer: {input}")
    payload = {"system": None, "user": "Answer: {input}"}
    assert independent_canonical(payload) == '{"system":null,"user":"Answer: {input}"}'
    assert independent_sha256(payload) == GOLDEN["prompt-plain"] == plain.sha256

    user = 'Reply as {"answer": "' + KANNADA_WORD + '"} to: {input}'
    payload = {"system": "Be brief.", "user": user}
    canonical = independent_canonical(payload)
    assert KANNADA_WORD in canonical and "\\u0c89" not in canonical  # raw UTF-8, never escapes
    assert independent_sha256(payload) == GOLDEN["prompt-system-unicode"]
    assert PromptTemplate(system="Be brief.", user=user).sha256 == GOLDEN["prompt-system-unicode"]


def test_request_hash_is_sha256_of_the_full_request_dump_with_nulls_for_unset_params():
    unset = GenerationRequest(
        request_id="c1", model="m", messages=(Message(role="user", content="hi"),)
    )
    payload = {
        "messages": [{"content": "hi", "role": "user"}],
        "model": "m",
        "params": {
            "max_tokens": None,
            "response_format": None,
            "seed": None,
            "stop": None,
            "temperature": None,
            "top_p": None,
        },
        "request_id": "c1",
    }
    assert independent_canonical(payload) == (
        '{"messages":[{"content":"hi","role":"user"}],"model":"m","params":{"max_tokens":null,'
        '"response_format":null,"seed":null,"stop":null,"temperature":null,"top_p":null},'
        '"request_id":"c1"}'
    )
    assert independent_sha256(payload) == GOLDEN["request-unset-params"]
    assert request_sha256(unset) == GOLDEN["request-unset-params"]

    full = GenerationRequest(
        request_id="qa-hi-001",
        model="fake-model",
        messages=(
            Message(role="system", content="Be brief."),
            Message(role="user", content="Answer: " + HINDI_WORD + "?"),
        ),
        params=GenerationParams(
            temperature=0.0,
            top_p=0.9,
            max_tokens=64,
            seed=7,
            stop=("END", "।"),
            response_format="json_object",
        ),
    )
    payload = {
        "messages": [
            {"content": "Be brief.", "role": "system"},
            {"content": "Answer: " + HINDI_WORD + "?", "role": "user"},
        ],
        "model": "fake-model",
        "params": {
            "max_tokens": 64,
            "response_format": "json_object",
            "seed": 7,
            "stop": ["END", "।"],
            "temperature": 0.0,
            "top_p": 0.9,
        },
        "request_id": "qa-hi-001",
    }
    assert HINDI_WORD in independent_canonical(payload)  # raw UTF-8, never escapes
    assert independent_sha256(payload) == GOLDEN["request-all-params-unicode"]
    assert request_sha256(full) == GOLDEN["request-all-params-unicode"]
