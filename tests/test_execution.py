import itertools
import socket
from pathlib import Path

import pytest

from ds_helpers import FIXTURES, qa, write_dataset
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import build_requests, execute_run, resume_run
from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationParams,
    GenerationSuccess,
)
from niriksha.core.provenance import SoftwareInfo
from niriksha.core.runner import ProviderContractViolation
from niriksha.core.runstore import (
    CorruptRunError,
    PersistenceError,
    PromptTemplate,
    ResumeError,
    RunConfig,
    read_manifest,
    read_results,
    request_sha256,
)
from niriksha.providers.fake import FakeProvider
from run_helpers import NOW

SOFTWARE = SoftwareInfo(
    niriksha_version="0.0.0",
    python_version="3.13.3",
    pydantic_version="2.13.5",
    platform="win32",
    git_commit="c" * 40,
    git_dirty=False,
)


class Crash(Exception):
    """Stands in for a process dying or a programming error mid-run."""


class Flaky(FakeProvider):
    """A FakeProvider that raises Crash on its Nth call (1-based). Same class for both sessions."""

    def __init__(self, *args, crash_on=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.crash_on = crash_on

    def generate(self, request):
        if self.crash_on is not None and len(self.calls) + 1 == self.crash_on:
            raise Crash("simulated crash")
        return super().generate(request)


def ticking_clock():
    ticks = itertools.count()
    return lambda: next(ticks) * 0.25  # every provider call measures exactly 0.25 s


def config(run_id="run1", splits=("dev",), model="m", user="Answer: {input}", **params):
    return RunConfig(
        run_id=run_id,
        splits=splits,
        model=model,
        prompt=PromptTemplate(user=user),
        params=GenerationParams(**params),
    )


@pytest.fixture
def ten(tmp_path):
    """A 10-case QA dataset, all in the dev split."""
    cases = [qa(id=f"c{i:02d}", question=f"Question {i}?") for i in range(10)]
    return load_dataset(write_dataset(tmp_path, cases, dirname="ten"))


def run(cfg, dataset, provider, runs, **kwargs):
    kwargs.setdefault("clock", ticking_clock())
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("software", SOFTWARE)
    return execute_run(cfg, dataset, provider, runs, **kwargs)


def resume(cfg, dataset, provider, runs, **kwargs):
    kwargs.setdefault("clock", ticking_clock())
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("software", SOFTWARE)
    return resume_run(cfg, dataset, provider, runs, **kwargs)


def snapshot(run_dir: Path) -> dict:
    return {p.name: p.read_bytes() for p in sorted(run_dir.iterdir())}


def ids(run_dir: Path) -> list[str]:
    return [line.request_id for line in read_results(run_dir).lines]


# -- a complete run -------------------------------------------------------------------------------


def test_complete_run_with_the_fake_provider(tmp_path, ten):
    provider = FakeProvider()
    summary = run(config(), ten, provider, tmp_path / "runs")
    assert (summary.total, summary.already_recorded, summary.executed) == (10, 0, 10)
    assert len(provider.calls) == 10

    results = read_results(summary.run_dir).lines
    assert [r.request_id for r in results] == [f"c{i:02d}" for i in range(10)]
    for i, line in enumerate(results):
        assert line.execution.result.output_text == f"Answer: Question {i}?"
        assert line.execution.elapsed_s == 0.25
        assert line.recorded_at == NOW
    for (_case, request), line in zip(build_requests(config(), ten), results, strict=True):
        assert line.request_sha256 == request_sha256(request)


def test_manifest_records_what_was_run(tmp_path, ten):
    summary = run(config(temperature=0.0), ten, FakeProvider(), tmp_path / "runs")
    manifest = read_manifest(summary.run_dir)
    assert manifest.run_id == "run1" and manifest.created_at == NOW
    assert manifest.dataset.content_sha256 == ten.content_sha256
    assert (manifest.dataset.name, manifest.dataset.task) == ("ds", "short_answer_qa")
    assert (manifest.selection.splits, manifest.selection.case_count) == (("dev",), 10)
    assert manifest.prompt.user == "Answer: {input}" and len(manifest.prompt.sha256) == 64
    assert manifest.provider.name == "fake"
    assert manifest.provider.implementation == "niriksha.providers.fake.FakeProvider"
    assert manifest.requested_model == "m" and manifest.params.temperature == 0.0
    assert manifest.software == SOFTWARE


def test_run_files_contain_no_absolute_paths_or_home_directory(tmp_path, ten):
    summary = run(config(), ten, FakeProvider(), tmp_path / "runs")
    text = "".join(p.read_text(encoding="utf-8") for p in summary.run_dir.iterdir())
    assert str(tmp_path) not in text and str(Path.home()) not in text


def test_extraction_dataset_uses_the_source_text_as_input(tmp_path):
    dataset = load_dataset(FIXTURES / "tiny_extraction")
    summary = run(
        config(splits=("dev",), user="Extract: {input}"), dataset, FakeProvider(), tmp_path
    )
    outputs = [r.execution.result.output_text for r in read_results(summary.run_dir).lines]
    assert outputs == ["Extract: Asha is 30 years old.", "Extract: Ravi, aged 25, lives in Pune."]


def test_json_braces_in_the_prompt_survive(tmp_path, ten):
    cfg = config(user='Reply as {"answer": "..."} to: {input}')
    summary = run(cfg, ten, FakeProvider(), tmp_path)
    first = read_results(summary.run_dir).lines[0].execution.result.output_text
    assert first == 'Reply as {"answer": "..."} to: Question 0?'


def test_selected_splits_run_in_dataset_order_not_config_order(tmp_path):
    dataset = load_dataset(FIXTURES / "tiny_qa")
    summary = run(config(splits=("test", "dev")), dataset, FakeProvider(), tmp_path)
    assert ids(summary.run_dir) == ["qa-en-001", "qa-en-002", "qa-hi-001", "qa-hi-002"]
    assert read_manifest(summary.run_dir).selection.splits == ("test", "dev")


def test_empty_selection_is_refused_before_anything_is_created(tmp_path, ten):
    provider = FakeProvider()
    with pytest.raises(ValueError, match="no cases"):
        run(config(splits=("holdout",)), ten, provider, tmp_path / "runs")
    assert provider.calls == () and not (tmp_path / "runs").exists()


def test_hindi_and_kannada_cases_round_trip(tmp_path):
    dataset = load_dataset(FIXTURES / "tiny_qa")
    summary = run(config(splits=("test", "holdout")), dataset, FakeProvider(), tmp_path)
    outputs = {
        r.request_id: r.execution.result.output_text for r in read_results(summary.run_dir).lines
    }
    assert outputs["qa-hi-001"] == "Answer: भारत की राजधानी क्या है?"
    assert outputs["qa-kn-001"].startswith("Answer: ಕರ್ನಾಟಕದ")


# -- single attempt, failures, errors -------------------------------------------------------------


def test_one_provider_call_per_case_and_failures_are_recorded_not_retried(tmp_path, ten):
    script = ["ok"] * 10
    script[3] = FailureKind.TIMEOUT
    provider = FakeProvider(script)
    summary = run(config(), ten, provider, tmp_path)
    assert summary.executed == 10 and len(provider.calls) == 10
    results = read_results(summary.run_dir).lines
    assert isinstance(results[3].execution.result, GenerationFailure)
    assert results[3].execution.result.kind is FailureKind.TIMEOUT
    assert all(isinstance(r.execution.result, GenerationSuccess) for r in results[:3])


def test_duplicate_run_id_is_refused_before_any_provider_call(tmp_path, ten):
    run(config(), ten, FakeProvider(), tmp_path)
    before = snapshot(tmp_path / "run1")
    provider = FakeProvider()
    with pytest.raises(FileExistsError):
        run(config(model="other"), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(tmp_path / "run1") == before


def test_provider_programming_error_propagates_and_leaves_a_resumable_run(tmp_path, ten):
    with pytest.raises(Crash):
        run(config(), ten, Flaky(crash_on=1), tmp_path)
    assert ids(tmp_path / "run1") == []
    assert (tmp_path / "run1" / "manifest.json").exists()
    summary = resume(config(), ten, Flaky(), tmp_path)
    assert summary.executed == 10


def test_contract_violation_propagates_and_persists_nothing_for_that_case(tmp_path, ten):
    class Tamper(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.request_id == "c02":
                return result.model_copy(update={"request_id": "someone-else"})
            return result

    with pytest.raises(ProviderContractViolation, match="request_id"):
        run(config(), ten, Tamper(), tmp_path)
    assert ids(tmp_path / "run1") == ["c00", "c01"]


def test_metadata_mutated_by_a_provider_is_refused_at_the_persistence_boundary(tmp_path, ten):
    class Sneaky(FakeProvider):
        def generate(self, request):
            result = super().generate(request)
            if request.request_id == "c01":
                result.provider_metadata["authorization"] = "Bearer sentinel-token"
            return result

    with pytest.raises(PersistenceError) as info:
        run(config(), ten, Sneaky(), tmp_path)
    assert "sentinel-token" not in str(info.value)
    assert ids(tmp_path / "run1") == ["c00"]


def test_secret_guard_is_applied_during_runs(tmp_path):
    guard = "FAKE-GUARD-VALUE-0002"
    cases = [qa(id="a"), qa(id="b", question=f"leaky {guard}")]
    dataset = load_dataset(write_dataset(tmp_path, cases, dirname="leaky"))
    with pytest.raises(PersistenceError) as info:
        run(config(), dataset, FakeProvider(), tmp_path / "runs", secret_values=(guard,))
    assert guard not in str(info.value)
    assert ids(tmp_path / "runs" / "run1") == ["a"]


def test_too_short_guard_values_are_refused_before_any_side_effect(tmp_path, ten):
    provider = FakeProvider()
    with pytest.raises(ValueError, match="at least 8"):
        run(config(), ten, provider, tmp_path / "runs", secret_values=("short",))
    assert provider.calls == () and not (tmp_path / "runs").exists()
    with pytest.raises(ValueError, match="at least 8"):
        resume(config(), ten, provider, tmp_path / "runs", secret_values=("short",))


def test_everything_runs_offline_with_the_socket_block_active(tmp_path, ten):
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket.create_connection(("127.0.0.1", 9))
    assert run(config(), ten, FakeProvider(), tmp_path).executed == 10


# -- interruption and resume ----------------------------------------------------------------------


def interrupted(tmp_path, dataset, *, crash_on=6, script=None, cfg=None):
    cfg = cfg or config()
    with pytest.raises(Crash):
        run(cfg, dataset, Flaky(script, crash_on=crash_on), tmp_path)
    return tmp_path / cfg.run_id


def test_interrupted_run_resumes_with_no_duplicate_or_missing_cases(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten)
    assert ids(run_dir) == [f"c{i:02d}" for i in range(5)]

    provider = Flaky()
    summary = resume(config(), ten, provider, tmp_path)
    assert (summary.total, summary.already_recorded, summary.executed) == (10, 5, 5)
    assert [r.request_id for r in provider.calls] == [f"c{i:02d}" for i in range(5, 10)]
    assert ids(run_dir) == [f"c{i:02d}" for i in range(10)]  # each exactly once, in order


def test_resume_never_retries_recorded_failures(tmp_path, ten):
    script = ["ok", "ok", FailureKind.SERVER_ERROR, "ok", "ok", "ok"]
    run_dir = interrupted(tmp_path, ten, script=script)
    provider = Flaky()
    resume(config(), ten, provider, tmp_path)
    assert "c02" not in [r.request_id for r in provider.calls]
    stored = {r.request_id: r for r in read_results(run_dir).lines}
    assert stored["c02"].execution.result.kind is FailureKind.SERVER_ERROR


def test_resuming_a_complete_run_makes_no_calls_and_changes_nothing(tmp_path, ten):
    summary = run(config(), ten, FakeProvider(), tmp_path)
    before = snapshot(summary.run_dir)
    provider = FakeProvider()
    again = resume(config(), ten, provider, tmp_path)
    assert (again.already_recorded, again.executed) == (10, 0) and provider.calls == ()
    assert snapshot(summary.run_dir) == before


def test_manifest_is_never_modified_by_resume(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten)
    manifest_before = (run_dir / "manifest.json").read_bytes()
    resume(config(), ten, Flaky(), tmp_path)
    assert (run_dir / "manifest.json").read_bytes() == manifest_before


def test_torn_final_line_is_truncated_and_its_case_runs_again(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten, crash_on=4)
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(b'{"request_id": "c03", "request_sha256": "ab')
    provider = Flaky()
    summary = resume(config(), ten, provider, tmp_path)
    assert (summary.already_recorded, summary.executed) == (3, 7)
    assert provider.calls[0].request_id == "c03"
    assert ids(run_dir) == [f"c{i:02d}" for i in range(10)]
    assert read_results(run_dir).torn_tail_bytes == 0


def test_a_complete_last_line_missing_its_newline_is_treated_as_torn(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten, crash_on=4)
    path = run_dir / "results.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    provider = Flaky()
    resume(config(), ten, provider, tmp_path)
    assert provider.calls[0].request_id == "c02"
    assert ids(run_dir) == [f"c{i:02d}" for i in range(10)]


def test_corrupt_line_in_the_middle_is_an_error_and_nothing_changes(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten)
    lines = (run_dir / "results.jsonl").read_bytes().split(b"\n")
    lines[1] = b"garbage"
    (run_dir / "results.jsonl").write_bytes(b"\n".join(lines))
    before, provider = snapshot(run_dir), Flaky()
    with pytest.raises(CorruptRunError, match="line 2"):
        resume(config(), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(run_dir) == before


def test_duplicate_result_ids_are_rejected(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten, crash_on=3)
    path = run_dir / "results.jsonl"
    first = path.read_bytes().split(b"\n")[0]
    path.write_bytes(path.read_bytes() + first + b"\n")
    before, provider = snapshot(run_dir), Flaky()
    with pytest.raises(CorruptRunError, match="duplicate"):
        resume(config(), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(run_dir) == before


def test_foreign_result_ids_are_rejected(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten, crash_on=3)
    summary_lines = read_results(run_dir).lines
    foreign = summary_lines[0].model_copy(deep=True)
    object.__setattr__(foreign, "request_id", "not-in-dataset")
    object.__setattr__(foreign.execution.result, "request_id", "not-in-dataset")
    with open(run_dir / "results.jsonl", "ab") as handle:
        handle.write(foreign.model_dump_json().encode() + b"\n")
    before, provider = snapshot(run_dir), Flaky()
    with pytest.raises(ResumeError, match="not in the selection"):
        resume(config(), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(run_dir) == before


def test_stored_request_hash_is_checked_against_the_rebuilt_request(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten, crash_on=3)
    path = run_dir / "results.jsonl"
    first_hash = read_results(run_dir).lines[0].request_sha256
    path.write_bytes(path.read_bytes().replace(first_hash.encode(), b"0" * 64, 1))
    before, provider = snapshot(run_dir), Flaky()
    with pytest.raises(ResumeError, match="request hash"):
        resume(config(), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(run_dir) == before


def test_resuming_a_run_that_does_not_exist_is_an_error_and_creates_nothing(tmp_path, ten):
    with pytest.raises(ResumeError, match="does not exist"):
        resume(config(), ten, FakeProvider(), tmp_path / "runs")
    assert not (tmp_path / "runs" / "run1").exists()


# -- refusals -------------------------------------------------------------------------------------


def changed_dataset(tmp_path):
    cases = [qa(id=f"c{i:02d}", question=f"Different {i}?") for i in range(10)]
    return load_dataset(write_dataset(tmp_path, cases, dirname="changed"))


def software_with(**changes):
    return SoftwareInfo(**{**SOFTWARE.model_dump(), **changes})


# (field expected in the error, config changes, provider, software, use a different dataset)
REFUSALS = {
    "dataset content": ("dataset", {}, None, None, True),
    "selection": ("selection", {"splits": ("dev", "test")}, None, None, False),
    "prompt": ("prompt", {"user": "Different: {input}"}, None, None, False),
    "model": ("requested_model", {"model": "other-model"}, None, None, False),
    "parameters": ("params", {"temperature": 0.7}, None, None, False),
    "provider name": ("provider", {}, FakeProvider(name="other"), None, False),
    "provider class": ("provider", {}, type("Other", (FakeProvider,), {})(), None, False),
    "niriksha version": (
        "software.niriksha_version", {}, None, software_with(niriksha_version="9.9.9"), False,
    ),
    "python version": (
        "software.python_version", {}, None, software_with(python_version="3.11.0"), False,
    ),
    "pydantic version": (
        "software.pydantic_version", {}, None, software_with(pydantic_version="2.7.0"), False,
    ),
    "platform": ("software.platform", {}, None, software_with(platform="linux"), False),
    "git commit": ("software.git_commit", {}, None, software_with(git_commit="d" * 40), False),
    "git commit unknown": (
        "software.git_commit", {}, None, software_with(git_commit=None, git_dirty=None), False,
    ),
}  # fmt: skip


@pytest.mark.parametrize("label", list(REFUSALS))
def test_resume_is_refused_when_the_run_no_longer_matches_its_manifest(tmp_path, ten, label):
    field, changes, provider, software, other_dataset = REFUSALS[label]
    run_dir = interrupted(tmp_path, ten, cfg=config(temperature=0.0))
    before = snapshot(run_dir)
    provider = provider or FakeProvider()
    new_config = config(**{"temperature": 0.0, **changes})
    dataset = changed_dataset(tmp_path) if other_dataset else ten
    with pytest.raises(ResumeError) as info:
        resume(new_config, dataset, provider, tmp_path, software=software or SOFTWARE)
    assert field in str(info.value), label
    assert "Different" not in str(info.value)  # field names only, never prompt text
    assert provider.calls == () and snapshot(run_dir) == before


def test_a_changed_dirty_flag_alone_does_not_block_resume(tmp_path, ten):
    interrupted(tmp_path, ten, cfg=config(temperature=0.0))
    dirty = SoftwareInfo(**{**SOFTWARE.model_dump(), "git_dirty": True})
    summary = resume(config(temperature=0.0), ten, Flaky(), tmp_path, software=dirty)
    assert summary.executed == 5


def test_a_manifest_moved_under_another_run_id_is_refused(tmp_path, ten):
    run_dir = interrupted(tmp_path, ten)
    run_dir.rename(tmp_path / "run2")
    with pytest.raises(ResumeError, match="run_id"):
        resume(config("run2"), ten, FakeProvider(), tmp_path)


def test_extraction_runs_resume_too(tmp_path):
    dataset = load_dataset(FIXTURES / "tiny_extraction")
    cfg = config(splits=("dev", "test"))
    with pytest.raises(Crash):
        run(cfg, dataset, Flaky(crash_on=2), tmp_path)
    summary = resume(cfg, dataset, Flaky(), tmp_path)
    assert (summary.already_recorded, summary.executed) == (1, 2)
    assert ids(tmp_path / "run1") == ["ex-en-001", "ex-en-002", "ex-hi-001"]


# -- regression: manifest guard, stored provider/model, failed appends (pre-commit audit) ---------


def test_secret_in_the_prompt_template_is_refused_before_anything_is_created(tmp_path, ten):
    guard = "FAKE-GUARD-VALUE-0004"
    provider = FakeProvider()
    with pytest.raises(PersistenceError) as info:
        run(
            config(user=f"key {guard} then {{input}}"),
            ten,
            provider,
            tmp_path / "runs",
            secret_values=(guard,),
        )
    assert guard not in str(info.value)
    assert provider.calls == () and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("field", ["requested_model", "provider"])
def test_resume_refuses_stored_results_from_another_model_or_provider(tmp_path, ten, field):
    run_dir = interrupted(tmp_path, ten, crash_on=3)
    path = run_dir / "results.jsonl"
    original = f'"{field}":"{"m" if field == "requested_model" else "fake"}"'
    assert original.encode() in path.read_bytes()
    path.write_bytes(path.read_bytes().replace(original.encode(), f'"{field}":"other"'.encode(), 1))
    before, provider = snapshot(run_dir), Flaky()
    with pytest.raises(ResumeError, match="different provider or model"):
        resume(config(), ten, provider, tmp_path)
    assert provider.calls == () and snapshot(run_dir) == before


class Mutating(FakeProvider):
    """Mutates one result's metadata after construction, but only when told to."""

    def __init__(self, *args, mutate_on=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutate_on = mutate_on

    def generate(self, request):
        result = super().generate(request)
        if request.request_id == self.mutate_on:
            result.provider_metadata["authorization"] = "Bearer sentinel-token"
        return result


def test_refused_append_leaves_a_clean_resumable_run_and_the_case_runs_again(tmp_path, ten):
    with pytest.raises(PersistenceError):
        run(config(), ten, Mutating(mutate_on="c03"), tmp_path)
    run_dir = tmp_path / "run1"
    raw = (run_dir / "results.jsonl").read_bytes()
    assert raw.endswith(b"\n") and read_results(run_dir).torn_tail_bytes == 0
    assert ids(run_dir) == ["c00", "c01", "c02"]  # no partial or misleading record for c03

    provider = Mutating()
    summary = resume(config(), ten, provider, tmp_path)
    assert (summary.already_recorded, summary.executed) == (3, 7)
    assert provider.calls[0].request_id == "c03"
    assert ids(run_dir) == [f"c{i:02d}" for i in range(10)]
