"""Runner-level retries: bounded, deterministic, recorded attempt by attempt. No real sleeping."""

import itertools
import json

import pytest
from pydantic import ValidationError

from ds_helpers import qa, write_dataset
from e2e_support import make_config
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run, resume_run
from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationRequest,
    GenerationSuccess,
    Message,
)
from niriksha.core.runner import (
    NO_RETRY,
    AttemptRecord,
    ExecutionRecord,
    ProviderContractViolation,
    RetryPolicy,
    run_one,
    run_requests,
)
from niriksha.core.runstore import read_results
from niriksha.providers.fake import FakeProvider
from score_helpers import deterministic

REQUEST = GenerationRequest(
    request_id="r1", model="m", messages=(Message(role="user", content="q"),)
)
THREE = RetryPolicy(max_attempts=3)
T, R, S, C = (
    FailureKind.TIMEOUT,
    FailureKind.RATE_LIMIT,
    FailureKind.SERVER_ERROR,
    FailureKind.CONNECTION_ERROR,
)


def ticking():
    return itertools.count(0, 0.5).__next__  # each call to generate() then measures 0.5 s


class Sleeps:
    def __init__(self):
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class ResultProvider:
    """Plays back callables that build a result for the request they are given."""

    name = "scripted"

    def __init__(self, *builders):
        self._builders = list(builders)
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return self._builders[len(self.calls) - 1](request)


def failing(kind, **metadata):
    return lambda r: GenerationFailure(
        request_id=r.request_id,
        provider="scripted",
        requested_model=r.model,
        kind=kind,
        message="scripted failure",
        provider_metadata=metadata,
    )


def succeeding(text="ok"):
    return lambda r: GenerationSuccess(
        request_id=r.request_id, provider="scripted", requested_model=r.model, output_text=text
    )


# -- the default is one attempt and an unchanged record -------------------------------------------


def test_the_default_is_a_single_attempt_with_an_unchanged_record():
    sleeps = Sleeps()
    provider = FakeProvider([S, "never reached"])
    record = run_one(provider, REQUEST, clock=ticking(), sleep=sleeps)
    assert NO_RETRY.max_attempts == 1 and RetryPolicy().max_attempts == 1
    assert len(provider.calls) == 1 and sleeps.waits == []
    assert record.attempts is None and isinstance(record.result, GenerationFailure)
    assert "attempts" not in json.loads(record.model_dump_json())  # byte-identical to before
    assert (
        record.model_dump_json()
        == ExecutionRecord(result=record.result, elapsed_s=0.5).model_dump_json()
    )


def test_a_record_without_attempts_loads_from_a_legacy_line():
    legacy = '{"result": ' + FakeProvider(["x"]).generate(REQUEST).model_dump_json()
    legacy += ', "elapsed_s": 0.5}'
    assert ExecutionRecord.model_validate_json(legacy).attempts is None


# -- retrying -------------------------------------------------------------------------------------


def test_success_after_failures_records_every_attempt_and_the_waits():
    sleeps, provider = Sleeps(), FakeProvider([T, R, "answer"])
    record = run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert len(provider.calls) == 3 and sleeps.waits == [1.0, 2.0]
    assert isinstance(record.result, GenerationSuccess) and record.result.output_text == "answer"
    assert [a.waited_s for a in record.attempts] == [0.0, 1.0, 2.0]
    assert [a.elapsed_s for a in record.attempts] == [0.5, 0.5, 0.5]
    assert [getattr(a.result, "kind", None) for a in record.attempts] == [T, R, None]
    assert record.elapsed_s == 0.5  # the final attempt only, not the waits
    assert ExecutionRecord.model_validate_json(record.model_dump_json()) == record


def test_the_limit_is_three_attempts_and_the_last_failure_is_recorded():
    sleeps, provider = Sleeps(), FakeProvider([S, S, S, "never"])
    record = run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert len(provider.calls) == 3 and sleeps.waits == [1.0, 2.0]
    assert isinstance(record.result, GenerationFailure) and record.result.kind is S
    assert len(record.attempts) == 3 and record.attempts[-1].result == record.result


def test_one_attempt_is_recorded_when_a_retry_policy_is_used_but_nothing_failed():
    record = run_one(FakeProvider(["fine"]), REQUEST, clock=ticking(), retry=THREE, sleep=Sleeps())
    assert len(record.attempts) == 1 and record.attempts[0].waited_s == 0.0


@pytest.mark.parametrize(
    "kind",
    [
        FailureKind.AUTH_ERROR,
        FailureKind.INVALID_REQUEST,
        FailureKind.MALFORMED_RESPONSE,
        FailureKind.UNSUPPORTED_CAPABILITY,
        FailureKind.INTERNAL_ERROR,
    ],
)
def test_permanent_failures_are_never_retried(kind):
    sleeps, provider = Sleeps(), FakeProvider([kind, "never"])
    record = run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert len(provider.calls) == 1 and sleeps.waits == []
    assert record.result.kind is kind and len(record.attempts) == 1


@pytest.mark.parametrize("kind", [T, R, S, C])
def test_transient_failures_are_retried(kind):
    sleeps, provider = Sleeps(), FakeProvider([kind, "ok"])
    record = run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert len(provider.calls) == 2 and sleeps.waits == [1.0]
    assert isinstance(record.result, GenerationSuccess)


def test_backoff_doubles_and_is_capped():
    sleeps = Sleeps()
    policy = RetryPolicy(max_attempts=5, backoff_base_s=10.0, backoff_cap_s=30.0)
    run_one(FakeProvider([T] * 5), REQUEST, clock=ticking(), retry=policy, sleep=sleeps)
    assert sleeps.waits == [10.0, 20.0, 30.0, 30.0]


def test_no_wait_ever_exceeds_the_maximum_wait():
    sleeps = Sleeps()
    policy = RetryPolicy(max_attempts=3, backoff_base_s=50.0, backoff_cap_s=300.0, max_wait_s=60.0)
    run_one(FakeProvider([T, T, T]), REQUEST, clock=ticking(), retry=policy, sleep=sleeps)
    assert sleeps.waits == [50.0, 60.0]  # the second backoff would be 100 s


def test_the_retry_set_can_be_narrowed():
    policy = RetryPolicy(max_attempts=3, retry_on=frozenset({R}))
    provider = FakeProvider([T, "never"])
    run_one(provider, REQUEST, clock=ticking(), retry=policy, sleep=Sleeps())
    assert len(provider.calls) == 1


# -- Retry-After ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("asked", "expected_wait"),
    [(5.0, 5.0), (0.5, 1.0), (0.0, 1.0), (60.0, 60.0), (2, 2.0)],
)
def test_retry_after_raises_the_wait_up_to_the_bound(asked, expected_wait):
    sleeps = Sleeps()
    provider = ResultProvider(failing(R, retry_after_s=asked), succeeding())
    run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert sleeps.waits == [expected_wait]


def test_a_retry_after_beyond_the_bound_stops_retrying_without_sleeping():
    sleeps = Sleeps()
    provider = ResultProvider(failing(R, retry_after_s=61.0), succeeding())
    record = run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert len(provider.calls) == 1 and sleeps.waits == []
    assert record.result.kind is R and len(record.attempts) == 1


@pytest.mark.parametrize("junk", ["soon", True, -3.0])
def test_a_malformed_retry_after_is_ignored(junk):
    sleeps = Sleeps()
    provider = ResultProvider(failing(R, retry_after_s=junk), succeeding())
    run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=sleeps)
    assert sleeps.waits == [1.0]


# -- contract and records -------------------------------------------------------------------------


def test_every_attempt_is_checked_against_the_provider_contract():
    def wrong_id(request):
        return GenerationSuccess(
            request_id="someone-else",
            provider="scripted",
            requested_model=request.model,
            output_text="x",
        )

    provider = ResultProvider(failing(T), wrong_id)
    with pytest.raises(ProviderContractViolation):
        run_one(provider, REQUEST, clock=ticking(), retry=THREE, sleep=Sleeps())


def test_run_requests_applies_the_policy_to_each_request():
    requests = [
        GenerationRequest(
            request_id=f"r{i}", model="m", messages=(Message(role="user", content="q"),)
        )
        for i in range(2)
    ]
    sleeps, provider = Sleeps(), FakeProvider([T, "a", S, "b"])
    records = run_requests(provider, requests, clock=ticking(), retry=THREE, sleep=sleeps)
    assert [len(r.attempts) for r in records] == [2, 2] and sleeps.waits == [1.0, 1.0]


def _failure():
    return FakeProvider([T]).generate(REQUEST)


def _success():
    return FakeProvider(["x"]).generate(REQUEST)


@pytest.mark.parametrize(
    "attempts",
    [
        (),  # empty
        (AttemptRecord(result=_failure(), elapsed_s=0.5, waited_s=0.0),),  # last != result
        (
            AttemptRecord(result=_success(), elapsed_s=0.5, waited_s=0.0),  # earlier success
            AttemptRecord(result=_success(), elapsed_s=0.5, waited_s=1.0),
        ),
        (AttemptRecord(result=_success(), elapsed_s=0.5, waited_s=2.0),),  # first has a wait
        (AttemptRecord(result=_success(), elapsed_s=0.9, waited_s=0.0),),  # elapsed differs
    ],
)
def test_inconsistent_attempt_lists_are_refused(attempts):
    with pytest.raises(ValidationError):
        ExecutionRecord(result=_success(), elapsed_s=0.5, attempts=attempts)


@pytest.mark.parametrize(
    "bad",
    [
        {"max_attempts": 0},
        {"max_attempts": 11},
        {"backoff_base_s": -1.0},
        {"backoff_base_s": float("nan")},
        {"backoff_cap_s": float("inf")},
        {"max_wait_s": 1000.0},
        {"retry_on": {"timeout"}},
    ],
)
def test_an_invalid_policy_is_refused(bad):
    with pytest.raises(ValidationError):
        RetryPolicy(**bad)


# -- through execute_run and resume_run -----------------------------------------------------------


def _dataset(tmp_path, n=3):
    directory = write_dataset(tmp_path, [qa(id=f"q{i}") for i in range(n)], dirname="ds")
    return load_dataset(directory), directory


def test_execute_run_defaults_to_no_retries_and_unchanged_lines(tmp_path):
    dataset, _ = _dataset(tmp_path)
    provider, sleeps = FakeProvider([T, "b", "c"]), Sleeps()
    summary = execute_run(
        make_config("r"), dataset, provider, tmp_path / "runs", sleep=sleeps, **deterministic()
    )
    assert len(provider.calls) == 3 and sleeps.waits == []
    for line in read_results(summary.run_dir).lines:
        assert line.execution.attempts is None
    text = (summary.run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert "attempts" not in text


def test_an_enabled_policy_is_applied_and_each_retried_result_keeps_its_attempts(tmp_path):
    dataset, _ = _dataset(tmp_path)
    provider, sleeps = FakeProvider([T, "a", "b", S, S, S]), Sleeps()
    summary = execute_run(
        make_config("r"),
        dataset,
        provider,
        tmp_path / "runs",
        retry=THREE,
        sleep=sleeps,
        **deterministic(),
    )
    lines = read_results(summary.run_dir).lines
    assert summary.executed == 3 and len(provider.calls) == 6
    assert [len(line.execution.attempts) for line in lines] == [2, 1, 3]
    assert isinstance(lines[2].execution.result, GenerationFailure)  # the final failure is kept
    assert sleeps.waits == [1.0, 1.0, 2.0]


def test_resume_never_retries_a_recorded_failure(tmp_path):
    dataset, _ = _dataset(tmp_path)
    config = make_config("r")
    execute_run(
        config,
        dataset,
        FakeProvider([S] * 9),
        tmp_path / "runs",
        retry=THREE,
        sleep=Sleeps(),
        **deterministic(),
    )
    again, sleeps = FakeProvider(), Sleeps()
    summary = resume_run(
        config, dataset, again, tmp_path / "runs", retry=THREE, sleep=sleeps, **deterministic()
    )
    assert summary.executed == 0 and summary.already_recorded == 3
    assert again.calls == () and sleeps.waits == []


def test_resume_applies_the_policy_only_to_cases_that_are_run_now(tmp_path):
    dataset, _ = _dataset(tmp_path)
    config = make_config("r")
    first = FakeProvider([T, "a"])  # case 1 is retried and recorded; case 2 finds no script left
    with pytest.raises(RuntimeError, match="exhausted"):
        execute_run(
            config,
            dataset,
            first,
            tmp_path / "runs",
            retry=THREE,
            sleep=Sleeps(),
            **deterministic(),
        )
    sleeps, rest = Sleeps(), FakeProvider([S, "b", "c"])
    summary = resume_run(
        config, dataset, rest, tmp_path / "runs", retry=THREE, sleep=sleeps, **deterministic()
    )
    assert summary.already_recorded == 1 and summary.executed == 2
    assert len(rest.calls) == 3 and sleeps.waits == [1.0]  # the policy applied to case 2 only
    lines = read_results(summary.run_dir).lines
    assert [len(line.execution.attempts) for line in lines] == [2, 2, 1]
