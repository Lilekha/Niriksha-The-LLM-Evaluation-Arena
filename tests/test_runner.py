import inspect
import math
import socket
import time

import pytest
from pydantic import ValidationError

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationRequest,
    GenerationSuccess,
    Message,
)
from niriksha.core.runner import ExecutionRecord, ProviderContractViolation, run_one, run_requests
from niriksha.providers.fake import FakeProvider


def request(request_id="r1", model="m", content="hello"):
    return GenerationRequest(
        request_id=request_id, model=model, messages=(Message(role="user", content=content),)
    )


class FakeClock:
    """Deterministic stand-in for time.perf_counter."""

    def __init__(self, start=100.0):
        self.now = start
        self.reads = 0

    def __call__(self):
        self.reads += 1
        return self.now


class SlowProvider:
    """Wraps a FakeProvider and advances a FakeClock during each call."""

    def __init__(self, clock, steps, inner=None):
        self.name = "fake"
        self._clock, self._steps, self._inner = clock, list(steps), inner or FakeProvider()

    def generate(self, req):
        self._clock.now += self._steps.pop(0)
        return self._inner.generate(req)


class Tampering:
    """Returns the fake provider's result with some fields overwritten (a contract bug)."""

    name = "fake"

    def __init__(self, **overrides):
        self._overrides, self._inner = overrides, FakeProvider()

    def generate(self, req):
        return self._inner.generate(req).model_copy(update=self._overrides)


class Raising:
    name = "fake"

    def __init__(self, exc):
        self._exc = exc

    def generate(self, req):
        raise self._exc


def test_results_come_back_in_input_order():
    ids = [f"r{i}" for i in range(5)]
    records = run_requests(FakeProvider([f"out-{i}" for i in range(5)]), [request(i) for i in ids])
    assert [r.result.request_id for r in records] == ids
    assert [r.result.output_text for r in records] == [f"out-{i}" for i in range(5)]


def test_accepts_any_iterable_and_empty_input():
    provider = FakeProvider()
    assert run_requests(provider, (request(f"r{i}") for i in range(3)))[2].result.request_id == "r2"
    assert run_requests(FakeProvider(), []) == []


def test_exactly_one_provider_call_per_request():
    provider = FakeProvider()
    requests = [request(f"r{i}") for i in range(4)]
    run_requests(provider, requests)
    assert provider.calls == tuple(requests)


@pytest.mark.parametrize(
    "kind", [FailureKind.TIMEOUT, FailureKind.RATE_LIMIT, FailureKind.SERVER_ERROR]
)
def test_retryable_failure_is_not_retried(kind):
    # If the runner retried, the second scripted item ("would succeed") would be consumed.
    provider = FakeProvider([kind, "would succeed"])
    [record] = run_requests(provider, [request()])
    assert isinstance(record.result, GenerationFailure) and record.result.kind is kind
    assert len(provider.calls) == 1


def test_failure_stays_structured_and_run_continues():
    provider = FakeProvider(["a", FailureKind.MALFORMED_RESPONSE, "c"])
    records = run_requests(provider, [request(f"r{i}") for i in range(3)])
    assert [type(r.result) for r in records] == [
        GenerationSuccess,
        GenerationFailure,
        GenerationSuccess,
    ]
    assert records[1].result.kind is FailureKind.MALFORMED_RESPONSE
    assert len(provider.calls) == 3


def test_elapsed_time_is_measured_around_the_provider_call():
    clock = FakeClock()
    provider = SlowProvider(clock, [2.5, 0.25, 7.0])
    records = run_requests(provider, [request(f"r{i}") for i in range(3)], clock=clock)
    assert [r.elapsed_s for r in records] == [2.5, 0.25, 7.0]
    assert clock.reads == 6  # two reads per request, none elsewhere


def test_failures_are_timed_too():
    clock = FakeClock()
    provider = SlowProvider(clock, [1.5], inner=FakeProvider([FailureKind.TIMEOUT]))
    [record] = run_requests(provider, [request()], clock=clock)
    assert isinstance(record.result, GenerationFailure) and record.elapsed_s == 1.5


def test_default_clock_is_time_perf_counter():
    # Pins the monotonic clock: a wall-clock default (e.g. time.time) must fail this test.
    assert inspect.signature(run_requests).parameters["clock"].default is time.perf_counter


def test_run_with_default_clock_records_non_negative_elapsed():
    [record] = run_requests(FakeProvider(), [request()])
    assert record.elapsed_s >= 0


def test_invalid_item_later_in_the_input_fails_before_any_provider_call():
    provider = FakeProvider()

    def requests():
        yield request("r1")
        yield GenerationRequest(
            request_id="", model="m", messages=(Message(role="user", content="x"),)
        )

    with pytest.raises(ValidationError):
        run_requests(provider, requests())
    assert provider.calls == ()


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"request_id": "other"}, "request_id"),
        ({"requested_model": "other"}, "requested_model"),
        ({"provider": "impostor"}, "provider"),
    ],
)
def test_mismatch_is_raised_explicitly_and_not_corrected(overrides, field):
    with pytest.raises(ProviderContractViolation, match=field):
        run_requests(Tampering(**overrides), [request("r1", "m")])


def test_mismatch_on_a_later_request_is_still_caught():
    class OnlySecondWrong:
        name = "fake"

        def __init__(self):
            self._inner, self._n = FakeProvider(), 0

        def generate(self, req):
            self._n += 1
            result = self._inner.generate(req)
            return result.model_copy(update={"request_id": "zzz"}) if self._n == 2 else result

    with pytest.raises(ProviderContractViolation, match="request_id"):
        run_requests(OnlySecondWrong(), [request("r1"), request("r2")])


@pytest.mark.parametrize("bad", [None, {"status": "ok"}, "text"])
def test_non_result_return_value_is_a_violation(bad):
    class Returns:
        name = "fake"

        def generate(self, req):
            return bad

    with pytest.raises(ProviderContractViolation, match="expected GenerationSuccess"):
        run_requests(Returns(), [request()])


@pytest.mark.parametrize(
    "exc", [ValueError("bug"), KeyError("k"), TypeError("t"), ZeroDivisionError()]
)
def test_provider_programming_errors_propagate_unchanged(exc):
    with pytest.raises(type(exc)) as info:
        run_requests(Raising(exc), [request()])
    assert info.value is exc
    assert not isinstance(info.value, ProviderContractViolation)


def test_exhausted_fake_script_propagates_as_an_error_not_a_failure():
    with pytest.raises(RuntimeError, match="exhausted"):
        run_requests(FakeProvider(["only one"]), [request("r1"), request("r2")])


def test_runs_offline_with_the_socket_block_active():
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket.create_connection(("127.0.0.1", 9))  # proves the conftest fixture is active
    assert len(run_requests(FakeProvider(), [request()])) == 1


def test_execution_record_round_trips_through_json():
    [record] = run_requests(FakeProvider([FailureKind.TIMEOUT]), [request()])
    assert ExecutionRecord.model_validate_json(record.model_dump_json()) == record


@pytest.mark.parametrize("elapsed", [-0.001, math.nan, math.inf])
def test_execution_record_rejects_invalid_elapsed(elapsed):
    result = FakeProvider(["x"]).generate(request())
    with pytest.raises(ValidationError):
        ExecutionRecord(result=result, elapsed_s=elapsed)


def test_execution_record_is_frozen_and_closed():
    result = FakeProvider(["x"]).generate(request())
    record = ExecutionRecord(result=result, elapsed_s=0.0)
    with pytest.raises(ValidationError):
        record.elapsed_s = 1.0
    with pytest.raises(ValidationError):
        ExecutionRecord(result=result, elapsed_s=0.0, cost=0)


def test_run_one_matches_a_single_item_run_requests():
    one = run_one(FakeProvider(["x"]), request("r1"), clock=FakeClock())
    [many] = run_requests(FakeProvider(["x"]), [request("r1")], clock=FakeClock())
    assert one == many


def test_run_one_makes_one_call_and_checks_the_contract():
    provider = FakeProvider([FailureKind.TIMEOUT, "unused"])
    record = run_one(provider, request("r1"))
    assert isinstance(record.result, GenerationFailure) and len(provider.calls) == 1
    with pytest.raises(ProviderContractViolation, match="request_id"):
        run_one(Tampering(request_id="other"), request("r1"))
