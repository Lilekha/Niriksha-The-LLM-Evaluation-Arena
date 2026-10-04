"""Minimal sequential runner: one provider call per request, timed by the runner.

Timing lives here, not in the generation schemas or in providers. Measuring every call with the
same monotonic clock around ``provider.generate`` keeps latency comparable across providers; a
provider's own timing would be self-reported and, for the fake provider, scripted.

Retries are opt-in and owned here, never by a provider (docs/adr/0002, docs/adr/0009). The default
``RetryPolicy`` makes exactly one attempt and leaves the record exactly as before. With a policy of
several attempts, only a failure of a retryable kind is retried, at most ``max_attempts`` times,
after a bounded, deterministic wait, and every attempt is kept in ``ExecutionRecord.attempts``.

Not in this module: concurrency and run state.
"""

import math
import time
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationRequest,
    GenerationResult,
    GenerationSuccess,
)
from niriksha.core.provider import Provider

Sleep = Callable[[float], None]
RETRY_AFTER_KEY = "retry_after_s"  # provider_metadata key a provider uses to report Retry-After
_DEFAULT_RETRY_ON = frozenset(
    {
        FailureKind.TIMEOUT,
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER_ERROR,
        FailureKind.CONNECTION_ERROR,
    }
)


class ProviderContractViolation(RuntimeError):
    """A provider returned a result that breaks the Provider contract.

    This is a bug in the provider, not a model outcome, so it is raised rather than recorded as a
    ``GenerationFailure`` and the result is never corrected.
    """


class RetryPolicy(BaseModel):
    """How often and how long to retry a failed call. The default is one attempt: no retries.

    Only a failure whose kind is in ``retry_on`` is retried. The wait before retry ``n`` is
    ``min(backoff_base_s * 2 ** (n - 1), backoff_cap_s)``, with no jitter. A provider that reports
    a ``retry_after_s`` makes the wait at least that long; if it asks for more than ``max_wait_s``
    the runner stops retrying and records the failure instead of sleeping that long.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_attempts: int = Field(default=1, ge=1, le=10)
    backoff_base_s: float = Field(default=1.0, ge=0, le=60, allow_inf_nan=False)
    backoff_cap_s: float = Field(default=30.0, ge=0, le=300, allow_inf_nan=False)
    max_wait_s: float = Field(default=60.0, ge=0, le=300, allow_inf_nan=False)
    retry_on: frozenset[FailureKind] = _DEFAULT_RETRY_ON


NO_RETRY = RetryPolicy()  # one attempt: the default everywhere


class AttemptRecord(BaseModel):
    """One call to the provider within a retried request."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    result: GenerationResult
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)  # this call only, not the waits
    waited_s: float = Field(ge=0, allow_inf_nan=False)  # the wait requested before this call


class ExecutionRecord(BaseModel):
    """One request's outcome plus the runner's own measurement of the provider call.

    ``result`` and ``elapsed_s`` are always those of the final call. ``attempts`` is ``None`` when
    retries were not enabled (one call, nothing to add, and it is then left out of the serialized
    form, so such records are byte-identical to those written before retries existed). When a
    retry policy was used it lists every call in order: the last one is the final result and every
    earlier one is a failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    result: GenerationResult
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)  # monotonic clock, around generate() only
    attempts: tuple[AttemptRecord, ...] | None = None

    @model_validator(mode="after")
    def _attempts_are_consistent(self):
        if self.attempts is None:
            return self
        if not self.attempts:
            raise ValueError("attempts must not be empty")
        *earlier, last = self.attempts
        if last.result != self.result or last.elapsed_s != self.elapsed_s:
            raise ValueError("the final attempt must equal the recorded result and elapsed time")
        if any(not isinstance(attempt.result, GenerationFailure) for attempt in earlier):
            raise ValueError("every attempt before the last must be a failure")
        if self.attempts[0].waited_s != 0:
            raise ValueError("the first attempt has no wait")
        return self

    @model_serializer(mode="wrap")
    def _omit_unused_attempts(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if data.get("attempts") is None:
            data.pop("attempts", None)
        return data


def _check_contract(provider: Provider, request: GenerationRequest, result: object) -> None:
    if not isinstance(result, GenerationSuccess | GenerationFailure):
        raise ProviderContractViolation(
            f"provider {provider.name!r} returned {type(result).__name__} for request "
            f"{request.request_id!r}; expected GenerationSuccess or GenerationFailure"
        )
    expected = {
        "request_id": request.request_id,
        "requested_model": request.model,
        "provider": provider.name,
    }
    for field, want in expected.items():
        got = getattr(result, field)
        if got != want:
            raise ProviderContractViolation(
                f"provider {provider.name!r}: result {field} {got!r} does not match "
                f"expected {want!r} (request {request.request_id!r})"
            )


def _next_wait(policy: RetryPolicy, attempt: int, result: GenerationResult) -> float | None:
    """Seconds to wait before attempt ``attempt + 1``, or ``None`` to stop and keep this result."""
    if attempt >= policy.max_attempts:
        return None
    if not isinstance(result, GenerationFailure) or result.kind not in policy.retry_on:
        return None
    wait = min(policy.backoff_base_s * 2 ** (attempt - 1), policy.backoff_cap_s)
    asked = result.provider_metadata.get(RETRY_AFTER_KEY)
    if isinstance(asked, int | float) and not isinstance(asked, bool) and math.isfinite(asked):
        if asked > policy.max_wait_s:
            return None  # the provider wants longer than we are willing to wait
        wait = max(wait, asked, 0.0)
    return min(wait, policy.max_wait_s)


def run_one(
    provider: Provider,
    request: GenerationRequest,
    *,
    clock: Callable[[], float] = time.perf_counter,
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = time.sleep,
) -> ExecutionRecord:
    """Call the provider for ``request``, timing each call with ``clock``.

    With the default policy this is exactly one call. With ``retry.max_attempts > 1`` a failure of
    a retryable kind is retried after ``sleep(wait)``, at most ``max_attempts`` calls in total; the
    record then lists every attempt. A result whose request_id, requested_model or provider name
    does not match raises ``ProviderContractViolation`` (checked on every attempt). Any other
    exception from the provider propagates unchanged.

    A retry after a timeout can repeat a request the server already processed.
    """
    attempts: list[AttemptRecord] = []
    waited = 0.0
    while True:
        start = clock()
        result = provider.generate(request)
        elapsed = clock() - start
        _check_contract(provider, request, result)
        attempts.append(AttemptRecord(result=result, elapsed_s=elapsed, waited_s=waited))
        wait = _next_wait(retry, len(attempts), result)
        if wait is None:
            break
        sleep(wait)
        waited = wait
    if retry.max_attempts == 1:
        return ExecutionRecord(result=result, elapsed_s=elapsed)
    return ExecutionRecord(result=result, elapsed_s=elapsed, attempts=tuple(attempts))


def run_requests(
    provider: Provider,
    requests: Iterable[GenerationRequest],
    *,
    clock: Callable[[], float] = time.perf_counter,
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = time.sleep,
) -> list[ExecutionRecord]:
    """Call the provider once per request (more only under ``retry``), sequentially, in order.

    By default failures come back as records holding a ``GenerationFailure`` and are not retried.
    Contract violations and provider exceptions behave as in ``run_one``.
    """
    # Materialise first: an invalid item raises before any provider call is made.
    queued = list(requests)
    return [run_one(provider, request, clock=clock, retry=retry, sleep=sleep) for request in queued]
