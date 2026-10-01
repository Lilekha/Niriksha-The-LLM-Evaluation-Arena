"""Minimal sequential runner: one provider call per request, timed by the runner.

Timing lives here, not in the generation schemas or in providers. Measuring every call with the
same monotonic clock around ``provider.generate`` keeps latency comparable across providers; a
provider's own timing would be self-reported and, for the fake provider, scripted.

Not in this module (later milestones): retries, concurrency, persistence, run state. Because
nothing is persisted yet, an exception partway through a run discards the records collected so far.
"""

import time
from collections.abc import Callable, Iterable

from pydantic import BaseModel, ConfigDict, Field

from niriksha.core.generation import (
    GenerationFailure,
    GenerationRequest,
    GenerationResult,
    GenerationSuccess,
)
from niriksha.core.provider import Provider


class ProviderContractViolation(RuntimeError):
    """A provider returned a result that breaks the Provider contract.

    This is a bug in the provider, not a model outcome, so it is raised rather than recorded as a
    ``GenerationFailure`` and the result is never corrected.
    """


class ExecutionRecord(BaseModel):
    """One request's outcome plus the runner's own measurement of the provider call."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    result: GenerationResult
    elapsed_s: float = Field(ge=0, allow_inf_nan=False)  # monotonic clock, around generate() only


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


def run_one(
    provider: Provider,
    request: GenerationRequest,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> ExecutionRecord:
    """Make exactly one provider call for ``request`` and time it with ``clock``.

    A result whose request_id, requested_model or provider name does not match raises
    ``ProviderContractViolation``. Any other exception from the provider propagates unchanged.
    """
    start = clock()
    result = provider.generate(request)
    elapsed = clock() - start
    _check_contract(provider, request, result)
    return ExecutionRecord(result=result, elapsed_s=elapsed)


def run_requests(
    provider: Provider,
    requests: Iterable[GenerationRequest],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> list[ExecutionRecord]:
    """Call ``provider.generate`` exactly once per request, sequentially, in input order.

    Failures come back as records holding a ``GenerationFailure``; they are not retried.
    Contract violations and provider exceptions behave as in ``run_one``.
    """
    # Materialise first: an invalid item raises before any provider call is made.
    queued = list(requests)
    return [run_one(provider, request, clock=clock) for request in queued]
