"""The provider protocol. Concrete providers live in niriksha.providers and depend on this."""

from typing import Protocol, runtime_checkable

from niriksha.core.generation import GenerationRequest, GenerationResult


@runtime_checkable
class Provider(Protocol):
    """One model endpoint, configured by a named profile.

    Contract for ``generate``:

    - Makes exactly one attempt. It never retries; the runner owns retries and attempt records.
    - Returns ``GenerationFailure`` for expected provider or transport problems.
    - Does not catch programming errors or invalid input; those raise.
    - Never substitutes another model or provider.
    - Does not mutate the request.
    - Returns a result whose ``request_id`` and ``requested_model`` equal the request's.
      The runner verifies this (M1.2).
    - Does not measure authoritative latency; the runner times the call.
    """

    @property
    def name(self) -> str: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...
