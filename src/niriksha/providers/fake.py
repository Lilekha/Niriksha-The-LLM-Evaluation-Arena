"""Deterministic fake provider for offline tests and development.

It is not an inference service: no network, no model, no randomness, no files, no environment
variables. Results carry ``provider_metadata={"fake": True}`` so synthetic output is never mistaken
for a real model's. It never reports usage, because it has no tokenizer and must not invent numbers.
"""

from collections.abc import Sequence

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationRequest,
    GenerationResult,
    GenerationSuccess,
)

ScriptItem = str | FailureKind


class FakeProvider:
    """Returns scripted results in call order.

    - ``script=None`` (default): echo the last user message as the output text.
    - ``script=[...]``: one item per call, in order. A ``str`` is a successful output text;
      a ``FailureKind`` is a failure of that kind. Pass the enum member, not its string value,
      because a plain string is always treated as output text.
    - Calling past the end of the script raises ``RuntimeError``: a test that needs more calls than
      it scripted has a bug, and a hidden default would mask it.

    Every request received is kept in ``calls``, so tests can assert exactly how many provider
    calls were made.
    """

    def __init__(
        self,
        script: Sequence[ScriptItem] | None = None,
        *,
        name: str = "fake",
        returned_model: str = "fake-model-v0",
    ) -> None:
        if isinstance(script, str):  # also catches a bare FailureKind (a str subclass)
            raise TypeError(
                f"script must be a sequence of str or FailureKind, not a bare string; "
                f"use [{script!r}]"
            )
        if script is not None:
            for item in script:
                if not isinstance(item, str | FailureKind):
                    raise TypeError(f"script items must be str or FailureKind, got {item!r}")
        self._script = None if script is None else tuple(script)
        self._name = name
        self._returned_model = returned_model
        self._calls: list[GenerationRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def calls(self) -> tuple[GenerationRequest, ...]:
        return tuple(self._calls)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        index = len(self._calls)
        self._calls.append(request)
        if self._script is None:
            item: ScriptItem = next(
                m.content for m in reversed(request.messages) if m.role == "user"
            )
        elif index < len(self._script):
            item = self._script[index]
        else:
            raise RuntimeError(
                f"FakeProvider script exhausted: call {index + 1}, {len(self._script)} scripted"
            )

        common = {
            "request_id": request.request_id,
            "provider": self._name,
            "requested_model": request.model,
            "provider_metadata": {"fake": True},
        }
        if isinstance(item, FailureKind):  # check first: FailureKind is also a str
            return GenerationFailure(
                **common, kind=item, message=f"scripted {item.value} (fake provider)"
            )
        return GenerationSuccess(**common, returned_model=self._returned_model, output_text=item)
