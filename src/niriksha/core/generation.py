"""Provider-independent generation contract: request, result and failure types.

Design rules (see docs/adr/0002-provider-contract.md):

- Models are frozen, strict and reject unknown fields, so nothing is silently coerced or dropped.
- ``None`` means "unknown / not sent". It is never replaced by a default value or a zero.
- There is no timing field. The runner measures elapsed time around each provider call and
  records it in its attempt record (M1.2); adapters are not authoritative for latency.
- There is no cost field. Cost is derived later from usage and a dated price table.
- An invalid request raises ``pydantic.ValidationError`` at construction, before any attempt exists.
- Expected provider/transport problems are returned as ``GenerationFailure``. Programming errors
  are not failures and must propagate as exceptions.
"""

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, AfterValidator(_non_blank)]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Message(_Model):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)  # stored exactly as given, never stripped


class GenerationParams(_Model):
    """Decoding parameters. ``None`` means not sent: the provider default applies."""

    temperature: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    seed: int | None = None
    stop: tuple[Annotated[str, Field(min_length=1)], ...] | None = None
    response_format: Literal["text", "json_object"] | None = None


class GenerationRequest(_Model):
    request_id: NonBlank  # assigned by the caller (the runner)
    model: NonBlank
    messages: tuple[Message, ...] = Field(min_length=1)
    params: GenerationParams = GenerationParams()

    @model_validator(mode="after")
    def _has_user_message(self) -> "GenerationRequest":
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("messages must contain at least one user message")
        return self


class Usage(_Model):
    """Token counts as reported or estimated. ``None`` means not available."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(
        default=None, ge=0
    )  # need not equal the sum (provider-defined)
    source: Literal["reported", "estimated"]


class FailureKind(StrEnum):
    TIMEOUT = "timeout"  # no response within the configured timeout
    RATE_LIMIT = "rate_limit"  # provider throttled the request
    SERVER_ERROR = "server_error"  # provider-side 5xx
    MALFORMED_RESPONSE = "malformed_response"  # response could not be parsed into a result
    INVALID_REQUEST = "invalid_request"  # provider rejected a schema-valid request
    UNSUPPORTED_CAPABILITY = "unsupported_capability"  # needs a capability not declared
    INTERNAL_ERROR = "internal_error"  # adapter-detected fault; NOT a catch-all for exceptions


_RETRYABLE = frozenset({FailureKind.TIMEOUT, FailureKind.RATE_LIMIT, FailureKind.SERVER_ERROR})

# ponytail: substring heuristic for obvious secret-bearing keys; not a secret-handling guarantee.
_FORBIDDEN_KEY_PARTS = (
    "authorization",
    "api-key",
    "apikey",
    "secret",
    "password",
    "cookie",
    "bearer",
)

MetadataValue = str | int | float | bool | None


class _ResultBase(_Model):
    request_id: NonBlank
    provider: NonBlank  # provider profile name
    requested_model: NonBlank
    returned_model: NonBlank | None = None  # as reported by the server, if at all
    # Flat, JSON-safe provider details (e.g. a fingerprint). Never headers, keys or raw payloads.
    provider_metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    @field_validator("provider_metadata")
    @classmethod
    def _safe_metadata(cls, value: dict[str, MetadataValue]) -> dict[str, MetadataValue]:
        for key, item in value.items():
            normalized = key.lower().replace("_", "-")
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError(f"metadata key {key!r} looks secret-bearing")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"metadata value for {key!r} must be finite")
        return value


class GenerationSuccess(_ResultBase):
    status: Literal["ok"] = "ok"
    output_text: str  # may be empty; stored exactly as returned
    finish_reason: str | None = None  # raw provider string, not normalised
    usage: Usage | None = None


class GenerationFailure(_ResultBase):
    status: Literal["failed"] = "failed"
    kind: FailureKind
    message: NonBlank  # must not contain secrets or full raw payloads
    http_status: int | None = Field(default=None, ge=100, le=599)

    @property
    def retryable(self) -> bool:
        """Hint only. Whether and how often to retry is runner policy."""
        return self.kind in _RETRYABLE


GenerationResult = Annotated[GenerationSuccess | GenerationFailure, Field(discriminator="status")]

result_adapter: TypeAdapter[GenerationSuccess | GenerationFailure] = TypeAdapter(GenerationResult)
