"""OpenAI-compatible chat-completions adapter.

Design and limits: docs/adr/0009-openai-compatible-adapter-and-retries.md.

``OpenAICompatibleProvider`` makes exactly one HTTP request per ``generate`` call and never retries;
the runner owns retries (``HTTP_RETRY_POLICY`` is the policy intended for HTTP runs, and it must be
passed to ``execute_run`` explicitly). It returns a ``GenerationFailure`` for expected problems and
raises for programming and configuration errors. It never substitutes another model or endpoint.

Behaviour that is deliberately conservative:

- A request parameter the profile does not declare as supported, or more ``stop`` sequences than
  the profile allows, is refused with ``unsupported_capability`` before any HTTP request is made.
  Nothing is dropped silently. A parameter being *sent* does not mean the server honours it.
- A response that is not exactly what was asked for (one choice with string content, optional
  integer token counts) is ``malformed_response``. Nothing is repaired, and token usage that is
  missing stays ``None``; it is never replaced by zero. There is no cost.
- Failure messages are fixed text plus the status code and, when it is a short lowercase token, the
  server's ``error.type`` or ``error.code``. Free-form server text is never copied, because servers
  can echo prompts or fragments of keys. The key is read from an environment variable the profile
  names, is sent only in the ``Authorization`` header, and appears in no message, metadata or repr.
- Redirects are not followed, proxies and netrc from the environment are ignored (``trust_env`` is
  off), and a response larger than ``max_response_bytes`` is refused.

What this does not know: how any particular server behaves. Servers that call themselves
OpenAI-compatible differ (for example in ``seed``, ``response_format`` and the name of the
max-tokens field), which is why those are profile settings.
"""

import ipaddress
import json
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, NoReturn
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationParams,
    GenerationRequest,
    GenerationResult,
    GenerationSuccess,
    MetadataValue,
    NonBlank,
    Usage,
)
from niriksha.core.runner import RETRY_AFTER_KEY, RetryPolicy

API_STYLE = "openai_chat_completions"
# The retry policy intended for HTTP providers. The core default is one attempt; pass this to
# execute_run(retry=...) to enable three. A retry after a timeout can repeat a request that the
# server already processed (and may bill for).
HTTP_RETRY_POLICY = RetryPolicy(max_attempts=3)

DEFAULT_SUPPORTED_PARAMS = frozenset(
    {"temperature", "top_p", "max_tokens", "stop", "response_format"}
)
_ALL_PARAMS = frozenset(GenerationParams.model_fields)
_LOOPBACK_NAMES = frozenset({"localhost"})
_TOKEN = re.compile(r"^[a-z0-9_.-]{1,64}$")
_RETRY_AFTER = re.compile(r"^\d{1,9}(\.\d{1,6})?$")
_ERROR_BODY_LIMIT = 64 * 1024


class ProviderConfigError(Exception):
    """The adapter is misconfigured (for example a named environment variable is not set)."""


def _is_loopback(host: str) -> bool:
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpenAICompatibleProfile(BaseModel):
    """One endpoint. Holds the *name* of the key's environment variable, never a key."""

    # hide_input_in_errors: a validation error must not echo a URL that may hold credentials
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)

    name: NonBlank  # recorded as the provider name in manifests and results
    base_url: NonBlank  # e.g. http://127.0.0.1:8000/v1; the adapter appends /chat/completions
    api_key_env: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    timeout_s: float = Field(default=60.0, gt=0, le=3600, allow_inf_nan=False)  # per operation
    connect_timeout_s: float = Field(default=10.0, gt=0, le=600, allow_inf_nan=False)
    supported_params: frozenset[str] = DEFAULT_SUPPORTED_PARAMS
    max_stop_sequences: int = Field(default=4, ge=1, le=64)
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    max_response_bytes: int = Field(default=4 * 1024 * 1024, ge=1024, le=64 * 1024 * 1024)

    @field_validator("base_url")
    @classmethod
    def _plain_http_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("base_url must be an http or https URL with a host")
        if parts.username is not None or parts.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parts.query or parts.fragment:
            raise ValueError("base_url must not contain a query or a fragment")
        return value

    @field_validator("supported_params")
    @classmethod
    def _known_params(cls, value: frozenset[str]) -> frozenset[str]:
        unknown = value - _ALL_PARAMS
        if unknown:
            raise ValueError(f"unknown generation parameter(s): {', '.join(sorted(unknown))}")
        return value

    @model_validator(mode="after")
    def _no_key_over_plain_http(self):
        parts = urlsplit(self.base_url)
        if self.api_key_env and parts.scheme == "http" and not _is_loopback(parts.hostname or ""):
            raise ValueError("an API key must not be sent over plain http to a non-loopback host")
        return self

    @property
    def endpoint(self) -> str:
        """Scheme, host, port and path only: no credentials, query or fragment."""
        parts = urlsplit(self.base_url)
        return f"{parts.scheme}://{parts.netloc}{parts.path.rstrip('/')}"


class _Malformed(Exception):
    pass


def _reject_constant(name: str) -> NoReturn:
    raise _Malformed(f"{name} is not valid JSON")


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _Malformed("usage holds a token count that is not a non-negative integer")
    return value


def _usage(raw: Any) -> Usage | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise _Malformed("usage is not an object")
    counts = {
        name: _count(raw.get(name))
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if all(value is None for value in counts.values()):
        return None  # nothing was reported: unknown, not zero
    return Usage(**counts, source="reported")


def _optional_text(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise _Malformed(f"{what} is not a string")
    return value if value.strip() else None


class OpenAICompatibleProvider:
    """A ``Provider`` for a server that implements ``POST {base_url}/chat/completions``."""

    def __init__(
        self, profile: OpenAICompatibleProfile, *, environ: Mapping[str, str] | None = None
    ) -> None:
        self._profile = profile
        self._key: str | None = None
        if profile.api_key_env is not None:
            value = (os.environ if environ is None else environ).get(profile.api_key_env)
            if value is None or not value.strip():
                raise ProviderConfigError(
                    f"environment variable {profile.api_key_env} is not set or is empty"
                )
            if value != value.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
                raise ProviderConfigError(
                    f"environment variable {profile.api_key_env} has surrounding whitespace or "
                    "control characters"
                )
            self._key = value
        self._client = httpx.Client(
            base_url=profile.base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(profile.timeout_s, connect=profile.connect_timeout_s),
            follow_redirects=False,
            trust_env=False,
            # TLS verification is always on for https. For plain http there is no TLS to verify, and
            # redirects are never followed, so the CA bundle is not loaded needlessly.
            verify=profile.base_url.lower().startswith("https://"),
            headers={"User-Agent": "niriksha"},
        )

    @property
    def name(self) -> str:
        return self._profile.name

    @property
    def profile(self) -> OpenAICompatibleProfile:
        return self._profile

    def __repr__(self) -> str:
        return f"OpenAICompatibleProvider(name={self.name!r}, endpoint={self._profile.endpoint!r})"

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenAICompatibleProvider":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- results ----------------------------------------------------------------------------

    def _metadata(
        self, extra: Mapping[str, MetadataValue] | None = None
    ) -> dict[str, MetadataValue]:
        return {"endpoint": self._profile.endpoint, "api_style": API_STYLE, **(extra or {})}

    def _failure(
        self,
        request: GenerationRequest,
        kind: FailureKind,
        message: str,
        *,
        http_status: int | None = None,
        extra: Mapping[str, MetadataValue] | None = None,
    ) -> GenerationFailure:
        if self._key:  # defence in depth: messages are built from fixed text and never hold it
            message = message.replace(self._key, "[redacted]")
        return GenerationFailure(
            request_id=request.request_id,
            provider=self.name,
            requested_model=request.model,
            provider_metadata=self._metadata(extra),
            kind=kind,
            message=message,
            http_status=http_status,
        )

    # -- request ----------------------------------------------------------------------------

    def _unsupported(self, request: GenerationRequest) -> str | None:
        params = request.params
        set_names = {name for name in _ALL_PARAMS if getattr(params, name) is not None}
        unsupported = sorted(set_names - self._profile.supported_params)
        if unsupported:
            return "unsupported parameter(s) for this provider profile: " + ", ".join(unsupported)
        if params.stop is not None and len(params.stop) > self._profile.max_stop_sequences:
            return f"stop: at most {self._profile.max_stop_sequences} sequences are supported"
        return None

    def _body(self, request: GenerationRequest) -> dict[str, Any]:
        params = request.params
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
        }
        if params.temperature is not None:
            body["temperature"] = params.temperature
        if params.top_p is not None:
            body["top_p"] = params.top_p
        if params.max_tokens is not None:
            body[self._profile.max_tokens_field] = params.max_tokens
        if params.seed is not None:
            body["seed"] = params.seed
        if params.stop is not None:
            body["stop"] = list(params.stop)
        if params.response_format is not None:
            body["response_format"] = {"type": params.response_format}
        return body

    # -- the single attempt -----------------------------------------------------------------

    def generate(self, request: GenerationRequest) -> GenerationResult:
        refusal = self._unsupported(request)
        if refusal is not None:
            return self._failure(request, FailureKind.UNSUPPORTED_CAPABILITY, refusal)
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        payload = json.dumps(self._body(request), ensure_ascii=False).encode("utf-8")
        try:
            with self._client.stream(
                "POST", "chat/completions", content=payload, headers=headers
            ) as response:
                status = response.status_code
                retry_after = response.headers.get("retry-after")
                raw, too_large = self._read(response)
        except httpx.TimeoutException:
            return self._failure(
                request, FailureKind.TIMEOUT, "the request timed out before a response arrived"
            )
        except httpx.DecodingError:
            return self._failure(
                request, FailureKind.MALFORMED_RESPONSE, "the response body could not be decoded"
            )
        except httpx.TransportError as exc:
            return self._failure(
                request,
                FailureKind.CONNECTION_ERROR,
                f"no usable response from the server ({type(exc).__name__})",
            )
        return self._interpret(request, status, retry_after, raw, too_large)

    def _read(self, response: httpx.Response) -> tuple[bytes, bool]:
        limit = self._profile.max_response_bytes
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > limit:
                return b"", True
            chunks.append(chunk)
        return b"".join(chunks), False

    # -- interpreting the response ----------------------------------------------------------

    def _interpret(
        self,
        request: GenerationRequest,
        status: int,
        retry_after: str | None,
        raw: bytes,
        too_large: bool,
    ) -> GenerationResult:
        if status == 200:
            if too_large:
                return self._failure(
                    request,
                    FailureKind.MALFORMED_RESPONSE,
                    "the response is larger than the profile allows",
                    http_status=status,
                )
            try:
                return self._success(request, raw)
            except _Malformed as exc:
                return self._failure(
                    request, FailureKind.MALFORMED_RESPONSE, str(exc), http_status=status
                )
        kind, text = self._classify(status)
        extra: dict[str, MetadataValue] = {}
        if kind in (FailureKind.RATE_LIMIT, FailureKind.SERVER_ERROR) and retry_after:
            candidate = retry_after.strip()
            if _RETRY_AFTER.fullmatch(candidate):
                extra[RETRY_AFTER_KEY] = float(candidate)
        detail = None if too_large or len(raw) > _ERROR_BODY_LIMIT else self._error_detail(raw)
        message = f"HTTP {status}: {text}" + (f" (error: {detail})" if detail else "")
        return self._failure(request, kind, message, http_status=status, extra=extra)

    @staticmethod
    def _classify(status: int) -> tuple[FailureKind, str]:
        if status in (401, 403):
            return FailureKind.AUTH_ERROR, "authentication failed or is not permitted"
        if status == 408:
            return FailureKind.TIMEOUT, "the server reported a request timeout"
        if status == 429:
            return FailureKind.RATE_LIMIT, "rate limited"
        if status >= 500:
            return FailureKind.SERVER_ERROR, "server error"
        if 300 <= status < 400:
            return FailureKind.INVALID_REQUEST, "unexpected redirect (redirects are not followed)"
        if status >= 400:
            return FailureKind.INVALID_REQUEST, "the server rejected the request"
        return FailureKind.MALFORMED_RESPONSE, "unexpected status"

    @staticmethod
    def _error_detail(raw: bytes) -> str | None:
        """A short lowercase ``error.type``/``error.code`` token if the body has one, else None."""
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            return None
        error = document.get("error") if isinstance(document, dict) else None
        if not isinstance(error, dict):
            return None
        tokens = [
            error[key]
            for key in ("type", "code")
            if isinstance(error.get(key), str) and _TOKEN.fullmatch(error[key])
        ]
        return "/".join(tokens) or None

    def _success(self, request: GenerationRequest, raw: bytes) -> GenerationSuccess:
        try:
            document = json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
        except _Malformed:
            raise
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise _Malformed("the response body is not valid UTF-8 JSON") from None
        if not isinstance(document, dict):
            raise _Malformed("the response is not a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _Malformed("the response does not hold exactly one choice")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise _Malformed("the choice has no message object")
        content = message.get("content")
        if not isinstance(content, str):
            raise _Malformed("message.content is not a string")
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise _Malformed("finish_reason is not a string")
        usage = _usage(document.get("usage"))
        returned_model = _optional_text(document.get("model"), "model")
        fingerprint = _optional_text(document.get("system_fingerprint"), "system_fingerprint")
        extra: dict[str, MetadataValue] = (
            {} if fingerprint is None else {"system_fingerprint": fingerprint}
        )
        return GenerationSuccess(
            request_id=request.request_id,
            provider=self.name,
            requested_model=request.model,
            returned_model=returned_model,
            provider_metadata=self._metadata(extra),
            output_text=content,
            finish_reason=finish_reason,
            usage=usage,
        )
