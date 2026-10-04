"""The OpenAI-compatible adapter against a local fake HTTP server. No real network, no real keys."""

import ast
import json
import socket
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

import niriksha.providers.openai_compatible as adapter_module
from http_support import BLOCKED, FakeServer, Reply, closed_port_url, completion
from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationParams,
    GenerationRequest,
    GenerationSuccess,
    Message,
    Usage,
)
from niriksha.core.provider import Provider
from niriksha.core.runner import NO_RETRY, run_one
from niriksha.providers.openai_compatible import (
    DEFAULT_SUPPORTED_PARAMS,
    HTTP_RETRY_POLICY,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
    ProviderConfigError,
)

KEY = "sk-test-SECRETVALUE-12345"  # a deliberately fake key
TEXT = "नमस्ते ಕನ್ನಡ"
ENV_NAME = "NIRIKSHA_TEST_VARIABLE"  # the environment variable that holds the fake key
LEAK = "hunter2"  # a value that must never be echoed


def make_request(**params):
    return GenerationRequest(
        request_id="r1",
        model="my-model",
        messages=(Message(role="system", content="S"), Message(role="user", content=TEXT)),
        params=GenerationParams(**params),
    )


@pytest.fixture
def make(server):
    created = []

    def _make(*, environ=None, base_url=None, **profile):
        provider = OpenAICompatibleProvider(
            OpenAICompatibleProfile(name="local", base_url=base_url or server.base_url, **profile),
            environ=environ,
        )
        created.append(provider)
        return provider

    yield _make
    for provider in created:
        provider.close()


# -- request and response mapping -----------------------------------------------------------------


def test_a_successful_generation_maps_the_request_and_the_response(server, make):
    server.enqueue(
        Reply(
            200,
            completion(
                "Paris",
                model="served-1",
                usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                system_fingerprint="fp_1",
            ),
        )
    )
    result = make().generate(make_request(temperature=0.0, max_tokens=16))
    assert isinstance(result, GenerationSuccess)
    assert (result.request_id, result.provider, result.requested_model) == (
        "r1",
        "local",
        "my-model",
    )
    assert result.output_text == "Paris" and result.returned_model == "served-1"
    assert result.finish_reason == "stop"
    assert result.usage == Usage(
        prompt_tokens=5, completion_tokens=2, total_tokens=7, source="reported"
    )
    assert result.provider_metadata == {
        "endpoint": server.base_url,
        "api_style": "openai_chat_completions",
        "system_fingerprint": "fp_1",
    }
    (sent,) = server.requests
    assert sent.path == "/v1/chat/completions"
    assert sent.json() == {
        "model": "my-model",
        "messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": TEXT},
        ],
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 16,
    }
    assert sent.headers["content-type"] == "application/json"
    assert "authorization" not in sent.headers
    assert TEXT.encode("utf-8") in sent.body and b"\\u" not in sent.body  # sent as UTF-8, unescaped


@pytest.mark.parametrize(
    ("params", "profile", "expected"),
    [
        ({}, {}, {}),
        ({"top_p": 0.9}, {}, {"top_p": 0.9}),
        ({"stop": ("a", "b")}, {}, {"stop": ["a", "b"]}),
        ({"response_format": "json_object"}, {}, {"response_format": {"type": "json_object"}}),
        ({"response_format": "text"}, {}, {"response_format": {"type": "text"}}),
        (
            {"max_tokens": 9},
            {"max_tokens_field": "max_completion_tokens"},
            {"max_completion_tokens": 9},
        ),
        ({"seed": 7}, {"supported_params": DEFAULT_SUPPORTED_PARAMS | {"seed"}}, {"seed": 7}),
    ],
)
def test_only_the_parameters_that_are_set_are_sent(server, make, params, profile, expected):
    server.enqueue(Reply(200, completion()))
    make(**profile).generate(make_request(**params))
    body = server.requests[0].json()
    assert {k: v for k, v in body.items() if k not in ("model", "messages", "stream")} == expected


def test_the_key_is_sent_only_as_a_bearer_header(server, make):
    server.enqueue(Reply(200, completion()))
    make(api_key_env=ENV_NAME, environ={ENV_NAME: KEY}).generate(make_request())
    (sent,) = server.requests
    assert sent.headers["authorization"] == f"Bearer {KEY}"
    assert KEY.encode() not in sent.body


@pytest.mark.parametrize(
    ("text", "finish"),
    [("", "stop"), ("  keep\nspace  ", "length"), ("x", None)],
)
def test_content_is_kept_exactly_and_finish_reason_is_raw(server, make, text, finish):
    server.enqueue(Reply(200, completion(text, finish_reason=finish)))
    result = make().generate(make_request())
    assert isinstance(result, GenerationSuccess)
    assert result.output_text == text and result.finish_reason == finish


@pytest.mark.parametrize(
    ("document", "model"),
    [
        (completion(model="served"), "served"),
        ({**completion(), "model": None}, None),
        ({**completion(), "model": "  "}, None),
        ({k: v for k, v in completion().items() if k != "model"}, None),
    ],
)
def test_the_returned_model_is_kept_when_given_and_unknown_otherwise(server, make, document, model):
    server.enqueue(Reply(200, document))
    result = make().generate(make_request())
    assert result.returned_model == model and result.requested_model == "my-model"


def test_a_different_returned_model_is_recorded_not_rejected(server, make):
    server.enqueue(Reply(200, completion(model="something-else")))
    assert make().generate(make_request()).returned_model == "something-else"


# -- usage is never invented ----------------------------------------------------------------------


@pytest.mark.parametrize("usage", [None, {}, {"extra": 1}])
def test_missing_usage_stays_unknown_not_zero(server, make, usage):
    document = completion()
    if usage is not None:
        document["usage"] = usage
    server.enqueue(Reply(200, document))
    assert make().generate(make_request()).usage is None


def test_partial_and_zero_usage_are_reported_as_given(server, make):
    server.enqueue(
        Reply(200, completion(usage={"prompt_tokens": 3})),
        Reply(
            200, completion(usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        ),
    )
    provider = make()
    partial = provider.generate(make_request()).usage
    zeros = provider.generate(make_request()).usage
    assert partial == Usage(
        prompt_tokens=3, completion_tokens=None, total_tokens=None, source="reported"
    )
    assert zeros == Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0, source="reported")


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1},
        {"prompt_tokens": "3"},
        {"completion_tokens": True},
        {"total_tokens": 1.5},
        [1, 2],
        "many",
    ],
)
def test_invalid_usage_fails_closed(server, make, usage):
    server.enqueue(Reply(200, completion(usage=usage)))
    result = make().generate(make_request())
    assert isinstance(result, GenerationFailure) and result.kind is FailureKind.MALFORMED_RESPONSE
    assert result.http_status == 200


# -- unsupported parameters never reach the network -----------------------------------------------


def test_an_unsupported_parameter_is_refused_before_any_request(server, make):
    result = make().generate(make_request(seed=424242))
    assert isinstance(result, GenerationFailure)
    assert result.kind is FailureKind.UNSUPPORTED_CAPABILITY and result.http_status is None
    assert "seed" in result.message and "424242" not in result.message
    assert server.requests == []


def test_every_unsupported_parameter_is_named_sorted(server, make):
    provider = make(supported_params=frozenset({"temperature"}))
    result = provider.generate(make_request(temperature=0.1, top_p=0.5, seed=1, max_tokens=3))
    assert result.kind is FailureKind.UNSUPPORTED_CAPABILITY
    assert result.message.endswith("max_tokens, seed, top_p") and server.requests == []


def test_response_format_text_still_needs_support(server, make):
    provider = make(supported_params=frozenset({"temperature"}))
    result = provider.generate(make_request(response_format="text"))
    assert result.kind is FailureKind.UNSUPPORTED_CAPABILITY and server.requests == []


def test_too_many_stop_sequences_are_refused_and_the_limit_is_allowed(server, make):
    server.enqueue(Reply(200, completion()))
    provider = make()
    refused = provider.generate(make_request(stop=("a", "b", "c", "d", "e")))
    assert refused.kind is FailureKind.UNSUPPORTED_CAPABILITY and server.requests == []
    assert isinstance(provider.generate(make_request(stop=("a", "b", "c", "d"))), GenerationSuccess)
    assert len(server.requests) == 1


def test_a_profile_cannot_declare_an_unknown_parameter():
    with pytest.raises(ValidationError):
        OpenAICompatibleProfile(
            name="p", base_url="http://127.0.0.1:1/v1", supported_params=frozenset({"frequency"})
        )


# -- malformed responses fail closed --------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"\xff\xfe",
        b"[]",
        b"null",
        b"{}",
        b'{"choices": NaN}',
        {"choices": []},
        {"choices": [completion()["choices"][0]] * 2},
        {"choices": [1]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"content": None}}]},
        {"choices": [{"message": {"content": 5}}]},
        {"choices": [{"message": {"content": ["LEAK-12345"]}}]},
        {"choices": [{"message": {"content": "x"}, "finish_reason": 5}]},
        completion(model=7),
        pytest.param(b"[" * 200_000, id="deeply-nested"),
    ],
)
def test_a_response_that_is_not_what_was_asked_for_is_malformed(server, make, body):
    server.enqueue(Reply(200, body))
    result = make().generate(make_request())
    assert isinstance(result, GenerationFailure) and result.kind is FailureKind.MALFORMED_RESPONSE
    assert result.http_status == 200 and "LEAK" not in result.message


def test_an_oversized_response_is_refused(server, make):
    provider = make(max_response_bytes=1024)
    server.enqueue(Reply(200, completion("x" * 5000)), Reply(500, "y" * 5000))
    big = provider.generate(make_request())
    assert big.kind is FailureKind.MALFORMED_RESPONSE and "larger" in big.message
    assert provider.generate(make_request()).kind is FailureKind.SERVER_ERROR  # still classified


# -- HTTP statuses --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, FailureKind.INVALID_REQUEST),
        (404, FailureKind.INVALID_REQUEST),
        (413, FailureKind.INVALID_REQUEST),
        (418, FailureKind.INVALID_REQUEST),
        (422, FailureKind.INVALID_REQUEST),
        (401, FailureKind.AUTH_ERROR),
        (403, FailureKind.AUTH_ERROR),
        (408, FailureKind.TIMEOUT),
        (429, FailureKind.RATE_LIMIT),
        (500, FailureKind.SERVER_ERROR),
        (502, FailureKind.SERVER_ERROR),
        (503, FailureKind.SERVER_ERROR),
        (504, FailureKind.SERVER_ERROR),
        (529, FailureKind.SERVER_ERROR),
        (201, FailureKind.MALFORMED_RESPONSE),
        (204, FailureKind.MALFORMED_RESPONSE),
    ],
)
def test_http_statuses_map_to_typed_failures(server, make, status, kind):
    server.enqueue(Reply(status, b"" if status == 204 else {"error": {"type": "some_error"}}))
    result = make().generate(make_request())
    assert isinstance(result, GenerationFailure)
    assert result.kind is kind and result.http_status == status
    assert result.retryable is (
        kind
        in {
            FailureKind.TIMEOUT,
            FailureKind.RATE_LIMIT,
            FailureKind.SERVER_ERROR,
            FailureKind.CONNECTION_ERROR,
        }
    )
    assert len(server.requests) == 1  # the adapter makes exactly one attempt


def test_a_redirect_is_not_followed_and_no_header_is_forwarded(server, make):
    other = FakeServer()
    try:
        server.enqueue(Reply(302, headers={"Location": f"{other.base_url}/stolen"}))
        provider = make(api_key_env="K", environ={"K": KEY})
        result = provider.generate(make_request())
        assert result.kind is FailureKind.INVALID_REQUEST and result.http_status == 302
        assert "redirect" in result.message
        assert other.requests == [] and len(server.requests) == 1
    finally:
        other.stop()


def test_a_short_error_token_is_kept_and_free_text_is_not(server, make):
    body = {
        "error": {"type": "invalid_request_error", "code": "model_not_found", "message": "ZZZ-FREE"}
    }
    server.enqueue(Reply(404, body), Reply(400, {"error": {"type": "Bad Type!", "message": "ZZZ"}}))
    provider = make()
    first = provider.generate(make_request())
    assert "invalid_request_error/model_not_found" in first.message and "ZZZ" not in first.message
    second = provider.generate(make_request())
    assert "error:" not in second.message and "ZZZ" not in second.message


@pytest.mark.parametrize(
    ("header", "expected"),
    [("2", 2.0), ("0", 0.0), ("1.5", 1.5), (" 3 ", 3.0)],
)
def test_retry_after_seconds_are_reported_for_rate_limits_and_server_errors(
    server, make, header, expected
):
    server.enqueue(
        Reply(429, b"", {"Retry-After": header}), Reply(503, b"", {"Retry-After": header})
    )
    provider = make()
    for _ in range(2):
        assert provider.generate(make_request()).provider_metadata["retry_after_s"] == expected


@pytest.mark.parametrize(
    "header",
    ["Wed, 21 Oct 2026 07:28:00 GMT", "soon", "-1", "9999999999999", "1e3", ""],
)
def test_unusable_retry_after_values_are_ignored(server, make, header):
    server.enqueue(Reply(429, b"", {"Retry-After": header}))
    assert "retry_after_s" not in make().generate(make_request()).provider_metadata


def test_retry_after_is_not_reported_for_other_failures(server, make):
    server.enqueue(Reply(400, b"", {"Retry-After": "5"}))
    assert "retry_after_s" not in make().generate(make_request()).provider_metadata


# -- transport failures ---------------------------------------------------------------------------


def test_a_server_that_never_answers_times_out_quickly(blackhole):
    with OpenAICompatibleProvider(
        OpenAICompatibleProfile(name="p", base_url=blackhole, timeout_s=0.3, connect_timeout_s=1.0)
    ) as provider:
        started = time.perf_counter()
        result = provider.generate(make_request())
    assert isinstance(result, GenerationFailure) and result.kind is FailureKind.TIMEOUT
    assert result.http_status is None and time.perf_counter() - started < 5


def test_a_refused_connection_is_a_connection_error(loopback_only):
    with OpenAICompatibleProvider(
        OpenAICompatibleProfile(name="p", base_url=closed_port_url(), timeout_s=2.0)
    ) as provider:
        result = provider.generate(make_request())
    assert isinstance(result, GenerationFailure) and result.kind is FailureKind.CONNECTION_ERROR
    assert result.http_status is None and result.retryable


def test_a_dropped_connection_is_a_connection_error_after_one_request(server, make):
    server.enqueue(Reply(drop=True))
    result = make().generate(make_request())
    assert isinstance(result, GenerationFailure) and result.kind is FailureKind.CONNECTION_ERROR
    assert len(server.requests) == 1


# -- secrets --------------------------------------------------------------------------------------


def test_the_key_appears_nowhere_even_when_the_server_echoes_it(server, make):
    echo = {
        "error": {
            "message": f"Incorrect API key provided: {KEY}",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }
    server.enqueue(Reply(401, echo), Reply(401, {"error": {"code": KEY}}))
    provider = make(api_key_env=ENV_NAME, environ={ENV_NAME: KEY})
    first = provider.generate(make_request())
    second = provider.generate(make_request())
    assert first.kind is FailureKind.AUTH_ERROR and first.http_status == 401
    assert "invalid_request_error/invalid_api_key" in first.message
    for result in (first, second):
        assert KEY not in result.message and KEY not in result.model_dump_json()
    assert KEY not in repr(provider) and KEY not in repr(provider.profile)
    assert KEY not in str(provider.profile.model_dump())
    assert server.requests[0].headers["authorization"] == f"Bearer {KEY}"


# -- configuration --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://127.0.0.1/v1",
        "http:///v1",
        "127.0.0.1/v1",
        "http://user:" + LEAK + "@127.0.0.1/v1",
        "http://127.0.0.1/v1?api_key=" + LEAK,
        "http://127.0.0.1/v1#" + LEAK,
    ],
)
def test_unsafe_or_malformed_base_urls_are_refused_without_echoing_them(url):
    with pytest.raises(ValidationError) as caught:
        OpenAICompatibleProfile(name="p", base_url=url)
    assert LEAK not in str(caught.value)


@pytest.mark.parametrize(
    ("url", "key_env", "ok"),
    [
        ("http://example.com/v1", "K", False),  # a key over plain http to a remote host
        ("http://example.com/v1", None, True),
        ("https://example.com/v1", "K", True),
        ("http://127.0.0.1:8000/v1", "K", True),
        ("http://localhost:8000/v1", "K", True),
        ("http://[::1]:8000/v1", "K", True),
    ],
)
def test_a_key_is_never_configured_for_plain_http_to_a_remote_host(url, key_env, ok):
    if ok:
        OpenAICompatibleProfile(name="p", base_url=url, api_key_env=key_env)
    else:
        with pytest.raises(ValidationError):
            OpenAICompatibleProfile(name="p", base_url=url, api_key_env=key_env)


@pytest.mark.parametrize("name", ["1BAD", "a-b", "has space", "", "KEY=VALUE"])
def test_the_key_variable_must_be_a_valid_name(name):
    with pytest.raises(ValidationError):
        OpenAICompatibleProfile(name="p", base_url="https://example.com/v1", api_key_env=name)


def test_the_endpoint_never_holds_more_than_scheme_host_port_and_path():
    profile = OpenAICompatibleProfile(name="p", base_url="https://api.example.com:8443/v1/")
    assert profile.endpoint == "https://api.example.com:8443/v1"


def test_a_missing_or_unusable_key_variable_is_a_configuration_error_naming_only_the_variable():
    profile = OpenAICompatibleProfile(name="p", base_url="https://example.com/v1", api_key_env="K")
    for environ in ({}, {"K": ""}, {"K": "   "}):
        with pytest.raises(ProviderConfigError, match="K"):
            OpenAICompatibleProvider(profile, environ=environ)
    for bad in (" padded ", "line\nbreak", "tab\tinside"):
        with pytest.raises(ProviderConfigError) as caught:
            OpenAICompatibleProvider(profile, environ={"K": bad})
        assert bad.strip() not in str(caught.value) or bad.strip() == "K"


def test_proxy_variables_and_redirects_from_the_environment_are_ignored(server, make, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")  # nothing listens there
    server.enqueue(Reply(200, completion()))
    assert isinstance(make().generate(make_request()), GenerationSuccess)


def test_the_default_profile_lists_the_conservative_parameters():
    assert DEFAULT_SUPPORTED_PARAMS == {
        "temperature",
        "top_p",
        "max_tokens",
        "stop",
        "response_format",
    }
    assert "seed" not in DEFAULT_SUPPORTED_PARAMS


# -- the contract and the retry policy ------------------------------------------------------------


def test_the_adapter_satisfies_the_provider_contract_and_the_runner(server, make):
    server.enqueue(Reply(200, completion("ok")))
    provider = make()
    assert isinstance(provider, Provider) and provider.name == "local"
    record = run_one(provider, make_request())  # the runner's contract check passes
    assert isinstance(record.result, GenerationSuccess) and record.attempts is None
    assert len(server.requests) == 1


def test_the_http_retry_policy_is_explicit_and_separate_from_the_core_default():
    assert NO_RETRY.max_attempts == 1 and HTTP_RETRY_POLICY.max_attempts == 3
    assert HTTP_RETRY_POLICY.retry_on == {
        FailureKind.TIMEOUT,
        FailureKind.RATE_LIMIT,
        FailureKind.SERVER_ERROR,
        FailureKind.CONNECTION_ERROR,
    }


def test_close_is_idempotent_and_the_provider_is_a_context_manager(server):
    provider = OpenAICompatibleProvider(OpenAICompatibleProfile(name="p", base_url=server.base_url))
    with provider as same:
        assert same is provider
    provider.close()


# -- safety nets ----------------------------------------------------------------------------------


def test_the_loopback_exception_does_not_open_the_network(server):
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.getaddrinfo("example.com", 443)
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection(("localhost.example.com", 80), timeout=0.1)


def test_without_the_fixture_even_loopback_is_blocked():
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)


def test_the_adapter_imports_core_and_http_libraries_but_never_scorers_or_the_core_imports_it():
    tree = ast.parse(Path(adapter_module.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not {m for m in imported if m.startswith(("niriksha.scorers", "niriksha.providers"))}
    assert "httpx" in imported
    core = Path(adapter_module.__file__).parents[1] / "core"
    for path in core.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "openai_compatible" not in text.replace("docs/adr/0009-openai-compatible", ""), (
            path.name
        )
        assert "import httpx" not in text and "from httpx" not in text, path.name


def test_the_secret_free_json_of_a_result_round_trips(server, make):
    server.enqueue(Reply(200, completion("ok", usage={"prompt_tokens": 1})))
    result = make().generate(make_request())
    assert json.loads(result.model_dump_json())["provider_metadata"]["api_style"] == (
        "openai_chat_completions"
    )
