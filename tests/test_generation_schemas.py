import math

import pytest
from pydantic import ValidationError

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationParams,
    GenerationRequest,
    GenerationSuccess,
    Message,
    Usage,
    result_adapter,
)

USER = Message(role="user", content="hi")


def req(**overrides):
    fields = {"request_id": "r1", "model": "m", "messages": (USER,)}
    return GenerationRequest(**{**fields, **overrides})


def ok(**overrides):
    fields = {"request_id": "r1", "provider": "fake", "requested_model": "m", "output_text": "x"}
    return GenerationSuccess(**{**fields, **overrides})


def fail(**overrides):
    fields = {
        "request_id": "r1",
        "provider": "fake",
        "requested_model": "m",
        "kind": FailureKind.TIMEOUT,
        "message": "timed out",
    }
    return GenerationFailure(**{**fields, **overrides})


def test_minimal_valid_objects():
    assert req().params == GenerationParams()
    assert ok().status == "ok" and ok().usage is None
    assert fail().status == "failed"


def test_empty_output_text_is_valid():
    assert ok(output_text="").output_text == ""


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": ""},
        {"model": "   "},
        {"request_id": ""},
        {"messages": ()},
        {"messages": (Message(role="system", content="s"),)},
        {"params": {"temperature": -0.1}},
        {"params": {"temperature": math.nan}},
        {"params": {"temperature": math.inf}},
        {"params": {"temperature": "0.5"}},
        {"params": {"top_p": 0}},
        {"params": {"top_p": 1.5}},
        {"params": {"max_tokens": 0}},
        {"params": {"seed": True}},
        {"params": {"stop": ("",)}},
        {"params": {"response_format": "xml"}},
        {"params": {"unknown_param": 1}},
        {"unknown_field": 1},
    ],
)
def test_invalid_requests_raise(overrides):
    with pytest.raises(ValidationError):
        req(**overrides)


@pytest.mark.parametrize(
    "kwargs", [{"role": "tool", "content": "x"}, {"role": "user", "content": ""}]
)
def test_invalid_messages_raise(kwargs):
    with pytest.raises(ValidationError):
        Message(**kwargs)


def test_messages_are_not_stripped():
    assert Message(role="user", content="  padded \n").content == "  padded \n"


@pytest.mark.parametrize(
    "build",
    [
        lambda: ok(usage=Usage(prompt_tokens=-1, source="reported")),
        lambda: Usage(source="guess"),
        lambda: Usage(),  # source is required: usage must say where numbers came from
        lambda: ok(provider=""),
        lambda: ok(returned_model=" "),
        lambda: ok(unknown_field=1),
        lambda: fail(http_status=99),
        lambda: fail(http_status=600),
        lambda: fail(message="  "),
        lambda: fail(kind="timeout"),  # strict: pass the enum, not a bare string
        lambda: ok(provider_metadata={"authorization": "x"}),
        lambda: ok(provider_metadata={"X_API_KEY": "x"}),
        lambda: ok(provider_metadata={"Set-Cookie": "x"}),
        lambda: ok(provider_metadata={"k": ["not", "flat"]}),
        lambda: ok(provider_metadata={"k": math.nan}),
    ],
)
def test_invalid_results_raise(build):
    with pytest.raises(ValidationError):
        build()


def test_models_are_frozen():
    with pytest.raises(ValidationError):
        req().model = "other"


def test_usage_none_fields_stay_none():
    usage = Usage(source="reported")
    assert usage.prompt_tokens is None and usage.total_tokens is None


def test_total_tokens_need_not_equal_sum():
    Usage(prompt_tokens=1, completion_tokens=1, total_tokens=5, source="reported")


@pytest.mark.parametrize(
    "result", [ok(), ok(usage=Usage(source="reported"), returned_model="m-2"), fail()]
)
def test_result_json_round_trip(result):
    parsed = result_adapter.validate_json(result.model_dump_json())
    assert parsed == result
    assert type(parsed) is type(result)


def test_success_round_trip_keeps_unknowns_as_none():
    parsed = result_adapter.validate_json(ok().model_dump_json())
    assert parsed.usage is None and parsed.returned_model is None and parsed.finish_reason is None


def test_request_round_trip_keeps_unset_params_none():
    request = req(params=GenerationParams(temperature=0.0, stop=("END",)))
    parsed = GenerationRequest.model_validate_json(request.model_dump_json())
    assert parsed == request
    assert parsed.params.top_p is None and parsed.params.max_tokens is None


@pytest.mark.parametrize(
    "payload",
    [
        '{"request_id":"r","provider":"p","requested_model":"m","output_text":"x"}',  # no status
        '{"status":"maybe","request_id":"r","provider":"p","requested_model":"m"}',
        '{"status":"ok","request_id":"r","provider":"p","requested_model":"m",'
        '"output_text":"x","kind":"timeout"}',  # failure field on a success
    ],
)
def test_bad_result_payloads_rejected(payload):
    with pytest.raises(ValidationError):
        result_adapter.validate_json(payload)


def test_failure_kinds_are_exactly_the_approved_seven():
    assert {k.value for k in FailureKind} == {
        "timeout",
        "rate_limit",
        "server_error",
        "malformed_response",
        "invalid_request",
        "unsupported_capability",
        "internal_error",
    }


@pytest.mark.parametrize("kind", list(FailureKind))
def test_every_failure_kind_constructs_and_round_trips(kind):
    failure = fail(kind=kind)
    assert result_adapter.validate_json(failure.model_dump_json()) == failure


@pytest.mark.parametrize(
    ("kind", "retryable"),
    [
        (FailureKind.TIMEOUT, True),
        (FailureKind.RATE_LIMIT, True),
        (FailureKind.SERVER_ERROR, True),
        (FailureKind.MALFORMED_RESPONSE, False),
        (FailureKind.INVALID_REQUEST, False),
        (FailureKind.UNSUPPORTED_CAPABILITY, False),
        (FailureKind.INTERNAL_ERROR, False),
    ],
)
def test_retryable_hint(kind, retryable):
    assert fail(kind=kind).retryable is retryable


def test_schemas_have_no_cost_or_timing_fields():
    for model in (GenerationSuccess, GenerationFailure, Usage):
        names = set(model.model_fields)
        assert not {n for n in names if "cost" in n or "latency" in n or n == "started_at"}
