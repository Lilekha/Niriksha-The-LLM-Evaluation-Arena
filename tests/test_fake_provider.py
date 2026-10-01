import pytest

from niriksha.core.generation import (
    FailureKind,
    GenerationFailure,
    GenerationRequest,
    GenerationSuccess,
    Message,
    result_adapter,
)
from niriksha.core.provider import Provider
from niriksha.providers.fake import FakeProvider


def request(request_id="r1", model="m", content="hello"):
    return GenerationRequest(
        request_id=request_id, model=model, messages=(Message(role="user", content=content),)
    )


def test_satisfies_protocol_with_stable_name():
    provider = FakeProvider()
    assert isinstance(provider, Provider)
    assert provider.name == "fake"
    assert FakeProvider(name="other").name == "other"


def test_echo_mode_returns_last_user_message():
    multi = GenerationRequest(
        request_id="r1",
        model="m",
        messages=(
            Message(role="user", content="first"),
            Message(role="assistant", content="ignored"),
            Message(role="user", content="last"),
            Message(role="assistant", content="also ignored"),
        ),
    )
    assert FakeProvider().generate(multi).output_text == "last"


def test_scripted_success_has_exact_expected_fields():
    result = FakeProvider(["scripted text"]).generate(request("r9", "model-x"))
    assert result == GenerationSuccess(
        request_id="r9",
        provider="fake",
        requested_model="model-x",
        returned_model="fake-model-v0",
        output_text="scripted text",
        provider_metadata={"fake": True},
    )
    # nothing invented: no usage and no finish reason
    assert result.usage is None and result.finish_reason is None


def test_returned_model_is_configurable_and_differs_from_requested():
    result = FakeProvider(["x"], returned_model="fake-snapshot-2").generate(request(model="alias"))
    assert (result.requested_model, result.returned_model) == ("alias", "fake-snapshot-2")


@pytest.mark.parametrize("kind", list(FailureKind))
def test_scripted_failure_is_returned_not_raised(kind):
    result = FakeProvider([kind]).generate(request("r2", "m"))
    assert isinstance(result, GenerationFailure)
    assert (result.kind, result.request_id, result.requested_model) == (kind, "r2", "m")
    assert result.returned_model is None
    assert result_adapter.validate_json(result.model_dump_json()) == result


def test_plain_string_is_output_text_even_if_it_names_a_failure_kind():
    result = FakeProvider(["timeout"]).generate(request())
    assert isinstance(result, GenerationSuccess) and result.output_text == "timeout"


def test_script_is_consumed_in_order_then_exhaustion_is_loud():
    provider = FakeProvider(["a", FailureKind.RATE_LIMIT, "c"])
    kinds = [type(provider.generate(request(f"r{i}"))).__name__ for i in range(3)]
    assert kinds == ["GenerationSuccess", "GenerationFailure", "GenerationSuccess"]
    with pytest.raises(RuntimeError, match="exhausted"):
        provider.generate(request("r3"))


def test_empty_script_is_not_echo_mode():
    with pytest.raises(RuntimeError, match="exhausted"):
        FakeProvider([]).generate(request())


def test_deterministic_across_instances_and_repeats():
    script = ["a", FailureKind.TIMEOUT]
    first = [FakeProvider(script).generate(request("r1")) for _ in range(2)]
    second = FakeProvider(script)
    assert first[0] == first[1] == second.generate(request("r1"))


def test_calls_record_requests_in_order_without_mutating_them():
    provider = FakeProvider()
    sent = [request("r1", content="a"), request("r2", content="b")]
    snapshot = [r.model_dump() for r in sent]
    for r in sent:
        provider.generate(r)
    assert provider.calls == tuple(sent)
    assert [r.model_dump() for r in sent] == snapshot


def test_calls_view_is_a_copy():
    provider = FakeProvider()
    provider.generate(request())
    assert isinstance(provider.calls, tuple) and len(provider.calls) == 1


@pytest.mark.parametrize("bare", ["my answer", "ab", "", FailureKind.TIMEOUT])
def test_bare_string_script_is_rejected(bare):
    with pytest.raises(TypeError, match="bare string"):
        FakeProvider(bare)


def test_sequences_of_strings_and_failure_kinds_are_still_accepted():
    for script in (["a", FailureKind.TIMEOUT], ("a", "b"), []):
        FakeProvider(script)


@pytest.mark.parametrize("bad", [[1], [None], [b"bytes"], ["ok", 3.5]])
def test_invalid_script_items_rejected_at_construction(bad):
    with pytest.raises(TypeError):
        FakeProvider(bad)
