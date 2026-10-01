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

REQUEST = GenerationRequest(
    request_id="r1", model="m", messages=(Message(role="user", content="hi"),)
)


class EchoStub:
    """Test-only stub; real providers live in niriksha.providers."""

    name = "stub"

    def generate(self, request):
        return GenerationSuccess(
            request_id=request.request_id,
            provider=self.name,
            requested_model=request.model,
            output_text=request.messages[-1].content,
        )


class FailingStub:
    name = "failing"

    def __init__(self, kind):
        self.kind = kind

    def generate(self, request):
        return GenerationFailure(
            request_id=request.request_id,
            provider=self.name,
            requested_model=request.model,
            kind=self.kind,
            message="scripted failure",
        )


class NoGenerate:
    name = "broken"


def test_stub_satisfies_protocol():
    assert isinstance(EchoStub(), Provider)


def test_object_without_generate_does_not_satisfy_protocol():
    assert not isinstance(NoGenerate(), Provider)


def test_success_result_validates_and_matches_request():
    result = EchoStub().generate(REQUEST)
    assert result_adapter.validate_python(result) == result
    assert (result.request_id, result.requested_model) == ("r1", "m")


@pytest.mark.parametrize("kind", list(FailureKind))
def test_failure_results_validate_for_every_kind(kind):
    result = FailingStub(kind).generate(REQUEST)
    assert isinstance(result, GenerationFailure)
    assert result_adapter.validate_python(result) == result
