import math

import pytest
from pydantic import ValidationError

from niriksha.core.scoring import ScoreRecord, ScoreStatus


def scored(**overrides):
    fields = {
        "request_id": "r1",
        "metric": "normalized_exact_match",
        "metric_version": "0.1.0",
        "status": ScoreStatus.SCORED,
        "value": 1.0,
    }
    return ScoreRecord(**{**fields, **overrides})


def not_scored(**overrides):
    fields = {
        "request_id": "r1",
        "metric": "normalized_exact_match",
        "metric_version": "0.1.0",
        "status": ScoreStatus.NOT_SCORED,
        "reason": "generation failed: timeout",
    }
    return ScoreRecord(**{**fields, **overrides})


def test_the_two_statuses_are_exactly_scored_and_not_scored():
    assert {s.value for s in ScoreStatus} == {"scored", "not_scored"}


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_a_scored_record_carries_a_value_in_the_unit_interval(value):
    record = scored(value=value)
    assert record.value == value and record.reason is None and record.details == {}


def test_a_not_scored_record_has_no_value_and_a_reason():
    record = not_scored(details={"failure_kind": "timeout"})
    assert record.value is None and record.reason == "generation failed: timeout"


@pytest.mark.parametrize(
    "build",
    [
        lambda: scored(value=None),  # scored needs a number
        lambda: scored(value=-0.01),
        lambda: scored(value=1.01),
        lambda: scored(value=math.nan),
        lambda: scored(value=math.inf),
        lambda: not_scored(value=0.0),  # never invent a number for an unscorable result
        lambda: not_scored(reason=None),
        lambda: not_scored(reason="  "),
        lambda: scored(status="scored"),  # strict: pass the enum, not a bare string
        lambda: scored(metric="Bad Metric"),
        lambda: scored(metric=""),
        lambda: scored(metric_version="1.0"),
        lambda: scored(request_id=" "),
        lambda: scored(unknown=1),
        lambda: scored(details={"k": math.nan}),
        lambda: scored(details={"k": [1]}),
        lambda: scored(details={"k": {"nested": 1}}),
    ],
)
def test_invalid_records_are_rejected(build):
    with pytest.raises(ValidationError):
        build()


def test_records_are_frozen_and_round_trip_through_json():
    record = scored(details={"matched_answer_index": 1, "empty_output": False, "x": None})
    with pytest.raises(ValidationError):
        record.value = 0.0
    assert ScoreRecord.model_validate_json(record.model_dump_json()) == record
    failed = not_scored(details={"failure_kind": "timeout"})
    assert ScoreRecord.model_validate_json(failed.model_dump_json()) == failed
