"""Field-level exact match for structured extraction. Definition: docs/metrics/field_exact_match.md.

Separate from JSON Schema validation (M2.2): it compares values, it does not check structure rules.
"""

import unicodedata
from typing import Any

from niriksha.core.dataset import ExtractionCase
from niriksha.core.generation import GenerationResult, GenerationSuccess
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.scorers._common import (
    failure_of,
    normalize_text,
    not_scored_for_failure,
    parse_json_strict,
)

METRIC = "field_exact_match"
VERSION = "0.1.0"
TASK = "json_extraction"


def _nfc(key: str) -> str:
    return unicodedata.normalize("NFC", key)


def values_equal(expected: Any, actual: Any) -> bool:
    """Strict, recursive JSON equality. No coercion between JSON types.

    - Booleans equal only booleans (``True`` is not ``1``). ``null`` equals only ``null``.
    - Numbers equal numbers by value (JSON has one number type, so ``30`` equals ``30.0``).
      A number never equals a string, so ``30`` and ``"30"`` differ.
    - Strings equal strings after ``normalize_text(casefold=False)``: Unicode NFC and whitespace
      collapsed and stripped; case and punctuation matter.
    - Arrays are equal if they have the same length and equal elements in the same order.
    - Objects must have the same key set (keys compared after NFC) with equal values: a nested
      object is compared in full, so an extra nested key makes it unequal.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected is actual
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, str) or isinstance(actual, str):
        return (
            isinstance(expected, str)
            and isinstance(actual, str)
            and normalize_text(expected, casefold=False) == normalize_text(actual, casefold=False)
        )
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return expected == actual
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            values_equal(e, a) for e, a in zip(expected, actual, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = {_nfc(k): v for k, v in expected.items()}
        actual_keys = {_nfc(k): v for k, v in actual.items()}
        if len(expected_keys) != len(expected) or len(actual_keys) != len(actual):
            return False  # keys that collide after NFC are ambiguous
        return expected_keys.keys() == actual_keys.keys() and all(
            values_equal(value, actual_keys[key]) for key, value in expected_keys.items()
        )
    return False


def _details(
    *, expected: int, matched: int, missing: int, mismatched: int, extra: int, is_object: bool
) -> dict[str, int | bool]:
    return {
        "output_is_json_object": is_object,
        "expected_fields": expected,
        "matched_fields": matched,
        "missing_fields": missing,
        "mismatched_fields": mismatched,
        "extra_fields": extra,
        "exact_object_match": is_object and matched == expected and extra == 0,
    }


def score_case(case: ExtractionCase, request_id: str, result: GenerationResult) -> ScoreRecord:
    if not isinstance(case, ExtractionCase):
        raise TypeError(f"{METRIC} scores extraction cases, got {type(case).__name__}")
    failure = failure_of(result)
    if failure is not None:
        return not_scored_for_failure(request_id, METRIC, VERSION, failure)
    assert isinstance(result, GenerationSuccess)

    expected = case.expected
    count = len(expected)
    outcome = parse_json_strict(result.output_text)
    output = outcome.value if outcome.ok and isinstance(outcome.value, dict) else None
    output_keys = None if output is None else {_nfc(k): v for k, v in output.items()}
    if output_keys is None or len(output_keys) != len(output):
        # Not valid JSON, not an object, or keys that collide after NFC: no field can be credited.
        return ScoreRecord(
            request_id=request_id,
            metric=METRIC,
            metric_version=VERSION,
            status=ScoreStatus.SCORED,
            value=0.0,
            details=_details(
                expected=count, matched=0, missing=count, mismatched=0, extra=0, is_object=False
            ),
        )

    matched = missing = mismatched = 0
    expected_keys = set()
    for key, expected_value in expected.items():
        normalized = _nfc(key)
        expected_keys.add(normalized)
        if normalized not in output_keys:
            missing += 1
        elif values_equal(expected_value, output_keys[normalized]):
            matched += 1
        else:
            mismatched += 1
    extra = len(set(output_keys) - expected_keys)
    # An expected object with no fields is matched exactly by an empty output object, else not.
    value = matched / count if count else (1.0 if not output_keys else 0.0)
    return ScoreRecord(
        request_id=request_id,
        metric=METRIC,
        metric_version=VERSION,
        status=ScoreStatus.SCORED,
        value=value,
        details=_details(
            expected=count,
            matched=matched,
            missing=missing,
            mismatched=mismatched,
            extra=extra,
            is_object=True,
        ),
    )
