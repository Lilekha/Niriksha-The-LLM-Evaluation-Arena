"""Helpers shared by the scorers: text normalisation, strict JSON parsing, not-scored records.

Scorers are pure and deterministic. They read stored outputs and dataset values only, never call a
provider, and know nothing about which provider produced an output.
"""

import json
import unicodedata
from dataclasses import dataclass
from typing import Any

from niriksha.core.generation import GenerationFailure, GenerationResult
from niriksha.core.scoring import ScoreRecord, ScoreStatus


def normalize_text(text: str, *, casefold: bool) -> str:
    """Unicode NFC, optional ``str.casefold``, NFC again, then collapse whitespace and strip.

    - NFC (never NFKC) makes canonically equivalent sequences compare equal, which matters for
      Devanagari and Kannada. NFC alone does not fold compatibility characters (fullwidth
      letters, ligatures).
    - ``casefold`` is Python's locale-independent full case folding. Scripts without case
      (Devanagari, Kannada) are unchanged. Some Latin characters change: German sharp s becomes
      ss, and the fi ligature becomes fi. Fullwidth letters are lowercased but stay fullwidth.
    - Whitespace is whatever ``str.split`` treats as whitespace (including no-break space). Runs
      collapse to one space and the ends are stripped. Zero-width joiner and non-joiner are not
      whitespace and are kept, because they change the meaning of Indic text.
    - Punctuation, digits and everything else are left exactly as they are.
    """
    text = unicodedata.normalize("NFC", text)
    if casefold:
        text = unicodedata.normalize("NFC", text.casefold())
    return " ".join(text.split())


def not_scored_for_failure(
    request_id: str, metric: str, version: str, failure: GenerationFailure
) -> ScoreRecord:
    """The record for a request whose generation failed: no value, a stated reason."""
    return ScoreRecord(
        request_id=request_id,
        metric=metric,
        metric_version=version,
        status=ScoreStatus.NOT_SCORED,
        value=None,
        reason=f"generation failed: {failure.kind.value}",
        details={"failure_kind": failure.kind.value},
    )


def failure_of(result: GenerationResult) -> GenerationFailure | None:
    return result if isinstance(result, GenerationFailure) else None


# -- strict JSON parsing ---------------------------------------------------------------------------


class _DuplicateKey(ValueError):
    pass


class _NonFinite(ValueError):
    pass


class _OutOfRange(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise _NonFinite(name)


def _finite_float(text: str) -> float:
    """A float literal as a finite double, or ``_OutOfRange``.

    Overflow (``1e999``) would become infinity and underflow (``1e-400``, a non-zero literal)
    would silently become 0.0, which would then compare equal to a real zero. Both are rejected
    rather than converted. A literal whose mantissa is all zeros (``0.0``, ``-0.0``, ``0e-400``)
    is a real zero and is accepted.
    """
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise _OutOfRange(text)
    if value == 0.0 and any(ch in "123456789" for ch in text.lower().split("e")[0]):
        raise _OutOfRange(text)
    return value


@dataclass(frozen=True)
class ParseOutcome:
    ok: bool
    value: Any = None
    failure: str | None = None  # None when ok
    top_level_type: str | None = None  # None when not ok


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    return "number"


def parse_json_strict(text: str) -> ParseOutcome:
    """Parse ``text`` exactly as it was stored, with these rules (see docs/metrics/):

    - Valid means Python's ``json.loads`` accepts the whole string: surrounding JSON whitespace is
      fine; prose, markdown fences, trailing commas, single quotes and a leading byte-order mark are
      not. Nothing is stripped, extracted or repaired.
    - Empty or whitespace-only output is invalid (``empty_output``).
    - ``NaN``, ``Infinity`` and ``-Infinity`` are invalid (``non_finite_number``). A number literal
      that overflows to infinity (``1e999``) or underflows to zero (``1e-400``) is invalid too
      (``number_out_of_range``): it would otherwise be silently replaced by a different number.
    - An object with a repeated key is invalid (``duplicate_key``).
    - Over-deep nesting and integers beyond Python's digit limit are invalid
      (``too_deeply_nested`` and ``invalid_json``).
    - Any JSON value is acceptable at the top level; callers decide whether they need an object.
    """
    if not text.strip():
        return ParseOutcome(ok=False, failure="empty_output")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except _DuplicateKey:
        return ParseOutcome(ok=False, failure="duplicate_key")
    except _NonFinite:
        return ParseOutcome(ok=False, failure="non_finite_number")
    except _OutOfRange:
        return ParseOutcome(ok=False, failure="number_out_of_range")
    except RecursionError:
        return ParseOutcome(ok=False, failure="too_deeply_nested")
    except ValueError:  # includes json.JSONDecodeError and the integer digit limit
        return ParseOutcome(ok=False, failure="invalid_json")
    return ParseOutcome(ok=True, value=value, top_level_type=_type_name(value))
