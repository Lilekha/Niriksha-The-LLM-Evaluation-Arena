"""JSON Schema validity. Definition: docs/metrics/json_schema_validity.md.

Checks the stored output against the dataset's own ``output_schema`` (the only schema; it is
covered by the dataset hash). Draft 2020-12 only, no ``format`` assertion, local references only,
no network. Nothing is repaired, coerced or substituted.
"""

from collections.abc import Iterator
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing.exceptions import Unresolvable

from niriksha.core.dataset import ExtractionCase
from niriksha.core.generation import GenerationResult, GenerationSuccess
from niriksha.core.runload import LoadedRun
from niriksha.core.scorestore import ScoreArtifactError
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.scorers._common import failure_of, not_scored_for_failure, parse_json_strict

METRIC = "json_schema_validity"
VERSION = "0.1.0"
TASK = "json_extraction"
DIRECTION = "higher_is_better"  # 1.0 is the desired outcome; see the metric document
DIALECT = "https://json-schema.org/draft/2020-12/schema"

_REFS = ("$ref", "$dynamicRef")


def _walk(node: Any) -> Iterator[tuple[str, Any]]:
    """Every (key, value) pair in every object of the schema, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield key, value
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def build_validator(schema: Any) -> Draft202012Validator:
    """A validator for ``schema``, or ``ScoreArtifactError`` if the schema cannot be used offline.

    The checks are textual and conservative: a key named ``$id`` or ``$ref`` is refused wherever it
    appears, even as a property name. That keeps the guarantee simple: nothing can be fetched.
    ``check_schema`` also rejects an invalid ``pattern`` (it asserts ``format: regex``).
    Recursion that needs no instance depth to overflow is caught by the probes at the end; one
    that only a particular output shape reaches cannot be told from an over-deep output.
    """
    if not isinstance(schema, dict):
        raise ScoreArtifactError("the dataset has no usable output_schema (not a JSON object)")
    if schema.get("$schema", DIALECT) != DIALECT:
        raise ScoreArtifactError(f"unsupported $schema {schema['$schema']!r}; only {DIALECT} is")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ScoreArtifactError(
            f"output_schema is not a valid 2020-12 schema: {exc.message}"
        ) from exc
    for key, value in _walk(schema):
        if key == "$id":
            raise ScoreArtifactError(
                "output_schema uses $id; only a self-contained schema is allowed"
            )
        if key in _REFS and not (isinstance(value, str) and value.startswith("#")):
            raise ScoreArtifactError(f"output_schema has a non-local {key} ({value!r})")
    validator = Draft202012Validator(schema)
    # Probe with instances that have nothing to descend into. A schema that overflows the stack or
    # cannot resolve a reference on one of these is defective whatever the model outputs, so it is
    # told apart from an over-deep output (which has to be deep to overflow) and raised here.
    for probe in (None, True, 0, "", [], {}):
        try:
            list(validator.iter_errors(probe))
        except RecursionError as exc:
            raise ScoreArtifactError("output_schema recurses without consuming the output") from exc
        except Unresolvable as exc:
            raise ScoreArtifactError(f"output_schema has an unresolvable reference: {exc}") from exc
    return validator


def prepare(run: LoadedRun) -> Draft202012Validator:
    """Called once per run by the scoring workflow, before any case is scored."""
    return build_validator(run.dataset.meta.output_schema)


def _details(
    failure: str | None,
    count: int | None = None,
    keyword: str | None = None,
    path: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_valid": failure is None,
        "failure": failure,
        "error_count": count,
        "first_error_keyword": keyword,
        "first_error_schema_path": path,
    }


def score_case(
    case: ExtractionCase, request_id: str, result: GenerationResult, validator: Draft202012Validator
) -> ScoreRecord:
    if not isinstance(case, ExtractionCase):
        raise TypeError(f"{METRIC} scores extraction cases, got {type(case).__name__}")
    failure = failure_of(result)
    if failure is not None:
        return not_scored_for_failure(request_id, METRIC, VERSION, failure)
    assert isinstance(result, GenerationSuccess)
    outcome = parse_json_strict(result.output_text)
    value, details = 0.0, _details(outcome.failure)
    if outcome.ok:
        try:
            # Sorted by schema location, never by library iteration order. Only schema-derived
            # text goes into details: never the output and never a path into the output.
            errors = sorted(
                validator.iter_errors(outcome.value),
                key=lambda e: (tuple(map(str, e.absolute_schema_path)), e.validator),
            )
        except Unresolvable as exc:
            raise ScoreArtifactError(f"output_schema has an unresolvable reference: {exc}") from exc
        except RecursionError:
            details = _details("too_deeply_nested")
        else:
            if errors:
                path = "/".join(map(str, errors[0].absolute_schema_path))
                details = _details("schema_violation", len(errors), errors[0].validator, path)
            else:
                value, details = 1.0, _details(None, 0)
    return ScoreRecord(
        request_id=request_id,
        metric=METRIC,
        metric_version=VERSION,
        status=ScoreStatus.SCORED,
        value=value,
        details=details,
    )
