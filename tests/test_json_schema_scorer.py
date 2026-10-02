"""json_schema_validity as a pure function: valid, invalid, malformed, Unicode, offline, schemas."""

import json
import socket

import pytest

from ds_helpers import FIXTURES
from niriksha.core.dataset import ExtractionCase, load_dataset
from niriksha.core.generation import FailureKind, GenerationFailure, GenerationSuccess
from niriksha.core.scorestore import ScoreArtifactError
from niriksha.core.scoring import ScoreStatus
from niriksha.scorers import json_schema
from niriksha.scorers.json_schema import DIALECT, build_validator, score_case

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "role": {"enum": ["admin", "user"]},
        "score": {"type": "number", "minimum": 0},
        "code": {"type": "string", "pattern": "^[A-Z]{2}$", "maxLength": 2},
    },
    "required": ["name", "age"],
}
CASE = ExtractionCase(
    id="e1",
    split="dev",
    language="en",
    script="Latn",
    origin="original",
    license="MIT",
    text="T",
    expected={"name": "Asha", "age": 30},
)


def ok(text):
    return GenerationSuccess(request_id="r", provider="fake", requested_model="m", output_text=text)


def score(output, schema=SCHEMA):
    result = output if isinstance(output, GenerationFailure) else ok(output)
    return score_case(CASE, "r", result, build_validator(schema))


def check(output, value, failure=None, keyword=None, schema=SCHEMA):
    record = score(output, schema)
    assert record.status is ScoreStatus.SCORED
    assert record.value == value
    assert record.details["failure"] == failure
    assert record.details["schema_valid"] is (value == 1.0)
    assert record.details["first_error_keyword"] == keyword
    assert record.metric_version == "0.1.0" and record.metric == "json_schema_validity"
    return record


# -- conforming and violating JSON ----------------------------------------------------------------


def test_a_conforming_object_scores_one():
    record = check('{"name": "Asha", "age": 30}', 1.0)
    assert record.details["error_count"] == 0


def test_extra_properties_are_allowed_unless_the_schema_forbids_them():
    check('{"name": "A", "age": 1, "extra": true}', 1.0)
    strict = {**SCHEMA, "additionalProperties": False}
    record = check(
        '{"name": "A", "age": 1, "extra": true}',
        0.0,
        "schema_violation",
        "additionalProperties",
        strict,
    )
    assert record.details["first_error_schema_path"] == "additionalProperties"


def test_a_missing_required_property_scores_zero():
    record = check('{"name": "Asha"}', 0.0, "schema_violation", "required")
    assert record.details["first_error_schema_path"] == "required"


def test_a_wrong_property_type_scores_zero_and_names_the_schema_location():
    record = check('{"name": "Asha", "age": "30"}', 0.0, "schema_violation", "type")
    assert record.details["first_error_schema_path"] == "properties/age/type"


@pytest.mark.parametrize(
    ("output", "keyword"),
    [
        ('{"name": "A", "age": 1, "role": "root"}', "enum"),
        ('{"name": "A", "age": 1, "score": -1}', "minimum"),
        ('{"name": "A", "age": 1, "code": "ABC"}', "maxLength"),
        ('{"name": "A", "age": 1, "code": "ab"}', "pattern"),
    ],
)
def test_enum_and_constraint_violations_score_zero(output, keyword):
    check(output, 0.0, "schema_violation", keyword)


@pytest.mark.parametrize("output", ["[]", '"text"', "42", "true", "null"])
def test_valid_json_of_the_wrong_top_level_type_scores_zero_not_unparsable(output):
    check(output, 0.0, "schema_violation", "type")


def test_several_errors_are_counted_and_the_first_is_chosen_by_schema_path():
    record = check('{"age": "x", "score": -1}', 0.0, "schema_violation", "type")
    assert record.details["error_count"] == 3  # age type, score minimum, required name
    assert record.details["first_error_schema_path"] == "properties/age/type"


# -- unparsable output stays a scored 0.0; failed generation stays not scored ---------------------


@pytest.mark.parametrize(
    ("output", "failure"),
    [
        ("", "empty_output"),
        ("   \n", "empty_output"),
        ('Sure! {"name": "A", "age": 1}', "invalid_json"),
        ('```json\n{"name": "A", "age": 1}\n```', "invalid_json"),
        ('{"name": "A", "age": 1,}', "invalid_json"),
        ('{"name": "A", "name": "B", "age": 1}', "duplicate_key"),
        ('{"name": "A", "age": NaN}', "non_finite_number"),
        ('{"name": "A", "age": 1, "score": 1e999}', "number_out_of_range"),
    ],
)
def test_unparsable_output_scores_zero_with_the_parse_failure_code(output, failure):
    record = check(output, 0.0, failure)
    assert record.details["error_count"] is None


def test_the_output_is_never_repaired_or_coerced():
    check('{"name": "A", "age": "30"}', 0.0, "schema_violation", "type")  # no str -> int
    check('{"name": "A", "age": 1} trailing', 0.0, "invalid_json")  # nothing extracted


@pytest.mark.parametrize("kind", list(FailureKind))
def test_a_failed_generation_is_not_scored(kind):
    failure = GenerationFailure(
        request_id="r", provider="fake", requested_model="m", kind=kind, message="x"
    )
    record = score(failure)
    assert record.status is ScoreStatus.NOT_SCORED and record.value is None
    assert record.reason == f"generation failed: {kind.value}"


# -- Unicode, numbers -----------------------------------------------------------------------------


def test_hindi_and_kannada_strings_properties_and_patterns():
    schema = {
        "type": "object",
        "properties": {
            "नाम": {"type": "string", "pattern": "^[ऀ-ॿ ]+$"},
            "ಹೆಸರು": {"type": "string", "enum": ["ಆಶಾ", "ರಾಮ"]},
        },
        "required": ["नाम", "ಹೆಸರು"],
    }
    check(json.dumps({"नाम": "आशा कुमारी", "ಹೆಸರು": "ಆಶಾ"}, ensure_ascii=False), 1.0, schema=schema)
    check(json.dumps({"नाम": "आशा", "ಹೆಸರು": "ಆಶಾ"}), 1.0, schema=schema)  # \u escapes
    check(
        json.dumps({"नाम": "Asha", "ಹೆಸರು": "ಆಶಾ"}, ensure_ascii=False),
        0.0,
        "schema_violation",
        "pattern",
        schema,
    )
    check(
        json.dumps({"नाम": "आशा", "ಹೆಸರು": "ಸೀತಾ"}, ensure_ascii=False),
        0.0,
        "schema_violation",
        "enum",
        schema,
    )


def test_max_length_counts_code_points_not_grapheme_clusters():
    schema = {"type": "string", "maxLength": 2}
    check('"ಕ್"', 1.0, schema=schema)  # Kannada ka + virama: 2 code points
    check('"ಕ್ಷ"', 0.0, "schema_violation", "maxLength", schema)  # one cluster, 3


def test_integers_booleans_and_floats():
    check('{"name": "A", "age": 30.0}', 1.0)  # 30.0 is an integer in 2020-12
    check('{"name": "A", "age": 30.5}', 0.0, "schema_violation", "type")
    check('{"name": "A", "age": true}', 0.0, "schema_violation", "type")  # bool is not an integer
    check('{"name": "A", "age": 1, "score": true}', 0.0, "schema_violation", "type")


def test_format_is_an_annotation_and_is_not_checked():
    schema = {"type": "string", "format": "email"}
    check('"definitely not an email"', 1.0, schema=schema)


# -- details never carry output text --------------------------------------------------------------


def test_details_contain_no_output_text_or_output_paths():
    marker = "LEAKED-TEXT-आशा"
    for output in (
        json.dumps({"name": marker, "age": marker, "extra-key": marker}, ensure_ascii=False),
        json.dumps({marker: 1}, ensure_ascii=False),
        json.dumps([marker], ensure_ascii=False),
    ):
        record = score(output)
        assert "LEAKED" not in json.dumps(record.details, ensure_ascii=False)


def test_schema_derived_path_through_a_local_reference():
    schema = {
        "$defs": {"person": {"type": "object", "required": ["name"]}},
        "type": "array",
        "items": {"$ref": "#/$defs/person"},
    }
    check('[{"name": "A"}]', 1.0, schema=schema)
    check("[{}]", 0.0, "schema_violation", "required", schema)


# -- the schema is checked before anything is scored ----------------------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "nonsense"},
        {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"},
        {"$schema": "https://json-schema.org/draft/2019-09/schema"},
        {"$ref": "https://example.com/schema.json"},
        {"$ref": "other.json"},
        {"properties": {"a": {"$dynamicRef": "https://example.com/x"}}},
        {"$id": "https://example.com/s", "type": "object"},
        {"properties": {"a": {"$id": "x"}}},
        {"pattern": "("},
        {"patternProperties": {"[": {}}},
    ],
)
def test_an_unusable_schema_raises_a_typed_error(schema):
    with pytest.raises(ScoreArtifactError):
        build_validator(schema)


def test_a_non_object_schema_raises_a_typed_error():
    for bad in (None, True, [], "x"):
        with pytest.raises(ScoreArtifactError):
            build_validator(bad)


def test_the_declared_dialect_is_accepted():
    build_validator({"$schema": DIALECT, "type": "object"})


def test_an_unresolvable_local_reference_is_a_schema_defect_not_a_zero():
    with pytest.raises(ScoreArtifactError, match="unresolvable"):
        build_validator({"$ref": "#/$defs/missing"})  # found by the probe instances
    reachable_by_data = {"properties": {"a": {"$ref": "#/$defs/missing"}}}
    build_validator(reachable_by_data)  # the probes do not reach it
    with pytest.raises(ScoreArtifactError, match="unresolvable"):
        score('{"a": 1}', reachable_by_data)  # found when a case reaches it


def test_a_remote_reference_cannot_reach_the_network(monkeypatch):
    calls = []

    def blocked(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network used")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    with pytest.raises(ScoreArtifactError):
        build_validator({"$ref": "https://example.com/schema.json"})
    assert not calls


@pytest.mark.parametrize(
    "schema",
    [
        {"$ref": "#"},
        {"$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"},
        {"allOf": [{"$ref": "#"}]},
    ],
)
def test_a_schema_that_recurses_without_consuming_output_is_a_typed_error(schema):
    with pytest.raises(ScoreArtifactError, match="recurses"):
        build_validator(schema)


def nested(depth):
    return '{"c": [' * depth + "1" + "]}" * depth


def test_a_legitimate_recursive_schema_works_and_only_a_deep_output_is_too_deep():
    tree = {"properties": {"c": {"items": {"$ref": "#"}}}}
    check(nested(20), 1.0, schema=tree)
    check(nested(400), 0.0, "too_deeply_nested", schema=tree)  # overflows only because it is deep
    check(nested(400), 1.0, schema={"type": "object"})  # the same output is fine without recursion


# -- the fixtures' gold data validates against the declared schema --------------------------------


def test_extraction_fixture_gold_outputs_validate_against_their_output_schema():
    dataset = load_dataset(FIXTURES / "tiny_extraction")
    validator = build_validator(dataset.meta.output_schema)
    assert dataset.cases
    for case in dataset.cases:
        assert not list(validator.iter_errors(case.expected)), case.id
        record = score_case(
            case, case.id, ok(json.dumps(case.expected, ensure_ascii=False)), validator
        )
        assert record.value == 1.0


def test_the_scorer_module_declares_its_identity():
    assert (json_schema.METRIC, json_schema.VERSION, json_schema.TASK) == (
        "json_schema_validity",
        "0.1.0",
        "json_extraction",
    )
