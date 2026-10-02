"""The three scorers as pure functions: expected cases, near-misses, malformed output, Unicode."""

import sys
import unicodedata

import pytest

from ds_helpers import FIXTURES
from niriksha.core.dataset import ExtractionCase, QACase, load_dataset
from niriksha.core.generation import FailureKind, GenerationFailure, GenerationSuccess
from niriksha.core.scoring import ScoreStatus
from niriksha.scorers import exact_match, field_match, json_parse
from niriksha.scorers._common import normalize_text, parse_json_strict

DEVANAGARI_NUKTA_PRECOMPOSED = "\u0958"  # NFC form is the two-code-point sequence below
DEVANAGARI_NUKTA_DECOMPOSED = "\u0915\u093c"
KANNADA_O_DECOMPOSED = "\u0c95\u0cc6\u0cc2"  # NFC composes the vowel sign into U+0CCA
KANNADA_O_COMPOSED = "\u0c95\u0cca"
KSSA = "\u0915\u094d\u0937"
KSSA_WITH_ZWJ = "\u0915\u094d\u200d\u0937"


def qa_case(*answers):
    return QACase(
        id="q1",
        split="dev",
        language="en",
        script="Latn",
        origin="original",
        license="MIT",
        question="Q?",
        answers=tuple(answers),
    )


def extraction_case(expected):
    return ExtractionCase(
        id="e1",
        split="dev",
        language="en",
        script="Latn",
        origin="original",
        license="MIT",
        text="T",
        expected=expected,
    )


def ok(text, request_id="q1"):
    return GenerationSuccess(
        request_id=request_id, provider="fake", requested_model="m", output_text=text
    )


def failed(kind, request_id="q1"):
    return GenerationFailure(
        request_id=request_id,
        provider="fake",
        requested_model="m",
        kind=kind,
        message="scripted failure",
    )


def em(answers, text):
    return exact_match.score_case(qa_case(*answers), "q1", ok(text))


def fm(expected, text):
    return field_match.score_case(extraction_case(expected), "e1", ok(text, "e1"))


def jv(text):
    return json_parse.score_case(extraction_case({"a": 1}), "e1", ok(text, "e1"))


def test_the_unicode_constants_are_what_the_tests_rely_on():
    """Guards the tests themselves: if a tool or editor ever normalised these literals, the
    Unicode cases below would silently stop testing anything."""
    nfc = unicodedata.normalize
    assert DEVANAGARI_NUKTA_PRECOMPOSED != DEVANAGARI_NUKTA_DECOMPOSED
    assert nfc("NFC", DEVANAGARI_NUKTA_PRECOMPOSED) == DEVANAGARI_NUKTA_DECOMPOSED
    assert KANNADA_O_DECOMPOSED != KANNADA_O_COMPOSED
    assert nfc("NFC", KANNADA_O_DECOMPOSED) == KANNADA_O_COMPOSED
    assert chr(0x200D) in KSSA_WITH_ZWJ and chr(0x200D) not in KSSA


# -- identity of the metrics ----------------------------------------------------------------------


def test_metric_names_versions_and_tasks_are_stable():
    assert (exact_match.METRIC, exact_match.VERSION, exact_match.TASK) == (
        "normalized_exact_match",
        "0.1.0",
        "short_answer_qa",
    )
    assert (json_parse.METRIC, json_parse.VERSION, json_parse.TASK) == (
        "json_parse_validity",
        "0.1.0",
        "json_extraction",
    )
    assert (field_match.METRIC, field_match.VERSION, field_match.TASK) == (
        "field_exact_match",
        "0.1.0",
        "json_extraction",
    )


@pytest.mark.parametrize("kind", list(FailureKind))
@pytest.mark.parametrize(
    "score",
    [
        lambda f: exact_match.score_case(qa_case("a"), "q1", f),
        lambda f: json_parse.score_case(extraction_case({"a": 1}), "q1", f),
        lambda f: field_match.score_case(extraction_case({"a": 1}), "q1", f),
    ],
    ids=["exact_match", "json_parse", "field_match"],
)
def test_a_failed_generation_is_not_scored_and_never_a_zero(score, kind):
    record = score(failed(kind))
    assert record.status is ScoreStatus.NOT_SCORED and record.value is None
    assert record.reason == f"generation failed: {kind.value}"
    assert record.details == {"failure_kind": kind.value}


def test_scorers_refuse_a_case_of_the_wrong_task():
    with pytest.raises(TypeError):
        exact_match.score_case(extraction_case({"a": 1}), "q1", ok("x"))
    with pytest.raises(TypeError):
        json_parse.score_case(qa_case("a"), "q1", ok("x"))
    with pytest.raises(TypeError):
        field_match.score_case(qa_case("a"), "q1", ok("x"))


# -- text normalisation ---------------------------------------------------------------------------


def test_normalize_text_is_nfc_not_nfkc_and_keeps_punctuation_and_joiners():
    assert normalize_text("  A \t B\n\nC  ", casefold=False) == "A B C"
    assert normalize_text("A\u00a0B", casefold=False) == "A B"  # no-break space is whitespace
    assert normalize_text("Paris.", casefold=True) == "paris."
    assert normalize_text("\uff30", casefold=False) == "\uff30"  # fullwidth letter is not folded
    assert normalize_text("\ufb01", casefold=False) == "\ufb01"  # NFC does not fold it
    assert normalize_text("\ufb01", casefold=True) == "fi"  # full case folding does
    assert normalize_text(KSSA_WITH_ZWJ, casefold=True) == KSSA_WITH_ZWJ  # joiner kept
    assert normalize_text("Stra\u00dfe", casefold=True) == "strasse"
    once = normalize_text("  \u0958 X\u0301  ", casefold=True)
    assert normalize_text(once, casefold=True) == once  # idempotent


# -- normalized exact match -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answers", "text", "index"),
    [
        (["Paris"], "Paris", 0),
        (["Paris"], "  paris  \n", 0),
        (["Paris"], "PARIS", 0),
        (["New Delhi"], "new   delhi", 0),
        (["New Delhi"], "New\tDelhi", 0),
        (["New Delhi", "Delhi"], "DELHI", 1),  # any accepted answer counts; the first match wins
        (["Stra\u00dfe"], "STRASSE", 0),  # full case folding, documented
        (["fish"], "\ufb01sh", 0),  # casefold also folds the fi ligature, documented
        ([DEVANAGARI_NUKTA_DECOMPOSED], DEVANAGARI_NUKTA_PRECOMPOSED, 0),  # canonical equivalence
        ([DEVANAGARI_NUKTA_PRECOMPOSED], DEVANAGARI_NUKTA_DECOMPOSED, 0),
        ([KANNADA_O_COMPOSED], KANNADA_O_DECOMPOSED, 0),
        ([KANNADA_O_DECOMPOSED], KANNADA_O_COMPOSED, 0),
    ],
)
def test_exact_match_accepts_equal_answers_after_the_documented_normalisation(answers, text, index):
    record = em(answers, text)
    assert record.status is ScoreStatus.SCORED and record.value == 1.0
    assert record.details == {"matched_answer_index": index, "empty_output": False}


@pytest.mark.parametrize(
    ("answers", "text"),
    [
        (["Paris"], "Paris."),  # punctuation is never silently removed
        (["Paris"], "The capital is Paris"),
        (["Paris"], "Pariss"),
        (["Paris"], "Pari"),
        (["Paris"], "\u00c9paris"),
        (["e"], "\u00e9"),  # accents matter
        (["366"], "366 days"),
        (["366"], "\uff13\uff16\uff16"),  # fullwidth digits are different characters
        (["Paris"], "\uff30\uff21\uff32\uff29\uff33"),  # no NFKC folding
        (["Paris"], "Pa\u200bris"),  # a zero-width space is not whitespace and is not removed
        ([KSSA], KSSA_WITH_ZWJ),  # a joiner changes the text and is never stripped
        ([KSSA_WITH_ZWJ], KSSA),
        (["Paris"], "Paris\u0301"),  # an extra combining mark
    ],
)
def test_exact_match_rejects_near_misses(answers, text):
    record = em(answers, text)
    assert record.status is ScoreStatus.SCORED and record.value == 0.0
    assert record.details["matched_answer_index"] is None


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_output_is_scored_zero_and_flagged(text):
    record = em(["Paris"], text)
    assert record.value == 0.0 and record.details["empty_output"] is True


def test_exact_match_on_the_fixture_cases_covers_english_hindi_and_kannada():
    cases = {c.id: c for c in load_dataset(FIXTURES / "tiny_qa").cases}
    for case_id, variant in [
        ("qa-en-001", "  paris "),
        ("qa-hi-001", None),  # the stored first answer, then one with a no-break space inside
        ("qa-kn-001", None),
        ("qa-hi-002", "new delhi"),
    ]:
        case = cases[case_id]
        text = variant or case.answers[0]
        assert exact_match.score_case(case, case_id, ok(text, case_id)).value == 1.0
    hindi = cases["qa-hi-001"]
    spaced = hindi.answers[0].replace(" ", "\u00a0")
    assert exact_match.score_case(hindi, "qa-hi-001", ok(spaced, "qa-hi-001")).value == 1.0
    assert (
        exact_match.score_case(
            hindi, "qa-hi-001", ok("\u092e\u0941\u0902\u092c\u0908", "qa-hi-001")
        ).value
        == 0.0
    )
    assert (
        exact_match.score_case(cases["qa-kn-001"], "qa-kn-001", ok("Bengaluru", "qa-kn-001")).value
        == 0.0
    )


# -- JSON parse validity --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "top"),
    [
        ('{"a": 1}', "object"),
        ("[1, 2, 3]", "array"),
        ('"text"', "string"),
        ("42", "number"),
        ("-1.5e3", "number"),
        ("true", "boolean"),
        ("null", "null"),
        ('  \n{"a": 1}\t\r\n', "object"),  # JSON whitespace around the value is allowed
        ('{"name": "\\u0906\\u0936\\u093e"}', "object"),
        ('"\\ud800"', "string"),  # an escaped lone surrogate is syntactically valid
        ('{"unrelated": true}', "object"),  # parse validity is not correctness and not a schema
        ("0.0", "number"),
        ("-0.0", "number"),
        ("0e-400", "number"),  # an all-zero mantissa is a real zero, however small the exponent
        ("5e-324", "number"),  # the smallest positive double is representable
        ("1e308", "number"),
    ],
)
def test_valid_json_scores_one(text, top):
    record = jv(text)
    assert record.status is ScoreStatus.SCORED and record.value == 1.0
    assert record.details == {"top_level_type": top, "failure": None}


@pytest.mark.parametrize(
    ("text", "failure"),
    [
        ("", "empty_output"),
        ("   \n", "empty_output"),
        ('Sure! Here it is: {"a": 1}', "invalid_json"),  # prose is not stripped or extracted
        ('```json\n{"a": 1}\n```', "invalid_json"),  # a markdown fence is not repaired
        ('{"a": 1} and some prose', "invalid_json"),
        ('{"a": 1,}', "invalid_json"),  # trailing comma
        ("{'a': 1}", "invalid_json"),  # single quotes
        ('{"a": 1', "invalid_json"),  # truncated
        ("{a: 1}", "invalid_json"),
        ('{"a": 1}{"b": 2}', "invalid_json"),  # two documents
        ('\ufeff{"a": 1}', "invalid_json"),  # a leading byte-order mark
        ('{"a": NaN}', "non_finite_number"),
        ('{"a": Infinity}', "non_finite_number"),
        ('{"a": -Infinity}', "non_finite_number"),
        ('{"a": 1e999}', "number_out_of_range"),  # overflows to infinity
        ('{"a": -1e999}', "number_out_of_range"),
        ('{"a": 1e-400}', "number_out_of_range"),  # underflows to 0.0: would equal a real zero
        ('{"a": -1.5E-400}', "number_out_of_range"),
        ("[1e-999]", "number_out_of_range"),
        ('{"a": 1, "a": 2}', "duplicate_key"),
        ('{"outer": {"k": 1, "k": 2}}', "duplicate_key"),
        pytest.param(
            "1" * 5000,
            "invalid_json",
            id="integer-beyond-digit-limit",
            marks=pytest.mark.skipif(
                not 0 < sys.get_int_max_str_digits() < 5000,
                reason="the interpreter's integer digit limit is disabled or above 5000",
            ),
        ),
        pytest.param(
            "[" * 200_000 + "]" * 200_000,
            "too_deeply_nested",
            id="nested-200000-deep",
            marks=pytest.mark.skipif(
                sys.getrecursionlimit() > 10_000,
                reason="the recursion limit was raised; 200000 levels might not be refused",
            ),
        ),
    ],
)
def test_invalid_json_scores_zero_with_a_named_failure(text, failure):
    record = jv(text)
    assert record.status is ScoreStatus.SCORED  # unparsable output is a score, not "not scored"
    assert record.value == 0.0
    assert record.details == {"top_level_type": None, "failure": failure}


def test_interpreter_failures_map_to_named_codes_whatever_the_interpreter_limits(monkeypatch):
    """The mapping is tested with injected errors, so it holds on any supported Python version
    and any recursion or integer-digit limit; the real-input cases above are skipped when those
    limits are not the defaults."""

    def too_deep(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    def too_many_digits(*args, **kwargs):
        raise ValueError("Exceeds the limit (4300 digits) for integer string conversion")

    monkeypatch.setattr("niriksha.scorers._common.json.loads", too_deep)
    assert parse_json_strict("[[[]]]").failure == "too_deeply_nested"
    monkeypatch.setattr("niriksha.scorers._common.json.loads", too_many_digits)
    assert parse_json_strict("1" * 10).failure == "invalid_json"


def test_parse_json_strict_returns_the_parsed_value_only_when_valid():
    assert parse_json_strict('{"a": [1, 2]}').value == {"a": [1, 2]}
    assert parse_json_strict("nope").value is None


# -- field-level exact match ----------------------------------------------------------------------


def details(**overrides):
    base = {
        "output_is_json_object": True,
        "expected_fields": 2,
        "matched_fields": 2,
        "missing_fields": 0,
        "mismatched_fields": 0,
        "extra_fields": 0,
        "exact_object_match": True,
    }
    return {**base, **overrides}


EXPECTED = {"name": "Asha", "age": 30}


def test_an_identical_object_scores_one_and_is_an_exact_object_match():
    record = fm(EXPECTED, '{"name": "Asha", "age": 30}')
    assert record.value == 1.0 and record.details == details()


def test_value_is_the_fraction_of_expected_fields_matched():
    both_wrong = fm(EXPECTED, '{"name": "Ravi", "age": 31}')
    assert both_wrong.value == 0.0
    assert both_wrong.details == details(
        matched_fields=0, mismatched_fields=2, exact_object_match=False
    )
    half = fm(EXPECTED, '{"name": "Asha", "age": 31}')
    assert half.value == 0.5 and half.details["matched_fields"] == 1
    missing = fm(EXPECTED, '{"name": "Asha"}')
    assert missing.value == 0.5
    assert missing.details == details(matched_fields=1, missing_fields=1, exact_object_match=False)
    third = fm({"a": 1, "b": 2, "c": 3}, '{"a": 1, "b": 0, "c": 0}')
    assert third.value == pytest.approx(1 / 3)


def test_extra_keys_do_not_lower_the_value_but_are_reported():
    record = fm(EXPECTED, '{"name": "Asha", "age": 30, "city": "Pune"}')
    assert record.value == 1.0
    assert record.details == details(extra_fields=1, exact_object_match=False)


@pytest.mark.parametrize(
    ("expected", "actual", "equal"),
    [
        (30, 30, True),
        (30, 30.0, True),  # JSON has one number type
        (30, "30", False),  # a number is never a string
        ("30", 30, False),
        (1, True, False),  # a boolean is never a number
        (True, 1, False),
        (0, False, False),
        (True, True, True),
        (True, False, False),
        (None, None, True),
        (None, 0, False),
        (None, "", False),
        (None, False, False),
        ("", None, False),
        ([1, 2], [1, 2], True),
        ([1, 2], [2, 1], False),  # array order matters
        ([1, 2], [1, 2, 3], False),
        ([], [], True),
        ({"a": {"b": [1, None]}}, {"a": {"b": [1, None]}}, True),
        ({"a": {"b": 1}}, {"a": {"b": 1, "c": 2}}, False),  # a nested object is compared in full
        ({"a": 1}, {"a": 1, "b": 1}, False),
        ({"a": 1}, [1], False),
        ({"a": 1}, "a", False),
        ("Asha", "asha", False),  # case matters for field values
        ("Asha", "Asha.", False),  # so does punctuation
        ("Asha", "  Asha \n", True),  # whitespace is collapsed and stripped
        ("New Delhi", "New   Delhi", True),
        (2**53 + 1, float(2**53), False),  # no precision-losing coercion
    ],
)
def test_values_are_compared_strictly_with_no_type_coercion(expected, actual, equal):
    assert field_match.values_equal(expected, actual) is equal


def test_unicode_strings_and_keys_compare_after_nfc_but_never_fold_or_strip_joiners():
    assert field_match.values_equal(DEVANAGARI_NUKTA_DECOMPOSED, DEVANAGARI_NUKTA_PRECOMPOSED)
    assert field_match.values_equal(KANNADA_O_COMPOSED, KANNADA_O_DECOMPOSED)
    assert not field_match.values_equal(KSSA, KSSA_WITH_ZWJ)
    record = fm({KANNADA_O_COMPOSED: "x"}, f'{{"{KANNADA_O_DECOMPOSED}": "x"}}')
    assert record.value == 1.0  # keys are matched after NFC too
    assert fm({"Name": 1}, '{"name": 1}').value == 0.0  # but key case matters


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        '["name", "Asha"]',  # valid JSON but not an object
        '"Asha"',
        "42",
        "null",
        'Sure: {"name": "Asha", "age": 30}',  # prose is not stripped
        '{"name": "Asha", "age": NaN}',
        '{"name": "Asha", "name": "Asha", "age": 30}',  # duplicate key
        '{"name": "Asha", "age": 30',
    ],
)
def test_output_that_is_not_a_usable_json_object_scores_zero(text):
    record = fm(EXPECTED, text)
    assert record.status is ScoreStatus.SCORED and record.value == 0.0
    assert record.details == details(
        output_is_json_object=False,
        matched_fields=0,
        missing_fields=2,
        exact_object_match=False,
    )


def test_output_keys_that_collide_after_nfc_are_unusable():
    text = f'{{"{KANNADA_O_COMPOSED}": 1, "{KANNADA_O_DECOMPOSED}": 2}}'
    record = fm({KANNADA_O_COMPOSED: 1}, text)
    assert record.value == 0.0 and record.details["output_is_json_object"] is False


def test_null_is_not_the_same_as_an_absent_key():
    expected = {"middle_name": None, "name": "Asha"}
    absent = fm(expected, '{"name": "Asha"}')
    assert absent.value == 0.5 and absent.details["missing_fields"] == 1
    present = fm(expected, '{"name": "Asha", "middle_name": null}')
    assert present.value == 1.0 and present.details["missing_fields"] == 0
    wrong = fm(expected, '{"name": "Asha", "middle_name": ""}')
    assert wrong.value == 0.5 and wrong.details["mismatched_fields"] == 1


@pytest.mark.parametrize("text", ['{"x": 1e999}', '{"x": 1e-400}', '{"x": NaN}', '{"x": Infinity}'])
def test_numbers_that_cannot_be_represented_never_earn_credit(text):
    # 1e-400 would silently become 0.0 and match an expected 0; the whole output is refused instead.
    record = fm({"x": 0}, text)
    assert record.value == 0.0
    assert (
        record.details["output_is_json_object"] is False and record.details["matched_fields"] == 0
    )


@pytest.mark.parametrize("text", ['{"x": 0}', '{"x": 0.0}', '{"x": -0.0}', '{"x": 0e-400}'])
def test_real_zeros_match_an_expected_zero(text):
    assert fm({"x": 0}, text).value == 1.0


def test_an_expected_object_with_no_fields():
    assert fm({}, "{}").value == 1.0
    non_empty = fm({}, '{"a": 1}')
    assert non_empty.value == 0.0 and non_empty.details["extra_fields"] == 1
    assert fm({}, "[]").value == 0.0


def test_field_match_on_the_extraction_fixture_covers_english_and_hindi():
    cases = {c.id: c for c in load_dataset(FIXTURES / "tiny_extraction").cases}
    english, hindi = cases["ex-en-001"], cases["ex-hi-001"]
    text = '{"name": "Asha", "age": 30}'
    assert field_match.score_case(english, "ex-en-001", ok(text, "ex-en-001")).value == 1.0
    hindi_text = f'{{"name": "{hindi.expected["name"]}", "age": 30}}'
    assert field_match.score_case(hindi, "ex-hi-001", ok(hindi_text, "ex-hi-001")).value == 1.0
    as_string = f'{{"name": "{hindi.expected["name"]}", "age": "30"}}'
    assert field_match.score_case(hindi, "ex-hi-001", ok(as_string, "ex-hi-001")).value == 0.5
