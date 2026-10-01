import hashlib
import json

import pytest

import niriksha.core.dataset as dataset_module
from ds_helpers import EXTRACTION_SCHEMA, FIXTURES, ext, qa, write_dataset
from niriksha.core.dataset import (
    DatasetError,
    canonical_json_bytes,
    compute_content_sha256,
    load_dataset,
)

# Golden values. If these change, dataset identity changed: that is a deliberate, versioned act
# (bump HASH_VERSION, update fixtures and docs), never a silent side effect of a refactor.
GOLDEN_QA = "e6486282fd5d6089bea7a5f260c42e5f7aa15226a34f56986cfccc8fb8c782e4"
GOLDEN_EXTRACTION = "a5939377e14a4f65399cab44b03f91210a009519ac0f0cbe84b10465bd4fe5df"


def test_golden_hashes_of_the_fixtures():
    assert load_dataset(FIXTURES / "tiny_qa").content_sha256 == GOLDEN_QA
    assert load_dataset(FIXTURES / "tiny_extraction").content_sha256 == GOLDEN_EXTRACTION


def test_canonical_json_form_is_exactly_as_documented():
    value = {"b": 1, "a": [True, None, "é", {"z": 0, "y": 1}]}
    assert canonical_json_bytes(value) == '{"a":[true,null,"é",{"y":1,"z":0}],"b":1}'.encode()
    with pytest.raises(ValueError):
        canonical_json_bytes({"a": float("nan")})


def test_hash_payload_is_exactly_the_documented_document():
    cases = [qa(id="a"), qa(id="b")]
    expected = hashlib.sha256(
        json.dumps(
            {
                "cases": cases,
                "hash_version": 1,
                "output_schema": None,
                "schema_version": 1,
                "task": "short_answer_qa",
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    assert compute_content_sha256("short_answer_qa", None, cases) == expected


def test_hashing_is_repeatable():
    assert load_dataset(FIXTURES / "tiny_qa").content_sha256 == (
        load_dataset(FIXTURES / "tiny_qa").content_sha256
    )


def _reserialise(cases, *, reverse_keys=False, **dumps_kwargs):
    lines = []
    for case in cases:
        items = list(case.items())[::-1] if reverse_keys else list(case.items())
        lines.append(json.dumps(dict(items), **dumps_kwargs).encode("utf-8"))
    return lines


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("reversed key order", {"reverse_keys": True}),
        ("extra spaces", {"separators": (" , ", " : ")}),
        ("escaped non-ascii", {"ensure_ascii": True}),
        ("raw non-ascii", {"ensure_ascii": False}),
        ("CRLF line endings", {"eol": b"\r\n"}),
        ("no final newline", {"trailing_newline": False}),
    ],
)
def test_formatting_differences_do_not_change_the_hash(tmp_path, label, kwargs):
    cases = [qa(id="a", question="हिंदी प्रश्न?"), qa(id="b", answers=["é", "x"])]
    pin = compute_content_sha256("short_answer_qa", None, cases)
    options = dict(kwargs)
    eol = options.pop("eol", b"\n")
    trailing = options.pop("trailing_newline", True)
    lines = _reserialise(cases, **options)
    directory = write_dataset(
        tmp_path, lines, pin=pin, eol=eol, trailing_newline=trailing, dirname="variant"
    )
    assert load_dataset(directory).content_sha256 == pin, label


def _hash(cases, task="short_answer_qa", schema=None):
    return compute_content_sha256(task, schema, cases)


BASE = [qa(id="a"), qa(id="b"), qa(id="c")]


@pytest.mark.parametrize(
    ("label", "changed"),
    [
        ("changed value", [qa(id="a", question="Other?"), qa(id="b"), qa(id="c")]),
        ("changed id", [qa(id="a"), qa(id="b"), qa(id="d")]),
        ("changed split", [qa(id="a", split="test"), qa(id="b"), qa(id="c")]),
        ("changed license", [qa(id="a", license="CC0-1.0"), qa(id="b"), qa(id="c")]),
        ("reordered", [qa(id="b"), qa(id="a"), qa(id="c")]),
        ("removed case", [qa(id="a"), qa(id="b")]),
        ("added case", [*BASE, qa(id="d")]),
        ("type change", [qa(id="a", answers=["1"]), qa(id="b"), qa(id="c")]),
        ("explicit default", [qa(id="a", code_mixed=False), qa(id="b"), qa(id="c")]),
    ],
)
def test_content_changes_change_the_hash(label, changed):
    assert _hash(changed) != _hash(BASE), label


def test_json_types_are_part_of_identity():
    as_int = [ext(expected={"name": "A", "age": 30})]
    as_str = [ext(expected={"name": "A", "age": "30"})]
    as_float = [ext(expected={"name": "A", "age": 30.0})]  # 30 and 30.0 differ; avoid floats
    hashes = {_hash(c, "json_extraction", EXTRACTION_SCHEMA) for c in (as_int, as_str, as_float)}
    assert len(hashes) == 3


def test_task_and_output_schema_are_part_of_identity():
    cases = [ext()]
    base = _hash(cases, "json_extraction", EXTRACTION_SCHEMA)
    assert _hash(cases, "short_answer_qa", None) != base
    assert _hash(cases, "json_extraction", {**EXTRACTION_SCHEMA, "required": ["name"]}) != base


def test_hash_and_schema_versions_are_part_of_identity(monkeypatch):
    before = _hash(BASE)
    monkeypatch.setattr(dataset_module, "HASH_VERSION", 2)
    assert _hash(BASE) != before
    monkeypatch.undo()
    monkeypatch.setattr(dataset_module, "SCHEMA_VERSION", 2)
    assert _hash(BASE) != before


def test_descriptive_metadata_is_not_part_of_the_hash(tmp_path):
    pin = _hash(BASE)
    directory = write_dataset(
        tmp_path, BASE, pin=pin, name="renamed", version="9.9.9", description="different text"
    )
    assert load_dataset(directory).content_sha256 == pin


def test_pin_mismatch_is_rejected_and_reports_the_computed_hash(tmp_path):
    directory = write_dataset(tmp_path, [qa(id="a", question="Edited?")], pin=_hash([qa(id="a")]))
    with pytest.raises(DatasetError) as info:
        load_dataset(directory)
    computed = _hash([qa(id="a", question="Edited?")])
    issue = info.value.issues[0]
    assert issue.file == "ds/dataset.json" and issue.field == "content_sha256"
    assert computed in issue.message
    assert "bump 'version'" in issue.message


def test_reordering_a_pinned_dataset_is_detected(tmp_path):
    pin = _hash(BASE)
    directory = write_dataset(tmp_path, [BASE[1], BASE[0], BASE[2]], pin=pin)
    with pytest.raises(DatasetError, match="does not match"):
        load_dataset(directory)
