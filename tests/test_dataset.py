import json

import pytest

from ds_helpers import FIXTURES, MISSING, ext, qa, write_dataset
from niriksha.core.dataset import MAX_ISSUES, DatasetError, ExtractionCase, QACase, load_dataset


def issues_for(tmp_path, cases, **kwargs):
    with pytest.raises(DatasetError) as info:
        load_dataset(write_dataset(tmp_path, cases, **kwargs))
    return info.value


# -- valid datasets -------------------------------------------------------------------------------


def test_qa_fixture_loads_in_file_order():
    dataset = load_dataset(FIXTURES / "tiny_qa")
    assert [c.id for c in dataset.cases] == [
        "qa-en-001",
        "qa-en-002",
        "qa-hi-001",
        "qa-kn-001",
        "qa-hi-002",
    ]
    assert all(isinstance(c, QACase) for c in dataset.cases)
    assert dataset.meta.task == "short_answer_qa" and dataset.meta.output_schema is None
    assert {c.language for c in dataset.cases} == {"en", "hi", "kn"}
    assert dataset.cases[4].code_mixed and dataset.cases[4].script == "Latn"


def test_extraction_fixture_loads():
    dataset = load_dataset(FIXTURES / "tiny_extraction")
    assert all(isinstance(c, ExtractionCase) for c in dataset.cases)
    assert dataset.cases[0].expected == {"name": "Asha", "age": 30}
    assert dataset.meta.output_schema is not None


def test_select_filters_by_split_and_keeps_file_order():
    dataset = load_dataset(FIXTURES / "tiny_qa")
    assert [c.id for c in dataset.select(["test", "dev"])] == [
        "qa-en-001",
        "qa-en-002",
        "qa-hi-001",
        "qa-hi-002",
    ]
    assert dataset.select(["nonexistent"]) == ()


def test_input_text_is_the_question_or_the_source_text():
    assert (
        load_dataset(FIXTURES / "tiny_qa").cases[0].input_text == "What is the capital of France?"
    )
    assert load_dataset(FIXTURES / "tiny_extraction").cases[0].input_text == "Asha is 30 years old."


def test_a_missing_final_newline_is_fine(tmp_path):
    assert len(load_dataset(write_dataset(tmp_path, [qa()], trailing_newline=False)).cases) == 1


def test_crlf_and_lf_files_load_identically(tmp_path):
    cases = [qa(id="a"), qa(id="b")]
    lf = load_dataset(write_dataset(tmp_path, cases, dirname="lf"))
    crlf = load_dataset(write_dataset(tmp_path, cases, eol=b"\r\n", dirname="crlf"))
    assert lf.cases == crlf.cases and lf.content_sha256 == crlf.content_sha256


def test_unicode_line_separator_inside_a_string_does_not_split_the_record(tmp_path):
    text = "line one line two"
    dataset = load_dataset(write_dataset(tmp_path, [qa(question=text)]))
    assert len(dataset.cases) == 1 and dataset.cases[0].question == text


def test_loading_never_modifies_the_files(tmp_path):
    directory = write_dataset(tmp_path, [qa()], pin="f" * 64)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(DatasetError):
        load_dataset(directory)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


# -- invalid records ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "field"),
    [
        (qa(extra_field=1), "extra_field"),
        (qa(question=MISSING), "question"),
        (qa(answers=MISSING), "answers"),
        (qa(id="bad id!"), "id"),
        (qa(id=""), "id"),
        (qa(id="x" * 65), "id"),
        (qa(split="train"), "split"),
        (qa(language="fr"), "language"),
        (qa(script="Cyrl"), "script"),
        (qa(language="hi", script="Knda"), "(record)"),
        (qa(language="en", script="Deva"), "(record)"),
        (qa(language="kn", script="Deva"), "(record)"),
        (qa(origin="public"), "(record)"),
        (qa(origin="web"), "origin"),
        (qa(license=" "), "license"),
        (qa(license=MISSING), "license"),
        (qa(question="  "), "question"),
        (qa(answers=[]), "answers"),
        (qa(answers=["ok", " "]), "answers.1"),
        (qa(answers="Paris"), "answers"),
        (qa(code_mixed="yes"), "code_mixed"),
        (qa(code_mixed=1), "code_mixed"),
    ],
)
def test_invalid_qa_record_is_rejected_with_file_line_and_field(tmp_path, case, field):
    error = issues_for(tmp_path, [case])
    issue = error.issues[0]
    assert issue.file == "ds/cases.jsonl" and issue.line == 1
    assert issue.field == field


@pytest.mark.parametrize(
    ("case", "field"),
    [
        (ext(expected=[1, 2]), "expected"),
        (ext(expected="Asha"), "expected"),
        (ext(expected=MISSING), "expected"),
        (ext(text=MISSING), "text"),
        (ext(question="not an extraction field"), "question"),
    ],
)
def test_invalid_extraction_record_is_rejected(tmp_path, case, field):
    assert issues_for(tmp_path, [case], task="json_extraction").issues[0].field == field


def test_public_origin_with_a_source_is_accepted(tmp_path):
    case = qa(origin="public", source="example-corpus v1", license="CC-BY-4.0")
    assert len(load_dataset(write_dataset(tmp_path, [case])).cases) == 1


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b'{"id": "a"', "invalid JSON"),
        (b"not json", "invalid JSON"),
        (b"[1, 2]", "must be a JSON object"),
        (b'"string"', "must be a JSON object"),
        (b"\xff\xfe", "invalid UTF-8"),
        (b'{"id":"a","id":"b"}', "duplicate key"),
        (b'{"id":"a","answers":[NaN]}', "NaN"),
        (b'{"id":"a","answers":[Infinity]}', "Infinity"),
        (b'{"id":"a","answers":[-Infinity]}', "Infinity"),
        (b'{"id":"a","answers":[1e999]}', "out of range"),
        (b"", "blank line"),
        (b"   \t", "blank line"),
    ],
)
def test_malformed_lines_are_rejected_not_skipped(tmp_path, raw, message):
    error = issues_for(tmp_path, [qa(id="ok"), raw])
    assert any(i.line == 2 and message in i.message for i in error.issues), str(error)


def test_non_nfc_text_is_rejected_not_normalised(tmp_path):
    error = issues_for(tmp_path, [qa(question="café")])
    assert error.issues[0].field == "question" and "NFC" in error.issues[0].message


def test_non_nfc_in_nested_expected_data_is_rejected(tmp_path):
    case = ext(expected={"name": "José", "age": 3})
    error = issues_for(tmp_path, [case], task="json_extraction")
    assert error.issues[0].field == "expected.name"


def test_lone_surrogate_is_rejected(tmp_path):
    raw = json.dumps(qa()).replace('"Q?"', '"\\ud800"').encode()
    assert "surrogate" in issues_for(tmp_path, [raw]).issues[0].message


def test_utf8_bom_in_cases_file_is_rejected(tmp_path):
    directory = write_dataset(tmp_path, [qa()])
    path = directory / "cases.jsonl"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    with pytest.raises(DatasetError, match="byte-order mark"):
        load_dataset(directory)


def test_duplicate_case_id_names_the_first_line(tmp_path):
    error = issues_for(tmp_path, [qa(id="same"), qa(id="other"), qa(id="same")])
    assert error.issues[0].line == 3 and "line 1" in error.issues[0].message


def test_extra_blank_line_at_the_end_is_an_error(tmp_path):
    directory = write_dataset(tmp_path, [qa()])
    path = directory / "cases.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(DatasetError) as info:
        load_dataset(directory)
    assert info.value.issues[0].line == 2 and "blank" in info.value.issues[0].message


def test_empty_cases_file_is_rejected(tmp_path):
    assert "no cases" in issues_for(tmp_path, []).issues[0].message


def test_all_problems_are_reported_together(tmp_path):
    error = issues_for(tmp_path, [qa(split="x"), b"nope", qa(id="ok"), qa(license="")])
    assert [i.line for i in error.issues] == [1, 2, 4]


def test_issue_list_is_capped_and_marked_truncated(tmp_path):
    error = issues_for(tmp_path, [qa(split="x")] * (MAX_ISSUES + 10))
    assert len(error.issues) == MAX_ISSUES and error.truncated
    assert "more issues omitted" in str(error)


def test_exactly_the_cap_is_not_marked_truncated(tmp_path):
    error = issues_for(tmp_path, [qa(split="x")] * MAX_ISSUES)
    assert len(error.issues) == MAX_ISSUES and not error.truncated


def test_error_text_does_not_echo_record_contents_or_local_paths(tmp_path):
    error = issues_for(tmp_path, [qa(code_mixed="SENTINEL-VALUE")])
    assert "SENTINEL-VALUE" not in str(error)
    assert str(tmp_path) not in str(error)


# -- invalid metadata and layout ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "field"),
    [
        ({"version": "1.0"}, "version"),
        ({"name": "Bad Name"}, "name"),
        ({"schema_version": 2}, "schema_version"),
        ({"task": "summarisation"}, "task"),
        ({"content_sha256": "abc"}, "content_sha256"),
        ({"content_sha256": "F" * 64}, "content_sha256"),
        ({"description": MISSING}, "description"),
        ({"unknown": 1}, "unknown"),
        ({"output_schema": {"type": "object"}}, "(record)"),
    ],
)
def test_invalid_qa_metadata_is_rejected(tmp_path, overrides, field):
    directory = write_dataset(tmp_path, [qa()], meta_overrides=overrides)
    with pytest.raises(DatasetError) as info:
        load_dataset(directory)
    issue = info.value.issues[0]
    assert issue.file == "ds/dataset.json" and issue.field == field


def test_extraction_dataset_without_output_schema_is_rejected(tmp_path):
    directory = write_dataset(
        tmp_path, [ext()], task="json_extraction", meta_overrides={"output_schema": MISSING}
    )
    with pytest.raises(DatasetError, match="needs an output_schema"):
        load_dataset(directory)


def test_metadata_must_be_a_json_object_in_nfc_without_bom(tmp_path):
    directory = write_dataset(tmp_path, [qa()])
    meta = directory / "dataset.json"
    meta.write_bytes(b"[]")
    with pytest.raises(DatasetError, match="JSON object"):
        load_dataset(directory)
    meta.write_bytes(b"\xef\xbb\xbf{}")
    with pytest.raises(DatasetError, match="byte-order mark"):
        load_dataset(directory)
    meta.write_bytes(json.dumps({"name": "café"}, ensure_ascii=False).encode())
    with pytest.raises(DatasetError, match="NFC"):
        load_dataset(directory)


def test_missing_files_and_directories_are_reported(tmp_path):
    with pytest.raises(DatasetError, match="not a directory"):
        load_dataset(tmp_path / "nope")
    directory = write_dataset(tmp_path, [qa()])
    (directory / "cases.jsonl").unlink()
    with pytest.raises(DatasetError, match="file not found"):
        load_dataset(directory)
    (directory / "dataset.json").unlink()
    with pytest.raises(DatasetError, match="file not found"):
        load_dataset(directory)
