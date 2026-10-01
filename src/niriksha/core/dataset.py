"""Local dataset loading, strict validation and content hashing.

Layout of a dataset directory::

    dataset.json   metadata, task, schema version and the pinned content hash
    cases.jsonl    one case per line; file order is the dataset order

Nothing is skipped or rewritten: a malformed record, a non-NFC string or a hash mismatch is
an error.
Files are read as bytes and split on LF only (``str.splitlines`` would also split on U+2028, which
is legal inside a JSON string). CRLF is tolerated.

Dataset identity (see docs/adr/0003-dataset-identity-and-run-persistence.md)
---------------------------------------------------------------------------
``content_sha256`` is SHA-256 over the UTF-8 bytes of this canonical JSON document::

    {"cases": [<each case line as parsed, in file order>],
     "hash_version": 1,
     "output_schema": <object for json_extraction, else null>,
     "schema_version": 1,
     "task": "<task>"}

serialised with ``json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False,
allow_nan=False)``. This is a project-defined canonical form, not RFC 8785. Cases are hashed as
parsed from the file, not as model dumps, so adding an optional schema field later does not change
existing hashes; the flip side is that omitting a defaulted field and writing its default hash
differently. ``name``, ``version``, ``description`` and the stored hash are not hashed. Whitespace,
key order, CRLF vs LF and a final newline do not matter; case order, values and JSON types do
(``1`` and ``1.0`` differ, so avoid floats in gold data).
"""

import hashlib
import json
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationError,
    model_validator,
)

SCHEMA_VERSION = 1
HASH_VERSION = 1
MAX_ISSUES = 20
META_FILE = "dataset.json"
CASES_FILE = "cases.jsonl"
_BOM = b"\xef\xbb\xbf"

Task = Literal["short_answer_qa", "json_extraction"]
Split = Literal["dev", "test", "holdout"]
Language = Literal["en", "hi", "kn"]
Script = Literal["Latn", "Deva", "Knda"]
# Romanised Hindi or Kannada is the language with script Latn.
_ALLOWED_SCRIPTS: dict[str, set[str]] = {
    "en": {"Latn"},
    "hi": {"Deva", "Latn"},
    "kn": {"Knda", "Latn"},
}


def _non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


NonBlank = Annotated[str, AfterValidator(_non_blank)]
CaseId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


# -- errors ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetIssue:
    file: str
    line: int | None
    field: str | None
    message: str

    def __str__(self) -> str:
        where = self.file + (f":{self.line}" if self.line is not None else "")
        return f"{where}" + (f" [{self.field}]" if self.field else "") + f": {self.message}"


class DatasetError(Exception):
    """The dataset could not be loaded. ``issues`` lists every problem found (capped)."""

    def __init__(self, issues: Sequence[DatasetIssue], *, truncated: bool = False) -> None:
        self.issues = tuple(issues)
        self.truncated = truncated
        lines = [f"{len(self.issues)} dataset issue(s):", *(f"  {i}" for i in self.issues)]
        if truncated:
            lines.append(f"  ... more issues omitted after the first {MAX_ISSUES}")
        super().__init__("\n".join(lines))


# -- models ---------------------------------------------------------------------------------------


class _CaseBase(_Model):
    id: CaseId
    split: Split
    language: Language
    script: Script
    code_mixed: bool = False
    origin: Literal["original", "public"]
    source: NonBlank | None = None
    license: NonBlank

    @model_validator(mode="after")
    def _consistent(self):
        if self.script not in _ALLOWED_SCRIPTS[self.language]:
            allowed = sorted(_ALLOWED_SCRIPTS[self.language])
            raise ValueError(
                f"script {self.script!r} is not valid for language {self.language!r}; "
                f"allowed: {allowed}"
            )
        if self.origin == "public" and self.source is None:
            raise ValueError("a public case needs a source")
        return self


class QACase(_CaseBase):
    question: NonBlank
    answers: tuple[NonBlank, ...] = Field(min_length=1)

    @property
    def input_text(self) -> str:
        return self.question


class ExtractionCase(_CaseBase):
    text: NonBlank
    expected: dict[str, JsonValue]

    @property
    def input_text(self) -> str:
        return self.text


Case = QACase | ExtractionCase
_CASE_MODELS: dict[str, type[_CaseBase]] = {
    "short_answer_qa": QACase,
    "json_extraction": ExtractionCase,
}


class DatasetMeta(_Model):
    schema_version: Literal[1]
    name: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9._-]{0,63}$")]
    version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
    task: Task
    description: str
    content_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    output_schema: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _schema_matches_task(self):
        if self.task == "json_extraction" and self.output_schema is None:
            raise ValueError("a json_extraction dataset needs an output_schema")
        if self.task != "json_extraction" and self.output_schema is not None:
            raise ValueError("output_schema is only allowed for json_extraction datasets")
        return self


@dataclass(frozen=True)
class Dataset:
    meta: DatasetMeta
    cases: tuple[Case, ...]
    content_sha256: str  # recomputed from the files, and equal to meta.content_sha256

    def select(self, splits: Sequence[str]) -> tuple[Case, ...]:
        """Cases whose split is in ``splits``, in dataset (file) order."""
        wanted = set(splits)
        return tuple(case for case in self.cases if case.split in wanted)


# -- hashing --------------------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_content_sha256(task: str, output_schema: dict | None, raw_cases: Sequence[dict]) -> str:
    """The dataset identity hash; see the module docstring for the exact payload."""
    payload = {
        "hash_version": HASH_VERSION,
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "output_schema": output_schema,
        "cases": list(raw_cases),
    }
    return sha256_hex(canonical_json_bytes(payload))


# -- parsing --------------------------------------------------------------------------------------


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _finite_float(text: str) -> float:
    value = float(text)
    if value in (float("inf"), float("-inf")):
        raise ValueError(f"number {text} is out of range")
    return value


def _parse_json(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
    )


def _text_problems(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Strings that are not NFC or not encodable as UTF-8 (a lone surrogate), with their path."""
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            yield path, "string contains a lone surrogate and cannot be encoded as UTF-8"
            return
        if not unicodedata.is_normalized("NFC", value):
            yield path, "string is not Unicode NFC (normalise it in the source file)"
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _text_problems(key, f"{path}.{key}" if path else str(key))
            yield from _text_problems(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _text_problems(item, f"{path}[{index}]")


def _field_path(loc: tuple) -> str:
    return ".".join(str(part) for part in loc) or "(record)"


def _model_issues(error: ValidationError, file: str, line: int | None) -> list[DatasetIssue]:
    # include_input=False: error text must not echo record contents back.
    return [
        DatasetIssue(file, line, _field_path(e["loc"]), e["msg"])
        for e in error.errors(include_input=False, include_url=False, include_context=False)
    ]


def _decode(data: bytes, file: str, line: int | None) -> tuple[str | None, DatasetIssue | None]:
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, DatasetIssue(file, line, None, f"invalid UTF-8 at byte {exc.start}")


def _load_meta(directory: Path) -> tuple[DatasetMeta, dict]:
    file = f"{directory.name}/{META_FILE}"
    path = directory / META_FILE
    if not path.is_file():
        raise DatasetError([DatasetIssue(file, None, None, "file not found")])
    data = path.read_bytes()
    if data.startswith(_BOM):
        raise DatasetError([DatasetIssue(file, 1, None, "UTF-8 byte-order mark is not allowed")])
    text, issue = _decode(data, file, None)
    if issue:
        raise DatasetError([issue])
    try:
        raw = _parse_json(text)
    except ValueError as exc:
        raise DatasetError([DatasetIssue(file, None, None, f"invalid JSON: {exc}")]) from None
    if not isinstance(raw, dict):
        raise DatasetError([DatasetIssue(file, None, None, "must be a JSON object")])
    problems = [DatasetIssue(file, None, p or "(root)", m) for p, m in _text_problems(raw)]
    if problems:
        raise DatasetError(problems[:MAX_ISSUES], truncated=len(problems) > MAX_ISSUES)
    try:
        return DatasetMeta.model_validate_json(text), raw
    except ValidationError as exc:
        issues = _model_issues(exc, file, None)
        raise DatasetError(issues[:MAX_ISSUES], truncated=len(issues) > MAX_ISSUES) from None


def _load_cases(directory: Path, task: str) -> tuple[list[Case], list[dict], list[DatasetIssue]]:
    file = f"{directory.name}/{CASES_FILE}"
    path = directory / CASES_FILE
    if not path.is_file():
        return [], [], [DatasetIssue(file, None, None, "file not found")]
    data = path.read_bytes()
    if data.startswith(_BOM):
        return [], [], [DatasetIssue(file, 1, None, "UTF-8 byte-order mark is not allowed")]

    model = _CASE_MODELS[task]
    lines = data.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()  # a final trailing newline is the only allowed empty element

    cases: list[Case] = []
    raw_cases: list[dict] = []
    issues: list[DatasetIssue] = []
    first_seen: dict[str, int] = {}
    for number, raw_line in enumerate(lines, start=1):
        if len(issues) > MAX_ISSUES:
            break
        if not raw_line.strip(b" \t\r"):
            issues.append(DatasetIssue(file, number, None, "blank line (records are not skipped)"))
            continue
        text, issue = _decode(raw_line, file, number)
        if issue:
            issues.append(issue)
            continue
        try:
            raw = _parse_json(text)
        except ValueError as exc:
            issues.append(DatasetIssue(file, number, None, f"invalid JSON: {exc}"))
            continue
        if not isinstance(raw, dict):
            issues.append(DatasetIssue(file, number, None, "record must be a JSON object"))
            continue
        text_issues = [
            DatasetIssue(file, number, p or "(record)", m) for p, m in _text_problems(raw)
        ]
        if text_issues:
            issues.extend(text_issues)
            continue
        try:
            case = model.model_validate_json(text)
        except ValidationError as exc:
            issues.extend(_model_issues(exc, file, number))
            continue
        if case.id in first_seen:
            msg = f"duplicate case id {case.id!r} (first seen on line {first_seen[case.id]})"
            issues.append(DatasetIssue(file, number, "id", msg))
            continue
        first_seen[case.id] = number
        cases.append(case)
        raw_cases.append(raw)
    if not lines and not issues:
        issues.append(DatasetIssue(file, None, None, "dataset has no cases"))
    return cases, raw_cases, issues


def load_dataset(directory: str | Path) -> Dataset:
    """Load, validate and hash-check a dataset from a local directory.

    Raises ``DatasetError`` listing every problem found (capped at ``MAX_ISSUES``). Never edits
    the files.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DatasetError([DatasetIssue(directory.name, None, None, "not a directory")])
    meta, _ = _load_meta(directory)
    cases, raw_cases, issues = _load_cases(directory, meta.task)
    if issues:
        raise DatasetError(issues[:MAX_ISSUES], truncated=len(issues) > MAX_ISSUES)
    computed = compute_content_sha256(meta.task, meta.output_schema, raw_cases)
    if computed != meta.content_sha256:
        raise DatasetError(
            [
                DatasetIssue(
                    f"{directory.name}/{META_FILE}",
                    None,
                    "content_sha256",
                    f"pinned hash does not match the cases (computed {computed}). If the change is "
                    "intended, bump 'version', add a changelog entry and update the pin; otherwise "
                    "restore the original cases.",
                )
            ]
        )
    return Dataset(meta=meta, cases=tuple(cases), content_sha256=computed)
