"""Helpers for building throwaway datasets in tmp_path. Test code only."""

import json
from pathlib import Path

from niriksha.core.dataset import compute_content_sha256

FIXTURES = Path(__file__).parent / "fixtures" / "datasets"
MISSING = object()  # pass as a value to delete a key from a case

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def _build(base: dict, overrides: dict) -> dict:
    merged = {**base, **overrides}
    return {k: v for k, v in merged.items() if v is not MISSING}


def qa(**overrides) -> dict:
    return _build(
        {
            "id": "q1",
            "split": "dev",
            "language": "en",
            "script": "Latn",
            "origin": "original",
            "license": "MIT",
            "question": "Q?",
            "answers": ["A"],
        },
        overrides,
    )


def ext(**overrides) -> dict:
    return _build(
        {
            "id": "e1",
            "split": "dev",
            "language": "en",
            "script": "Latn",
            "origin": "original",
            "license": "MIT",
            "text": "Asha is 30.",
            "expected": {"name": "Asha", "age": 30},
        },
        overrides,
    )


def write_dataset(
    root: Path,
    cases: list,
    *,
    task: str = "short_answer_qa",
    pin: str | None = None,
    name: str = "ds",
    version: str = "0.1.0",
    description: str = "test",
    eol: bytes = b"\n",
    trailing_newline: bool = True,
    meta_overrides: dict | None = None,
    dirname: str = "ds",
) -> Path:
    """Write dataset.json + cases.jsonl. ``cases`` items are dicts or raw bytes.

    The pin is computed from the dict cases unless given (or any case is raw bytes).
    """
    directory = root / dirname
    directory.mkdir(parents=True)
    schema = EXTRACTION_SCHEMA if task == "json_extraction" else None
    lines = [
        c if isinstance(c, bytes) else json.dumps(c, ensure_ascii=False).encode() for c in cases
    ]
    if pin is None:
        raw = [c for c in cases if isinstance(c, dict)]
        pin = compute_content_sha256(task, schema, raw) if len(raw) == len(cases) else "0" * 64
    meta = {
        "schema_version": 1,
        "name": name,
        "version": version,
        "task": task,
        "description": description,
        "content_sha256": pin,
    }
    if schema is not None:
        meta["output_schema"] = schema
    meta.update(meta_overrides or {})
    meta = {k: v for k, v in meta.items() if v is not MISSING}
    (directory / "dataset.json").write_bytes(json.dumps(meta, ensure_ascii=False).encode())
    body = eol.join(lines) + (eol if trailing_newline and lines else b"")
    (directory / "cases.jsonl").write_bytes(body)
    return directory
