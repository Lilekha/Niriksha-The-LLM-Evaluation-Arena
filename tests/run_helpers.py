"""Builders for run-storage tests. Test code only."""

import json
from datetime import UTC, datetime

from niriksha.core.generation import GenerationRequest, GenerationSuccess, Message
from niriksha.core.runner import ExecutionRecord
from niriksha.core.runstore import RunManifest, RunResultLine, request_sha256

NOW = datetime(2026, 10, 1, 9, 30, tzinfo=UTC)
HASH_A = "a" * 64
HASH_B = "b" * 64


def make_request(request_id="r1", model="m", content="hi") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id, model=model, messages=(Message(role="user", content=content),)
    )


def make_success(request_id="r1", output="out", **kwargs) -> GenerationSuccess:
    return GenerationSuccess(
        request_id=request_id, provider="fake", requested_model="m", output_text=output, **kwargs
    )


def make_line(request_id="r1", output="out", elapsed=0.5, result=None) -> RunResultLine:
    result = result or make_success(request_id, output)
    return RunResultLine(
        request_id=request_id,
        request_sha256=request_sha256(make_request(request_id)),
        recorded_at=NOW,
        execution=ExecutionRecord(result=result, elapsed_s=elapsed),
    )


def manifest_data(run_id="run1") -> dict:
    return {
        "manifest_version": 1,
        "run_id": run_id,
        "created_at": "2026-10-01T09:30:00Z",
        "dataset": {
            "name": "tiny-qa",
            "version": "0.1.0",
            "task": "short_answer_qa",
            "schema_version": 1,
            "content_sha256": HASH_A,
        },
        "selection": {"splits": ["dev"], "case_count": 2, "case_ids_sha256": HASH_B},
        "prompt": {"system": None, "user": "Answer: {input}", "sha256": HASH_A},
        "provider": {"name": "fake", "implementation": "niriksha.providers.fake.FakeProvider"},
        "requested_model": "m",
        "params": {"temperature": 0.0},
        "software": {
            "niriksha_version": "0.0.0",
            "python_version": "3.13.3",
            "pydantic_version": "2.13.5",
            "platform": "win32",
            "git_commit": "c" * 40,
            "git_dirty": False,
        },
    }


def make_manifest(run_id="run1", **overrides) -> RunManifest:
    return RunManifest.model_validate_json(json.dumps({**manifest_data(run_id), **overrides}))
