"""A whole run through the adapter and the runner's retries, then scoring. Local server only."""

import httpx
import pytest

from e2e_support import make_config
from http_support import Reply, completion
from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run, resume_run
from niriksha.core.generation import FailureKind, GenerationFailure, GenerationSuccess
from niriksha.core.report import aggregate
from niriksha.core.runload import load_run
from niriksha.core.runstore import read_manifest, read_results
from niriksha.providers.openai_compatible import (
    HTTP_RETRY_POLICY,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
)
from niriksha.scorers.artifacts import score_run_to_artifact, verify_artifact
from score_helpers import SPLITS, copy_fixture, deterministic

KEY = "sk-test-WORKFLOW-SECRET-98765"  # a deliberately fake key
METRIC = "normalized_exact_match"
ENV_NAME = "NIRIKSHA_TEST_VARIABLE"


class Sleeps:
    def __init__(self):
        self.waits: list[float] = []

    def __call__(self, seconds):
        self.waits.append(seconds)


@pytest.fixture
def setup(tmp_path, server):
    dataset_dir = copy_fixture(tmp_path, "tiny_qa")
    dataset = load_dataset(dataset_dir)
    config = make_config("http-run", SPLITS["tiny_qa"], model="served-model")
    answers = [case.answers[0] for case in dataset.select(config.splits)]
    provider = OpenAICompatibleProvider(
        OpenAICompatibleProfile(name="local-test", base_url=server.base_url, api_key_env=ENV_NAME),
        environ={ENV_NAME: KEY},
    )
    yield tmp_path, dataset_dir, dataset, config, answers, provider
    provider.close()


def ok(text):
    return Reply(200, completion(text, model="served-model-2025"))


def test_a_run_with_retries_records_attempts_scores_and_verifies(setup, server):
    tmp_path, dataset_dir, dataset, config, answers, provider = setup
    server.enqueue(
        Reply(429, {"error": {"type": "rate_limit_error"}}, {"Retry-After": "2"}),
        ok(answers[0]),  # case 1: retried once after the server's Retry-After, then right
        ok("a wrong answer"),  # case 2: wrong, not retried
        Reply(500, "boom"),
        Reply(500, "boom"),
        Reply(500, "boom"),  # case 3: three attempts, still failing
        ok(answers[3]),  # case 4: right
        Reply(401, {"error": {"message": f"bad key {KEY}", "code": "invalid_api_key"}}),  # case 5
    )
    sleeps = Sleeps()
    summary = execute_run(
        config,
        dataset,
        provider,
        tmp_path / "runs",
        retry=HTTP_RETRY_POLICY,
        sleep=sleeps,
        secret_values=(KEY,),
        **deterministic(),
    )
    assert summary.executed == 5 and len(server.requests) == 8
    assert sleeps.waits == [2.0, 1.0, 2.0]  # Retry-After, then the 1 s and 2 s backoffs
    assert all(r.headers["authorization"] == f"Bearer {KEY}" for r in server.requests)

    lines = read_results(summary.run_dir).lines
    assert [len(line.execution.attempts) for line in lines] == [2, 1, 3, 1, 1]
    results = [line.execution.result for line in lines]
    assert (
        isinstance(results[0], GenerationSuccess)
        and results[0].returned_model == "served-model-2025"
    )
    assert isinstance(results[2], GenerationFailure) and results[2].kind is FailureKind.SERVER_ERROR
    assert isinstance(results[4], GenerationFailure) and results[4].kind is FailureKind.AUTH_ERROR
    assert [a.result.kind for a in lines[0].execution.attempts[:1]] == [FailureKind.RATE_LIMIT]
    assert lines[0].execution.attempts[0].result.provider_metadata["retry_after_s"] == 2.0
    assert all(r.provider_metadata["endpoint"] == server.base_url for r in results)

    manifest = read_manifest(summary.run_dir)
    assert manifest.provider.name == "local-test" and manifest.requested_model == "served-model"
    assert manifest.provider.implementation == (
        "niriksha.providers.openai_compatible.OpenAICompatibleProvider"
    )
    for path in summary.run_dir.iterdir():  # the key is in no persisted file
        assert KEY.encode() not in path.read_bytes(), path.name

    load_run(summary.run_dir, dataset_dir)  # the run verifies as a whole
    scores = tmp_path / "scores"
    outcome = score_run_to_artifact(summary.run_dir, dataset_dir, scores, METRIC)
    verify_artifact(outcome.artifact, summary.run_dir, dataset_dir)
    records = outcome.artifact.records
    assert [r.value for r in records] == [1.0, 0.0, None, 1.0, None]
    assert records[2].reason == "generation failed: server_error"
    assert records[4].reason == "generation failed: auth_error"
    assert aggregate(outcome.artifact).mean == pytest.approx(2 / 3)
    for path in scores.iterdir():
        assert KEY.encode() not in path.read_bytes()


def test_resume_does_not_call_the_server_again_for_recorded_failures(setup, server):
    tmp_path, dataset_dir, dataset, config, answers, provider = setup
    server.enqueue(*[Reply(500, "boom")] * 15)  # five cases, three attempts each
    execute_run(
        config,
        dataset,
        provider,
        tmp_path / "runs",
        retry=HTTP_RETRY_POLICY,
        sleep=Sleeps(),
        **deterministic(),
    )
    sent = len(server.requests)
    assert sent == 15
    summary = resume_run(
        config,
        dataset,
        provider,
        tmp_path / "runs",
        retry=HTTP_RETRY_POLICY,
        sleep=Sleeps(),
        **deterministic(),
    )
    assert summary.executed == 0 and len(server.requests) == sent


def test_without_a_retry_policy_each_case_is_one_request_and_lines_have_no_attempts(setup, server):
    tmp_path, _, dataset, config, answers, provider = setup
    server.enqueue(*[ok(a) for a in answers[:4]], Reply(503, "down"))
    sleeps = Sleeps()
    summary = execute_run(
        config, dataset, provider, tmp_path / "runs", sleep=sleeps, **deterministic()
    )
    assert len(server.requests) == 5 and sleeps.waits == []
    assert "attempts" not in (summary.run_dir / "results.jsonl").read_text(encoding="utf-8")
    assert all(line.execution.attempts is None for line in read_results(summary.run_dir).lines)


def test_the_key_is_refused_if_it_would_reach_a_run_file(setup, server):
    """The run store's own guard still applies to anything the adapter might ever echo."""
    tmp_path, _, dataset, config, answers, provider = setup
    server.enqueue(ok(f"the answer is {KEY}"))  # a model that happens to output the key
    with pytest.raises(Exception, match="secret"):
        execute_run(
            config, dataset, provider, tmp_path / "runs", secret_values=(KEY,), **deterministic()
        )


def test_https_profiles_verify_certificates_and_plain_http_loads_no_ca_bundle(monkeypatch):
    seen = []
    real = httpx.Client

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", spy)
    for url in ("https://api.example.com/v1", "http://127.0.0.1:1/v1"):
        OpenAICompatibleProvider(OpenAICompatibleProfile(name="p", base_url=url)).close()
    assert [kw["verify"] for kw in seen] == [True, False]
    assert all(kw["trust_env"] is False and kw["follow_redirects"] is False for kw in seen)
