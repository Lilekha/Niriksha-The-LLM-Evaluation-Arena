"""The live smoke harness, tested offline against the local fake server. No real network."""

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from http_support import Reply, completion
from niriksha.core.generation import GenerationParams, GenerationRequest, Message
from niriksha.core.runload import load_run
from niriksha.core.runstore import read_results

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "live_smoke.py"
spec = importlib.util.spec_from_file_location("live_smoke", SCRIPT)
smoke = importlib.util.module_from_spec(spec)
sys.modules["live_smoke"] = smoke
spec.loader.exec_module(smoke)

KEY = "sk-test-SMOKE-HARNESS-24680"  # a deliberately fake key
VARIABLE = "NIRIKSHA_TEST_VARIABLE"
MODEL_TEXT = "ZZ-MODEL-TEXT-MUST-NOT-BE-PRINTED"
LEAK = "hunter2"  # a value that must never be echoed
NOW = datetime(2026, 10, 4, 9, 30, tzinfo=UTC)
FAST = {"sleep": lambda seconds: None}


def env(server=None, **changes):
    values = {
        smoke.ENV_BASE_URL: server.base_url if server else "http://127.0.0.1:9/v1",
        smoke.ENV_MODEL: "test-model",
        smoke.ENV_KEY_VARIABLE: VARIABLE,
        VARIABLE: KEY,
    }
    values.update(changes)
    return {k: v for k, v in values.items() if v is not None}


def request():
    return GenerationRequest(
        request_id="r1",
        model="m",
        messages=(Message(role="user", content="q"),),
        params=GenerationParams(temperature=0.0, max_tokens=8),
    )


def replies_for_a_full_run():
    answers = [Reply(200, completion(MODEL_TEXT, usage={"total_tokens": 10})) for _ in range(5)]
    answers.append(Reply(200, completion('{"name": "Asha", "age": 30}')))
    answers.append(Reply(200, completion(MODEL_TEXT)))
    answers.append(Reply(200, completion('{"name": "आशा", "age": 30}')))
    return answers


# -- importing and configuration ------------------------------------------------------------------


def test_importing_the_script_makes_no_request_and_writes_nothing(tmp_path):
    assert smoke.__name__ == "live_smoke" and [t.label for t in smoke.TASKS] == ["qa", "extraction"]
    assert sum(t.cases for t in smoke.TASKS) == 8  # the whole budget
    assert not (smoke.REPO / "runs" / "smoke-qa").exists()


def test_the_caps_are_fixed_and_the_documented_values():
    assert smoke.MIN_INTERVAL_S == 3.0 and smoke.MAX_TIMEOUT_S == 300.0
    assert [t.max_tokens for t in smoke.TASKS] == [32, 128]
    assert smoke.NO_RETRY.max_attempts == 1
    source = SCRIPT.read_text(encoding="utf-8")
    assert "HTTP_RETRY_POLICY" not in source  # never the three-attempt policy


def test_settings_read_the_environment_and_default_to_both_tasks():
    settings = smoke.settings_from_env(env())
    assert settings.model == "test-model" and settings.key_variable == VARIABLE
    assert settings.max_tokens_field == "max_tokens" and settings.timeout_s == 30.0
    assert [t.label for t in settings.tasks] == ["qa", "extraction"]
    only = smoke.settings_from_env(env(**{smoke.ENV_TASKS: "extraction"}))
    assert [t.label for t in only.tasks] == ["extraction"]
    assert smoke.settings_from_env(env(**{smoke.ENV_KEY_VARIABLE: None})).key_variable is None


@pytest.mark.parametrize(
    ("changes", "mentions"),
    [
        ({smoke.ENV_BASE_URL: None}, smoke.ENV_BASE_URL),
        ({smoke.ENV_MODEL: None}, smoke.ENV_MODEL),
        ({smoke.ENV_MAX_TOKENS_FIELD: "tokens"}, smoke.ENV_MAX_TOKENS_FIELD),
        ({smoke.ENV_TIMEOUT: "abc"}, smoke.ENV_TIMEOUT),
        ({smoke.ENV_TIMEOUT: "0"}, smoke.ENV_TIMEOUT),
        ({smoke.ENV_TIMEOUT: "301"}, smoke.ENV_TIMEOUT),
        ({smoke.ENV_TASKS: "qa,other"}, smoke.ENV_TASKS),
        ({smoke.ENV_TASKS: "qa,qa"}, smoke.ENV_TASKS),
    ],
)
def test_invalid_settings_are_refused_naming_the_variable(changes, mentions):
    with pytest.raises(smoke.SmokeError, match=mentions):
        smoke.settings_from_env(env(**changes))


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/v1",  # remote and not https
        "ftp://example.com/v1",
        "https://user:" + LEAK + "@example.com/v1",
        "https://example.com/v1?api_key=" + LEAK,
        "https://example.com/v1#" + LEAK,
    ],
)
def test_unsafe_endpoints_are_refused_without_echoing_them(url):
    with pytest.raises(smoke.SmokeError) as caught:
        smoke.settings_from_env(env(**{smoke.ENV_BASE_URL: url, smoke.ENV_KEY_VARIABLE: None}))
    assert LEAK not in str(caught.value) and "example.com" not in str(caught.value)


def test_https_remote_and_plain_http_loopback_are_accepted():
    smoke.settings_from_env(env(**{smoke.ENV_BASE_URL: "https://api.example.com/v1"}))
    smoke.settings_from_env(env(**{smoke.ENV_BASE_URL: "http://localhost:11434/v1"}))
    smoke.settings_from_env(env(**{smoke.ENV_BASE_URL: "http://[::1]:11434/v1"}))


def test_an_unusable_key_variable_name_is_refused():
    with pytest.raises(smoke.SmokeError, match=smoke.ENV_KEY_VARIABLE):
        smoke.settings_from_env(env(**{smoke.ENV_KEY_VARIABLE: "not a name"}))


# -- plan, preflight, confirmation: all without any network ---------------------------------------


def test_the_plan_counts_exactly_eight_requests_and_prints_no_secret(tmp_path):
    plan = smoke.build_plan(smoke.settings_from_env(env()), lambda: NOW)
    assert plan.total_requests == 8
    assert [r.run_id for r in plan.runs] == [
        "smoke-qa-20261004t093000z",
        "smoke-extraction-20261004t093000z",
    ]
    text = "\n".join(smoke.describe_plan(plan))
    assert "NO retries" in text and "8 in total" in text and "test-model" in text
    assert "http://127.0.0.1:9/v1" in text and VARIABLE in text
    assert KEY not in text


def test_preflight_passes_without_network_and_names_a_missing_variable_only(tmp_path):
    plan = smoke.build_plan(smoke.settings_from_env(env()), lambda: NOW)
    smoke.preflight(plan, env(), tmp_path)  # the global socket guard is active: no network used
    with pytest.raises(smoke.SmokeError, match=VARIABLE) as caught:
        smoke.preflight(plan, env(**{VARIABLE: None}), tmp_path)
    assert KEY not in str(caught.value)
    with pytest.raises(smoke.SmokeError, match=VARIABLE):
        smoke.preflight(plan, env(**{VARIABLE: "  "}), tmp_path)


def test_an_existing_run_is_refused_not_resumed(tmp_path):
    plan = smoke.build_plan(smoke.settings_from_env(env()), lambda: NOW)
    (tmp_path / plan.runs[1].run_id).mkdir()
    with pytest.raises(smoke.SmokeError, match="already exists"):
        smoke.preflight(plan, env(), tmp_path)


@pytest.mark.parametrize("answer", ["", "7", "eight", "8 ", "y", "yes"])
def test_only_the_exact_request_count_confirms(answer):
    plan = smoke.build_plan(smoke.settings_from_env(env()), lambda: NOW)
    if answer.strip() == "8":
        smoke.confirm(plan, lambda prompt: answer)
        return
    with pytest.raises(smoke.SmokeError, match="no request was sent"):
        smoke.confirm(plan, lambda prompt: answer)


def test_end_of_input_aborts():
    plan = smoke.build_plan(smoke.settings_from_env(env()), lambda: NOW)

    def eof(prompt):
        raise EOFError

    with pytest.raises(smoke.SmokeError, match="no request was sent"):
        smoke.confirm(plan, eof)


# -- the budgeted provider ------------------------------------------------------------------------


def test_the_budget_refuses_the_next_call_before_any_http_request(server):
    provider = smoke.BudgetedProvider(
        smoke.settings_from_env(env(server)).profile(),
        environ=env(server),
        max_calls=2,
        sleep=lambda s: None,
    )
    server.enqueue(
        Reply(200, completion("a")), Reply(200, completion("b")), Reply(200, completion("c"))
    )
    try:
        provider.generate(request())
        provider.generate(request())
        with pytest.raises(smoke.BudgetExceeded):
            provider.generate(request())
        assert provider.calls == 2 and len(server.requests) == 2
    finally:
        provider.close()


def test_requests_are_spaced_by_the_minimum_interval(server):
    ticks = iter([0.0, 0.5, 0.5, 4.0, 4.0, 4.1])
    waits = []
    provider = smoke.BudgetedProvider(
        smoke.settings_from_env(env(server)).profile(),
        environ=env(server),
        max_calls=3,
        sleep=waits.append,
        clock=lambda: next(ticks),
    )
    server.enqueue(*[Reply(200, completion("x"))] * 3)
    try:
        for _ in range(3):
            provider.generate(request())
    finally:
        provider.close()
    assert waits == [pytest.approx(2.5)]  # 3.0 - 0.5 the first time; the second gap was long enough


# -- a whole run against the fake server ----------------------------------------------------------


def test_a_full_run_sends_exactly_eight_single_attempt_requests_and_leaks_nothing(server, tmp_path):
    server.enqueue(*replies_for_a_full_run())
    lines = []
    code = smoke.main(
        env(server),
        input_fn=lambda prompt: "8",
        out=lines.append,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    printed = "\n".join(lines)
    assert code == 0 and len(server.requests) == 8
    assert "requests made: 8" in printed and "No model text was printed" in printed
    assert KEY not in printed and MODEL_TEXT not in printed and "Asha" not in printed
    for body in (r.json() for r in server.requests):
        assert (
            body["temperature"] == 0.0
            and body["max_tokens"] in (32, 128)
            and body["stream"] is False
        )
        assert body["model"] == "test-model"
    assert [r.json()["max_tokens"] for r in server.requests] == [32] * 5 + [128] * 3
    for run in ("smoke-qa-20261004t093000z", "smoke-extraction-20261004t093000z"):
        run_dir = tmp_path / "runs" / run
        assert sorted(p.name for p in run_dir.iterdir()) == ["manifest.json", "results.jsonl"]
        assert "attempts" not in (run_dir / "results.jsonl").read_text(encoding="utf-8")
        assert all(line.execution.attempts is None for line in read_results(run_dir).lines)
    for directory in (tmp_path / "runs", tmp_path / "scores"):
        for path in directory.rglob("*"):
            if path.is_file():
                assert KEY.encode() not in path.read_bytes(), path.name
    assert len(list((tmp_path / "scores").iterdir())) == 4  # 1 metric for QA, 3 for extraction
    assert all("Authorization" not in line for line in lines)


def test_failures_are_recorded_once_never_retried(server, tmp_path):
    answers = replies_for_a_full_run()
    answers[0] = Reply(429, {"error": {"type": "rate_limit_error"}}, {"Retry-After": "1"})
    answers[1] = Reply(500, "boom")
    answers[7] = Reply(401, {"error": {"message": f"bad {KEY}", "code": "invalid_api_key"}})
    server.enqueue(*answers)
    lines = []
    code = smoke.main(
        env(server),
        input_fn=lambda prompt: "8",
        out=lines.append,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    printed = "\n".join(lines)
    assert code == 0 and len(server.requests) == 8  # a 429 and a 500 did not cause extra requests
    assert "rate_limit 1" in printed and "server_error 1" in printed and "auth_error 1" in printed
    assert KEY not in printed
    run_dir = tmp_path / "runs" / "smoke-qa-20261004t093000z"
    load_run(run_dir, smoke.FIXTURES / "tiny_qa")
    kinds = [
        getattr(line.execution.result, "kind", None) and line.execution.result.kind.value
        for line in read_results(run_dir).lines
    ]
    assert kinds[:2] == ["rate_limit", "server_error"]


def test_choosing_one_task_sends_only_its_requests(server, tmp_path):
    server.enqueue(*replies_for_a_full_run()[5:])
    code = smoke.main(
        env(server, **{smoke.ENV_TASKS: "extraction"}),
        input_fn=lambda prompt: "3",
        out=lambda line: None,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    assert code == 0 and len(server.requests) == 3
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["smoke-extraction-20261004t093000z"]


def test_a_wrong_confirmation_sends_nothing(server, tmp_path):
    lines = []
    code = smoke.main(
        env(server),
        input_fn=lambda prompt: "5",
        out=lines.append,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    assert code == 2 and server.requests == []
    assert not (tmp_path / "runs").exists() and any("no request was sent" in text for text in lines)


def test_a_configuration_problem_stops_before_any_request(server, tmp_path):
    lines = []
    code = smoke.main(
        env(server, **{VARIABLE: None}),
        input_fn=lambda prompt: pytest.fail("must not ask"),
        out=lines.append,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    assert code == 2 and server.requests == []
    assert any(VARIABLE in line for line in lines)


# -- the secret guards ----------------------------------------------------------------------------


def test_the_run_store_refuses_a_key_the_model_echoes_and_nothing_leaks(server, tmp_path):
    answers = replies_for_a_full_run()
    answers[2] = Reply(200, completion(f"the key is {KEY}"))
    server.enqueue(*answers)
    with pytest.raises(Exception, match="secret"):
        smoke.main(
            env(server),
            input_fn=lambda prompt: "8",
            out=lambda line: None,
            now=lambda: NOW,
            runs_dir=tmp_path / "runs",
            scores_dir=tmp_path / "scores",
            **FAST,
        )
    for path in (tmp_path / "runs").rglob("*"):
        if path.is_file():
            assert KEY.encode() not in path.read_bytes()


def test_the_scan_finds_a_key_in_any_persisted_file_and_does_not_print_it(tmp_path):
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "x.json").write_text(json.dumps({"a": KEY}), encoding="utf-8")
    with pytest.raises(smoke.SecretFound) as caught:
        smoke.scan_for_secret([tmp_path / "scores"], KEY)
    assert KEY not in str(caught.value) and "rotate" in str(caught.value)
    smoke.scan_for_secret([tmp_path / "scores"], "a-different-value")
    smoke.scan_for_secret([tmp_path / "missing"], KEY)
    smoke.scan_for_secret([tmp_path / "scores"], None)


def test_the_script_has_no_provider_registry_cli_or_dataset_parameter():
    source = SCRIPT.read_text(encoding="utf-8")
    for absent in ("argparse", "dotenv", "typer", "click"):
        assert absent not in source
    assert "NIRIKSHA_SMOKE_DATASET" not in source


# -- the harness checks what it promises ----------------------------------------------------------


def run_main(server, tmp_path, lines, **changes):
    return smoke.main(
        env(server, **changes),
        input_fn=lambda prompt: "8",
        out=lines.append,
        now=lambda: NOW,
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )


def test_every_run_is_loaded_and_every_artifact_is_verified(server, tmp_path, monkeypatch):
    seen = {"load_run": 0, "verify_artifact": 0}

    def counting(name):
        real = getattr(smoke, name)

        def wrapper(*args, **kwargs):
            seen[name] += 1
            return real(*args, **kwargs)

        return wrapper

    for name in seen:
        monkeypatch.setattr(smoke, name, counting(name))
    server.enqueue(*replies_for_a_full_run())
    assert run_main(server, tmp_path, []) == 0
    assert seen == {"load_run": 2, "verify_artifact": 4}  # 2 runs; 1 + 3 metrics


def test_the_budget_equals_exactly_the_planned_requests(server, tmp_path, monkeypatch):
    caps = []

    class Spy(smoke.BudgetedProvider):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            caps.append(self._max_calls)

    monkeypatch.setattr(smoke, "BudgetedProvider", Spy)
    server.enqueue(*replies_for_a_full_run())
    run_main(server, tmp_path, [])
    server.enqueue(*replies_for_a_full_run()[5:])
    smoke.main(
        env(server, **{smoke.ENV_TASKS: "extraction"}),
        input_fn=lambda prompt: "3",
        out=lambda line: None,
        now=lambda: datetime(2026, 10, 4, 10, 0, tzinfo=UTC),
        runs_dir=tmp_path / "runs",
        scores_dir=tmp_path / "scores",
        **FAST,
    )
    assert caps == [8, 3]


def test_a_key_that_reaches_a_persisted_file_stops_the_run_with_a_warning(
    server, tmp_path, monkeypatch
):
    real = smoke.score_run_to_artifact

    def leaky(run_dir, dataset_dir, scores_dir, metric, **kwargs):
        outcome = real(run_dir, dataset_dir, scores_dir, metric, **kwargs)
        (Path(scores_dir) / "leak.txt").write_text(KEY, encoding="utf-8")  # a simulated leak
        return outcome

    monkeypatch.setattr(smoke, "score_run_to_artifact", leaky)
    server.enqueue(*replies_for_a_full_run())
    lines = []
    assert run_main(server, tmp_path, lines) == 2
    printed = "\n".join(lines)
    assert "rotate the key" in printed and KEY not in printed
