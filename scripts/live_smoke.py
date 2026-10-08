"""Manual, opt-in smoke test of the OpenAI-compatible adapter against ONE real endpoint.

This is not part of pytest or CI and nothing imports it at run time. Running it sends REAL requests
to whatever endpoint you configure, so it only ever runs when a person decides to run it, after
checking the provider's free access, limits and data terms (docs/provider-validation.md). Starting
the script is not that decision: it prints the plan and waits for you to type the request count.

What it does, and does not:

- Sends only the two synthetic test fixtures (``tiny_qa``, ``tiny_extraction``): at most 8 requests,
  one attempt each (``NO_RETRY``: a failure costs one request and is never repeated), temperature 0,
  ``max_tokens`` 32 for QA and 128 for extraction, at least 3 s between requests. None of these
  caps can be raised from the environment. There is no dataset parameter: only the fixtures run.
- Reads the endpoint, model and the NAME of the variable that holds the key from environment
  variables (it does not load ``.env``): ``NIRIKSHA_SMOKE_BASE_URL``, ``NIRIKSHA_SMOKE_MODEL``,
  ``NIRIKSHA_SMOKE_KEY_VARIABLE`` (omit for a server that needs no key), and optionally
  ``NIRIKSHA_SMOKE_MAX_TOKENS_FIELD`` (``max_tokens`` or ``max_completion_tokens``),
  ``NIRIKSHA_SMOKE_TIMEOUT_S`` (up to 300) and ``NIRIKSHA_SMOKE_TASKS`` (``qa``, ``extraction``;
  both by default; choosing fewer only reduces the number of requests).
- Requires https for a remote host (plain http only for loopback, e.g. a local Ollama).
- Persists runs and scores through the normal run store and scorers under ``runs/`` and ``scores/``
  (both gitignored), passes the key to the run store's secret guard, and afterwards scans those
  directories for the key.
- Prints only counts, latencies, the returned model name, reported token usage and scores. It never
  prints model text, headers, the profile or the key, and it configures no logging.

Usage (PowerShell, in a shell where the key variable is set for this session only)::

    $env:NIRIKSHA_SMOKE_BASE_URL = "https://api.example.com/v1"
    $env:NIRIKSHA_SMOKE_MODEL = "the-exact-model-id"
    $env:NIRIKSHA_SMOKE_KEY_VARIABLE = "MY_PROVIDER_KEY"   # the NAME; that variable must be set
    python scripts/live_smoke.py
"""

import json
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from niriksha.core.dataset import load_dataset
from niriksha.core.execution import execute_run
from niriksha.core.generation import GenerationFailure, GenerationParams, GenerationSuccess
from niriksha.core.report import aggregate
from niriksha.core.runload import load_run
from niriksha.core.runner import NO_RETRY
from niriksha.core.runstore import PromptTemplate, RunConfig, read_results
from niriksha.providers.openai_compatible import (
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
    ProviderConfigError,
)
from niriksha.scorers.artifacts import score_run_to_artifact, verify_artifact

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "datasets"
MIN_INTERVAL_S = 3.0
DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 300.0
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ENV_BASE_URL = "NIRIKSHA_SMOKE_BASE_URL"
ENV_MODEL = "NIRIKSHA_SMOKE_MODEL"
ENV_KEY_VARIABLE = "NIRIKSHA_SMOKE_KEY_VARIABLE"
ENV_MAX_TOKENS_FIELD = "NIRIKSHA_SMOKE_MAX_TOKENS_FIELD"
ENV_TIMEOUT = "NIRIKSHA_SMOKE_TIMEOUT_S"
ENV_TASKS = "NIRIKSHA_SMOKE_TASKS"
_SAFE_TOKEN = re.compile(r"^[a-z_]{1,32}$")


class SmokeError(Exception):
    """An expected refusal. Messages never contain the key or a URL that was supplied."""


class BudgetExceeded(SmokeError):
    pass


class SecretFound(SmokeError):
    pass


@dataclass(frozen=True)
class SmokeTask:
    label: str
    fixture: str
    dataset_name: str
    splits: tuple[str, ...]
    cases: int
    max_tokens: int
    prompt: PromptTemplate
    metrics: tuple[str, ...]


TASKS = (
    SmokeTask(
        label="qa",
        fixture="tiny_qa",
        dataset_name="tiny-qa",
        splits=("dev", "test", "holdout"),
        cases=5,
        max_tokens=32,
        prompt=PromptTemplate(
            system="Answer with only the answer, no explanation.", user="Question: {input}"
        ),
        metrics=("normalized_exact_match",),
    ),
    SmokeTask(
        label="extraction",
        fixture="tiny_extraction",
        dataset_name="tiny-extraction",
        splits=("dev", "test"),
        cases=3,
        max_tokens=128,
        prompt=PromptTemplate(
            system="Reply with only a JSON object, no other text.",
            user='Text: {input}\nExtract {"name": <string>, "age": <integer>}.',
        ),
        metrics=("json_parse_validity", "json_schema_validity", "field_exact_match"),
    ),
)


# -- configuration and plan -----------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeSettings:
    base_url: str
    model: str
    key_variable: str | None
    max_tokens_field: str
    timeout_s: float
    tasks: tuple[SmokeTask, ...]

    def profile(self) -> OpenAICompatibleProfile:
        # Only what the script sends is declared supported; anything else would be refused.
        return OpenAICompatibleProfile(
            name="live-smoke",
            base_url=self.base_url,
            api_key_env=self.key_variable,
            timeout_s=self.timeout_s,
            connect_timeout_s=10.0,
            supported_params=frozenset({"temperature", "max_tokens"}),
            max_tokens_field=self.max_tokens_field,
        )


def _host(url: str) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(url).hostname or "").lower()


def settings_from_env(environ: Mapping[str, str]) -> SmokeSettings:
    base_url = environ.get(ENV_BASE_URL, "").strip()
    model = environ.get(ENV_MODEL, "").strip()
    if not base_url:
        raise SmokeError(f"{ENV_BASE_URL} is not set")
    if not model:
        raise SmokeError(f"{ENV_MODEL} is not set")
    key_variable = environ.get(ENV_KEY_VARIABLE, "").strip() or None
    field_name = environ.get(ENV_MAX_TOKENS_FIELD, "max_tokens").strip()
    if field_name not in ("max_tokens", "max_completion_tokens"):
        raise SmokeError(f"{ENV_MAX_TOKENS_FIELD} must be max_tokens or max_completion_tokens")
    try:
        timeout_s = float(environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT_S))
    except ValueError:
        raise SmokeError(f"{ENV_TIMEOUT} must be a number") from None
    if not 0 < timeout_s <= MAX_TIMEOUT_S:
        raise SmokeError(f"{ENV_TIMEOUT} must be greater than 0 and at most {MAX_TIMEOUT_S:g}")
    wanted = [t.strip() for t in environ.get(ENV_TASKS, "").split(",") if t.strip()]
    labels = [task.label for task in TASKS]
    if not wanted:
        tasks = TASKS
    elif all(label in labels for label in wanted) and len(set(wanted)) == len(wanted):
        tasks = tuple(task for task in TASKS if task.label in wanted)
    else:
        raise SmokeError(f"{ENV_TASKS} must be a comma-separated subset of: {', '.join(labels)}")
    settings = SmokeSettings(base_url, model, key_variable, field_name, timeout_s, tasks)
    if _host(base_url) not in LOOPBACK_HOSTS and not base_url.lower().startswith("https://"):
        raise SmokeError("a remote endpoint must use https (plain http is only for loopback)")
    try:
        settings.profile()
    except ValidationError:
        raise SmokeError(
            f"{ENV_BASE_URL} or {ENV_KEY_VARIABLE} is not usable (an http(s) URL without "
            "credentials, query or fragment; a plain variable name)"
        ) from None
    return settings


@dataclass(frozen=True)
class PlannedRun:
    task: SmokeTask
    run_id: str

    @property
    def requests(self) -> int:
        return self.task.cases


@dataclass(frozen=True)
class Plan:
    settings: SmokeSettings
    runs: tuple[PlannedRun, ...]

    @property
    def total_requests(self) -> int:
        return sum(run.requests for run in self.runs)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_plan(settings: SmokeSettings, now: Callable[[], datetime] = _utcnow) -> Plan:
    stamp = now().strftime("%Y%m%dt%H%M%Sz")  # run ids are lower case
    runs = []
    for task in settings.tasks:
        dataset = load_dataset(FIXTURES / task.fixture)
        selected = dataset.select(task.splits)
        if dataset.meta.name != task.dataset_name or len(selected) != task.cases:
            raise SmokeError(f"fixture {task.fixture} is not the expected synthetic dataset")
        runs.append(PlannedRun(task, f"smoke-{task.label}-{stamp}"))
    return Plan(settings, tuple(runs))


def preflight(plan: Plan, environ: Mapping[str, str], runs_dir: Path) -> None:
    """Checks that need no network. Raises ``SmokeError`` naming the problem, never a value."""
    for run in plan.runs:
        if (runs_dir / run.run_id).exists():
            raise SmokeError(f"run {run.run_id} already exists; nothing is resumed or overwritten")
    try:
        OpenAICompatibleProvider(plan.settings.profile(), environ=environ).close()
    except ProviderConfigError as exc:
        raise SmokeError(str(exc)) from None


def describe_plan(plan: Plan) -> list[str]:
    s = plan.settings
    profile = s.profile()
    key_text = f"from the environment variable named {s.key_variable}" if s.key_variable else "none"
    caps = ", ".join(f"{r.task.max_tokens} ({r.task.label})" for r in plan.runs)
    lines = [
        "LIVE smoke test plan (real requests will be sent)",
        f"  endpoint : {profile.endpoint}",
        f"  model    : {s.model}",
        f"  key      : {key_text}",
        f"  requests : {plan.total_requests} in total, one attempt each, NO retries",
        f"  tokens   : temperature 0; max tokens {caps}",
        f"  spacing  : at least {MIN_INTERVAL_S:g} s between requests; timeout {s.timeout_s:g} s",
        "  data     : the synthetic test fixtures only (no real or personal data)",
        "  output   : runs/ and scores/ (gitignored); no model text is printed",
    ]
    lines += [f"  run      : {r.run_id} ({r.requests} requests)" for r in plan.runs]
    return lines


def confirm(plan: Plan, input_fn: Callable[[str], str]) -> None:
    count = plan.total_requests
    try:
        answer = input_fn(f"Type {count} to send {count} real requests; anything else aborts: ")
    except EOFError:
        answer = ""
    if answer.strip() != str(count):
        raise SmokeError("aborted: no request was sent")


# -- the budgeted provider ------------------------------------------------------------------------


class BudgetedProvider(OpenAICompatibleProvider):
    """The adapter plus a hard cap on calls and a minimum spacing between them.

    The cap is checked before the request is built, so call N+1 never reaches the network.
    """

    def __init__(
        self,
        profile: OpenAICompatibleProfile,
        *,
        environ: Mapping[str, str],
        max_calls: int,
        min_interval_s: float = MIN_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(profile, environ=environ)
        self._max_calls = max_calls
        self._min_interval_s = min_interval_s
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None
        self.calls = 0

    def generate(self, request):
        if self.calls >= self._max_calls:
            raise BudgetExceeded(f"the budget of {self._max_calls} requests is spent")
        if self._last is not None:
            wait = self._min_interval_s - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self.calls += 1
        self._last = self._clock()
        return super().generate(request)


# -- running and reporting ------------------------------------------------------------------------


@dataclass
class RunStats:
    run_id: str
    task: str
    cases: int
    ok: int = 0
    empty_outputs: int = 0  # successes whose output_text is exactly "" (failures never count)
    failed: Counter = field(default_factory=Counter)  # failure kind -> count
    returned_models: set[str] = field(default_factory=set)
    finish_reasons: Counter = field(default_factory=Counter)
    elapsed_s: list[float] = field(default_factory=list)
    total_tokens: int | None = None  # None: the server reported none
    scores: dict[str, tuple[float | None, int, int]] = field(default_factory=dict)


@dataclass
class SmokeSummary:
    requests_made: int
    runs: list[RunStats]


def scan_for_secret(directories: list[Path], key: str | None) -> None:
    """Raise ``SecretFound`` if the key is in any file under the directories."""
    if not key:
        return
    needles = {key.encode("utf-8"), json.dumps(key)[1:-1].encode("utf-8")}
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and any(needle in path.read_bytes() for needle in needles):
                raise SecretFound(
                    f"the API key was found in a persisted file ({path.name}); delete the "
                    "runs and scores directories and rotate the key"
                )


def _collect(run_dir: Path, stats: RunStats) -> None:
    for line in read_results(run_dir).lines:
        result = line.execution.result
        stats.elapsed_s.append(line.execution.elapsed_s)
        if isinstance(result, GenerationSuccess):
            stats.ok += 1
            if result.output_text == "":
                stats.empty_outputs += 1
            if result.returned_model:
                stats.returned_models.add(result.returned_model)
            reason = result.finish_reason
            stats.finish_reasons[reason if reason and _SAFE_TOKEN.match(reason) else "other"] += 1
            if result.usage is not None and result.usage.total_tokens is not None:
                stats.total_tokens = (stats.total_tokens or 0) + result.usage.total_tokens
        elif isinstance(result, GenerationFailure):
            stats.failed[result.kind.value] += 1


def run_smoke(
    plan: Plan,
    *,
    environ: Mapping[str, str],
    runs_dir: Path,
    scores_dir: Path,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> SmokeSummary:
    s = plan.settings
    key = environ.get(s.key_variable) if s.key_variable else None
    provider = BudgetedProvider(
        s.profile(),
        environ=environ,
        max_calls=plan.total_requests,
        sleep=sleep,
        clock=clock,
    )
    runs: list[RunStats] = []
    try:
        for planned in plan.runs:
            task = planned.task
            dataset_dir = FIXTURES / task.fixture
            config = RunConfig(
                run_id=planned.run_id,
                splits=task.splits,
                model=s.model,
                prompt=task.prompt,
                params=GenerationParams(temperature=0.0, max_tokens=task.max_tokens),
            )
            summary = execute_run(
                config,
                load_dataset(dataset_dir),
                provider,
                runs_dir,
                retry=NO_RETRY,
                secret_values=(key,) if key else (),
            )
            load_run(summary.run_dir, dataset_dir)  # the run must verify as a whole
            stats = RunStats(planned.run_id, task.label, task.cases)
            _collect(summary.run_dir, stats)
            for metric in task.metrics:
                outcome = score_run_to_artifact(summary.run_dir, dataset_dir, scores_dir, metric)
                verify_artifact(outcome.artifact, summary.run_dir, dataset_dir)
                agg = aggregate(outcome.artifact)
                stats.scores[metric] = (agg.mean, agg.scored, agg.not_scored)
            runs.append(stats)
    finally:
        provider.close()
        scan_for_secret([runs_dir, scores_dir], key)
    return SmokeSummary(provider.calls, runs)


def format_summary(summary: SmokeSummary) -> list[str]:
    lines = [f"requests made: {summary.requests_made}"]
    for stats in summary.runs:
        lines.append(f"run {stats.run_id} ({stats.task}, {stats.cases} cases)")
        failed = ", ".join(f"{kind} {n}" for kind, n in sorted(stats.failed.items())) or "none"
        lines.append(f"  results: {stats.ok} ok; failed: {failed}")
        lines.append(f"  empty outputs: {stats.empty_outputs} of {stats.ok} ok")
        if stats.elapsed_s:
            lines.append(
                f"  latency s: min {min(stats.elapsed_s):.2f}, "
                f"mean {sum(stats.elapsed_s) / len(stats.elapsed_s):.2f}, "
                f"max {max(stats.elapsed_s):.2f}"
            )
        models = ", ".join(sorted(stats.returned_models)) or "not reported"
        lines.append(f"  returned model: {models}")
        if stats.finish_reasons:
            reasons = ", ".join(f"{k} {n}" for k, n in sorted(stats.finish_reasons.items()))
            lines.append(f"  finish reasons: {reasons}")
        usage = "not reported" if stats.total_tokens is None else str(stats.total_tokens)
        lines.append(f"  reported total tokens: {usage}")
        for metric, (mean, scored, not_scored) in stats.scores.items():
            mean_text = "unavailable" if mean is None else f"{mean:.3f}"
            lines.append(f"  {metric}: mean {mean_text} ({scored} scored, {not_scored} not scored)")
    lines.append("outputs: runs/ and scores/ (gitignored). No model text was printed.")
    return lines


def main(
    environ: Mapping[str, str] | None = None,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], datetime] = _utcnow,
    runs_dir: Path | None = None,
    scores_dir: Path | None = None,
) -> int:
    environ = os.environ if environ is None else environ
    runs_dir = runs_dir or REPO / "runs"
    scores_dir = scores_dir or REPO / "scores"
    try:
        plan = build_plan(settings_from_env(environ), now)
        preflight(plan, environ, runs_dir)
        for line in describe_plan(plan):
            out(line)
        confirm(plan, input_fn)
        summary = run_smoke(
            plan, environ=environ, runs_dir=runs_dir, scores_dir=scores_dir, sleep=sleep
        )
    except SmokeError as exc:
        out(f"smoke test stopped: {exc}")
        return 2
    for line in format_summary(summary):
        out(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
