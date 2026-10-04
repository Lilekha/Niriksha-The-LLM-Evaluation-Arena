"""Run orchestration: execute a dataset against a provider, persist results, and resume.

``execute_run`` and ``resume_run`` build and validate every request before the first provider call,
call the provider once per case through ``run_one`` (no retries, sequential), and append each result
to the run's ``results.jsonl`` as it completes.

Resume semantics and their limits are in docs/adr/0003-dataset-identity-and-run-persistence.md:

- only selected cases with no recorded result are run; a recorded failure counts as done;
- everything that defines the run must match the immutable manifest, or resume is refused;
- a call that was in flight when the process died is issued again, which with a real provider
  could be a duplicate billable call;
- one writer at a time is assumed; there is no locking.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from niriksha.core.dataset import Case, Dataset
from niriksha.core.generation import GenerationRequest
from niriksha.core.provenance import SoftwareInfo, collect_software_info
from niriksha.core.provider import Provider
from niriksha.core.runner import NO_RETRY, RetryPolicy, Sleep, run_one
from niriksha.core.runstore import (
    ManifestDataset,
    ManifestPrompt,
    ManifestProvider,
    ManifestSelection,
    ResumeError,
    RunConfig,
    RunManifest,
    RunResultLine,
    append_result,
    check_secret_values,
    create_run,
    ids_sha256,
    read_manifest,
    read_results,
    request_sha256,
    run_dir_path,
    truncate_torn_tail,
)

Clock = Callable[[], float]
Now = Callable[[], datetime]


@dataclass(frozen=True)
class RunSummary:
    run_dir: Path
    total: int  # selected cases
    already_recorded: int  # results found on disk before this call (0 for execute_run)
    executed: int  # provider calls made by this call


def _utcnow() -> datetime:
    return datetime.now(UTC)


def build_requests(config: RunConfig, dataset: Dataset) -> list[tuple[Case, GenerationRequest]]:
    """One request per selected case, in dataset order. Raises before any provider call."""
    cases = dataset.select(config.splits)
    if not cases:
        raise ValueError(f"no cases in the selected splits {list(config.splits)}")
    return [
        (
            case,
            GenerationRequest(
                request_id=case.id,
                model=config.model,
                messages=config.prompt.render(case.input_text),
                params=config.params,
            ),
        )
        for case in cases
    ]


def build_manifest(
    config: RunConfig,
    dataset: Dataset,
    provider: Provider,
    pairs: list[tuple[Case, GenerationRequest]],
    software: SoftwareInfo,
    created_at: datetime,
) -> RunManifest:
    return RunManifest(
        manifest_version=1,
        run_id=config.run_id,
        created_at=created_at,
        dataset=ManifestDataset(
            name=dataset.meta.name,
            version=dataset.meta.version,
            task=dataset.meta.task,
            schema_version=dataset.meta.schema_version,
            content_sha256=dataset.content_sha256,
        ),
        selection=ManifestSelection(
            splits=config.splits,
            case_count=len(pairs),
            case_ids_sha256=ids_sha256([case.id for case, _ in pairs]),
        ),
        prompt=ManifestPrompt(
            system=config.prompt.system, user=config.prompt.user, sha256=config.prompt.sha256
        ),
        provider=ManifestProvider(
            name=provider.name,
            implementation=f"{type(provider).__module__}.{type(provider).__qualname__}",
        ),
        requested_model=config.model,
        params=config.params,
        software=software,
    )


def _run_pending(
    run_dir: Path,
    provider: Provider,
    pairs: list[tuple[Case, GenerationRequest]],
    done: set[str],
    clock: Clock,
    now: Now,
    secret_values: tuple[str, ...],
    retry: RetryPolicy,
    sleep: Sleep,
) -> int:
    executed = 0
    for _, request in pairs:
        if request.request_id in done:
            continue
        record = run_one(provider, request, clock=clock, retry=retry, sleep=sleep)
        line = RunResultLine(
            request_id=request.request_id,
            request_sha256=request_sha256(request),
            recorded_at=now(),
            execution=record,
        )
        append_result(run_dir, line, secret_values=secret_values)
        executed += 1
    return executed


def execute_run(
    config: RunConfig,
    dataset: Dataset,
    provider: Provider,
    runs_dir: str | Path,
    *,
    clock: Clock = time.perf_counter,
    now: Now = _utcnow,
    software: SoftwareInfo | None = None,
    secret_values: tuple[str, ...] = (),
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = time.sleep,
) -> RunSummary:
    """Start a new run. Raises ``FileExistsError`` if ``config.run_id`` already exists.

    ``retry`` is the explicit retry policy (default: one attempt, no retries). It is not recorded in
    the manifest; each retried result records its own attempts.

    Any exception from the provider (a programming error, or a contract violation) propagates and
    leaves the results recorded so far on disk, so the run can be resumed.
    """
    check_secret_values(secret_values)
    pairs = build_requests(config, dataset)  # validates everything before any side effect
    manifest = build_manifest(
        config, dataset, provider, pairs, software or collect_software_info(), now()
    )
    run_dir = create_run(runs_dir, manifest, secret_values=secret_values)
    executed = _run_pending(
        run_dir, provider, pairs, set(), clock, now, secret_values, retry, sleep
    )
    return RunSummary(run_dir, total=len(pairs), already_recorded=0, executed=executed)


def _manifest_differences(expected: RunManifest, stored: RunManifest) -> list[str]:
    """Names of the fields that define the run and differ. Values are not echoed."""
    differences = []
    pairs = [
        ("run_id", expected.run_id, stored.run_id),
        ("dataset", expected.dataset, stored.dataset),
        ("selection", expected.selection, stored.selection),
        ("prompt", expected.prompt, stored.prompt),
        ("provider", expected.provider, stored.provider),
        ("requested_model", expected.requested_model, stored.requested_model),
        ("params", expected.params, stored.params),
    ]
    for name, want, have in pairs:
        if want != have:
            differences.append(name)
    return differences


def _software_differences(current: SoftwareInfo, stored: SoftwareInfo) -> list[str]:
    differences = [
        f"software.{name}"
        for name in ("niriksha_version", "python_version", "pydantic_version", "platform")
        if getattr(current, name) != getattr(stored, name)
    ]
    # The dirty flag is ignored: it changes constantly during development. The commits must be
    # equal: a known commit versus an unknown one is a difference (the code can no longer be tied
    # to the same revision), while two unknown commits are not.
    if current.git_commit != stored.git_commit:
        differences.append("software.git_commit")
    return differences


def resume_run(
    config: RunConfig,
    dataset: Dataset,
    provider: Provider,
    runs_dir: str | Path,
    *,
    clock: Clock = time.perf_counter,
    now: Now = _utcnow,
    software: SoftwareInfo | None = None,
    secret_values: tuple[str, ...] = (),
    retry: RetryPolicy = NO_RETRY,
    sleep: Sleep = time.sleep,
) -> RunSummary:
    """Run the selected cases that have no recorded result yet.

    Refuses (``ResumeError``) unless the dataset, selection, prompt, provider, model, parameters and
    software versions all match the stored manifest. Nothing on disk is changed until every check
    has passed; then an incomplete final line (a possible torn write) is truncated, and its case is
    run again because it has no complete result. Recorded failures are not retried; ``retry``
    applies only to the cases that are run now.
    """
    check_secret_values(secret_values)
    pairs = build_requests(config, dataset)
    run_dir = run_dir_path(runs_dir, config.run_id)
    if not run_dir.is_dir():
        raise ResumeError("run directory does not exist; use execute_run to start a run")
    stored = read_manifest(run_dir)
    current = software or collect_software_info()
    expected = build_manifest(config, dataset, provider, pairs, current, stored.created_at)

    differences = _manifest_differences(expected, stored)
    differences += _software_differences(current, stored.software)
    if differences:
        raise ResumeError(
            "cannot resume: the run no longer matches its manifest ("
            + ", ".join(differences)
            + "). Start a new run instead."
        )

    results = read_results(run_dir, allow_torn_tail=True)
    by_id = {request.request_id: request for _, request in pairs}
    for line in results.lines:
        request = by_id.get(line.request_id)
        if request is None:
            raise ResumeError(
                f"results contain request_id {line.request_id!r}, which is not in the selection"
            )
        if line.request_sha256 != request_sha256(request):
            raise ResumeError(
                f"stored request hash for {line.request_id!r} does not match the rebuilt request "
                "(the dataset or prompt rendering changed)"
            )
        result = line.execution.result
        if (
            result.provider != stored.provider.name
            or result.requested_model != stored.requested_model
        ):
            raise ResumeError(
                f"stored result for {line.request_id!r} was produced by a different provider or "
                "model than the manifest records"
            )

    truncate_torn_tail(run_dir, results.torn_tail_bytes)
    done = {line.request_id for line in results.lines}
    executed = _run_pending(run_dir, provider, pairs, done, clock, now, secret_values, retry, sleep)
    return RunSummary(run_dir, total=len(pairs), already_recorded=len(done), executed=executed)
