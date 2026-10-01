"""Support code for the end-to-end tests: crash doubles, a child-process entry point, a run audit.

The crash provider has to live in an importable module and be imported under the same name by the
test process and by the child process: a run manifest records the provider as ``module.qualname``,
and resume refuses a provider whose recorded implementation differs.
"""

import math
import os
import socket
import sys
from pathlib import Path

from niriksha.core.dataset import load_dataset
from niriksha.core.execution import build_requests, execute_run
from niriksha.core.generation import GenerationParams, GenerationSuccess
from niriksha.core.runstore import (
    MANIFEST_FILE,
    RESULTS_FILE,
    PromptTemplate,
    RunConfig,
    ids_sha256,
    read_manifest,
    read_results,
    request_sha256,
)
from niriksha.providers.fake import FakeProvider

CHILD_ENV_FLAG = "NIRIKSHA_E2E_CHILD"
SKIP_GUARD_FLAG = "NIRIKSHA_E2E_SKIP_GUARD"  # test-only: lets a negative control omit the guard
CRASH_EXIT_CODE = 137
GUARD_MISSING_EXIT_CODE = 86  # the child found no network guard when it first called the provider
TORN_FRAGMENT = b'{"request_id": "torn", "request_sha256": "ab'  # an incomplete JSONL record


def make_config(run_id: str, splits=("dev",)) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        splits=tuple(splits),
        model="m",
        prompt=PromptTemplate(user="Answer: {input}"),
        params=GenerationParams(temperature=0.0),
    )


class Crash(Exception):
    """Raised by ``CrashingProvider`` to model a crash inside the test process."""


class CrashingProvider(FakeProvider):
    """A FakeProvider that dies after ``crash_after`` completed calls.

    In-process it raises ``Crash``. With ``hard_exit=True`` it ends the process immediately with
    ``os._exit``, which models a killed process (no cleanup, no exception handling), after
    optionally appending an incomplete record to ``torn_path``. Hard exit is only honoured inside
    the child process started by the tests, so a mistake cannot terminate pytest itself.

    With ``require_network_guard=True`` (child only) every call first checks that the network guard
    is installed and exits with ``GUARD_MISSING_EXIT_CODE`` if it is not. Because the check runs
    inside ``execute_run``, it proves the guard was installed before the evaluation began.
    """

    def __init__(
        self,
        *args,
        crash_after=None,
        torn_path=None,
        hard_exit=False,
        require_network_guard=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (hard_exit or require_network_guard) and os.environ.get(CHILD_ENV_FLAG) != "1":
            raise RuntimeError("this option is only allowed inside the end-to-end child process")
        self.crash_after, self.torn_path, self.hard_exit = crash_after, torn_path, hard_exit
        self.require_network_guard = require_network_guard

    def generate(self, request):
        if self.require_network_guard and not network_guard_is_active():
            os._exit(GUARD_MISSING_EXIT_CODE)
        if self.crash_after is not None and len(self.calls) == self.crash_after:
            if self.hard_exit:
                if self.torn_path is not None:
                    with open(self.torn_path, "ab") as handle:
                        handle.write(TORN_FRAGMENT)
                os._exit(CRASH_EXIT_CODE)
            raise Crash("simulated crash")
        return super().generate(request)


class ViolatingProvider(FakeProvider):
    """Breaks the provider contract for one case, in one of four ways."""

    def __init__(self, *args, violate_on, kind, **kwargs):
        super().__init__(*args, **kwargs)
        self.violate_on, self.kind = violate_on, kind

    def generate(self, request):
        result = super().generate(request)
        if request.request_id != self.violate_on:
            return result
        if self.kind == "not_a_result":
            return {"status": "ok"}
        field = {
            "request_id": "someone-else",
            "requested_model": "another-model",
            "provider": "impostor",
        }[self.kind]
        return result.model_copy(update={self.kind: field})


def block_network() -> None:
    """The same guard as tests/conftest.py. That fixture patches only the pytest process, so the
    child interpreter installs its own to keep the end-to-end tests offline there too."""

    def blocked(*args, **kwargs):
        raise RuntimeError("network access is not allowed in tests")

    socket.socket.connect = blocked
    socket.getaddrinfo = blocked


def network_guard_is_active() -> bool:
    """True if ``block_network`` (or the conftest guard) is installed in this process.

    The probes use invalid arguments, so without the guard they fail locally on argument
    validation (``TypeError``, ``gaierror``) and nothing is sent anywhere; with it they raise the
    guard's ``RuntimeError`` first.
    """

    def intercepted(call) -> bool:
        try:
            call()
        except RuntimeError as error:
            return "network access is not allowed" in str(error)
        except Exception:
            return False  # failed for an ordinary reason: the guard did not intercept the call
        return False

    def connect_to_a_malformed_address():
        with socket.socket() as sock:
            sock.connect("not-an-address")

    return intercepted(connect_to_a_malformed_address) and intercepted(
        lambda: socket.getaddrinfo(None, None)
    )


def child_main(argv: list[str]) -> int:
    """Run ``execute_run`` through the public API with the real clock, time and software info,
    crashing hard after ``crash_after`` results. Arguments: dataset_dir runs_dir run_id
    crash_after torn(0|1) splits(comma separated).

    The network guard is installed first, before anything is loaded or run. ``SKIP_GUARD_FLAG``
    omits it, solely so a negative control can show that the provider's probe notices."""
    if os.environ.get(SKIP_GUARD_FLAG) != "1":
        block_network()
    dataset_dir, runs_dir, run_id, crash_after, torn, splits = argv
    results_path = Path(runs_dir) / run_id / RESULTS_FILE
    provider = CrashingProvider(
        crash_after=int(crash_after),
        torn_path=results_path if torn == "1" else None,
        hard_exit=True,
        require_network_guard=True,
    )
    execute_run(
        make_config(run_id, splits.split(",")), load_dataset(dataset_dir), provider, runs_dir
    )
    return 0


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def audit_run(run_dir: Path, config: RunConfig, dataset_dir: Path, *, expect_complete=True) -> None:
    """Recompute everything checkable about a finished run from its files and the dataset on disk.

    Raises AssertionError naming the first problem. Uses only the public APIs a scorer would use.
    """
    run_dir = Path(run_dir)
    names = {path.name for path in run_dir.iterdir()}
    _check(
        names <= {MANIFEST_FILE, RESULTS_FILE} and MANIFEST_FILE in names,
        f"persisted files are not exactly the documented set: {sorted(names)}",
    )

    manifest = read_manifest(run_dir)
    results = read_results(run_dir)  # strict: no torn tail, no corruption, no duplicate IDs
    dataset = load_dataset(dataset_dir)  # verifies the pin against the cases on disk
    pairs = build_requests(config, dataset)

    _check(manifest.run_id == config.run_id == run_dir.name, "run_id disagrees")
    _check(manifest.dataset.content_sha256 == dataset.content_sha256, "dataset hash disagrees")
    _check(manifest.dataset.version == dataset.meta.version, "dataset version disagrees")
    _check(manifest.prompt.sha256 == config.prompt.sha256, "prompt hash disagrees")
    _check(manifest.selection.case_count == len(pairs), "selection case count disagrees")
    _check(
        manifest.selection.case_ids_sha256 == ids_sha256([c.id for c, _ in pairs]),
        "selection hash disagrees",
    )

    expected_ids = [case.id for case, _ in pairs]
    stored_ids = [line.request_id for line in results.lines]
    if expect_complete:
        _check(stored_ids == expected_ids, "result count or order differs from the selection")
    else:
        _check(stored_ids == expected_ids[: len(stored_ids)], "results are not a selection prefix")

    by_id = {request.request_id: request for _, request in pairs}
    for line in results.lines:
        result = line.execution.result
        _check(
            line.request_sha256 == request_sha256(by_id[line.request_id]), "request hash differs"
        )
        _check(result.provider == manifest.provider.name, "stored provider differs from manifest")
        _check(result.requested_model == manifest.requested_model, "stored model differs")
        _check(math.isfinite(line.execution.elapsed_s) and line.execution.elapsed_s >= 0, "elapsed")
        if isinstance(result, GenerationSuccess):
            _check(result.usage is None, "the fake provider must not report usage")

    text = "".join((run_dir / name).read_text(encoding="utf-8") for name in names)
    _check(str(run_dir.parent) not in text and str(Path.home()) not in text, "a local path leaked")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(child_main(sys.argv[1:]))
