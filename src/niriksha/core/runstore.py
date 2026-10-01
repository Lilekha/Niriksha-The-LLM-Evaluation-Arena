"""Run configuration, manifest and on-disk run storage.

Layout::

    runs/<run_id>/manifest.json   written once with exclusive create, never modified
    runs/<run_id>/results.jsonl   append-only, one line per completed request

See docs/adr/0003-dataset-identity-and-run-persistence.md for the rationale and the limits.
The manifest is a provenance record: it says what was run and with what, not that the run can be
reproduced bit for bit.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from niriksha.core.dataset import NonBlank, Split, Task, canonical_json_bytes, sha256_hex
from niriksha.core.generation import GenerationParams, GenerationRequest, Message
from niriksha.core.provenance import SoftwareInfo
from niriksha.core.runner import ExecutionRecord

MANIFEST_VERSION = 1
MANIFEST_FILE = "manifest.json"
RESULTS_FILE = "results.jsonl"
MIN_SECRET_LENGTH = 8
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)}
)
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class RunStoreError(Exception):
    """Base class for run storage problems."""


class CorruptRunError(RunStoreError):
    """A stored run file is malformed. The message includes the line number where known."""


class ResumeError(RunStoreError):
    """The run cannot be resumed with the given inputs."""


class PersistenceError(RunStoreError):
    """A result was refused at the persistence boundary. Messages never include the value."""


# -- configuration --------------------------------------------------------------------------------


def validate_run_id(value: str) -> str:
    """Lowercase letters, digits, ``_`` and ``-`` (1-64 chars, not starting with ``_``/``-``).

    Lowercase only, so names cannot collide on a case-insensitive filesystem; no dots, so no
    trailing-dot or extension tricks; Windows reserved device names are rejected.
    """
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value):
        raise ValueError("run_id must match [a-z0-9][a-z0-9_-]{0,63}")
    if value in _RESERVED_NAMES:
        raise ValueError("run_id is a reserved device name on Windows")
    return value


RunId = Annotated[str, AfterValidator(validate_run_id)]


class PromptTemplate(_Model):
    """``user`` must contain ``{input}`` exactly once. It is replaced with ``str.replace`` (not
    ``str.format``), so literal braces such as JSON in the template stay as they are. ``system`` is
    not templated."""

    system: NonBlank | None = None
    user: NonBlank

    @field_validator("user")
    @classmethod
    def _one_placeholder(cls, value: str) -> str:
        if value.count("{input}") != 1:
            raise ValueError("user template must contain {input} exactly once")
        return value

    def render(self, input_text: str) -> tuple[Message, ...]:
        messages = [Message(role="user", content=self.user.replace("{input}", input_text))]
        if self.system is not None:
            messages.insert(0, Message(role="system", content=self.system))
        return tuple(messages)

    @property
    def sha256(self) -> str:
        return sha256_hex(canonical_json_bytes({"system": self.system, "user": self.user}))


class RunConfig(_Model):
    run_id: RunId
    splits: tuple[Split, ...] = Field(min_length=1)
    model: NonBlank
    prompt: PromptTemplate
    params: GenerationParams = GenerationParams()

    @field_validator("splits")
    @classmethod
    def _unique_splits(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("splits must not repeat")
        return value


# -- manifest -------------------------------------------------------------------------------------


class ManifestDataset(_Model):
    name: NonBlank
    version: NonBlank
    task: Task
    schema_version: int
    content_sha256: _Sha256


class ManifestSelection(_Model):
    splits: tuple[Split, ...]
    case_count: int = Field(ge=1)
    case_ids_sha256: _Sha256  # hash of the selected case ids, in dataset order


class ManifestPrompt(_Model):
    system: str | None
    user: str
    sha256: _Sha256


class ManifestProvider(_Model):
    name: NonBlank
    implementation: NonBlank  # module-qualified class name; never a path


class RunManifest(_Model):
    manifest_version: int
    run_id: RunId
    created_at: AwareDatetime
    dataset: ManifestDataset
    selection: ManifestSelection
    prompt: ManifestPrompt
    provider: ManifestProvider
    requested_model: NonBlank
    params: GenerationParams
    software: SoftwareInfo

    @field_validator("manifest_version")
    @classmethod
    def _known_version(cls, value: int) -> int:
        if value != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest_version {value}")
        return value


class RunResultLine(_Model):
    request_id: NonBlank
    request_sha256: _Sha256
    recorded_at: AwareDatetime  # wall clock, informational only
    execution: ExecutionRecord

    @model_validator(mode="after")
    def _ids_agree(self):
        if self.execution.result.request_id != self.request_id:
            raise ValueError("request_id does not match the stored result")
        return self


def request_sha256(request: GenerationRequest) -> str:
    return sha256_hex(canonical_json_bytes(request.model_dump(mode="json")))


def ids_sha256(case_ids: list[str]) -> str:
    return sha256_hex(canonical_json_bytes(case_ids))


# -- storage --------------------------------------------------------------------------------------


def _describe(error: ValidationError) -> str:
    """Locations and messages only. Never the input values, which may be sensitive."""
    items = error.errors(include_input=False, include_url=False, include_context=False)
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in items)


def run_dir_path(runs_dir: str | Path, run_id: str) -> Path:
    return Path(runs_dir) / validate_run_id(run_id)


def _write_new_file(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:  # "x": fail if it exists
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _refuse_secret_values(serialized: str, secret_values: tuple[str, ...], what: str) -> None:
    for secret in secret_values:
        forms = {secret, json.dumps(secret)[1:-1], json.dumps(secret, ensure_ascii=False)[1:-1]}
        if any(form in serialized for form in forms):
            raise PersistenceError(
                f"{what} contains a caller-supplied secret value; nothing written"
            )


def create_run(
    runs_dir: str | Path, manifest: RunManifest, *, secret_values: tuple[str, ...] = ()
) -> Path:
    """Claim ``runs/<run_id>/`` and write the manifest once. Never overwrites.

    Raises ``FileExistsError`` if the run directory already exists. Creating the directory is the
    claim, so two callers cannot both start the same run. A manifest containing one of
    ``secret_values`` (for example an API key pasted into the prompt template) is refused before
    anything is created; see ``append_result`` for what that guard is and is not.

    Directory-entry durability varies by platform (Windows cannot fsync a directory), so a crash
    right after creation may lose the run. A crash between creating the directory and finishing the
    manifest leaves an empty directory that blocks the run ID until it is removed by hand.
    """
    check_secret_values(secret_values)
    text = manifest.model_dump_json(indent=2)
    _refuse_secret_values(text, secret_values, "manifest")
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir_path(runs_dir, manifest.run_id)
    os.mkdir(run_dir)
    try:
        _write_new_file(run_dir / MANIFEST_FILE, (text + "\n").encode())
    except BaseException:
        (run_dir / MANIFEST_FILE).unlink(missing_ok=True)
        run_dir.rmdir()
        raise
    return run_dir


def check_secret_values(secret_values: tuple[str, ...]) -> None:
    for secret in secret_values:
        if len(secret) < MIN_SECRET_LENGTH:
            raise ValueError(
                f"secret guard values must be at least {MIN_SECRET_LENGTH} characters "
                "(shorter values would match ordinary text)"
            )


def append_result(
    run_dir: str | Path, line: RunResultLine, *, secret_values: tuple[str, ...] = ()
) -> None:
    """Revalidate ``line`` and append it to ``results.jsonl``, flushed and fsynced.

    A failed append leaves the file as it was: if writing, flushing or fsyncing raises, the new
    bytes are truncated away (best effort) so no partial or unconfirmed line remains. The result is
    then lost, and resuming issues that provider call again. An append is also refused, with
    nothing written, if the file already ends with an incomplete line, so a torn tail can never be
    merged into a corrupt middle line.

    Revalidation re-parses the serialised line, so every validator runs again. That catches data
    mutated after construction (a frozen model can still hold a mutable dict), such as a
    ``provider_metadata`` entry added later.

    ``secret_values`` is a backstop for values the caller holds in memory (for example API keys read
    from the environment): a line containing one is refused. It is not a secret detector and will
    not catch a secret the caller did not list, or one that has been transformed. Neither the
    refusal nor any error message includes the secret.
    """
    check_secret_values(secret_values)
    # warnings=False: pydantic's serializer warnings would print the offending values.
    serialized = line.model_dump_json(warnings=False)
    problem = None
    try:
        revalidated = RunResultLine.model_validate_json(serialized)
    except ValidationError as exc:
        problem = f"result failed revalidation; nothing written: {_describe(exc)}"
    else:
        if revalidated != line:  # e.g. pydantic writes a mutated NaN as null
            problem = "result changed when serialised; nothing written"
    if problem:  # raised outside the except block: no ValidationError (and its input) is chained
        raise PersistenceError(problem)
    _refuse_secret_values(serialized, secret_values, "result")

    data = serialized.encode("utf-8") + b"\n"  # bytes: no newline translation on Windows
    with open(Path(run_dir) / RESULTS_FILE, "a+b") as handle:
        size_before = handle.seek(0, os.SEEK_END)
        if size_before:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                raise PersistenceError(
                    f"{RESULTS_FILE} ends with an incomplete line; nothing written. "
                    "Resume the run to repair it"
                )
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            try:
                handle.truncate(size_before)  # best effort: leave the file as it was
            except OSError:
                pass
            raise
    # ponytail: reopens the file per result; fine for hundreds of results, keep one handle if not


def read_manifest(run_dir: str | Path) -> RunManifest:
    path = Path(run_dir) / MANIFEST_FILE
    if not path.is_file():
        raise CorruptRunError(f"{MANIFEST_FILE}: not found")
    try:
        return RunManifest.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise CorruptRunError(f"{MANIFEST_FILE}: invalid ({_describe(exc)})") from None


@dataclass(frozen=True)
class ResultsRead:
    lines: tuple[RunResultLine, ...]
    torn_tail_bytes: int  # bytes after the last newline; 0 when the file ends cleanly


def read_results(run_dir: str | Path, *, allow_torn_tail: bool = False) -> ResultsRead:
    """Read ``results.jsonl`` strictly. Never modifies the file.

    A final line without a newline may be a torn write: it is reported through ``torn_tail_bytes``
    when ``allow_torn_tail`` is set, and is an error otherwise. Corruption anywhere else, a blank
    line, or a duplicate ``request_id`` raises ``CorruptRunError`` with the line number.
    """
    path = Path(run_dir) / RESULTS_FILE
    if not path.exists():
        return ResultsRead((), 0)
    parts = path.read_bytes().split(b"\n")
    tail = parts.pop()  # b"" if the file is empty or ends with a newline
    if tail and not allow_torn_tail:
        raise CorruptRunError(
            f"{RESULTS_FILE} line {len(parts) + 1}: no trailing newline (possible torn write)"
        )
    lines: list[RunResultLine] = []
    first_line: dict[str, int] = {}
    for number, raw in enumerate(parts, start=1):
        if not raw.strip():
            raise CorruptRunError(f"{RESULTS_FILE} line {number}: blank line")
        try:
            line = RunResultLine.model_validate_json(raw)
        except ValidationError as exc:
            raise CorruptRunError(f"{RESULTS_FILE} line {number}: {_describe(exc)}") from None
        if line.request_id in first_line:
            raise CorruptRunError(
                f"{RESULTS_FILE} line {number}: duplicate request_id {line.request_id!r} "
                f"(first on line {first_line[line.request_id]})"
            )
        first_line[line.request_id] = number
        lines.append(line)
    return ResultsRead(tuple(lines), len(tail))


def truncate_torn_tail(run_dir: str | Path, torn_tail_bytes: int) -> None:
    """Remove an incomplete final line, as reported by ``read_results``."""
    if torn_tail_bytes <= 0:
        return
    with open(Path(run_dir) / RESULTS_FILE, "r+b") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.truncate(size - torn_tail_bytes)
        handle.flush()
        os.fsync(handle.fileno())
