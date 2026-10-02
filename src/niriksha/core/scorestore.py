"""Score artifacts: one metric applied to one verified run, persisted apart from the run.

An artifact is a pure function of the run files, the dataset and the scorer version. It carries no
timestamp, so rewriting it gives identical bytes and "is this the same artifact" is a content
comparison. It lives in a separate ``scores/`` tree, never inside ``runs/<run_id>/``, which keeps
holding exactly ``manifest.json`` and ``results.jsonl``.

File: ``<scores_dir>/<run_id>--<metric>--<version>.json``, one pretty-printed JSON document::

    artifact_version   1
    artifact_id        "<run_id>--<metric>--<version>" (must equal the file name)
    source             run_id; SHA-256 of the exact bytes of the run's manifest.json and
                       results.jsonl; dataset identity; selection (case count, case-ID hash);
                       prompt hash; provider name; requested model
    metric             name, version, task
    records_sha256     see below
    records            ScoreRecord objects in selection order

``records_sha256`` is SHA-256 (lowercase hex) of the UTF-8 bytes of this canonical JSON text::

    json.dumps([record.model_dump(mode="json") for record in records],
               sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

that is, the JSON array of the records in artifact order, each record an object with the keys
``details``, ``metric``, ``metric_version``, ``reason``, ``request_id``, ``status`` and ``value``
in that sorted order, no whitespace, non-ASCII text as raw UTF-8, ``null`` for absent values and
floats in Python's shortest round-trip form (``1.0`` is written ``1.0``, never ``1``).

Integrity, not authenticity. The hashes detect accidental corruption and drift between an artifact
and the run, dataset and scorer it claims to come from. Anyone who can write the files can rewrite
the artifact and every hash consistently, so nothing here proves who produced a score.

Persistence follows the run store: exclusive create, flush and fsync, and a failed write removes
its partial file. A process killed mid-write can leave a truncated file; readers refuse it and it
blocks the name until it is removed by hand. An existing artifact is never overwritten.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from niriksha.core.dataset import Task, canonical_json_bytes, sha256_hex
from niriksha.core.generation import NonBlank
from niriksha.core.runload import LoadedRun
from niriksha.core.runstore import MANIFEST_FILE, RESULTS_FILE, ManifestDataset, RunId, ids_sha256
from niriksha.core.scoring import MetricName, MetricVersion, ScoreRecord

ARTIFACT_VERSION = 1
SEPARATOR = "--"
BYTE_ORDER_MARK = chr(0xFEFF)  # an invisible character: never write it literally in source
MAX_ARTIFACT_ID_LENGTH = 150  # keeps the path comfortably inside Windows limits
_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ScoreArtifactError(Exception):
    """A score artifact is missing, malformed, corrupt, unsupported or cannot be verified."""


class ScoreArtifactMismatchError(ScoreArtifactError):
    """A well-formed artifact does not match the run, dataset or scorer it is checked against."""


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


def artifact_id(run_id: str, metric: str, version: str) -> str:
    identifier = f"{run_id}{SEPARATOR}{metric}{SEPARATOR}{version}"
    if len(identifier) > MAX_ARTIFACT_ID_LENGTH:
        raise ValueError(f"artifact id is longer than {MAX_ARTIFACT_ID_LENGTH} characters")
    return identifier


def artifact_path(scores_dir: str | Path, identifier: str) -> Path:
    return Path(scores_dir) / f"{identifier}.json"


def records_hash(records: tuple[ScoreRecord, ...] | list[ScoreRecord]) -> str:
    """``records_sha256``: the canonical form is specified in the module docstring."""
    return sha256_hex(canonical_json_bytes([record.model_dump(mode="json") for record in records]))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- schema ---------------------------------------------------------------------------------------


class ArtifactSelection(_Model):
    case_count: int = Field(ge=1)
    case_ids_sha256: _Sha256


class ArtifactSource(_Model):
    run_id: RunId
    run_manifest_sha256: _Sha256
    run_results_sha256: _Sha256
    dataset: ManifestDataset
    selection: ArtifactSelection
    prompt_sha256: _Sha256
    provider: NonBlank
    requested_model: NonBlank


class ArtifactMetric(_Model):
    name: MetricName
    version: MetricVersion
    task: Task


class ScoreArtifact(_Model):
    artifact_version: int
    artifact_id: NonBlank
    source: ArtifactSource
    metric: ArtifactMetric
    records_sha256: _Sha256
    records: tuple[ScoreRecord, ...]

    @model_validator(mode="after")
    def _self_consistent(self):
        if self.artifact_version != ARTIFACT_VERSION:
            raise ValueError(f"unsupported artifact_version {self.artifact_version}")
        try:
            expected = artifact_id(self.source.run_id, self.metric.name, self.metric.version)
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if self.artifact_id != expected:
            raise ValueError("artifact_id does not match the source run and metric")
        ids = [record.request_id for record in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("records contain a duplicate request_id")
        if len(ids) != self.source.selection.case_count:
            raise ValueError("the number of records differs from the selection case count")
        if ids_sha256(ids) != self.source.selection.case_ids_sha256:
            raise ValueError("the record case IDs do not match the selection hash")
        for record in self.records:
            if (record.metric, record.metric_version) != (self.metric.name, self.metric.version):
                raise ValueError("a record belongs to a different metric or metric version")
        if records_hash(self.records) != self.records_sha256:
            raise ValueError("records_sha256 does not match the records")
        return self


def source_from_run(run: LoadedRun) -> ArtifactSource:
    """The identity block for a run that already passed ``load_run``.

    The file hashes are taken from the run's files at this moment. A run is immutable once
    complete, so they only change if someone edits the files.
    """
    manifest = run.manifest
    return ArtifactSource(
        run_id=manifest.run_id,
        run_manifest_sha256=file_sha256(run.run_dir / MANIFEST_FILE),
        run_results_sha256=file_sha256(run.run_dir / RESULTS_FILE),
        dataset=manifest.dataset,
        selection=ArtifactSelection(
            case_count=manifest.selection.case_count,
            case_ids_sha256=manifest.selection.case_ids_sha256,
        ),
        prompt_sha256=manifest.prompt.sha256,
        provider=manifest.provider.name,
        requested_model=manifest.requested_model,
    )


def build_artifact(
    source: ArtifactSource, name: str, version: str, task: str, records: tuple[ScoreRecord, ...]
) -> ScoreArtifact:
    return ScoreArtifact(
        artifact_version=ARTIFACT_VERSION,
        artifact_id=artifact_id(source.run_id, name, version),
        source=source,
        metric=ArtifactMetric(name=name, version=version, task=task),
        records_sha256=records_hash(records),
        records=records,
    )


# -- persistence ----------------------------------------------------------------------------------


def serialize_artifact(artifact: ScoreArtifact) -> bytes:
    """The exact bytes written: indented JSON in field order, a final newline, UTF-8."""
    return (artifact.model_dump_json(indent=2) + "\n").encode("utf-8")


def write_artifact(scores_dir: str | Path, artifact: ScoreArtifact) -> Path:
    """Create the artifact file exclusively; raises ``FileExistsError`` if it already exists.

    Never overwrites, and never deletes a file it did not create: if the exclusive create itself
    fails for any reason (including a reason other than "exists"), whatever is at the path is left
    alone. Only a failure *after* this call created the file removes it (the partial file).

    Raises ``ScoreArtifactError`` if ``scores_dir`` cannot be used as a directory (for example it
    is a regular file) or if a directory occupies the artifact's path.
    """
    scores_dir = Path(scores_dir)
    try:
        scores_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if scores_dir.is_dir():
            raise
        raise ScoreArtifactError(
            f"scores_dir {scores_dir.name!r} cannot be created or is not a directory"
        ) from exc
    path = artifact_path(scores_dir, artifact.artifact_id)
    if path.is_dir():
        raise ScoreArtifactError(f"{path.name}: a directory occupies the artifact path")
    data = serialize_artifact(artifact)
    created = False
    try:
        with open(path, "xb") as handle:
            created = True  # from here on this call owns the file
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path


class _DuplicateKey(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _describe(error: ValidationError) -> str:
    """Locations and messages only, never the input values."""
    items = error.errors(include_input=False, include_url=False, include_context=False)
    return "; ".join(f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}" for e in items)


def read_artifact(path: str | Path) -> ScoreArtifact:
    """Read and strictly validate one artifact, without needing the run or any provider.

    Refuses: a missing file, invalid UTF-8, a byte-order mark, malformed JSON, trailing data,
    duplicate keys, NaN or Infinity, an unsupported ``artifact_version``, unknown or missing
    fields, invalid scores, duplicate, missing or unexpected case IDs, a mismatched
    ``records_sha256`` or ``artifact_id``, and a file whose name is not ``<artifact_id>.json``.
    """
    path = Path(path)
    if path.is_dir():
        raise ScoreArtifactError(f"{path.name}: is a directory, not an artifact file")
    if not path.is_file():
        raise ScoreArtifactError(f"{path.name}: not found")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ScoreArtifactError(f"{path.name}: invalid UTF-8 at byte {exc.start}") from exc
    if text.startswith(BYTE_ORDER_MARK):
        raise ScoreArtifactError(f"{path.name}: a byte-order mark is not allowed")
    try:
        document = json.loads(
            text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant
        )
    except _DuplicateKey as exc:
        raise ScoreArtifactError(f"{path.name}: duplicate JSON key") from exc
    except ValueError as exc:
        raise ScoreArtifactError(f"{path.name}: invalid JSON") from exc
    if not isinstance(document, dict):
        raise ScoreArtifactError(f"{path.name}: the document must be a JSON object")
    version = document.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ScoreArtifactError(
            f"{path.name}: unsupported artifact_version (this build reads {ARTIFACT_VERSION})"
        )
    problem = None
    try:
        artifact = ScoreArtifact.model_validate_json(text)
    except ValidationError as exc:
        problem = f"{path.name}: invalid artifact ({_describe(exc)})"
    if problem:  # raised outside the except block so no input values are chained
        raise ScoreArtifactError(problem)
    if path.name != f"{artifact.artifact_id}.json":
        raise ScoreArtifactError(f"{path.name}: the file name does not match its artifact_id")
    return artifact
