"""Offline run reader: load a persisted run together with its dataset and verify it.

``load_run`` is read-only. It never calls a provider, never touches the network and never modifies
or truncates a file (an unterminated final line is refused, not repaired; resuming is what repairs
it). It returns a ``LoadedRun`` only if every check below passes, and raises ``RunIntegrityError``
otherwise, keeping the original exception as ``__cause__`` where there is one.

Checks, in order:

1. The manifest and results files parse strictly (structure, types, no unknown fields, no corrupt
   or blank line, no duplicate request ID, no torn final line).
2. The manifest's ``run_id`` equals the run directory name.
3. The dataset loads and its recomputed hash equals its own pin (``load_dataset``), and the
   dataset identity recorded in the manifest (name, version, task, schema version, content hash)
   equals that dataset.
4. The prompt template stored in the manifest hashes to the stored prompt hash.
5. The selection rebuilt from the manifest's splits has the recorded case count and the recorded
   case-ID hash.
6. The requests rebuilt from the dataset, the manifest's prompt, model and parameters hash, one by
   one, to the request hash stored on each result line.
7. The results are exactly the selected cases: complete (the manifest's case count), no unexpected
   ID, in selection order, one per case.
8. Each stored result carries the manifest's provider name and requested model, and its own
   request ID.

These checks recompute hashes with the production helpers (``ids_sha256``, ``request_sha256``,
``PromptTemplate.sha256``, ``load_dataset``). They detect drift and tampering between a run's files
and its dataset, prompt and configuration. They cannot detect a defect in those helpers; the
independent tests in ``tests/test_runstore.py`` pin the algorithms. Nothing here authenticates the
files: a deliberate, internally consistent rewrite of both files would pass.
"""

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from niriksha.core.dataset import Case, Dataset, DatasetError, load_dataset
from niriksha.core.execution import build_requests
from niriksha.core.generation import GenerationRequest
from niriksha.core.runstore import (
    CorruptRunError,
    ManifestDataset,
    PromptTemplate,
    RunConfig,
    RunManifest,
    RunResultLine,
    ids_sha256,
    read_manifest,
    read_results,
    request_sha256,
)


class RunIntegrityError(Exception):
    """A persisted run is malformed, incomplete, inconsistent or tampered with."""


@dataclass(frozen=True)
class LoadedRun:
    """A run that passed ``load_run``: manifest, dataset and the three aligned sequences.

    ``cases``, ``requests`` and ``results`` have equal length and are in selection order, so
    ``results[i]`` is the stored outcome of ``requests[i]`` for ``cases[i]``. Build it only through
    ``load_run``: constructing one by hand bypasses every check above.
    """

    run_dir: Path
    manifest: RunManifest
    dataset: Dataset
    config: RunConfig
    cases: tuple[Case, ...]
    requests: tuple[GenerationRequest, ...]
    results: tuple[RunResultLine, ...]

    def __post_init__(self) -> None:
        aligned = len(self.cases) == len(self.requests) == len(self.results)
        if not aligned or any(
            not (case.id == request.request_id == line.request_id)
            for case, request, line in zip(self.cases, self.requests, self.results, strict=False)
        ):
            raise ValueError("cases, requests and results must be aligned")


def _fail(message: str, cause: BaseException | None = None) -> RunIntegrityError:
    error = RunIntegrityError(message)
    error.__cause__ = cause
    return error


def _rebuild_config(manifest: RunManifest, prompt: PromptTemplate) -> RunConfig:
    return RunConfig(
        run_id=manifest.run_id,
        splits=manifest.selection.splits,
        model=manifest.requested_model,
        prompt=prompt,
        params=manifest.params,
    )


def load_run(run_dir: str | Path, dataset_dir: str | Path) -> LoadedRun:
    """Read the run in ``run_dir`` and the dataset in ``dataset_dir``; verify; return both.

    Raises ``RunIntegrityError`` for any failed check (see the module docstring). Files are never
    modified.
    """
    run_dir = Path(run_dir)

    # 1. structure
    try:
        manifest = read_manifest(run_dir)
    except CorruptRunError as exc:
        raise _fail(f"manifest is unusable: {exc}", exc) from exc
    try:
        results = read_results(run_dir).lines  # strict: a torn final line is an error here
    except CorruptRunError as exc:
        raise _fail(f"results are unusable: {exc}", exc) from exc

    # 2. directory name
    if manifest.run_id != run_dir.name:
        raise _fail("the manifest run_id differs from the run directory name")

    # 3. dataset identity
    try:
        dataset = load_dataset(dataset_dir)
    except DatasetError as exc:
        raise _fail(
            f"the dataset cannot be loaded or no longer matches its pin: {exc}", exc
        ) from exc
    meta = dataset.meta
    current = ManifestDataset(
        name=meta.name,
        version=meta.version,
        task=meta.task,
        schema_version=meta.schema_version,
        content_sha256=dataset.content_sha256,
    )
    if manifest.dataset != current:
        differing = [
            f
            for f in ManifestDataset.model_fields
            if getattr(manifest.dataset, f) != getattr(current, f)
        ]
        raise _fail(f"the dataset differs from the one the run used ({', '.join(differing)})")

    # 4. prompt
    try:
        prompt = PromptTemplate(system=manifest.prompt.system, user=manifest.prompt.user)
    except ValidationError as exc:
        raise _fail("the prompt template stored in the manifest is invalid", exc) from exc
    if prompt.sha256 != manifest.prompt.sha256:
        raise _fail("the stored prompt text does not match the stored prompt hash")

    # 5. selection
    try:
        config = _rebuild_config(manifest, prompt)
    except ValidationError as exc:
        raise _fail("the run configuration stored in the manifest is invalid", exc) from exc
    cases = dataset.select(config.splits)
    if len(cases) != manifest.selection.case_count:
        raise _fail(
            f"the selection has {len(cases)} cases but the manifest records "
            f"{manifest.selection.case_count}"
        )
    selected_ids = [case.id for case in cases]
    if ids_sha256(selected_ids) != manifest.selection.case_ids_sha256:
        raise _fail("the selected case IDs do not match the stored selection hash")

    # 6. requests
    try:
        pairs = build_requests(config, dataset)
    except ValueError as exc:
        raise _fail(f"the requests cannot be rebuilt: {exc}", exc) from exc
    requests = tuple(request for _, request in pairs)

    # 7. completeness, identity, order
    stored_ids = [line.request_id for line in results]
    if len(results) < len(cases):
        raise _fail(f"the run is incomplete: {len(results)} of {len(cases)} results are stored")
    unexpected = [rid for rid in stored_ids if rid not in set(selected_ids)]
    if unexpected or len(results) > len(cases):
        raise _fail(f"the results contain a request_id outside the selection: {unexpected[:1]}")
    if stored_ids != selected_ids:
        raise _fail("the results are not in selection order")

    # 6 (per result) and 8
    for request, line in zip(requests, results, strict=True):
        if line.request_sha256 != request_sha256(request):
            raise _fail(f"the stored request hash for {line.request_id!r} does not match")
        result = line.execution.result
        if result.request_id != line.request_id:
            raise _fail(f"the stored result for {line.request_id!r} has another request_id")
        if result.provider != manifest.provider.name or (
            result.requested_model != manifest.requested_model
        ):
            raise _fail(
                f"the stored result for {line.request_id!r} was produced by a different provider "
                "or model than the manifest records"
            )

    return LoadedRun(
        run_dir=run_dir,
        manifest=manifest,
        dataset=dataset,
        config=config,
        cases=cases,
        requests=requests,
        results=tuple(results),
    )
