"""Scoring workflow: score a verified run into a persisted artifact, and verify an artifact.

Offline and read-only with respect to the run: no provider, no network, and nothing under the run
directory is written. Artifacts go to a separate ``scores_dir``.

Reuse rule. Scoring a run with a metric that already has an artifact never overwrites it. The
existing file is read strictly, then compared with the artifact the current code computes from the
verified run. It is reused only if they are equal, which establishes in one step that its structure
is valid, that it belongs to this exact run and dataset (including the run files' hashes), and
that the exact ``(metric, version)`` implementation reproduces its scores. A corrupt, mismatched
or unverifiable existing artifact raises a typed error and is left untouched.

There are no automatic version upgrades. An artifact is verified by the implementation registered
for its exact ``(metric, version)``; if none is registered it cannot be verified (it can still be
read and aggregated). A change to scoring rules must ship as a new metric version.
"""

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from niriksha.core.runload import LoadedRun, RunIntegrityError, load_run
from niriksha.core.scorestore import (
    ScoreArtifact,
    ScoreArtifactError,
    ScoreArtifactMismatchError,
    artifact_path,
    build_artifact,
    read_artifact,
    source_from_run,
    write_artifact,
)
from niriksha.core.scoring import ScoreRecord
from niriksha.scorers import exact_match, field_match, json_parse

# The implementation of every (metric name, version) this build can score and verify.
REGISTRY: dict[tuple[str, str], ModuleType] = {
    (module.METRIC, module.VERSION): module for module in (exact_match, json_parse, field_match)
}


@dataclass(frozen=True)
class ScoreOutcome:
    path: Path
    artifact: ScoreArtifact
    created: bool  # False when a verified existing artifact was reused and nothing was written


def available_metrics(task: str | None = None) -> tuple[tuple[str, str], ...]:
    """Registered ``(name, version)`` pairs, optionally only those that apply to ``task``."""
    return tuple(
        sorted(key for key, module in REGISTRY.items() if task is None or module.TASK == task)
    )


def resolve_scorer(metric: str, version: str | None = None) -> ModuleType:
    """The scorer for ``metric`` at exactly ``version``; with no version, the only registered one.

    Raises ``ScoreArtifactError`` for an unknown metric or version. It never substitutes another
    version.
    """
    versions = sorted(v for (name, v) in REGISTRY if name == metric)
    if not versions:
        known = ", ".join(sorted({name for name, _ in REGISTRY}))
        raise ScoreArtifactError(f"unknown metric {metric!r}; available: {known}")
    if version is None:
        if len(versions) != 1:
            raise ScoreArtifactError(f"metric {metric!r} has versions {versions}; pass one")
        version = versions[0]
    if (metric, version) not in REGISTRY:
        raise ScoreArtifactError(
            f"no scorer implements {metric!r} version {version!r}; available versions: {versions}"
        )
    return REGISTRY[(metric, version)]


def score_records(run: LoadedRun, scorer: ModuleType) -> tuple[ScoreRecord, ...]:
    task = run.dataset.meta.task
    if scorer.TASK != task:
        raise ScoreArtifactError(
            f"metric {scorer.METRIC!r} applies to {scorer.TASK!r} runs, but this run is {task!r}"
        )
    return tuple(
        scorer.score_case(case, line.request_id, line.execution.result)
        for case, line in zip(run.cases, run.results, strict=True)
    )


def artifact_for_run(run: LoadedRun, scorer: ModuleType) -> ScoreArtifact:
    """The artifact the current code computes for a verified run. Deterministic."""
    return build_artifact(
        source_from_run(run), scorer.METRIC, scorer.VERSION, scorer.TASK, score_records(run, scorer)
    )


def _differences(existing: ScoreArtifact, expected: ScoreArtifact) -> list[str]:
    """Names of the parts that differ (never their values)."""
    found = [
        f"source.{name}"
        for name in type(expected.source).model_fields
        if getattr(existing.source, name) != getattr(expected.source, name)
    ]
    if existing.metric != expected.metric:
        found.append("metric")
    if existing.records != expected.records:
        differing = [
            a.request_id for a, b in zip(existing.records, expected.records, strict=False) if a != b
        ]
        found.append(f"records (first differing case: {differing[0]})" if differing else "records")
    return found


def verify_artifact(artifact: ScoreArtifact, run_dir: str | Path, dataset_dir: str | Path) -> None:
    """Verify ``artifact`` against its source run and the scorer implementation. Raises if not.

    - ``ScoreArtifactError`` if no scorer implements the artifact's exact ``(metric, version)``
      (it cannot be verified).
    - ``ScoreArtifactMismatchError`` if the run itself fails ``load_run``, or the artifact differs
      from what the verified run and scorer produce (identity, run file hashes or scores).
    """
    scorer = resolve_scorer(artifact.metric.name, artifact.metric.version)
    try:
        run = load_run(run_dir, dataset_dir)
    except RunIntegrityError as exc:
        raise ScoreArtifactMismatchError(f"the source run cannot be verified: {exc}") from exc
    differences = _differences(artifact, artifact_for_run(run, scorer))
    if differences:
        raise ScoreArtifactMismatchError(
            "the artifact does not match its source run and scorer: " + "; ".join(differences)
        )


def _refuse_scores_inside_run(scores_dir: Path, run_dir: Path) -> None:
    resolved = scores_dir.resolve()
    run_resolved = run_dir.resolve()
    if resolved == run_resolved or run_resolved in resolved.parents:
        raise ScoreArtifactError("scores_dir must not be inside the run directory")


def score_run_to_artifact(
    run_dir: str | Path,
    dataset_dir: str | Path,
    scores_dir: str | Path,
    metric: str,
    *,
    version: str | None = None,
) -> ScoreOutcome:
    """Score the recorded outputs of a completed run with one metric and persist the result.

    Loads and verifies the run first (``RunIntegrityError`` propagates unchanged), scores, then
    creates ``<scores_dir>/<run_id>--<metric>--<version>.json`` exclusively. If that file already
    exists it is read, compared and reused only when it matches (see the module docstring);
    otherwise a typed error is raised and the file is left as it is.
    """
    run = load_run(run_dir, dataset_dir)
    scorer = resolve_scorer(metric, version)
    _refuse_scores_inside_run(Path(scores_dir), run.run_dir)
    expected = artifact_for_run(run, scorer)
    try:
        return ScoreOutcome(write_artifact(scores_dir, expected), expected, created=True)
    except FileExistsError:
        pass
    path = artifact_path(scores_dir, expected.artifact_id)
    existing = read_artifact(path)  # ScoreArtifactError if it is corrupt; never overwritten
    differences = _differences(existing, expected)
    if differences:
        raise ScoreArtifactMismatchError(
            f"{path.name} already exists and does not match this run and scorer "
            f"({'; '.join(differences)}); it was not overwritten"
        )
    return ScoreOutcome(path, existing, created=False)
