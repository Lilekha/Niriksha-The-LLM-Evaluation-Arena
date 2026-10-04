"""Compare two completed runs on one metric, from their verified score artifacts.

Offline and read-only: no provider, no network, nothing in the runs, datasets or scores is
written, and a missing artifact is never created (score the run first). Each side is loaded with
``load_run``, its artifact is read strictly and verified with ``verify_artifact`` against that run
and dataset, so a corrupt, tampered, mismatched or unverifiable artifact stops the comparison with
the existing typed error, prefixed with the side it came from. Errors name no filesystem paths.

The comparison itself (compatibility rules, semantics, rendering) is in
``niriksha.core.compare``. Nothing is saved: render the result and write it yourself.
"""

from dataclasses import dataclass
from pathlib import Path

from niriksha.core.compare import Comparison, ComparisonInput, build_comparison
from niriksha.core.runload import RunIntegrityError, load_run
from niriksha.core.scorestore import (
    ScoreArtifactError,
    artifact_id,
    artifact_path,
    read_artifact,
)
from niriksha.scorers.artifacts import resolve_scorer, verify_artifact


@dataclass(frozen=True)
class RunInput:
    """Where one run, its dataset and its score artifacts live."""

    run_dir: str | Path
    dataset_dir: str | Path
    scores_dir: str | Path


def _load(role: str, source: RunInput, metric: str, version: str) -> ComparisonInput:
    try:
        run = load_run(source.run_dir, source.dataset_dir)
        path = artifact_path(source.scores_dir, artifact_id(run.manifest.run_id, metric, version))
        if not path.is_file():
            raise ScoreArtifactError(
                f"no {metric} {version} score artifact for this run; score the run first"
            )
        artifact = read_artifact(path)
        verify_artifact(artifact, source.run_dir, source.dataset_dir)
    except (RunIntegrityError, ScoreArtifactError) as exc:
        raise type(exc)(f"{role}: {exc}") from exc
    return ComparisonInput(artifact=artifact, params=run.manifest.params)


def compare_runs(
    baseline: RunInput, candidate: RunInput, metric: str, *, version: str | None = None
) -> Comparison:
    """Compare ``candidate`` against ``baseline`` on ``metric`` at exactly ``version``.

    Raises ``ScoreArtifactError`` for an unknown metric or version or a missing artifact,
    ``ScoreArtifactMismatchError`` for an artifact that does not match its run, dataset or scorer,
    ``RunIntegrityError`` for a run that fails ``load_run``, and ``IncompatibleRunsError`` if the
    two runs are not comparable. Existing errors are re-raised with the side prefixed.
    """
    scorer = resolve_scorer(metric, version)
    return build_comparison(
        _load("baseline", baseline, scorer.METRIC, scorer.VERSION),
        _load("candidate", candidate, scorer.METRIC, scorer.VERSION),
        scorer.DIRECTION,
    )
