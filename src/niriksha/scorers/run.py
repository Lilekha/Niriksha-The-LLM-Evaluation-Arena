"""Score a loaded run offline. No provider is involved: only stored outputs and dataset values."""

from niriksha.core.runload import LoadedRun
from niriksha.core.scoring import ScoreRecord
from niriksha.scorers import exact_match, field_match, json_parse

# Which metrics apply to which dataset task. Each applicable metric yields exactly one record per
# selected case, so scored + not_scored equals the case count for every metric.
SCORERS_BY_TASK = {
    exact_match.TASK: (exact_match,),
    json_parse.TASK: (json_parse, field_match),
}


def score_run(run: LoadedRun) -> tuple[ScoreRecord, ...]:
    """Records in case order, and within a case in the fixed metric order of ``SCORERS_BY_TASK``.

    Deterministic: scoring the same run twice returns equal records.
    """
    scorers = SCORERS_BY_TASK[run.dataset.meta.task]
    return tuple(
        scorer.score_case(case, line.request_id, line.execution.result)
        for case, line in zip(run.cases, run.results, strict=True)
        for scorer in scorers
    )
