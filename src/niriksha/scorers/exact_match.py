"""Normalized exact match for short-answer QA (docs/metrics/normalized_exact_match.md)."""

from collections.abc import Sequence

from niriksha.core.dataset import QACase
from niriksha.core.generation import GenerationResult, GenerationSuccess
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.scorers._common import failure_of, normalize_text, not_scored_for_failure

METRIC = "normalized_exact_match"
VERSION = "0.1.0"
TASK = "short_answer_qa"
DIRECTION = "higher_is_better"  # 1.0 is the desired outcome; see the metric document


def matching_answer_index(output_text: str, answers: Sequence[str]) -> int | None:
    """Index of the first accepted answer equal to the output after normalisation, else None."""
    normalized_output = normalize_text(output_text, casefold=True)
    for index, answer in enumerate(answers):
        if normalize_text(answer, casefold=True) == normalized_output:
            return index
    return None


def score_case(case: QACase, request_id: str, result: GenerationResult) -> ScoreRecord:
    if not isinstance(case, QACase):
        raise TypeError(f"{METRIC} scores short-answer QA cases, got {type(case).__name__}")
    failure = failure_of(result)
    if failure is not None:
        return not_scored_for_failure(request_id, METRIC, VERSION, failure)
    assert isinstance(result, GenerationSuccess)
    index = matching_answer_index(result.output_text, case.answers)
    return ScoreRecord(
        request_id=request_id,
        metric=METRIC,
        metric_version=VERSION,
        status=ScoreStatus.SCORED,
        value=0.0 if index is None else 1.0,
        details={
            "matched_answer_index": index,
            "empty_output": not normalize_text(result.output_text, casefold=False),
        },
    )
