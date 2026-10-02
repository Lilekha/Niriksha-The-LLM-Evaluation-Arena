"""JSON parse validity. Definition: docs/metrics/json_parse_validity.md.

This measures only whether the output parses as JSON under the policy in ``parse_json_strict``. It
says nothing about JSON Schema compliance or about whether the content is right.
"""

from niriksha.core.dataset import ExtractionCase
from niriksha.core.generation import GenerationResult, GenerationSuccess
from niriksha.core.scoring import ScoreRecord, ScoreStatus
from niriksha.scorers._common import failure_of, not_scored_for_failure, parse_json_strict

METRIC = "json_parse_validity"
VERSION = "0.1.0"
TASK = "json_extraction"
DIRECTION = "higher_is_better"  # 1.0 is the desired outcome; see the metric document


def score_case(case: ExtractionCase, request_id: str, result: GenerationResult) -> ScoreRecord:
    if not isinstance(case, ExtractionCase):
        raise TypeError(f"{METRIC} scores extraction cases, got {type(case).__name__}")
    failure = failure_of(result)
    if failure is not None:
        return not_scored_for_failure(request_id, METRIC, VERSION, failure)
    assert isinstance(result, GenerationSuccess)
    outcome = parse_json_strict(result.output_text)
    return ScoreRecord(
        request_id=request_id,
        metric=METRIC,
        metric_version=VERSION,
        status=ScoreStatus.SCORED,
        value=1.0 if outcome.ok else 0.0,
        details={"top_level_type": outcome.top_level_type, "failure": outcome.failure},
    )
