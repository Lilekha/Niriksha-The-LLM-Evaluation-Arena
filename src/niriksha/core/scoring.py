"""The score record: one metric's outcome for one stored result.

Value semantics are fixed here, not left to each scorer:

- ``scored``: ``value`` is a finite number in [0, 1]. Each metric defines what 0 and 1 mean (see
  docs/metrics/).
- ``not_scored``: ``value`` is ``None`` and ``reason`` says why. A generation that failed is
  ``not_scored``. It is never dropped and never counted as an incorrect answer, and no number is
  invented for it.

For every applicable request each metric returns exactly one record, so
``scored + not_scored == applicable requests`` always holds.

Scores are not persisted in M2.1; this is an in-memory contract. ``details`` therefore holds only
small flat values (counts, flags, short enumerations) and never raw model output.
"""

import math
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from niriksha.core.generation import NonBlank

DetailValue = str | int | float | bool | None
MetricName = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
MetricVersion = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]


class ScoreStatus(StrEnum):
    SCORED = "scored"
    NOT_SCORED = "not_scored"


class ScoreRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    request_id: NonBlank
    metric: MetricName
    metric_version: MetricVersion
    status: ScoreStatus
    value: float | None = Field(default=None, allow_inf_nan=False)
    reason: NonBlank | None = None
    details: dict[str, DetailValue] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def _finite_details(cls, value: dict[str, DetailValue]) -> dict[str, DetailValue]:
        for key, item in value.items():
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"details value for {key!r} must be finite")
        return value

    @model_validator(mode="after")
    def _value_matches_status(self):
        if self.status is ScoreStatus.SCORED:
            if self.value is None or not 0.0 <= self.value <= 1.0:
                raise ValueError("a scored record needs a value in [0, 1]")
        else:
            if self.value is not None:
                raise ValueError("a not_scored record must not carry a value")
            if self.reason is None:
                raise ValueError("a not_scored record needs a reason")
        return self
