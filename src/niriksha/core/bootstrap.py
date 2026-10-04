"""Paired percentile bootstrap interval for the mean of per-case differences. Standard library only.

The method is fixed and named ``paired_percentile_bootstrap_v1``; changing any step below needs a
new method name.

1. Input: the per-case differences ``d`` (candidate minus baseline) of the cases scored in both
   runs, in selection order, ``n = len(d)``.
2. Generator: ``random.Random(seed)`` (Mersenne Twister, seeded with an int). Python guarantees
   that ``random()`` keeps producing the same sequence for the same seed across versions.
3. One resample draws ``n`` indices, each ``int(random() * n)`` (uniform with replacement), and its
   statistic is ``math.fsum(selected) / n``. ``n_resamples`` resamples are drawn one after another.
4. The resample means are sorted ascending as ``m[0..B-1]`` with ``B = n_resamples``. For a
   confidence of ``P`` percent (an integer, so the index arithmetic is exact) the interval is
   ``m[(B * (100 - P)) // 200]`` to ``m[ceil(B * (100 + P) / 200) - 1]`` inclusive. For
   ``B = 10000`` and ``P = 95`` that is ``m[250]`` to ``m[9749]``: 95% of the resample means.
   Negating every difference therefore gives exactly the negated and swapped interval.

Every step is IEEE-754 double arithmetic, exactly rounded ``fsum``, integer index arithmetic and a
sort, so the result should be identical across CPython versions and platforms. That rests on
Python's documented ``random()`` contract and is checked by pinned values in the tests; it is not
claimed for other interpreters.

What the interval is. It resamples the evaluated cases only, so it estimates how much the mean
difference could vary over that case sample. It does not capture variability across repeated
generations, other prompts or cases that were not evaluated, it assumes the cases are independent,
and it is not a significance test. A percentile bootstrap can undercover with few cases or when
most differences are ties; ``MINIMUM_PAIRED_CASES`` does not guarantee the nominal coverage.
Cost is ``n_resamples * n`` draws: about 0.25 s for 100 cases and 6 s for 2000.
"""

import math
import random
from collections.abc import Sequence

METHOD = "paired_percentile_bootstrap_v1"
CONFIDENCE_PERCENT = 95
N_RESAMPLES = 10_000
SEED = 0
MINIMUM_PAIRED_CASES = 30


def paired_bootstrap_interval(
    values: Sequence[float],
    *,
    confidence_percent: int = CONFIDENCE_PERCENT,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """The ``(lower, upper)`` percentile interval of the bootstrapped mean of ``values``."""
    n = len(values)
    if n < 1:
        raise ValueError("at least one value is required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("values must be finite")
    if not 1 <= confidence_percent <= 99:
        raise ValueError("confidence_percent must be an integer from 1 to 99")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    draw = random.Random(seed).random
    means = sorted(
        math.fsum([values[int(draw() * n)] for _ in range(n)]) / n for _ in range(n_resamples)
    )
    lower = means[(n_resamples * (100 - confidence_percent)) // 200]
    upper = means[-(-(n_resamples * (100 + confidence_percent)) // 200) - 1]
    return lower, upper
