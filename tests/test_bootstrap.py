"""The paired percentile bootstrap: pinned values, symmetry, exact cross-check, validation."""

import ast
import itertools
import math
import random
from fractions import Fraction
from pathlib import Path

import pytest

from niriksha.core import bootstrap
from niriksha.core.bootstrap import (
    CONFIDENCE_PERCENT,
    METHOD,
    MINIMUM_PAIRED_CASES,
    N_RESAMPLES,
    SEED,
    paired_bootstrap_interval,
)
from niriksha.core.compare import ConfidenceInterval

# 40 values in {-1, -0.5, 0, 0.5, 1}: mean exactly 0
MIXED = [float(((i * 7) % 5) - 2) / 2 for i in range(40)]
MOSTLY_UP = [1.0] * 25 + [0.0] * 15


def test_the_documented_constants():
    assert (CONFIDENCE_PERCENT, N_RESAMPLES, SEED, MINIMUM_PAIRED_CASES) == (95, 10_000, 0, 30)
    assert METHOD == "paired_percentile_bootstrap_v1"
    allowed = ConfidenceInterval.model_fields["method"].annotation.__args__
    assert allowed == (METHOD,)


def test_the_generator_contract_the_method_relies_on_is_pinned():
    # Python documents that random() keeps its sequence for the same int seed across versions.
    assert random.Random(0).random() == 0.8444218515250481
    assert random.Random(12345).random() == 0.41661987254534116


def test_pinned_intervals_for_fixed_inputs():
    assert paired_bootstrap_interval(MIXED) == (-0.225, 0.2125)
    assert paired_bootstrap_interval(MIXED, seed=7) == (-0.225, 0.225)
    assert paired_bootstrap_interval(MIXED, n_resamples=1000, confidence_percent=90) == (
        -0.2,
        0.175,
    )
    assert paired_bootstrap_interval(MOSTLY_UP) == (0.475, 0.775)


def test_the_result_is_deterministic():
    assert paired_bootstrap_interval(MIXED) == paired_bootstrap_interval(MIXED)
    assert paired_bootstrap_interval(MIXED) == paired_bootstrap_interval(list(MIXED))


@pytest.mark.parametrize("value", [1.0, -0.5, 0.0, 0.25])
def test_constant_differences_give_a_zero_width_interval(value):
    assert paired_bootstrap_interval([value] * 30) == (value, value)
    assert paired_bootstrap_interval([value]) == (value, value)  # n=1 is the caller's policy


@pytest.mark.parametrize("values", [MIXED, MOSTLY_UP, [0.0, 1.0, 1.0, -1.0, 0.5]])
def test_negating_the_differences_mirrors_the_interval_exactly(values):
    lower, upper = paired_bootstrap_interval(values)
    negated = paired_bootstrap_interval([-v for v in values])
    assert negated == (-upper, -lower)


def test_direction_of_the_interval_follows_the_data():
    assert paired_bootstrap_interval(MOSTLY_UP)[0] > 0
    assert paired_bootstrap_interval([-v for v in MOSTLY_UP])[1] < 0
    lower, upper = paired_bootstrap_interval([1.0] * 15 + [-1.0] * 15)
    assert lower < 0 < upper


def test_the_interval_is_close_to_the_normal_approximation_for_a_proportion():
    p, n = 25 / 40, 40
    half = 1.96 * math.sqrt(p * (1 - p) / n)
    lower, upper = paired_bootstrap_interval(MOSTLY_UP)
    assert lower == pytest.approx(p - half, abs=0.02)
    assert upper == pytest.approx(p + half, abs=0.02)


@pytest.mark.parametrize(
    "values", [[0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0]]
)
def test_the_simulated_interval_equals_the_exact_bootstrap_quantiles_for_four_cases(values):
    # All 4**4 = 256 resamples are equally likely: the exact bootstrap distribution.
    exact = sorted(
        sum(values[i] for i in picks) / 4 for picks in itertools.product(range(4), repeat=4)
    )
    exact_interval = (exact[(256 * 5) // 200], exact[-(-(256 * 195) // 200) - 1])
    assert paired_bootstrap_interval(values) == exact_interval


def test_the_endpoints_stay_inside_the_data_range_and_are_ordered():
    data = random.Random(5)
    for _ in range(20):
        values = [data.choice((-1.0, -0.25, 0.0, 0.5, 1.0)) for _ in range(data.randint(1, 60))]
        lower, upper = paired_bootstrap_interval(values, n_resamples=500)
        assert min(values) <= lower <= upper <= max(values)


def test_quantile_indexes_are_valid_for_every_resample_count_and_confidence():
    for resamples in range(1, 41):
        for percent in range(1, 100):
            lower, upper = paired_bootstrap_interval(
                [0.0, 0.5, 1.0], confidence_percent=percent, n_resamples=resamples
            )
            assert lower <= upper


def test_wider_confidence_never_gives_a_narrower_interval():
    narrow = paired_bootstrap_interval(MIXED, confidence_percent=50)
    wide = paired_bootstrap_interval(MIXED, confidence_percent=95)
    assert wide[0] <= narrow[0] and narrow[1] <= wide[1]


@pytest.mark.parametrize(
    "call",
    [
        lambda: paired_bootstrap_interval([]),
        lambda: paired_bootstrap_interval([0.0, float("nan")]),
        lambda: paired_bootstrap_interval([0.0, float("inf")]),
        lambda: paired_bootstrap_interval([0.0], confidence_percent=0),
        lambda: paired_bootstrap_interval([0.0], confidence_percent=100),
        lambda: paired_bootstrap_interval([0.0], n_resamples=0),
    ],
)
def test_invalid_input_is_refused(call):
    with pytest.raises(ValueError):
        call()


def test_the_module_uses_only_the_standard_library():
    tree = ast.parse(Path(bootstrap.__file__).read_text(encoding="utf-8"))
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert modules == {"math", "random", "collections"}


def reference(values, n_resamples, seed, percent):
    """The documented method written out independently, with exact rational sums."""
    n, rng = len(values), random.Random(seed)
    means = sorted(
        float(sum(Fraction(values[int(rng.random() * n)]) for _ in range(n))) / n
        for _ in range(n_resamples)
    )
    return (
        means[(n_resamples * (100 - percent)) // 200],
        means[-(-(n_resamples * (100 + percent)) // 200) - 1],
    )


# Sums that lose precision in a plain left-to-right float sum (a big value cancelling a tiny one).
CANCELLING = [1.0, -1.0] + [2.0 ** -(60 + i) for i in range(28)]


@pytest.mark.parametrize(
    ("values", "resamples", "seed", "percent"),
    [
        (MIXED, 1000, 0, 95),
        (MOSTLY_UP, 1000, 3, 90),
        (CANCELLING, 2000, 0, 1),  # the central 1%: dominated by the tiny, cancelling sums
        ([0.1, 0.2, 0.7, -0.3, 0.0, 0.5], 500, 9, 80),
    ],
)
def test_the_result_equals_an_independent_exact_arithmetic_reference(
    values, resamples, seed, percent
):
    assert paired_bootstrap_interval(
        values, n_resamples=resamples, seed=seed, confidence_percent=percent
    ) == reference(values, resamples, seed, percent)
