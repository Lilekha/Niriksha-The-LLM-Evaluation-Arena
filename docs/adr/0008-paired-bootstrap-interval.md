# ADR 0008: Paired bootstrap interval for the mean difference

Status: accepted. Date: 2026-10-04.

Covers M2.5: a 95% paired percentile bootstrap interval for the mean per-case difference, added to the paired summary of a run comparison. Offline and deterministic. Partly supersedes the "no confidence interval" statements in [ADR 0007](0007-run-comparison.md) (M2.4). Still no significance test, winner or ranking.

## Context
A paired mean difference on a limited set of cases says nothing about how much it could vary. An interval gives that, if it is computed honestly: only from cases scored in both runs, never with imputed scores, with a method fixed and named in advance, and with its limits stated.

## Decisions
1. **What is resampled.** The existing per-case deltas (`candidate - baseline`) of the cases scored in both runs, in selection order. Unscored cases never enter, so no score is imputed.
2. **Method `paired_percentile_bootstrap_v1`** (`niriksha.core.bootstrap`): 10,000 resamples, each of `n` indices drawn with `int(random.Random(seed).random() * n)` and averaged with `math.fsum`; the sorted resample means `m[0..B-1]` give the 95% interval `m[(B*5)//200]` to `m[ceil(B*195/200)-1]`, that is `m[250]` to `m[9749]`. Integer percent arithmetic keeps the indexes exact, and negating every delta gives exactly the negated and swapped interval (tested). Any change to a step needs a new method name.
3. **Fixed seed `0`.** Arbitrary, chosen once, never tuned, not derived from the data. It is recorded in the report. The interval depends on the order of the deltas, which is the fixed selection order.
4. **Standard library, no SciPy.** SciPy was considered because the project plan names it for statistics. Rejected for now:
   - SciPy 1.18 requires Python 3.12 or newer while this project supports 3.11 and CI tests 3.11 and 3.13, so the two CI legs would install different SciPy releases unless it were pinned below 1.18.
   - NumPy and SciPy add about 58 MB of native wheels for roughly ten lines of logic.
   - Python documents that `random()` keeps its sequence for the same integer seed. NumPy does not promise stable `Generator` streams across versions and SciPy changes its resampling internals between releases.
   - `scipy.stats.bootstrap` defaults to BCa and has its own percentile interpolation; the convention above is explicit and tested.
   - The algorithm is small and is cross-checked by tests against the exact bootstrap distribution for four cases, against pinned values and against a normal approximation. A move to NumPy or SciPy would be reconsidered if datasets reach tens of thousands of paired cases (cost is `10,000 * n` draws: about 0.25 s at 100 cases and 6 s at 2,000).
5. **Minimum 30 paired cases.** Below it the endpoints are null with the reason `too_few_paired_cases` (and `no_paired_scored_cases` with none). With one case every resample is identical, so the interval would be a falsely confident point. 30 is a judgement, not a theorem. In a one-off simulation (not committed: 600 trials, 400 resamples, two synthetic distributions of differences in {-1, 0, +1}, about one point of simulation error) the nominal 95% interval covered the true mean about 94% of the time at 30 cases when 70% of cases tie, and about 91% when 90% of cases tie; at 20 cases the second figure was about 79%, and at 50 about 94%. So the minimum does not guarantee nominal coverage, and the reports say so.
6. **Report.** `Summary.paired.confidence_interval` holds `method`, `confidence_level` (0.95), `n_resamples`, `seed`, `minimum_paired_cases`, `lower`, `upper` and `unavailable_reason`. The endpoints are finite JSON numbers or null; the model refuses inconsistent values. `COMPARISON_VERSION` is now 2, because the strict model no longer accepts reports without the new field. The settings are documented constants and are not parameters of `compare_runs`, so they cannot be tuned after seeing a result.
7. **Wording.** The report calls it an uncertainty estimate for the evaluated cases. It never says whether the interval contains zero, never declares a winner and never claims significance. A zero-width interval is reported with a note that the paired differences did not vary in the sample, which does not imply certainty.

## Limitations
- The interval resamples the evaluated cases only. It does not capture variability across repeated generations (for example temperature above zero), other prompts, or cases that were not evaluated.
- It assumes the cases are independent. Related cases (for example translations of one item) make it too narrow.
- A percentile bootstrap can undercover with few cases or when most differences are ties, which is typical for exact-match metrics.
- Reproducibility across Python versions and platforms rests on the documented `random()` contract and IEEE-754 arithmetic; pinned values in the tests check it on the versions CI runs. It is not claimed for other interpreters.
- There is no correction for comparing several metrics or run pairs, and the paired mean is of per-case values that are not weighted.

## Consequences
- Artifact, run and dataset formats and hashes are unchanged, and no dependency was added.
- Comparison JSON written under version 1 does not load under version 2.
