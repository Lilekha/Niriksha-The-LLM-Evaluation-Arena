# Metric: <name>

**Version:** <e.g. 0.1.0; bump when the definition or implementation changes>
**Status:** draft | implemented | validated
**Direction:** higher_is_better | lower_is_better (the scorer module's `DIRECTION` constant; say what the best value means)

## Purpose
What question this metric answers.

## Applicable task
Which tasks (for example JSON extraction, short-answer QA). State where it does not apply.

## Formula
Exact definition, including any normalization applied before comparison.

## Required inputs
Case fields, stored output fields, and any configuration the metric reads.

## Assumptions
What must hold for the number to mean what it appears to mean.

## Limitations
Known failure modes, biases, and cases where the metric misleads.

## Validation method
How the metric is checked: unit and adversarial tests, golden fixtures, agreement with human labels (sample size and agreement statistic), as applicable. Record the result when it exists.
