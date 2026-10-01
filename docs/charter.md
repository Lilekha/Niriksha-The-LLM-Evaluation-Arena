# Charter

Status: pre-alpha. Everything below describes intent, not existing functionality.

## Problem

Teams choosing between LLMs, including cloud APIs and small local models, mostly rely on public leaderboards. Those measure generic tasks, not a team's own task, output schema, languages or latency budget. After adoption, prompt changes and provider-side model updates can degrade quality silently.

## Intended users

- Developers shipping an LLM feature who need to compare models or catch regressions on their own cases.
- People comparing a cloud model with a small local model on cost, latency and quality.
- The author, as a deliberate exercise in AI-engineering practice (portfolio first, developer tool second).

## Goals

- Reproducible runs: every result traceable to model ID, parameters, dataset version, prompt version, scorer version and Git commit.
- Honest reporting: failures are recorded, never hidden by retries or silent substitution. Differences within noise are reported as such.
- Provider independence: the evaluation core does not depend on any provider.
- English first, with Hindi and Kannada supported by the dataset schema from the start.

## MVP scope (planned)

- Tasks: structured JSON extraction, short-answer QA.
- Deterministic scorers only: normalized exact match, JSON validity, schema validity.
- Metrics: correctness, error and timeout rates, latency P50/P95, tokens, estimated cost, with confidence intervals and paired comparisons.
- First provider: a deterministic fake provider. Real providers come after the core is validated.
- Storage: local files and SQLite. Output: a static report.

## Non-goals (for the MVP)

Dashboard, LLM-as-judge, hallucination and faithfulness scoring, CI quality gates, tracing UI, hosted service, multi-user features, BERT-based models, Kubernetes, queues or other infrastructure without a demonstrated need.

## Success criteria

- A run is reproducible from its manifest, and scoring can be repeated offline from stored raw outputs.
- Each shipped metric has a written definition and known limitations.
- At least one comparison is published with confidence intervals and an explicit statement of what the data cannot show.
- No result in the repository is fabricated or hardcoded.

## Why a small independent engine

Mature tools already exist (see [alternatives.md](alternatives.md); none have been evaluated yet). Niriksha does not aim to replace them. It is built from scratch, with established libraries for supporting functions, because the project's purpose is to demonstrate evaluation-engineering depth: run provenance, failure accounting, statistical care and evaluator validation. Embedding a framework would hide exactly those mechanisms. The trade-off is less breadth and more maintenance than adopting an existing tool, and no claim is made that this engine is better or novel.
