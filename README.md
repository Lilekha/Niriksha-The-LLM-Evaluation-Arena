# Niriksha — The LLM Evaluation Arena

A model-agnostic platform for evaluating, benchmarking, and observing large language models.

> Every model has its quirks. Put them to the test.

## Status

**Pre-alpha. No real-model evaluation has been implemented yet.**

Today the repository contains the project foundations (M0), the generation schemas, failure model and provider protocol (M1.1, [ADR 0002](docs/adr/0002-provider-contract.md)), a deterministic fake provider and a minimal sequential runner that times each call (M1.2), and local dataset loading with content hashing, provenance, run persistence and resume (M1.3, [ADR 0003](docs/adr/0003-dataset-identity-and-run-persistence.md)). All of it is tested offline. The fake provider returns scripted text, not model output, and the only datasets are synthetic test fixtures.

M2.1 adds an offline run reader that verifies a stored run against its dataset, and three deterministic scorers (normalized exact match, JSON parse validity, field-level exact match; [ADR 0004](docs/adr/0004-run-integrity-and-scoring-contract.md)). They score stored outputs only, so they measure string and value identity as defined in [docs/metrics/](docs/metrics/), not semantic correctness.

There is still no CLI, config-file loading, real provider adapter, retry logic, score persistence, aggregation or benchmark dataset. Nothing here has been run against any real model, and real evaluations are not possible yet. A run's manifest records what was run; it does not make a run reproducible.

## The problem

Choosing or swapping an LLM is usually decided by public leaderboards and impressions, which say little about *your* task, output schema, languages or latency budget. After deployment, prompt edits and model updates can degrade quality without any signal. Niriksha aims to make small, reproducible, honestly reported comparisons on your own tasks.

## Planned MVP (not yet implemented)

- Two tasks: **structured JSON extraction** and **short-answer question answering**.
- English first. The dataset schema is designed to carry Hindi and Kannada from day one.
- Planned metrics: task correctness, JSON validity and schema validity, error and timeout rates, latency (P50/P95), token usage, and estimated cost, with confidence intervals and paired comparisons. Each metric will get a written definition and its known limitations before it ships.
- Raw model outputs are stored separately from scores, so scoring can be repeated without new inference calls.

Out of the MVP: dashboard, LLM-as-judge and hallucination scoring, CI quality gates, tracing UI. See [docs/charter.md](docs/charter.md).

## Architecture (planned)

An independent evaluation core (runner, provider protocol, storage, manifests, scorers) that never imports a concrete provider. Providers are adapters configured through data files. The first provider will be a deterministic fake provider; real providers come later. See [docs/architecture.md](docs/architecture.md).

## Development posture

- **₹0 budget.** No paid API usage.
- **No network calls by default.** Real providers will require explicit opt-in. CI uses no secrets and no external model access.
- Secrets never go in the repository, logs, manifests or result databases. See [SECURITY.md](SECURITY.md).

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| M0 | Repository foundations and documentation (no evaluation code) | complete |
| M1 | Schemas, dataset loader, fake provider, runner, store, manifest, resume | implemented offline with the fake provider only (M1.1 to M1.3), integration-tested in M1.4; pending review |
| M2 | Deterministic scorers, aggregation with confidence intervals | in progress (M2.1: offline run reader and three deterministic scorers done; JSON Schema validity, aggregation and confidence intervals pending) |
| M3 | OpenAI-compatible adapter (tested locally first), then opt-in real and local-model runs | planned |
| M4 | English dataset v0.1, then Hindi and Kannada cases | planned |
| M5 | Reports, paired comparisons, first honest write-up | planned |

## Setup and tests

Requires Python 3.11 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1   |   macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest
```

Optional, to run the checks on each commit:

```bash
pre-commit install
```

## Licence

[MIT](LICENSE).
