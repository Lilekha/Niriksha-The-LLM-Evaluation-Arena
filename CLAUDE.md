# Niriksha — The LLM Evaluation Arena

> **Every model has its quirks. Put them to the test.**

## 1. Project Overview

Niriksha is an open-source LLM evaluation and observability platform designed to help developers systematically evaluate model behaviour across tasks, compare results, identify failure modes, and reproduce experiments.

**Primary goal:** Build a technically rigorous AI engineering portfolio project that is also useful as a developer tool.

**Current status:** Pre-alpha. No real-model evaluation has been implemented yet.

### MVP scope

The initial version focuses on two evaluation tasks:

1. **Structured JSON extraction** — evaluate JSON parsing, schema validity, and field-level correctness.
2. **Short-answer question answering** — evaluate answer correctness using exact match or documented normalisation rules.

The dataset schema must support English, Hindi, and Kannada. Initial implementation and evaluation will be English-first, with Hindi and Kannada datasets added later.

The planned provider architecture supports a fake provider, cloud-hosted models, and compatible local inference runtimes. Actual providers will be added incrementally after the core engine is validated.

## 2. Core Engineering Principles

* **Reproducibility:** Every experiment must record enough metadata to explain how its results were produced.
* **Evaluation integrity:** Never fabricate results, manipulate evaluation criteria to improve scores, or hide failures.
* **Provider independence:** The evaluation engine must not depend on any particular model vendor or inference provider.
* **Failure transparency:** Record errors, retries, timeouts, and rate limits explicitly.
* **Separation of concerns:** Model generation, result storage, scoring, aggregation, and reporting must remain distinct.
* **Minimal complexity:** Prefer a small, well-tested implementation over premature abstractions, infrastructure, or dashboards.
* **Honest reporting:** Distinguish implemented capabilities from planned work and synthetic test results from real-model results.

## 3. Architecture Rules

Follow the architecture described in `docs/architecture.md`.

### Core and providers

* `niriksha.core` contains the provider protocol, request/result contracts, runner, storage, and experiment metadata.
* `niriksha.providers` contains provider adapters.
* `niriksha.scorers` contains task-specific scoring logic.
* The core must never import from or depend on the providers package.
* Provider-specific configuration belongs in configuration files, not in core logic.
* Do not scatter vendor-specific conditionals throughout the engine.

### Generation and scoring

* Persist raw model outputs before scoring.
* Scoring must be repeatable without additional inference calls.
* A change to a scorer must not require regenerating model outputs.
* Preserve the original output; any output-processing transformation must be explicit, versioned, and recorded.

### Experiment provenance

Record relevant metadata, including dataset and prompt hashes, provider profile, requested and returned model identifiers, generation parameters, timestamps, code revision, dependency lockfile hash, and available usage information.

Unknown values must be represented honestly. Never report unknown cost as zero.

Consult the architecture documentation before making structural changes.

## 4. Provider and Network Policy

The initial API budget is **₹0**.

* Use the deterministic fake provider for early end-to-end development.
* Do not make external API calls by default.
* Do not download models or datasets without explicit approval.
* Real-provider integration requires explicit approval and a deliberate, bounded test.
* Never assume that an API, model, or free tier is free or available.
* Keep API keys in environment variables; never hardcode or commit them.
* Never log secrets or store their values in manifests, databases, or result files.
* Do not silently substitute a different model or provider when a request fails.
* Record every attempt, including retries and failures.
* Enforce configured request and token limits when real inference is introduced.

Mocked HTTP responses and fake-provider fixtures are acceptable for offline testing, but they do not establish that a real provider works.

## 5. Technology and Coding Standards

Use the project's existing configuration and approved dependencies.

Planned technologies include:

* Python 3.11 or a compatible version selected in `pyproject.toml`.
* Pydantic for data contracts and validation.
* PyYAML for configuration.
* Typer for the CLI when needed.
* `jsonschema` for JSON Schema validation.
* `httpx` for HTTP adapters.
* SQLite and JSONL for experiment storage.
* SciPy or suitable established statistical utilities when statistical analysis is introduced.
* Jinja2 for reports when reporting is implemented.
* pytest for tests.
* Ruff for linting and formatting.

Do not install every planned dependency upfront. Add dependencies only when required by the approved milestone.

Write clear, typed Python code. Prefer small modules, explicit interfaces, meaningful names, and useful error messages. Avoid unnecessary inheritance, global mutable state, and speculative abstractions.

Do not implement features merely because a library makes them easy to add.

## 6. Testing and Quality

* Write tests alongside implementation.
* Test expected behaviour, invalid inputs, edge cases, and failure paths.
* Use deterministic fixtures for repeatable tests.
* Avoid tests that depend on external APIs or network access.
* Test that sensitive values never appear in persisted outputs.
* Test architectural boundaries where practical.
* Run relevant pytest and Ruff checks after changes.
* Report actual test results; never claim tests passed unless they were run and passed.

Coverage is useful, but meaningful behavioural tests matter more than an arbitrary coverage percentage.

## 7. Dataset and Evaluation Integrity

* Every dataset case must have a stable identifier.
* Record dataset origin, source, licence, and version.
* Use public datasets only after checking and documenting their licences.
* Keep original test cases distinguishable from public data.
* Document assumptions, limitations, possible contamination, and known biases.
* Define scoring rules before comparing models.
* Do not silently normalise, strip, or rewrite model outputs.
* Keep benchmark changes versioned and explain changes in the dataset changelog.
* Do not claim statistical significance without an appropriate analysis.
* Clearly state when a sample is too small to support strong conclusions.

Refer to the dataset-card and metric templates under `docs/`.

## 8. Development Workflow

Niriksha is developed incrementally through explicit milestones.

Before making changes:

1. Inspect the current repository, relevant source files, and Git status.
2. Read the current milestone's scope and acceptance criteria.
3. Identify the smallest change that satisfies the requirement.
4. Preserve existing user work and unrelated changes.

While working:

* Implement only the currently approved milestone.
* Avoid unrelated refactoring and scope expansion.
* Do not overwrite existing documentation without reviewing it.
* Do not create speculative modules or features for future milestones.
* Ask before making a decision that materially changes architecture or scope.

After working:

1. Run relevant tests and lint checks.
2. Inspect the final diff.
3. Check for accidental secrets, generated files, and unrelated modifications.
4. Summarise files changed, design decisions, commands run, test results, and unresolved issues.
5. Stop at the milestone boundary and wait for approval.

## 9. Git and Repository Rules

* Preserve the existing Git history and remote configuration.
* Never reinitialise the repository.
* Never discard, reset, or overwrite unrelated user changes.
* Never commit or push without explicit approval.
* Never change GitHub repository settings or visibility without explicit approval.
* Keep `.env`, local databases, generated run outputs, and model weights out of version control.
* Keep `.env.example` limited to variable names and empty placeholders.

## 10. Scope Boundaries

Do not add these to the initial MVP without explicit approval:

* A dashboard or hosted multi-user service.
* Authentication and user management.
* LLM-as-judge scoring.
* Complex hallucination or faithfulness evaluation.
* Additional evaluation frameworks embedded into the core.
* Large-scale benchmarking or costly model comparisons.
* BERT-specific evaluation tracks.
* Automated model fallback.
* Complex distributed infrastructure.

The initial goal is a reliable, reproducible evaluation engine with a small benchmark and a clear results report.

## 11. Documentation and Communication

Keep documentation aligned with the implementation.

Use `README.md` for the project overview and current status, `docs/architecture.md` for design, and ADRs for material architectural decisions.

When explaining progress:

* Separate completed work from planned work.
* State assumptions and limitations.
* Identify anything that still requires verification.
* Do not claim real-model evaluation, provider compatibility, or production readiness without evidence.

**When in doubt:** preserve evaluation integrity, choose the simplest defensible implementation, and ask before expanding scope.
