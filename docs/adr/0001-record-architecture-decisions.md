# ADR 0001: Initial architecture decisions

Status: accepted. Date: 2026-10-01.

One record for the founding decisions. Later decisions get their own numbered files.

## 1. Portfolio-first project
- **Context:** The author is building AI-engineering skills; the evaluation space already has mature tools.
- **Decision:** Optimize for demonstrable engineering depth and honest write-ups. A useful developer tool is secondary.
- **Consequences:** Rigor and documentation outrank feature breadth. Usability bar is modest.

## 2. Hybrid implementation strategy
- **Context:** Embedding a framework would hide the mechanisms the project exists to demonstrate.
- **Decision:** Write the engine (runner, store, manifests, scorers, adapter protocol) ourselves. Use established libraries for supporting work (validation, HTTP, CLI, statistics).
- **Consequences:** More code to maintain; no claim of being better than existing tools.

## 3. Provider-independent core
- **Context:** Providers differ in parameters, limits, errors and pricing.
- **Decision:** The core defines a provider protocol and never imports an adapter. Provider configuration is data.
- **Consequences:** Adding a provider does not touch the core. An import-boundary test will enforce this.

## 4. Fake-provider-first development
- **Context:** Real APIs add cost, flakiness and rate limits.
- **Decision:** The first provider is a deterministic fake with scripted successes and failures.
- **Consequences:** M1 and M2 are fully testable offline. Real-provider compatibility is proven later, not assumed.

## 5. No network access by default
- **Context:** Budget is zero and secrets must stay safe.
- **Decision:** Nothing calls the network unless explicitly enabled. CI has no secrets or external model access.
- **Consequences:** Real-provider runs need an explicit flag and are never part of CI.

## 6. Initial API budget of ₹0
- **Context:** No funds for paid APIs; free-tier availability of any provider is unverified.
- **Decision:** Plan on zero spend. Groq is a candidate only after access, limits and terms are confirmed. Cost for providers with unknown billing is reported as unknown, never zero.
- **Consequences:** Datasets and repeat counts stay small, so confidence intervals will be wide. CPU-only hardware limits local models to small quantized ones.

## 7. English first; Hindi and Kannada in the schema
- **Context:** Indic and code-mixed evaluation is a planned focus, but data takes time to build properly.
- **Decision:** The first dataset is English. The schema carries language and script fields from day one (see the dataset card template).
- **Consequences:** Hindi and Kannada cases can be added without a schema change. Indic tokenization and normalization issues are deferred to the dataset milestone.

## 8. Two initial tasks
- **Context:** Both tasks have objective, cheap, deterministic scoring.
- **Decision:** Structured JSON extraction and short-answer QA. BERT-based work is out of the MVP.
- **Consequences:** No LLM judge is needed for the MVP. Free-form generation and faithfulness are deferred.

## 9. Original cases plus licensed public data
- **Context:** Public data may be in model training sets; licences vary.
- **Decision:** Mix original cases with public data whose licence is recorded per case. Document contamination uncertainty for public items.
- **Consequences:** Extra annotation work and a dataset card per dataset. Nothing is downloaded until its licence is reviewed.

## 10. MIT licence and public repository
- **Context:** The project is a portfolio piece meant to be read and reused.
- **Decision:** MIT licence, public GitHub repository.
- **Consequences:** Holdout data must not be published. Secrets hygiene is enforced from the first commit.

## Tooling choices (M0)
- **Ruff** for linting and formatting: one fast tool, configured in `pyproject.toml`.
- **detect-secrets** for secret scanning, via pre-commit and CI: pure Python and pip-installable, avoiding a Go toolchain on Windows. Gitleaks was considered and is a reasonable alternative.
- **Python 3.11+**, `src/` layout, Hatchling build backend, no runtime dependencies yet.
