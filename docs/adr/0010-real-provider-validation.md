# ADR 0010: Real-provider validation protocol

Status: accepted. Date: 2026-10-04.

Covers M3b stage B: the safeguards and tooling for a first live smoke test of the OpenAI-compatible adapter ([ADR 0009](0009-openai-compatible-adapter-and-retries.md)). No live request was made while this stage was built or tested, and none is part of this change. The provider research and the gate are recorded in [provider-validation.md](../provider-validation.md).

Update (2026-10-05): one live smoke test (Stage C) has since been run, with Groq and `openai/gpt-oss-20b`; its observations are in provider-validation.md. It does not change the decisions below, and each further live request still needs separate explicit approval.

## Context
ADR 0001 sets a ₹0 budget and says a provider is a candidate "only after access, limits and terms are confirmed". Nothing in the repository defined M3b, so this ADR fixes what the first live check may do.

## Decisions
1. **A zero-cost gate comes first.** A provider is used only if free access, payment-method requirements, overage behaviour, regional availability and data terms are adequately verified. Otherwise the live test is postponed. As of 2026-10-04 no provider fully passed from public documentation alone; Groq is a conditional preferred candidate, pending checks only the user can make in their own account. Unverified items stay marked unverified.
2. **Live use is a separate approval.** Building the harness does not authorize a request. Before any request, the endpoint, model, request plan, caps, data terms and open uncertainties are presented and approved, and the confirmation prompt in the script is not that approval.
3. **A script, not a library feature.** `scripts/live_smoke.py` is outside the package and outside pytest and CI. It adds no provider registry, no CLI framework, no config loader and no dependency. It reads the endpoint, the model and the *name* of the key variable from the environment, and does not load `.env`.
4. **Hard caps that the environment cannot raise.** At most 8 requests (the 5 `tiny_qa` and 3 `tiny_extraction` cases), temperature 0, `max_tokens` 32 and 128, at least 3 s between requests, a timeout of at most 300 s. `NIRIKSHA_SMOKE_TASKS` can only reduce the number of requests.
5. **One attempt, no retries.** `NO_RETRY` is used deliberately, not `HTTP_RETRY_POLICY`, so a failure costs one request and nothing repeats. A `BudgetedProvider` subclass of the adapter refuses call N+1 before any request is built. An existing run id is refused, never resumed.
6. **Synthetic data only.** The script has no dataset parameter and runs only the two test fixtures, checking their names and case counts.
7. **Explicit start.** The script prints the plan and requires typing the exact request count; end of input or any other answer aborts. A remote endpoint must use https (plain http only for loopback, for example a local Ollama).
8. **Secrets.** The key is read only from the environment variable the user names, is passed to the run store's `secret_values` guard, and the run and score directories are scanned for it afterwards (a hit stops the run with a warning that does not print it). Output contains counts, latencies, the returned model name, reported usage and scores only: no model text, headers, profile or key. No logging is configured.
9. **Honest recording.** Observations from a live run are written to `docs/provider-validation.md` as observed, including failures; adapter changes only follow a demonstrated defect and carry an offline regression test.

## Consequences and limits
- The run manifests of smoke runs record the provider implementation as the script's `BudgetedProvider`, which marks them as smoke runs.
- Writing tests for the harness found a real defect before any live call: run ids must be lower case, and the script's first timestamp format was not.
- Offline tests use the local fake server only and prove the caps, the single attempt per case, the confirmation and the secret handling; they do not show that any real provider behaves the same.
- A provider's terms, limits and model list change, so every statement in the validation record is dated and must be re-checked before use.
