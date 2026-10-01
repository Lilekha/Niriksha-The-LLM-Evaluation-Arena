# ADR 0002: Provider contract

Status: accepted. Date: 2026-10-01.

## Context
The core must stay provider-independent, record every failure honestly, and keep generation separate from scoring.

## Decisions
1. **Pydantic v2** (`pydantic>=2.7,<3`) is the only runtime dependency. Models are frozen, strict and reject unknown fields, so nothing is silently coerced or dropped.
2. **One attempt per call.** `Provider.generate()` is synchronous, makes a single attempt and never retries. The runner owns retries and stores every attempt.
3. **Expected failures are values.** Provider and transport problems come back as `GenerationFailure`, a member of a discriminated union with `GenerationSuccess`. Unexpected exceptions are not caught and converted; they propagate. `internal_error` is for faults an adapter itself detects, not a catch-all.
4. **Invalid requests fail early.** Schema validation errors raise at request construction, before any attempt is recorded. They are not model outcomes.
5. **Seven failure kinds:** `timeout`, `rate_limit`, `server_error`, `malformed_response`, `invalid_request`, `unsupported_capability`, `internal_error`. `auth_error` and `connection_error` are deferred to M3a, when real HTTP exists. The `retryable` property is a hint; retry policy belongs to the runner.
6. **Timing belongs to the runner.** Results carry no `started_at` or `latency_s`. The runner measures elapsed time around each call (M1.2), so latency is measured the same way for every provider. A provider-reported processing time can be added later as a separate, non-authoritative field.
7. **No cost field.** Cost is derived later from usage and a dated price table. Unknown usage stays `None`; unknown cost is never zero.
8. **No raw payload capture yet.** Adapters put short context in `message`. Bounded raw capture is revisited in M3 with real payloads, and full raw HTTP payloads and auth headers are never persisted.
9. **Metadata guard.** `provider_metadata` is flat, JSON-safe and rejects keys containing obvious secret words (authorization, api-key, secret, password, cookie, bearer). This is defence in depth, not a substitute for keeping secrets out of results.
10. **Boundary enforced by test.** `niriksha.core` must not import `niriksha.providers`, `niriksha.scorers`, or networking libraries. This is stricter than the minimum rule in `CLAUDE.md`.

## Consequences
- Adding a provider means implementing a two-member protocol; the core does not change.
- Behavioural contract tests (request-ID and model consistency, no hidden retries) arrive with the fake provider and runner in M1.2. M1.1 only tests the schemas and the protocol shape.
- Strict mode means callers pass real enum members and typed values; in Python, a list where a tuple is expected is an error.
