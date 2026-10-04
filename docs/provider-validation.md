# Real-provider validation (M3b)

**Status (updated 2026-10-05): the harness exists and one live smoke test has been run (see "Observations from live runs").** While the harness was built (Stage B), no live request was made, no account was created, no key was requested or used, and no model was downloaded. The one live smoke test sent 8 requests to Groq's OpenAI-compatible API with `openai/gpt-oss-20b`, using synthetic fixtures only. It shows connectivity and successful structured extraction on a tiny sample. It is not a general provider validation, a demonstrated QA result, or evidence of model quality: the QA outputs were empty, so the QA score measures nothing.

Every live request needs its own separate, explicit approval (Stage C below). Starting `scripts/live_smoke.py` is not that approval, and the one completed run does not authorize further runs.

## Where the facts come from
The provider facts below were read from public documentation on 2026-10-04 through a tool that returns summaries of the pages, not their raw text. Check the cited page before relying on a number. Where an official page did not state something it is marked **unverified**, and facts that came only from forum posts or search snippets are marked as such. Third-party "free API" listings were not used. Terms, limits and model lists change; re-check on the day.

## The zero-cost gate
The budget is ₹0 (ADR 0001). A provider qualifies only if **all** hold, verified for the account that will be used:
1. A free tier exists for this account, with no payment method required or attached, and exhausting it stops service instead of billing.
2. The data terms for that tier are acceptable for synthetic prompts (training use, retention, human review).
3. An OpenAI-compatible chat-completions endpoint is documented.
4. A non-reasoning instruct model is available (reasoning models can put thinking text in the content and break JSON scoring; the adapter cannot turn it off).
5. The free limits allow at least ten requests.
6. It is available in India.

If any item stays unverified or fails, the live test is postponed, not paid for.

## Candidates

### Groq: conditional preferred candidate, not a confirmed selection
| Item | Finding | Status |
|---|---|---|
| Free access | A "Free plan" exists, listed at $0 ([plans](https://console.groq.com/settings/billing/plans), [rate limits](https://console.groq.com/docs/rate-limits)). | Seen in an official search-result snippet, not verified by reading the official page directly; still needs verification |
| Payment method for the free plan | Not stated on any official page read. Upgrading to the paid Developer plan needs a payment method and "there's no immediate charge" ([billing FAQs](https://console.groq.com/docs/billing-faqs)). A community-forum answer, seen only as a search snippet, says the free plan needs no card and is never charged. | **Unverified officially** |
| Overage on the free plan | Spend limits exist only for paid accounts ([spend limits](https://console.groq.com/docs/spend-limits)). Exceeding free limits returns HTTP 429 with `retry-after` and `x-ratelimit-*` headers. Free-plan charges are not described either way. | Partly verified |
| Free limits for the intended model | Not in the public docs; they are on the account's limits page. One table row for `openai/gpt-oss-120b` shows 30 requests per minute, 1K per day, 8K tokens per minute and 200K per day, without saying which plan it is. | **Unverifiable without an account** |
| Regions | A community-forum snippet says everywhere except Greater China, Russia, Syria, Iran, North Korea and Cuba, which would include India. | **Unverified officially** |
| Data terms | By default Groq does not retain inference data; inputs and outputs may be logged temporarily for troubleshooting or abuse, up to 30 days; Zero Data Retention can be enabled by any customer ([your data](https://console.groq.com/docs/your-data)). The Services Agreement says Groq may not use Inputs or Outputs for training or fine-tuning without permission ([agreement](https://console.groq.com/docs/legal/services-agreement)); that it applies to free accounts was not confirmed. | Mostly verified |
| Endpoint | Base URL `https://api.groq.com/openai/v1`. Unsupported (HTTP 400): `logprobs`, `logit_bias`, `top_logprobs`, `messages[].name`; `n` must be 1; temperature 0 is converted to 1e-8 ([compatibility](https://console.groq.com/docs/openai)). `max_tokens` is "deprecated in favor of `max_completion_tokens`"; `seed` is best effort; `stop` allows 4 sequences ([API reference](https://console.groq.com/docs/api-reference)). Responses include `usage` and `finish_reason`. | Verified |
| Model | `llama-3.1-8b-instant` is listed as a production, non-reasoning model ([models](https://console.groq.com/docs/models)); `openai/gpt-oss-*` and the Qwen preview model are reasoning-type. Check [deprecations](https://console.groq.com/docs/deprecations) on the day. | Verified for the date read |
| `response_format` | The reference shows `json_schema`; `json_object` for this model was not confirmed, so the harness does not send it. | **Unverified** |

Three gate items can only be confirmed in the user's own account: no payment method requested or attached for the Free plan, the free limits for the chosen model, and that India is accepted at signup. If signup asks for a card, stop. Never upgrade to the Developer plan.

### Google Gemini API free tier: passes zero-cost, weaker on privacy
Billing is not required for the free tier (moving to paid needs a linked billing account and a minimum $5 prepay; [rate limits](https://ai.google.dev/gemini-api/docs/rate-limits), page dated 2026-09-02; [billing](https://ai.google.dev/gemini-api/docs/billing)); whether a card is needed at signup was not stated. India is on the available-regions list ([regions](https://ai.google.dev/gemini-api/docs/available-regions)). The data terms are the weak point: for unpaid services Google "uses the content you submit ... and any generated responses to provide, improve, and develop Google products", "human reviewers may read, annotate, and process" input and output, and the terms say not to submit sensitive, confidential or personal information ([terms](https://ai.google.dev/gemini-api/terms); [pricing](https://ai.google.dev/gemini-api/docs/pricing) shows "used to improve our products: yes" for free-tier models). Per-model free limits are only in AI Studio. The OpenAI-compatible layer is documented thinly (support for `max_tokens`, `seed`, `stop` and `response_format` was not stated on the page read), and thinking cannot be turned off on the 2.5 Pro and 3 models, where thinking tokens can consume a small token cap.

### OpenRouter: not recommended
Free models allow 20 requests per minute and 50 per day for accounts with under $10 purchased; no upfront payment is needed ([limits](https://openrouter.ai/docs/api-reference/limits)). Each free model has its own upstream provider and data policy, the defaults are not stated, and the free model list is volatile ([privacy](https://openrouter.ai/docs/features/privacy-and-logging)). Data terms are not adequately verified.

### Eliminated
- **Cerebras:** its docs say API access stays inactive without a payment method at sign-up ([rate limits](https://inference-docs.cerebras.ai/support/rate-limits)), so it fails the gate.
- **GitHub Models:** retired as of 2026-07-30 per GitHub's own docs.
- **Mistral (free Experiment plan):** the help center says the API and Studio data-sharing toggle can be disabled but users are not opted out by default ([help center](https://help.mistral.ai/en/articles/455207-can-i-opt-out-of-my-input-or-output-data-being-used-for-training)); phone verification and the free limits were learned only from third-party pages. Not adequately verified.

### Local Ollama: certainly zero-cost, not a cloud test
Base URL `http://localhost:11434/v1/`; the API key is "required but ignored"; supported chat fields include `seed`, `stop`, `temperature`, `top_p`, `max_tokens`, `response_format` and `stream`; `n`, `logit_bias`, `user` and `tool_choice` are not supported ([docs](https://docs.ollama.com/api/openai-compatibility)). No cost, no key, no data leaves the machine. On the author's machine Ollama is installed with no model downloaded; a download needs explicit approval, and a CPU-only machine can only run a very small model, with a slow first response (raise `NIRIKSHA_SMOKE_TIMEOUT_S`).

## Recommendation
1. Groq with `llama-3.1-8b-instant`, **only after** the three account-side checks above pass and are recorded here.
2. Otherwise local Ollama with one small model, after approving the download.
3. Otherwise postpone further live testing. The adapter then stays documented as not generally validated against a real service: the only live evidence is the one tiny smoke test recorded under "Observations from live runs".

## Stages
- **Stage A (done):** research and this record.
- **Stage B (done):** `scripts/live_smoke.py`, its offline tests and these documents. No live request was made in this stage.
- **Stage C (first run done 2026-10-04, see Observations; each further run needs separate explicit approval):** before any live request, the exact endpoint, model ID, request plan, token caps, the data terms for that tier and the open uncertainties are presented and approved. Passing the script's confirmation prompt does not count.

## Runbook (Stage C, after approval)
Run it yourself in a shell where the key is set for this session only (never in a file, never pasted into chat). In PowerShell:

```powershell
$env:NIRIKSHA_SMOKE_BASE_URL = "https://api.groq.com/openai/v1"   # only if approved
$env:NIRIKSHA_SMOKE_MODEL = "llama-3.1-8b-instant"                 # only if approved
$env:NIRIKSHA_SMOKE_KEY_VARIABLE = "GROQ_API_KEY"                  # the NAME of the variable
$env:NIRIKSHA_SMOKE_MAX_TOKENS_FIELD = "max_completion_tokens"
python scripts/live_smoke.py
```

The script prints the plan (endpoint, model, 8 requests, one attempt each, no retries, caps) and sends nothing until you type `8`. `NIRIKSHA_SMOKE_TASKS=extraction` or `qa` sends fewer requests. Afterwards check the provider dashboard shows no spend, then record the observations below, then revoke or delete the key.

Caps (not changeable from the environment): at most 8 requests (5 `tiny_qa` and 3 `tiny_extraction` cases), temperature 0, `max_tokens` 32 and 128, one attempt each, at least 3 s between requests, only the synthetic fixtures.

Provider identity in the records: smoke-run manifests record the provider name as `live-smoke` (and the implementation as the script's `BudgetedProvider`). The actual endpoint is recorded only in each result's per-result metadata (`endpoint`), not in the manifest, so the manifest does not independently identify the provider endpoint.

## What counts as success
Every request yields a result (a success or a typed failure); the request count equals the plan; both runs pass `load_run`; scoring and `verify_artifact` pass; token use stays within the caps; the key appears in no run or score file; the dashboard shows no spend. A failure to meet this is a finding, not something to hide.

## Observations from live runs

### First smoke test (2026-10-04): Groq, `openai/gpt-oss-20b`
Endpoint: Groq's OpenAI-compatible API. Model: `openai/gpt-oss-20b` (the server returned the same name). Runs: `smoke-qa-20261004t190318z` and `smoke-extraction-20261004t190318z`, 8 requests in total, one attempt each, temperature 0, `max_tokens` 32 for QA and 128 for extraction.

| Run | Requests | Result | Reported total tokens |
|---|---|---|---|
| QA (5 cases) | 5 of 5 succeeded | All five visible outputs were empty. All five had `finish_reason="length"` and used the full 32-token completion cap. Normalized exact match = 0.000. | 622 |
| Extraction (3 cases) | 3 of 3 succeeded | JSON parse validity, JSON Schema validity and field exact match were all 1.000 on this three-case sample. | 553 |

- **The QA score of 0.000 is not a measurement of QA accuracy.** No visible answer was produced, so there was nothing to score. An empty output from a successful generation is scored 0.0 by the metric's definition.
- **Persistence and verification:** both runs loaded successfully (`load_run`) and all four score artifacts passed verification (`verify_artifact`).
- **Hypothesis, not verified:** the model may have used its completion budget without producing visible text. Two things are consistent with this: the extraction requests reported far more completion tokens than their short visible outputs, and the QA requests reported exactly the cap with empty text. The artifacts do not contain the cause, and this has not been confirmed.
- **What these results show:** connectivity to the endpoint and successful structured extraction on a tiny sample. They do not show general model quality, QA ability, or how the adapter behaves with other models or settings.

## Limits of this validation
One model, one endpoint, one date; synthetic prompts only; three cases that do not support any quality claim; no retry behaviour is exercised on purpose (a real 429 is observed only if it happens); cost stays unknown because the adapter computes none.
