# configs

Reserved for provider profiles and run configurations (planned for M1 onward). Empty in M0.

Config files hold the *names* of environment variables for credentials, never the values.

## OpenAI-compatible profiles (M3a)

There is no config-file loader yet; a profile is built in Python and holds only the *name* of the key's environment variable:

```python
from niriksha.core.runner import RetryPolicy
from niriksha.providers.openai_compatible import (
    HTTP_RETRY_POLICY,
    OpenAICompatibleProfile,
    OpenAICompatibleProvider,
)

VARIABLE = "LOCAL_VLLM_VARIABLE"  # the NAME of the environment variable that holds the key

profile = OpenAICompatibleProfile(
    name="local-vllm",  # recorded as the provider name in manifests and results
    base_url="http://127.0.0.1:8000/v1",  # no credentials, query or fragment
    api_key_env=VARIABLE,  # leave this out if the server needs no key
)
provider = OpenAICompatibleProvider(profile)  # reads the variable now; raises if it is missing
# execute_run(config, dataset, provider, runs_dir, retry=HTTP_RETRY_POLICY, secret_values=(key,))
```

- Retries are off unless you pass `retry=`. `HTTP_RETRY_POLICY` is three attempts; retried failures are `timeout`, `rate_limit`, `server_error` and `connection_error`, with waits of 1 s then 2 s (at least the server's `Retry-After`, never more than 60 s). A retry after a timeout can repeat a request that the server already processed and may bill for; use `RetryPolicy(max_attempts=1)` to avoid it.
- `supported_params` defaults to `temperature`, `top_p`, `max_tokens`, `stop` and `response_format`. Any other parameter you set (for example `seed`) is refused before a request is sent unless the profile declares it. `max_tokens_field` can be `max_completion_tokens` for servers that require it.
- Servers that call themselves OpenAI-compatible differ; check yours. Nothing here has been run against a real service.
