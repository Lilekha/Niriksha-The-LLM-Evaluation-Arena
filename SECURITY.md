# Security

Niriksha is pre-alpha and has no real-provider code yet. These rules apply from the start.

## Secret handling

- API keys live only in environment variables or a local `.env` file (gitignored). `.env.example` lists variable names only.
- Config files store the **name** of the environment variable (for example `api_key_env: GROQ_API_KEY`), never its value.
- Keys must never be written to logs, run manifests, raw result files or result databases. This will be enforced by tests when the store exists (M1).
- Never paste a real key into an issue, pull request or test fixture.

## No network by default

The default install makes no network calls, and CI runs with no secrets and no external model access. Real providers will require explicit opt-in when they are added.

## Secret scanning

Tool: [detect-secrets](https://github.com/Yelp/detect-secrets), run through pre-commit and in CI. Chosen because it is a maintained pure-Python tool that installs with pip, avoiding a Go toolchain dependency on Windows. Gitleaks is a reasonable alternative. It is not configured here.

One-time setup: `pip install -e ".[dev]"` then `pre-commit install`. The baseline is `.secrets.baseline`; review any new finding before adding it to the baseline.

### Verification procedure

Use only a deliberately fake token.

1. Create an untracked file `fake_secret_test.txt` containing the line `aws_access_key_id = ` followed by the AWS documentation example key, which is `AKIA` + `IOSFODNN7EXAMPLE` joined together (written split here so this file does not itself trip the scanner).
2. Run `detect-secrets-hook --baseline .secrets.baseline fake_secret_test.txt`. It must exit non-zero and report a potential secret.
3. Delete the file. Run `git status` and confirm nothing from the test is untracked, staged or committed.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting (the repository's Security tab) if it is enabled. Otherwise open an issue asking for a private contact, without including any details of the vulnerability.
