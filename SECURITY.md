# Security

Niriksha is pre-alpha. It has an OpenAI-compatible HTTP adapter (M3a) and a manual live smoke script (M3b), but the adapter's tests exercise only a local fake server: the tests and CI make no real provider calls, use no real keys and need no credentials. One live smoke test (8 requests to Groq with synthetic fixtures, 2026-10-04) has been run outside CI; it shows connectivity only and is not a security or quality validation. The smoke script is outside CI and pytest, needs a separate explicit approval before each use, and is described in `docs/provider-validation.md`. These rules apply from the start.

## Secret handling

- API keys live only in environment variables or a local `.env` file (gitignored). `.env.example` lists variable names only.
- Config files store the **name** of the environment variable (for example `api_key_env: GROQ_API_KEY`), never its value.
- Keys must never be written to logs, run manifests, raw result files or result databases. The adapter reads a key from the environment variable a profile names, sends it only in the `Authorization` header, never copies server error text into messages (servers can echo keys), and refuses to send a key over plain http to a non-loopback host. Pass the key to `execute_run(secret_values=(...))` so the run store refuses to persist it. Tests check that the key appears in no message, metadata, repr, run file or score file.
- Never paste a real key into an issue, pull request or test fixture.

## No network by default

The default install makes no network calls, and CI runs with no secrets and no external model access. Real providers will require explicit opt-in when they are added.

## Secret scanning

Tool: [detect-secrets](https://github.com/Yelp/detect-secrets), run through pre-commit and in CI. Chosen because it is a maintained pure-Python tool that installs with pip, avoiding a Go toolchain dependency on Windows. Gitleaks is a reasonable alternative. It is not configured here.

Dataset content hashes (`"content_sha256"` in `dataset.json`) look like high-entropy secrets. The baseline therefore skips lines that consist solely of such a pin: the key, a 64-character lowercase hex value, an optional comma, and nothing else (the regex is anchored). Every other line is scanned normally, including a second secret placed beside a pin, a pin written in compact single-line JSON (keep `dataset.json` pretty-printed), a non-SHA-256 value, and any 64-hex value under a different key. `tests/test_secret_scan_config.py` checks both directions. The residual risk is a real 64-hex secret deliberately written as a pin value on its own line, which would not be flagged.

**Windows: run the scan in UTF-8 mode.** detect-secrets opens files with the locale encoding. On a cp1252 system it silently skips any file it cannot decode (in this repository, files containing Hindi or Kannada text), so a local scan can pass while CI, which uses UTF-8, fails. Set `PYTHONUTF8=1` first (PowerShell: `$env:PYTHONUTF8 = "1"`; bash: `export PYTHONUTF8=1`) for `detect-secrets-hook` and `pre-commit`.

One-time setup: `pip install -e ".[dev]"` then `pre-commit install`. The baseline is `.secrets.baseline`; review any new finding before adding it to the baseline.

### Verification procedure

Use only a deliberately fake token.

1. Create an untracked file `fake_secret_test.txt` containing the line `aws_access_key_id = ` followed by the AWS documentation example key, which is `AKIA` + `IOSFODNN7EXAMPLE` joined together (written split here so this file does not itself trip the scanner).
2. Run `detect-secrets-hook --baseline .secrets.baseline fake_secret_test.txt`. It must exit non-zero and report a potential secret.
3. Delete the file. Run `git status` and confirm nothing from the test is untracked, staged or committed.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting (the repository's Security tab) if it is enabled. Otherwise open an issue asking for a private contact, without including any details of the vulnerability.
