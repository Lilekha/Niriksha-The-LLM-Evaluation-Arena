# ADR 0003: Dataset identity and run persistence

Status: accepted. Date: 2026-10-01.

Covers M1.3: local dataset loading, content hashing, provenance, run storage and resume. Everything here is offline and uses the fake provider only. There is still no scorer, CLI, real provider integration or benchmark dataset.

## Context
A comparison is only meaningful if the exact cases, prompt and settings are known, benchmark cases cannot change unnoticed, and an interrupted run can continue without duplicating or losing results.

## Decisions

### Dataset format
1. A dataset is a local directory with `dataset.json` (schema version, name, version tag, task, description, pinned `content_sha256`, and `output_schema` for extraction) and `cases.jsonl` (one case per line; file order is dataset order). No YAML, no dataset framework: stdlib JSON plus Pydantic.
2. Tasks: `short_answer_qa` (`question`, `answers`) and `json_extraction` (`text`, `expected` object). Every case also carries `id`, `split` (`dev`/`test`/`holdout`), `language` (`en`/`hi`/`kn`), `script` (`Latn`/`Deva`/`Knda`), `code_mixed`, `origin` (`original`/`public`), `source` (required for public) and `license`.
3. Language and script must be consistent (`en`: Latn; `hi`: Deva or Latn; `kn`: Knda or Latn; romanised Hindi or Kannada is the language with script Latn). The language and script sets are closed until another is needed.
4. Loading is strict and rejects rather than repairs: malformed JSON, non-objects, duplicate JSON keys, `NaN`/`Infinity`, out-of-range numbers, invalid UTF-8, a byte-order mark, blank lines, unknown or missing fields, duplicate IDs, empty datasets, lone surrogates, and any string that is not Unicode NFC. Nothing is normalised automatically, which matters for Devanagari and Kannada.
5. Files are read as bytes and split on LF only (CRLF tolerated). `str.splitlines()` is not used, because U+2028 is legal inside a JSON string. Up to 20 issues are reported together, each with file, line and field. Error text never echoes record contents or absolute paths.

### Dataset identity
6. `content_sha256` is SHA-256 over canonical JSON of `{"cases": [raw parsed cases in file order], "hash_version": 1, "output_schema": ..., "schema_version": 1, "task": ...}`, serialised with `sort_keys=True`, separators `(",", ":")`, `ensure_ascii=False`, `allow_nan=False`, encoded as UTF-8. It is a project-defined form, not RFC 8785.
7. Cases are hashed as parsed from the file, not as model dumps, so adding an optional field with a default in a later schema does not change existing hashes. The cost: omitting a defaulted field and writing its default hash differently.
8. Name, version, description and the stored hash are excluded. Whitespace, key order, CRLF vs LF and a final newline do not affect the hash. Case order, values and JSON types do (`1` and `1.0` differ, so avoid floats in gold data).
9. Version and hash: the `version` tag is a human label and the hash is the identity. The loader recomputes the hash and rejects a mismatch, printing the computed value. Changing any case therefore forces a deliberate act: update the pin, bump `version` and add a changelog entry in the dataset card. The loader never edits files. It cannot detect a content change that reuses the old version number; the hash pin catches the content change but not the missing bump.
10. The hash is pinned by golden values in the tests. CI is configured to run them on Python 3.11 and 3.13, but CI has not run yet and only 3.13 has been run locally, so cross-version stability is not claimed.

### Runs, manifest and results
11. A run lives in `runs/<run_id>/` (gitignored): `manifest.json`, written once with exclusive create and never modified, and `results.jsonl`, append-only, one line per completed request.
12. `run_id` is chosen by the caller: `[a-z0-9][a-z0-9_-]{0,63}`, lowercase only (no collisions on case-insensitive filesystems), no dots, and Windows reserved device names rejected. Creating the directory is the claim, so an existing run is never overwritten and two starters cannot both win.
13. The manifest records: run ID and UTC creation time; dataset name, version, task, schema version and hash; selected splits, case count and a hash of the selected case IDs; the prompt template and its hash; provider name and implementation (module-qualified class name); requested model and generation parameters; niriksha, Python, pydantic and platform versions; git commit and dirty flag. Unavailable values are `null`. It stores no absolute path, username, hostname, API key or authorization header, and its models forbid extra fields.
14. A result line holds `request_id` (the case ID), `request_sha256` (hash of the rendered request), `recorded_at` (wall clock, informational) and the runner's `ExecutionRecord`. Timing stays owned by the runner (ADR 0002).
15. The prompt template's `{input}` is replaced with `str.replace`, never `str.format`, so JSON braces in prompts are safe.

### Persistence boundary and secrets
16. Before every append the result line is serialised, re-parsed and compared with the in-memory object. A frozen Pydantic model can still hold a mutable dict, so a metadata entry added after construction is caught by re-running all validators. The comparison also catches silent changes: pydantic writes a mutated `NaN` as `null`. Serialisation warnings are disabled because they would print the offending values. A refusal writes nothing and never includes the refused value. A refusal is raised outside the `except` block, so the validation error that holds the offending input is not chained to it.
17. An optional guard takes secret values the caller holds in memory and refuses any result line, and the run manifest (for example an API key pasted into the prompt template), containing one (raw and JSON-escaped forms). Values shorter than 8 characters are rejected as guards. It is a backstop only. It does not detect a secret the caller did not list, or one that was transformed, and it is not a comprehensive secret scanner. Neither refusals nor error messages include the secret.
18. Files are written as UTF-8 bytes with LF line endings, flushed and fsynced. A failed append (a write, flush or fsync error, or an interrupt) is truncated away on a best-effort basis, so no partial or unconfirmed line remains; the result is lost and resume issues that call again. An append is refused, with nothing written, if the file already ends with an incomplete line, so a torn tail can never be merged into a corrupt middle line.

### Resume
19. `resume_run` first checks that dataset hash, selection, prompt, provider name and implementation, requested model, parameters and `niriksha`/Python/pydantic/platform versions and git commit all match the immutable manifest. Otherwise it refuses with an error naming the differing fields (never their values). The dirty flag is ignored because it changes constantly during development. The git commits must be equal: a known commit versus an unknown one is a difference, while two unknown commits are not.
20. Only after those checks pass does it read the results. An unterminated final line is a possible torn write: it is truncated, and its case runs again because it has no complete result. A corrupt line anywhere else, a blank line, a duplicate request ID, an ID outside the selection, a stored request hash that differs from the rebuilt request, or a stored result whose provider or requested model differs from the manifest is an error. Nothing on disk changes if any check fails.
21. Only selected cases without a recorded result are run. A recorded failure is a completed attempt and is never retried by resume.

## Limits (deliberately not solved here)
- **Possible duplicate call.** If the process dies while a provider call is in flight, resume issues that call again. With a real provider this could be a second billable call. Recording an attempt-start marker is deferred until real providers exist.
- **Single writer.** Concurrent resumes of the same run are unsupported. There is no locking, because a stale lock file after a crash would be a worse failure than the one it prevents.
- **Durability.** The directory entry itself is not fsynced (Windows cannot), so a crash right after a run is created may lose it. A crash between creating the run directory and finishing the manifest leaves an empty directory that blocks that run ID until it is removed by hand.
- **Lost results.** If an append fails after the provider call succeeded, the result is not kept and resume calls the provider again (a possible duplicate billable call with a real provider).
- **Dataset objects.** `execute_run` trusts the `Dataset` it is given. The hash pin is verified by `load_dataset`, so build datasets only through it; a hand-built `Dataset` bypasses the pin.
- **Secret scanning trade-off.** To keep dataset hashes from tripping detect-secrets, the baseline skips lines that consist solely of a `"content_sha256"` pin (anchored regex: key, 64 lowercase hex characters, optional comma, nothing else). Nothing else is skipped, and `tests/test_secret_scan_config.py` locks this. Residual risk: a genuine 64-hex secret written as the value of a `content_sha256` pin line would not be flagged. A consequence is that a pin in compact (single-line) JSON is flagged, so keep `dataset.json` pretty-printed with the pin on its own line.
- **Provenance is not reproducibility.** A recorded dataset hash and manifest say what was run. They do not make the run reproducible: provider-side nondeterminism and silent model updates, hardware, unlocked dependencies (there is no lockfile hash yet), the dirty working tree and wall-clock effects remain outside what is captured.
- **One attempt per case per run.** There is no retry policy; duplicate request IDs within a run are impossible because case IDs are unique.
- **Partial runs.** An exception partway through leaves a valid partial run on disk, which resume continues.
- **Not validated yet:** `expected` against `output_schema` (needs `jsonschema`, arriving with the scorers), and `output_schema` is only checked to be a JSON object.

## Consequences
- Benchmark edits are visible and deliberate; the hash pin is the enforcement.
- Runs are inspectable with any text tool, and re-scoring later needs no new inference calls.
- Config files, a CLI and provider profiles are future work; `RunConfig` is a Python object for now.
- Windows specifics handled: lowercase run IDs and reserved names, explicit LF, bytes-level reads, no directory fsync.
