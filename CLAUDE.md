# CLAUDE.md

Contract between this repository and any AI agent contributing to it.

## Required reading

1. This file. 2. [`README.md`](README.md). 3. [`ARCHITECTURE.md`](ARCHITECTURE.md).
4. [`SECURITY.md`](SECURITY.md).

## Mission

`pytest-llm-evals` makes LLM and agent evaluation a normal pytest run: local
models by default, deterministic re-runs, gates that fail CI.

## The rules

1. **No network in tests.** Use `pytest_llm_evals.testing.FakeLLM`, the
   `fake_server` fixture or a fake SDK module. Never a hosted model, never ollama.
2. **Plugin behaviour is tested with pytester** (`tests/test_plugin.py`):
   outcomes, exit codes, terminal lines and report files.
3. **Untrusted model output.** Graded text goes through the judge's delimited
   prompt; never format it into an instruction. Reports escape it.
4. **No secrets on disk.** Cache entries go through `cache.redact`; never add a
   credential to a cache key, a report, a log line or an exception message.
5. **Scores are in [0, 1]**; scorers with partial credit take a `threshold`.
6. New options get a flag, an ini key, a README row and a CHANGELOG entry.
7. Never add `# noqa`, `# nosec` or `# type: ignore` to the package, skip a
   test, or loosen lint/type/coverage config. Fix the code.

## Done means

`poetry run coverage run -m pytest && poetry run coverage report` (>= 85%),
`pytest examples`, `ruff check`, `ruff format --check`, `mypy` (strict),
`bandit`, `pip-audit` all green; docs and CHANGELOG updated.

## Git

Small commits, imperative subjects, never force-push `main`, never commit
secrets, `.env` or `.llm-eval-cache/` of this repository.
