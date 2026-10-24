# Contributing to pytest-llm-evals

## TL;DR

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install poetry
poetry install -E anthropic
poetry run coverage run -m pytest && poetry run coverage report
poetry run pytest examples
poetry run ruff check pytest_llm_evals tests examples && poetry run ruff format --check pytest_llm_evals tests examples
poetry run mypy pytest_llm_evals
poetry run bandit -c pyproject.toml -q -r pytest_llm_evals
poetry run pip-audit
```

No model is needed: tests use scripted fakes and a local HTTP stub. To try the
example against a real local model, install [ollama](https://ollama.com),
`ollama pull qwen3`, then `LLM_EVAL_EXAMPLES_LIVE=1 poetry run pytest examples`.

## Adding a scorer

1. A pure function in `pytest_llm_evals/scores.py` returning `Score(name,
   value in [0, 1], threshold, details)`; export it from `__init__.py`.
2. Unit tests in `tests/test_scores.py`, including the failure case.
3. A row in the README scorer table and a CHANGELOG entry.

## Adding an option

Add it to `OPTIONS` in `plugin.py` (it becomes a flag and an ini key), test it
through pytester, document it in the README configuration table.

## Releases

1. Open a release PR that bumps `pyproject.toml`, `__init__.__version__` and
   `CHANGELOG.md`.
2. Merge it.
3. Tag the merge commit `vX.Y.Z` and push the tag; `publish.yml` checks that
   the tag matches the version, runs the tests, scans the artefacts for tokens,
   publishes to PyPI with trusted publishing and creates the GitHub Release.

The PyPI pending publisher (owner `MartinCastroAlvarez`, repository
`pytest-llm-evals`, workflow `publish.yml`, environment `pypi`) must exist
before the first tag.
