# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### To do before the first PyPI release
- Register the PyPI pending publisher for `publish.yml` (environment `pypi`),
  then tag `v0.1.0`.

## [0.1.0] — unreleased

### Added
- pytest plugin registered through the `pytest11` entry point.
- `@pytest.mark.llm_eval(dataset=..., repeat, k, min_passes, min_score, tags)`;
  YAML / JSON / JSONL golden sets parametrize the `eval_case` argument.
- Fixtures: `llm`, `llm_judge`, `llm_eval`, `eval_case`, `llm_eval_session`,
  and overridable `llm_client` / `llm_judge_client`.
- Scorers: exact match, contains / not contains, regex, JSON Schema validity,
  numeric tolerance, ROUGE-N, ROUGE-L, BLEU, embedding similarity (pluggable,
  local fastembed extra), agent tool-call trajectory (exact, in order, subset,
  superset, argument matching).
- LLM-as-judge: pointwise rubric and pairwise comparison, hidden reasoning,
  strict JSON verdicts, N samples with median / mean / majority, position swap,
  content-hashed delimiters, injection heuristic with flag or penalize policy.
- Providers: ollama (default, local), any OpenAI-compatible server, Anthropic
  (optional extra).
- Response cache keyed by provider + model + messages + parameters + sample +
  attempt, with `auto` / `record` / `replay` / `off` modes and secret redaction.
- Cost / token / latency budgets that fail the run and refuse further calls;
  config-driven pricing table (local providers cost 0, unpriced calls warned).
- Repeat-N with pass@1 and unbiased pass@k; per-case and aggregate thresholds.
- Terminal summary, JSON and HTML reports, JUnit properties, baseline
  comparison that fails on regression.
- Example: evals for a tiny local RAG function.
