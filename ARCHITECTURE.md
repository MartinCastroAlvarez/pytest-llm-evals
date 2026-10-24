# Architecture

```text
pytest_llm_evals/
  plugin.py    pytest hooks: options (flag + ini), marker, dataset parametrization,
               fixtures, the repeat/pass@k runner (pytest_pyfunc_call), gates, reports
  dataset.py   EvalCase + YAML/JSON/JSONL loading, validation, rootdir confinement
  scores.py    Score + pure scorers (string, regex, JSON Schema, numeric, ROUGE, BLEU,
               embeddings, tool-call trajectories)
  judge.py     Rubric, Judge (pointwise + pairwise), prompt builders, verdict parsing,
               injection heuristic
  llm.py       Completion, LLMClient protocol, ollama / OpenAI-compatible / Anthropic
  cache.py     ResponseCache (files under .llm-eval-cache/), CachedClient, redaction
  usage.py     Usage, Budget, UsageLedger (thread-safe), TrackedClient
  pricing.py   Price, PricingTable (built-in + YAML/JSON file, free providers)
  results.py   EvalRecorder (the llm_eval fixture), CaseResult, pass@k, JSON/HTML
               reports, baseline comparison
  testing.py   FakeLLM, FakeEmbedder (for users' and our own tests)
```

## Call path

```text
test -> llm fixture = TrackedClient(CachedClient(llm_client))
          TrackedClient: refuse if a budget is already crossed
          CachedClient:  key = sha256(provider, model, messages, params, sample, attempt)
                         hit  -> Completion(cached=True)
                         miss -> provider client -> write entry (redacted)
          TrackedClient: price (0 if cached), add to session ledger + current case,
                         raise BudgetExceededError if a budget is now crossed
```

`llm_judge` is the same stack with role `judge`, so judge spend is budgeted and
reported separately (`usage.by_role`).

## Running a case

`pytest_pyfunc_call` takes over every test that has the `llm_eval` marker or
uses the `llm_eval` / `eval_case` fixtures (sync functions only):

1. Resolve `repeat`, `k`, `min_passes` (marker, then `--llm-eval-repeat`).
2. For each attempt: set `current_attempt` (part of the cache key, so attempts
   are distinct, replayable samples), clear the recorder, call the test.
   An exception is a failed attempt with score 0; otherwise the attempt passes
   when no recorded score is below its threshold and the mean meets the case's
   `min_score`. Budget and replay-miss errors stop immediately.
3. Store a `CaseResult` (mean attempt score, passes, pass@1, pass@k, scores of
   the last attempt, usage, output) and add JUnit `user_properties`.
4. Re-raise the last exception or `pytest.fail` with the score lines when
   passes < `min_passes`.

## Session end

`pytest_sessionfinish` evaluates the gates (mean score vs
`--llm-eval-min-score`, ledger violations, baseline regressions), writes the
reports, and sets `session.exitstatus` to `TESTS_FAILED` when a gate failed on
an otherwise green run. `pytest_terminal_summary` prints the table.

## Design choices

- **Soft scores.** A case records every score before failing, so reports show
  the whole picture and a baseline can compare numbers, not just pass/fail.
- **Cache as the determinism layer.** Local models are not bit-reproducible;
  recorded responses are. `replay` mode makes CI independent of any model.
- **Content-hashed nonces.** A random nonce would make the judge prompt
  uncacheable; a hash of the graded text is stable and still unforgeable from
  inside that text.
- **No heavy dependencies.** ROUGE, BLEU and cosine are pure Python; HTTP uses
  `http.client`; embeddings and Anthropic are extras. Core deps: pytest,
  PyYAML, jsonschema.
