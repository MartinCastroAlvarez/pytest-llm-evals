# Examples

`rag.py` is a tiny retrieval-augmented QA function; `test_rag_evals.py`
evaluates it with a golden set (`evals/qa.yaml`), string/regex/ROUGE/JSON-schema
scorers, an LLM judge with a rubric, a pairwise judge facing a prompt
injection, a trajectory check and a repeat-3 JSON contract.

```bash
pytest examples --llm-eval-json=llm-eval-report.json --llm-eval-min-score=0.8
pytest examples --llm-cache=replay --llm-eval-baseline=llm-eval-report.json   # free re-run
```

By default `conftest.py` plugs in deterministic stand-ins (`local_models.py`):
an extractive reader and an overlap judge, so the example runs in CI with no
model. With ollama installed:

```bash
ollama pull qwen3
LLM_EVAL_EXAMPLES_LIVE=1 pytest examples --llm-model qwen3 --llm-judge-model qwen3
```
