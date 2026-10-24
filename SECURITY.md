# Security

pytest-llm-evals runs inside other people's test suites. It reads datasets,
calls models, writes a cache and reports, and asks a model to grade text
produced by another model. Design goal: nothing a dataset, a model response or
a graded output contains can make it read outside the project, run code, leak
credentials to disk or logs, or talk the judge into a passing grade.

## Reporting a vulnerability

Open a private security advisory at
<https://github.com/MartinCastroAlvarez/pytest-llm-evals/security/advisories/new>.
Do **not** open a public issue for a suspected vulnerability. We acknowledge
within 5 business days and aim to ship a fix within 30 days; you are credited
in the advisory unless you ask otherwise.

Supported versions: the latest minor release.

## 1. Threat model

**Assets.** The developer's API keys, the machine running the tests, the
integrity of eval verdicts (a CI gate), and any data inside prompts.

| Actor | Controls | Must not be able to |
| --- | --- | --- |
| Output of the app under test | every graded string | steer the judge's score or winner, close the data block, inject markup into reports |
| Judge model | its raw answer | pass an out-of-scale score, smuggle reasoning into reports |
| Dataset author / a dataset from elsewhere | YAML/JSON/JSONL content, the marker path | execute code via YAML tags, read files outside the rootdir, exhaust memory |
| Anyone with read access to the cache or reports | files under `.llm-eval-cache/`, JSON/HTML reports | find API keys or token-shaped secrets |
| A hostile or broken model server | HTTP responses | crash the run with unbounded bodies, get credentials echoed into exceptions |

## 2. Controls

| Threat | Control |
| --- | --- |
| Prompt injection into the judge | Graded text, request and reference are wrapped in `<<<LABEL nonce>>> ... <<<END LABEL nonce>>>`; the nonce is a SHA-256 prefix of the graded content, so the content cannot contain its own closing marker; `<<<` and `>>>` inside it are defanged. The system prompt declares block content to be data and attempts to steer the grade a defect. `looks_like_injection` flags common phrasing and `on_injection="penalize"` gives such candidates the minimum. Tested with a modelled aligned judge, a gullible judge, forged markers and a naive prompt for contrast (`tests/test_judge.py`). |
| Grade inflation by a broken judge | Verdicts are strict JSON with an integer inside the rubric scale; anything else is an invalid sample, never clamped. Several samples are aggregated by median or majority (ties go low). Pairwise verdicts must survive a position swap. |
| Chain-of-thought leakage | `<think>` blocks are stripped; only a one-sentence reason (300 characters max) reaches reports. |
| Secrets in the cache | Cache keys are built from provider, model, messages and sampling parameters only; parameters whose names look like keys, tokens, secrets, passwords or authorization are dropped; every stored string is redacted for `sk-`, `sk-ant-`, `ghp_`/`github_pat_`, `AKIA`, Slack, Google API keys and bearer tokens. Tested with a real key in the environment, in the prompt and in the response. |
| Secrets in errors and reprs | Keys are read from the environment into a private attribute, sent only in the `Authorization` header, and never included in exception messages or `repr`. Anthropic SDK errors are reduced to their class name. |
| Code execution from datasets | YAML uses `safe_load` (a `!!python/object` tag is a parse error); JSON uses `json`. No `pickle`, `eval` or `exec` anywhere. |
| Path traversal | A marker's dataset path must resolve (after symlinks) inside the pytest rootdir; cache keys must be 64 hex characters; symlinked cache entries are ignored. |
| Resource exhaustion | Datasets max 16 MiB / 50 000 cases; cache entries 8 MiB; HTTP responses 8 MiB; scorer inputs 1 M characters; regex patterns 2 000 characters; reports read back max 64 MiB. |
| Report injection | The HTML report escapes every value; JSON reports are written with `json`. |
| Runaway spend | Budgets refuse every further model call once crossed, before the request is sent. Unknown hosted models are reported as unpriced. |
| Non-HTTP URLs | Provider URLs must be `http` or `https`; requests go through `http.client` with a timeout. |

## 3. Out of scope

- The security of the app under test and of the model servers themselves.
- Prompts and responses you deliberately commit (`.llm-eval-cache/` is
  plain JSON: review it like any fixture before committing it; redaction is a
  safety net, not a guarantee for arbitrary secrets).
- A judge model that ignores its system prompt entirely: the mitigations reduce
  injection risk; they cannot make a model obey. Use `on_injection="penalize"`,
  several samples and pairwise checks for adversarial evals.

## 4. Zero-cost defaults

Nothing in this repository calls a paid service. The default provider is a
local ollama model; hosted adapters run only when you configure them with your
own key. Tests and CI use scripted fakes and a local HTTP stub.
