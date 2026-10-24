"""The pytest plugin: options, marker, fixtures, repeat/pass@k, gates and reports.

Registered through the ``pytest11`` entry point, so installing the package is
enough. Every option is also an ini key (``[tool.pytest.ini_options]``).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import pytest

from pytest_llm_evals.cache import CACHE_MODES
from pytest_llm_evals.cache import CachedClient
from pytest_llm_evals.cache import CacheMissError
from pytest_llm_evals.cache import ResponseCache
from pytest_llm_evals.cache import current_attempt
from pytest_llm_evals.dataset import EvalCase
from pytest_llm_evals.dataset import load_dataset
from pytest_llm_evals.dataset import resolve_dataset_path
from pytest_llm_evals.judge import Judge
from pytest_llm_evals.llm import LLMClient
from pytest_llm_evals.llm import make_client
from pytest_llm_evals.pricing import PricingTable
from pytest_llm_evals.results import CaseResult
from pytest_llm_evals.results import EvalRecorder
from pytest_llm_evals.results import build_report
from pytest_llm_evals.results import compare_baseline
from pytest_llm_evals.results import load_report
from pytest_llm_evals.results import pass_at_k
from pytest_llm_evals.results import render_html
from pytest_llm_evals.results import summarise
from pytest_llm_evals.results import write_json
from pytest_llm_evals.usage import Budget
from pytest_llm_evals.usage import BudgetExceededError
from pytest_llm_evals.usage import TrackedClient
from pytest_llm_evals.usage import Usage
from pytest_llm_evals.usage import UsageLedger
from pytest_llm_evals.usage import current_usage

MARKER = (
    "llm_eval(dataset=None, *, repeat=1, k=None, min_passes=1, min_score=None, tags=None): "
    "an LLM eval; with a dataset the test is parametrized over its cases via `eval_case`"
)

_Conv = Callable[[str], Any]


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise ValueError("must be >= 1")
    return number


def _non_negative(value: str) -> float:
    number = float(value)
    if number < 0:
        raise ValueError("must be >= 0")
    return number


def _unit(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("must be within [0, 1]")
    return number


def _cache_mode(value: str) -> str:
    if value not in CACHE_MODES:
        raise ValueError(f"must be one of {', '.join(CACHE_MODES)}")
    return value


# name, converter, default, help
OPTIONS: list[tuple[str, _Conv, Any, str]] = [
    ("llm_provider", str, "ollama", "provider for the `llm` fixture: ollama, openai, anthropic"),
    ("llm_model", str, None, "model for the `llm` fixture (provider default when unset)"),
    ("llm_base_url", str, None, "base URL for ollama / OpenAI-compatible servers"),
    ("llm_judge_provider", str, None, "provider for `llm_judge` (defaults to --llm-provider)"),
    ("llm_judge_model", str, None, "judge model (ollama default: qwen3)"),
    ("llm_judge_base_url", str, None, "judge base URL (defaults to --llm-base-url)"),
    ("llm_judge_samples", _positive_int, 1, "judge samples per verdict (aggregated by median)"),
    ("llm_cache", _cache_mode, "auto", "response cache mode: auto, record, replay, off"),
    ("llm_cache_dir", str, ".llm-eval-cache", "cache directory, relative to the rootdir"),
    ("llm_budget_usd", _non_negative, None, "fail the run when live LLM spend exceeds this (USD)"),
    ("llm_budget_tokens", _positive_int, None, "fail the run when live tokens exceed this"),
    ("llm_budget_latency", _non_negative, None, "fail a call slower than this many seconds"),
    ("llm_pricing", str, None, "YAML/JSON pricing table (USD per 1M tokens)"),
    ("llm_eval_min_score", _unit, None, "fail the run when the mean case score is below this"),
    ("llm_eval_repeat", _positive_int, 1, "default repeat count for eval tests (pass@k)"),
    ("llm_eval_json", str, None, "write a JSON report here"),
    ("llm_eval_html", str, None, "write an HTML report here"),
    ("llm_eval_baseline", str, None, "fail on regression vs. this saved JSON report"),
    ("llm_eval_max_regression", _unit, 0.02, "allowed score drop vs. the baseline"),
]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("llm-evals", "LLM evals (pytest-llm-evals)")
    for name, _conv, default, help_text in OPTIONS:
        group.addoption(
            "--" + name.replace("_", "-"),
            dest=name,
            default=None,
            help=f"{help_text} [ini: {name}; default: {default}]",
        )
        parser.addini(name, help=help_text, type="string", default="")


def _option(config: pytest.Config, name: str) -> Any:
    for opt, conv, default, _ in OPTIONS:
        if opt != name:
            continue
        raw = config.getoption(name)
        if raw is None:
            raw = config.getini(name) or None
        if raw is None:
            return default
        try:
            return conv(str(raw))
        except ValueError as exc:
            raise pytest.UsageError(f"--{name.replace('_', '-')}={raw!r}: {exc}") from None
    raise KeyError(name)


@dataclass
class EvalSession:
    """Session state: options, the usage ledger, the cache and the results."""

    config: pytest.Config
    ledger: UsageLedger
    cache: ResponseCache
    results: list[CaseResult] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    report_paths: list[Path] = field(default_factory=list)

    def option(self, name: str) -> Any:
        return _option(self.config, name)

    def path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.config.rootpath / path

    def wrap(self, client: LLMClient, role: str) -> TrackedClient:
        """Cache, price and budget-check ``client``."""
        return TrackedClient(CachedClient(client, self.cache), self.ledger, role=role)


SESSION_KEY = pytest.StashKey[EvalSession]()
RECORDER_KEY = pytest.StashKey[EvalRecorder]()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", MARKER)
    pricing_file = _option(config, "llm_pricing")
    root = config.rootpath
    pricing = PricingTable()
    if pricing_file:
        path = Path(pricing_file)
        try:
            pricing = PricingTable.from_file(path if path.is_absolute() else root / path)
        except (OSError, ValueError) as exc:
            raise pytest.UsageError(f"--llm-pricing: {exc}") from None
    budget = Budget(
        max_usd=_option(config, "llm_budget_usd"),
        max_tokens=_option(config, "llm_budget_tokens"),
        max_latency_s=_option(config, "llm_budget_latency"),
    )
    cache_dir = Path(_option(config, "llm_cache_dir"))
    cache = ResponseCache(
        cache_dir if cache_dir.is_absolute() else root / cache_dir, _option(config, "llm_cache")
    )
    config.stash[SESSION_KEY] = EvalSession(config, UsageLedger(budget, pricing), cache)


def _session(config: pytest.Config) -> EvalSession:
    return config.stash[SESSION_KEY]


# --------------------------------------------------------------------------- #
# Parametrization from datasets
# --------------------------------------------------------------------------- #
def _dataset_spec(marker: pytest.Mark) -> Any:
    if "dataset" in marker.kwargs:
        return marker.kwargs["dataset"]
    return marker.args[0] if marker.args else None


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    marker = metafunc.definition.get_closest_marker("llm_eval")
    if marker is None:
        return
    spec = _dataset_spec(marker)
    if spec is None:
        return
    if "eval_case" not in metafunc.fixturenames:
        raise ValueError(
            f"{metafunc.definition.nodeid}: @pytest.mark.llm_eval(dataset=...) needs an "
            "`eval_case` argument"
        )
    path = resolve_dataset_path(
        spec, base=metafunc.definition.path.parent, root=metafunc.config.rootpath
    )
    cases = load_dataset(path, tags=marker.kwargs.get("tags") or ())
    metafunc.parametrize("eval_case", cases, ids=[c.id for c in cases])


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def llm_eval_session(pytestconfig: pytest.Config) -> EvalSession:
    """Session state: ``ledger`` (usage and budgets), ``cache``, ``results``."""
    return _session(pytestconfig)


@pytest.fixture(scope="session")
def llm_client(llm_eval_session: EvalSession) -> LLMClient:
    """The raw client the ``llm`` fixture wraps. Override it to plug in your own."""
    s = llm_eval_session
    return make_client(s.option("llm_provider"), s.option("llm_model"), s.option("llm_base_url"))


@pytest.fixture(scope="session")
def llm_judge_client(llm_eval_session: EvalSession) -> LLMClient:
    """The raw judge client. Override it to plug in your own judge model."""
    s = llm_eval_session
    return make_client(
        s.option("llm_judge_provider") or s.option("llm_provider"),
        s.option("llm_judge_model"),
        s.option("llm_judge_base_url") or s.option("llm_base_url"),
    )


@pytest.fixture
def llm(llm_client: LLMClient, llm_eval_session: EvalSession) -> TrackedClient:
    """Cached, priced, budgeted client. ``llm("prompt")`` returns text;
    ``llm.complete(messages)`` returns a :class:`Completion`."""
    return llm_eval_session.wrap(llm_client, "app")


@pytest.fixture
def llm_judge(llm_judge_client: LLMClient, llm_eval_session: EvalSession) -> Judge:
    """A :class:`Judge` on the (cached, budgeted) judge client."""
    return Judge(
        llm_eval_session.wrap(llm_judge_client, "judge"),
        samples=llm_eval_session.option("llm_judge_samples"),
    )


def _case_of(item: pytest.Item) -> EvalCase | None:
    callspec = getattr(item, "callspec", None)
    case = callspec.params.get("eval_case") if callspec is not None else None
    return case if isinstance(case, EvalCase) else None


def _recorder(item: pytest.Item, session: EvalSession) -> EvalRecorder:
    existing = item.stash.get(RECORDER_KEY, None)
    if existing is not None:
        return existing
    marker = item.get_closest_marker("llm_eval")
    case = _case_of(item)
    min_score = case.min_score if case and case.min_score is not None else None
    if min_score is None and marker is not None:
        min_score = marker.kwargs.get("min_score")
    recorder = EvalRecorder(item.nodeid, case=case, min_score=min_score, ledger=session.ledger)
    item.stash[RECORDER_KEY] = recorder
    return recorder


@pytest.fixture
def llm_eval(request: pytest.FixtureRequest, llm_eval_session: EvalSession) -> EvalRecorder:
    """Record scores for this case: ``llm_eval.add(contains(out, "Paris"))``."""
    return _recorder(request.node, llm_eval_session)


@pytest.fixture
def eval_case(request: pytest.FixtureRequest) -> EvalCase:
    """The current golden case (parametrized by ``@pytest.mark.llm_eval(dataset=...)``)."""
    pytest.fail(
        f"{request.node.nodeid} uses `eval_case` without @pytest.mark.llm_eval(dataset=...)",
        pytrace=False,
    )


# --------------------------------------------------------------------------- #
# Running an eval: repeat-N, pass@k, soft score assertions
# --------------------------------------------------------------------------- #
def _is_eval(item: pytest.Function) -> bool:
    if item.get_closest_marker("llm_eval") is not None:
        return True
    return "llm_eval" in item.fixturenames or "eval_case" in item.fixturenames


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool | None:
    if not _is_eval(pyfuncitem) or inspect.iscoroutinefunction(pyfuncitem.obj):
        return None
    session = _session(pyfuncitem.config)
    marker = pyfuncitem.get_closest_marker("llm_eval")
    kwargs = marker.kwargs if marker is not None else {}
    repeat = int(kwargs.get("repeat") or session.option("llm_eval_repeat"))
    k = int(kwargs.get("k") or repeat)
    min_passes = int(kwargs.get("min_passes", 1))
    if repeat < 1 or not 1 <= min_passes <= repeat or k < 1:
        raise pytest.UsageError(f"{pyfuncitem.nodeid}: need repeat >= 1, 1 <= min_passes <= repeat")
    recorder = _recorder(pyfuncitem, session)
    funcargs = pyfuncitem.funcargs
    testargs = {arg: funcargs[arg] for arg in pyfuncitem._fixtureinfo.argnames}
    usage = Usage()
    usage_token = current_usage.set(usage)
    attempt_scores: list[float] = []
    passes = 0
    last_exc: BaseException | None = None
    problems: list[str] = []
    last_scores: list[Any] = []
    try:
        for attempt in range(repeat):
            attempt_token = current_attempt.set(attempt)
            recorder._reset()
            ok = False
            raised = False
            try:
                pyfuncitem.obj(**testargs)
            except (BudgetExceededError, CacheMissError) as exc:
                problems = [f"{type(exc).__name__}: {exc}"]
                last_scores = list(recorder.scores)
                raise
            except (pytest.skip.Exception, pytest.xfail.Exception):
                raise
            except (Exception, pytest.fail.Exception) as exc:
                last_exc, raised = exc, True
                problems = [f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"]
            else:
                problems = recorder.failures()
                ok = not problems
            finally:
                current_attempt.reset(attempt_token)
            mean = recorder.mean
            attempt_scores.append(0.0 if raised else (mean if mean is not None else float(ok)))
            last_scores = list(recorder.scores)
            if ok:
                passes += 1
                last_exc = None
    except (BudgetExceededError, CacheMissError):
        attempt_scores.append(0.0)
        _store(pyfuncitem, session, recorder, attempt_scores, 0, k, problems, last_scores, usage)
        raise
    finally:
        current_usage.reset(usage_token)
    _store(
        pyfuncitem,
        session,
        recorder,
        attempt_scores,
        passes,
        k,
        problems,
        last_scores,
        usage,
        min_passes=min_passes,
    )
    if passes >= min_passes:
        return True
    if last_exc is not None:
        raise last_exc
    pytest.fail(
        "LLM eval below threshold:\n  "
        + "\n  ".join(problems)
        + (
            f"\n  passes {passes}/{len(attempt_scores)} < min_passes {min_passes}"
            if repeat > 1
            else ""
        ),
        pytrace=False,
    )


def _store(
    item: pytest.Function,
    session: EvalSession,
    recorder: EvalRecorder,
    attempt_scores: list[float],
    passes: int,
    k: int,
    problems: list[str],
    scores: list[Any],
    usage: Usage,
    *,
    min_passes: int = 1,
) -> None:
    n = len(attempt_scores)
    case = recorder.case
    score = sum(attempt_scores) / n if n else 0.0
    passed = passes >= min_passes
    result = CaseResult(
        nodeid=item.nodeid,
        case_id=case.id if case else None,
        dataset=case.dataset if case else None,
        score=round(score, 4),
        passed=passed,
        attempts=n,
        passes=passes,
        k=min(k, n) if n else k,
        pass_at_1=round(pass_at_k(n, passes, 1), 4) if n else 0.0,
        pass_at_k=round(pass_at_k(n, passes, k), 4) if n else 0.0,
        min_score=recorder.min_score,
        scores=[CaseResult.score_dict(s) for s in scores],
        usage=usage.to_dict(),
        output=CaseResult.clip_output(recorder.output),
        failure=None if passed else "; ".join(problems)[:1000] or None,
    )
    session.results.append(result)
    item.user_properties.extend(
        [
            ("llm_eval_score", f"{result.score:.4f}"),
            ("llm_eval_pass_at_k", f"{result.pass_at_k:.4f}"),
            ("llm_eval_tokens", str(usage.total_tokens)),
            ("llm_eval_cost_usd", f"{usage.cost_usd:.6f}"),
        ]
    )


# --------------------------------------------------------------------------- #
# Session gates and reports
# --------------------------------------------------------------------------- #
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    state = config.stash.get(SESSION_KEY, None)
    if state is None or (not state.results and not state.ledger.total.calls):
        return
    min_score = state.option("llm_eval_min_score")
    summary = summarise(state.results, state.ledger.total)
    mean = summary["mean_score"]
    if min_score is not None and mean is not None and mean < min_score:
        state.gate_failures.append(f"mean score {mean:.3f} < --llm-eval-min-score {min_score:.2f}")
    state.gate_failures.extend(f"budget: {v}" for v in state.ledger.violations)
    baseline = state.option("llm_eval_baseline")
    if baseline:
        try:
            state.regressions = compare_baseline(
                state.results,
                load_report(state.path(baseline)),
                max_regression=state.option("llm_eval_max_regression"),
            )
        except (OSError, ValueError) as exc:
            state.gate_failures.append(f"baseline {baseline}: {exc}")
        state.gate_failures.extend(f"regression: {r}" for r in state.regressions)
    gates = {
        "min_score": min_score,
        "budget_usd": state.ledger.budget.max_usd,
        "budget_tokens": state.ledger.budget.max_tokens,
        "budget_latency_s": state.ledger.budget.max_latency_s,
        "baseline": baseline,
        "cache_mode": state.cache.mode,
        "failures": list(state.gate_failures),
    }
    report = build_report(state.results, state.ledger.total, gates=gates)
    json_path = state.option("llm_eval_json")
    if json_path:
        state.report_paths.append(write_json(report, state.path(json_path)))
    html_path = state.option("llm_eval_html")
    if html_path:
        target = state.path(html_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_html(report), encoding="utf-8")
        state.report_paths.append(target)
    if state.gate_failures and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


MAX_SUMMARY_ROWS = 40


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    state = config.stash.get(SESSION_KEY, None)
    if state is None or (not state.results and not state.ledger.total.calls):
        return
    tr = terminalreporter
    tr.write_sep("=", "llm-evals summary")
    results = sorted(state.results, key=lambda r: (r.passed, r.score))
    if results:
        width = min(70, max(len(r.nodeid) for r in results))
        tr.write_line(
            f"{'case':<{width}}  {'score':>5}  {'pass':>5}  {'pass@k':>6}  "
            f"{'tokens':>7}  {'usd':>8}"
        )
        for r in results[:MAX_SUMMARY_ROWS]:
            node = r.nodeid if len(r.nodeid) <= width else "..." + r.nodeid[-(width - 3) :]
            line = (
                f"{node:<{width}}  {r.score:5.3f}  {r.passes:>2}/{r.attempts:<2}  "
                f"{r.pass_at_k:6.2f}  {r.usage.get('total_tokens', 0):>7}  "
                f"{r.usage.get('cost_usd', 0.0):8.4f}"
            )
            tr.write_line(line, red=not r.passed, green=r.passed)
        if len(results) > MAX_SUMMARY_ROWS:
            tr.write_line(f"... {len(results) - MAX_SUMMARY_ROWS} more (see --llm-eval-json)")
    summary = summarise(state.results, state.ledger.total)
    u = state.ledger.total
    mean = "n/a" if summary["mean_score"] is None else f"{summary['mean_score']:.3f}"
    tr.write_line(
        f"cases: {summary['cases']} ({summary['passed']} passed, {summary['failed']} failed), "
        f"mean score {mean}"
    )
    tr.write_line(
        f"llm calls: {u.calls} ({u.cached_calls} cached, cache mode {state.cache.mode}), "
        f"tokens {u.total_tokens} ({u.live_tokens} live), cost ${u.cost_usd:.4f}, "
        f"max latency {u.max_latency_s:.2f}s"
    )
    if u.unpriced_calls:
        tr.write_line(
            f"warning: {u.unpriced_calls} live call(s) to models with no price "
            "(add them with --llm-pricing)",
            yellow=True,
        )
    for failure in state.gate_failures:
        tr.write_line(f"FAILED gate: {failure}", red=True, bold=True)
    for path in state.report_paths:
        tr.write_line(f"report: {path}")
