"""Per-case results, the ``llm_eval`` recorder, pass@k, reports and baselines."""

from __future__ import annotations

import html
import json
import math
import statistics
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from pytest_llm_evals.dataset import EvalCase
from pytest_llm_evals.llm import Completion
from pytest_llm_evals.scores import Score
from pytest_llm_evals.usage import Usage
from pytest_llm_evals.usage import UsageLedger

__all__ = [
    "CaseResult",
    "EvalRecorder",
    "compare_baseline",
    "load_report",
    "pass_at_k",
    "render_html",
    "summarise",
]

REPORT_VERSION = 1
MAX_OUTPUT_CHARS = 2000
MAX_REPORT_BYTES = 64 * 1024 * 1024


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k (Chen et al., 2021): chance that k of n samples include a pass.

    >>> round(pass_at_k(5, 1, 1), 2), round(pass_at_k(5, 1, 5), 2)
    (0.2, 1.0)
    """
    if not 0 <= c <= n or k < 1:
        raise ValueError("need 0 <= c <= n and k >= 1")
    k = min(k, n)
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


class EvalRecorder:
    """The ``llm_eval`` fixture: collect scores for the running case.

    Scores are soft assertions: every one is recorded and reported, and the
    test fails at the end if any score is below its threshold or the mean is
    below the case's ``min_score``.
    """

    def __init__(
        self,
        nodeid: str,
        *,
        case: EvalCase | None = None,
        min_score: float | None = None,
        ledger: UsageLedger | None = None,
    ) -> None:
        self.nodeid = nodeid
        self.case = case
        self.min_score = min_score
        self.ledger = ledger or UsageLedger()
        self.scores: list[Score] = []
        self.output: Any = None

    def add(self, score: Score) -> Score:
        """Record ``score`` and return it (so ``assert llm_eval.add(s)`` also works)."""
        if not isinstance(score, Score):
            raise TypeError("llm_eval.add() takes a Score; use llm_eval.score(name, value)")
        self.scores.append(score)
        return score

    def score(self, name: str, value: float, *, threshold: float = 0.5, **details: Any) -> Score:
        """Record a custom metric in [0, 1]."""
        return self.add(Score(name, value, threshold, details))

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float = 0.0,
        provider: str = "custom",
        cached: bool = False,
    ) -> None:
        """Account for a call your app made with its own client (budgets apply)."""
        self.ledger.record(
            Completion(
                text="",
                model=model,
                provider=provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_s=latency_s,
                cached=cached,
            ),
            role="app",
        )

    @property
    def mean(self) -> float | None:
        if not self.scores:
            return None
        return statistics.fmean(s.value for s in self.scores)

    def failures(self) -> list[str]:
        problems = [str(s) for s in self.scores if not s.passed]
        mean = self.mean
        if self.min_score is not None and mean is not None and mean < self.min_score:
            problems.append(f"mean score {mean:.3f} < case min_score {self.min_score:.2f}")
        return problems

    def _reset(self) -> None:
        self.scores = []
        self.output = None


@dataclass
class CaseResult:
    nodeid: str
    case_id: str | None
    dataset: str | None
    score: float
    passed: bool
    attempts: int
    passes: int
    k: int
    pass_at_1: float
    pass_at_k: float
    min_score: float | None = None
    scores: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    output: str | None = None
    failure: str | None = None

    @staticmethod
    def score_dict(score: Score) -> dict[str, Any]:
        details = {
            k: v
            for k, v in score.details.items()
            if k
            in {
                "reason",
                "winner",
                "missing",
                "errors",
                "injection_suspected",
                "samples",
                "invalid_samples",
                "actual",
                "expected",
            }
        }
        return {
            "name": score.name,
            "value": round(score.value, 4),
            "threshold": score.threshold,
            "passed": score.passed,
            "details": json.loads(json.dumps(details, default=str)),
        }

    @staticmethod
    def clip_output(output: Any) -> str | None:
        if output is None:
            return None
        text = output if isinstance(output, str) else json.dumps(output, default=str)
        return text if len(text) <= MAX_OUTPUT_CHARS else text[:MAX_OUTPUT_CHARS] + "..."


def summarise(results: Sequence[CaseResult], usage: Usage) -> dict[str, Any]:
    scores = [r.score for r in results]
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "mean_score": round(statistics.fmean(scores), 4) if scores else None,
        "mean_pass_at_1": round(statistics.fmean(r.pass_at_1 for r in results), 4)
        if results
        else None,
        "mean_pass_at_k": round(statistics.fmean(r.pass_at_k for r in results), 4)
        if results
        else None,
        "usage": usage.to_dict(),
    }


def build_report(
    results: Sequence[CaseResult],
    usage: Usage,
    *,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": REPORT_VERSION,
        "tool": "pytest-llm-evals",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": summarise(results, usage),
        "gates": dict(gates),
        "cases": [asdict(r) for r in results],
    }


def write_json(report: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8"
    )
    return path


def load_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError(f"report {path} is larger than 64 MiB")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != REPORT_VERSION:
        raise ValueError(f"{path} is not a pytest-llm-evals report (version {REPORT_VERSION})")
    if not isinstance(data.get("cases"), list):
        raise ValueError(f"{path} has no 'cases' list")
    return data


def compare_baseline(
    results: Sequence[CaseResult], baseline: Mapping[str, Any], *, max_regression: float
) -> list[str]:
    """Regressions against a saved report: a case that passed and now fails, a
    case score that dropped by more than ``max_regression``, or a mean drop
    larger than ``max_regression``. New or removed cases are not regressions."""
    old = {c["nodeid"]: c for c in baseline["cases"] if isinstance(c, Mapping) and "nodeid" in c}
    problems: list[str] = []
    shared_old: list[float] = []
    shared_new: list[float] = []
    for r in results:
        prev = old.get(r.nodeid)
        if prev is None:
            continue
        prev_score = float(prev.get("score", 0.0))
        shared_old.append(prev_score)
        shared_new.append(r.score)
        if prev.get("passed") and not r.passed:
            problems.append(f"{r.nodeid}: passed in baseline, fails now")
        elif r.score < prev_score - max_regression:
            problems.append(f"{r.nodeid}: score {r.score:.3f} < baseline {prev_score:.3f}")
    if shared_old:
        old_mean, new_mean = statistics.fmean(shared_old), statistics.fmean(shared_new)
        if new_mean < old_mean - max_regression:
            problems.append(
                f"mean score {new_mean:.3f} < baseline {old_mean:.3f} "
                f"(tolerance {max_regression:.3f})"
            )
    return problems


_HTML_STYLE = """
body{font:14px/1.45 system-ui,sans-serif;margin:2rem;color:#1b1f2a;background:#fbfbfd}
h1{font-size:1.4rem}table{border-collapse:collapse;width:100%}
th,td{padding:.4rem .6rem;border-bottom:1px solid #e3e5ec;text-align:left;vertical-align:top}
th{background:#f0f2f7}.pass{color:#0a7a3b}.fail{color:#b3261e;font-weight:600}
.num{text-align:right;font-variant-numeric:tabular-nums}code{font-size:12px}
.chips span{display:inline-block;margin:0 .3rem .2rem 0;padding:0 .4rem;border-radius:4px;
background:#eef}
.chips span.f{background:#fde7e5}
"""


def render_html(report: Mapping[str, Any]) -> str:
    """A self-contained HTML report; every value is escaped."""
    e = html.escape
    s = report["summary"]
    u = s["usage"]
    rows = []
    for c in report["cases"]:
        chips = "".join(
            f'<span class="{"" if sc["passed"] else "f"}">{e(sc["name"])} {sc["value"]:.2f}</span>'
            for sc in c["scores"]
        )
        status = (
            '<span class="pass">pass</span>' if c["passed"] else '<span class="fail">FAIL</span>'
        )
        rows.append(
            "<tr>"
            f"<td><code>{e(c['nodeid'])}</code></td><td>{status}</td>"
            f"<td class=num>{c['score']:.3f}</td>"
            f"<td class=num>{c['passes']}/{c['attempts']}</td>"
            f"<td class=num>{c['pass_at_k']:.2f} (k={c['k']})</td>"
            f"<td class=chips>{chips}</td>"
            f"<td class=num>{c['usage'].get('total_tokens', 0)}</td>"
            f"<td class=num>{c['usage'].get('cost_usd', 0):.4f}</td>"
            f"<td>{e(c['failure'] or '')}</td>"
            "</tr>"
        )
    gates = "".join(
        f"<li>{e(str(k))}: {e(str(v))}</li>" for k, v in report.get("gates", {}).items()
    )
    mean = "n/a" if s["mean_score"] is None else f"{s['mean_score']:.3f}"
    return (
        "<!doctype html><html><head><meta charset=utf-8>"
        "<title>pytest-llm-evals report</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        f"<h1>pytest-llm-evals report</h1><p>{e(report['created_at'])} &middot; "
        f"{s['cases']} cases, {s['passed']} passed, {s['failed']} failed, "
        f"mean score {mean} &middot; "
        f"{u['calls']} LLM calls ({u['cached_calls']} cached), {u['total_tokens']} tokens, "
        f"${u['cost_usd']:.4f}</p><ul>{gates}</ul>"
        "<table><thead><tr><th>case</th><th>status</th><th>score</th><th>passes</th>"
        "<th>pass@k</th><th>scores</th><th>tokens</th><th>USD</th><th>failure</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></body></html>\n"
    )
