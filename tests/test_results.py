from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytest_llm_evals import EvalRecorder
from pytest_llm_evals import Score
from pytest_llm_evals import pass_at_k
from pytest_llm_evals.results import CaseResult
from pytest_llm_evals.results import build_report
from pytest_llm_evals.results import compare_baseline
from pytest_llm_evals.results import load_report
from pytest_llm_evals.results import render_html
from pytest_llm_evals.results import write_json
from pytest_llm_evals.testing import FakeLLM
from pytest_llm_evals.usage import Usage


def result(nodeid: str, score: float, passed: bool) -> CaseResult:
    return CaseResult(
        nodeid, None, None, score, passed, 1, int(passed), 1, float(passed), float(passed)
    )


def test_pass_at_k() -> None:
    assert pass_at_k(10, 0, 3) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert pass_at_k(5, 1, 1) == pytest.approx(0.2)
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    assert pass_at_k(3, 1, 9) == 1.0
    with pytest.raises(ValueError):
        pass_at_k(2, 3, 1)


def test_recorder() -> None:
    rec = EvalRecorder("t", min_score=0.8)
    assert rec.mean is None and rec.failures() == []
    rec.add(Score("a", 0.9, 0.5))
    rec.score("b", 0.5, threshold=0.4)
    assert rec.mean == pytest.approx(0.7)
    assert rec.failures() == ["mean score 0.700 < case min_score 0.80"]
    with pytest.raises(TypeError):
        rec.add(0.5)  # type: ignore[arg-type]


def test_baseline_comparison() -> None:
    base = {
        "cases": [
            {"nodeid": "a", "score": 0.9, "passed": True},
            {"nodeid": "b", "score": 0.9, "passed": False},
            "junk",
        ]
    }
    assert (
        compare_baseline(
            [result("a", 0.89, True), result("new", 0.0, False)], base, max_regression=0.02
        )
        == []
    )
    problems = compare_baseline(
        [result("a", 0.5, True), result("b", 0.9, False)], base, max_regression=0.1
    )
    assert problems[0].startswith("a: score 0.500 < baseline 0.900")
    assert problems[-1].startswith("mean score 0.700 < baseline 0.900")
    assert compare_baseline([result("x", 1.0, True)], base, max_regression=0) == []


def test_report_round_trip_and_html(tmp_path: Path) -> None:
    r = result("t::x", 0.5, False)
    r.scores = [CaseResult.score_dict(Score("s", 0.5, 1.0, {"reason": "r", "secret": "drop"}))]
    r.output = CaseResult.clip_output("y" * 5000)
    report = build_report([r], Usage(), gates={"failures": ["<b>"]})
    path = write_json(report, tmp_path / "sub" / "r.json")
    loaded = load_report(path)
    assert loaded["cases"][0]["scores"][0]["details"] == {"reason": "r"}
    assert (
        loaded["cases"][0]["output"].endswith("...") and len(loaded["cases"][0]["output"]) == 2003
    )
    page = render_html(loaded)
    assert "&lt;b&gt;" in page and "FAIL" in page
    empty = render_html(build_report([], Usage(), gates={}))
    assert "mean score n/a" in empty
    assert CaseResult.clip_output(None) is None and CaseResult.clip_output({"a": 1}) == '{"a": 1}'


@pytest.mark.parametrize("content", [{"version": 99, "cases": []}, {"version": 1}, [1]])
def test_load_report_rejects_foreign_files(tmp_path: Path, content: object) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_report(path)


def test_load_report_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_json(build_report([], Usage(), gates={}), tmp_path / "r.json")
    monkeypatch.setattr("pytest_llm_evals.results.MAX_REPORT_BYTES", 5)
    with pytest.raises(ValueError, match="larger than"):
        load_report(path)


def test_fake_llm_modes() -> None:
    mapping = FakeLLM({"France": "Paris"}, default="?")
    assert mapping.complete([{"role": "user", "content": "Peru"}]).text == "?"
    seq = FakeLLM(["one"], default="done")
    msgs = [{"role": "user", "content": "x"}]
    assert [seq.complete(msgs).text for _ in range(2)] == ["one", "done"]
