from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from pytest_llm_evals import Score
from pytest_llm_evals import ToolCall
from pytest_llm_evals import bleu
from pytest_llm_evals import contains
from pytest_llm_evals import embedding_similarity
from pytest_llm_evals import exact_match
from pytest_llm_evals import json_valid
from pytest_llm_evals import not_contains
from pytest_llm_evals import numeric_close
from pytest_llm_evals import regex_match
from pytest_llm_evals import rouge_l
from pytest_llm_evals import rouge_n
from pytest_llm_evals import trajectory_match
from pytest_llm_evals.scores import cosine
from pytest_llm_evals.scores import normalize_text
from pytest_llm_evals.testing import FakeEmbedder


def test_score_is_clamped_and_truthy() -> None:
    assert Score("x", 1.7).value == 1.0
    assert Score("x", -1).value == 0.0
    assert Score("x", math.nan).value == 0.0
    assert bool(Score("x", 0.6, 0.5))
    assert not Score("x", 0.4, 0.5)
    assert "FAIL" in str(Score("x", 0.4, 0.5))
    assert "pass" in str(Score("x", 0.9, 0.5))


def test_exact_match_normalises() -> None:
    assert exact_match("  Paris. ", "paris")
    assert not exact_match("Paris", "paris", normalize=False)
    assert exact_match("Lyon", ["Paris", "lyon"])
    assert exact_match({"a": 1}, {"a": 1})
    assert normalize_text("ＰＡＲＩＳ!") == "paris"


def test_contains_and_not_contains() -> None:
    s = contains("Paris is in France", ["paris", "france", "berlin"])
    assert s.value == pytest.approx(2 / 3)
    assert s.details["missing"] == ["berlin"]
    assert not s
    assert contains("Paris", ["berlin", "paris"], mode="any")
    assert not contains("Paris", "paris", case_sensitive=True)
    assert contains("a b", ["a", "z"], threshold=0.5)
    with pytest.raises(ValueError):
        contains("x", [])
    assert not_contains("All good", ["password", "api key"])
    leak = not_contains("my PASSWORD is", ["password"])
    assert not leak and leak.details["present"] == ["password"]


def test_regex_match() -> None:
    assert regex_match("Order #1234 confirmed", r"#\d{4}")
    assert not regex_match("Order 1234", r"\d{4}", fullmatch=True)
    assert regex_match("1234", r"\d{4}", fullmatch=True)
    with pytest.raises(ValueError):
        regex_match("x", "a" * 3000)


def test_json_valid_with_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
        "required": ["city"],
    }
    ok = json_valid('```json\n{"city": "Paris", "pop": 2}\n```', schema)
    assert ok and ok.details["value"]["city"] == "Paris"
    bad = json_valid('{"pop": "many"}', schema)
    assert not bad
    assert any("city" in e for e in bad.details["errors"])
    assert any(e.startswith("pop") for e in bad.details["errors"])
    assert not json_valid("not json")
    assert json_valid({"already": "parsed"})
    assert json_valid("[1, 2]")


def test_numeric_close() -> None:
    assert numeric_close("About 1,234.5 km", 1234.5)
    assert numeric_close("3.14159", math.pi, rel_tol=1e-3)
    assert not numeric_close("3.0", math.pi, rel_tol=1e-3)
    assert numeric_close(10, 10.4, abs_tol=0.5)
    assert not numeric_close("no digits", 1).passed
    assert not numeric_close(True, 1)
    assert numeric_close("-2e3", -2000)


def test_rouge_and_bleu() -> None:
    assert rouge_n("the cat sat", "the cat sat down").value == pytest.approx(6 / 7)
    assert rouge_n("the cat sat", "the cat sat", n=2).value == 1.0
    assert rouge_n("", "x").value == 0.0
    assert rouge_l("the cat sat on the mat", "the cat on the mat").value > 0.8
    assert rouge_l("a b c", "x y z").value == 0.0
    perfect = bleu("the quick brown fox jumps", "the quick brown fox jumps")
    assert perfect.value == pytest.approx(1.0)
    partial = bleu("the quick brown dog", ["the quick brown fox jumps", "a fast fox"])
    assert 0 < partial.value < 1
    assert bleu("zzz", "the fox").value == 0.0
    assert bleu("", "the fox").value == 0.0
    short = bleu("the quick", "the quick brown fox jumps over")
    assert short.value < bleu("the quick brown fox", "the quick brown fox jumps over").value


def test_embedding_similarity() -> None:
    emb = FakeEmbedder()
    assert embedding_similarity("the capital is Paris", "Paris is the capital", embedder=emb)
    assert not embedding_similarity("bananas", "Paris is the capital", embedder=emb)
    assert cosine([0, 0], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        cosine([1], [1, 2])


@dataclass
class _Call:
    name: str
    args: dict[str, object]


def test_tool_call_coercion() -> None:
    assert ToolCall.coerce("search") == ToolCall("search")
    assert ToolCall.coerce({"name": "a", "input": {"q": 1}}).args == {"q": 1}
    openai = {"type": "function", "function": {"name": "get", "arguments": '{"id": 3}'}}
    assert ToolCall.coerce(openai) == ToolCall("get", {"id": 3})
    assert ToolCall.coerce({"name": "x", "arguments": "not json"}).args == {"_raw": "not json"}
    assert ToolCall.coerce(_Call("obj", {"k": "v"})) == ToolCall("obj", {"k": "v"})
    with pytest.raises(TypeError):
        ToolCall.coerce(42)


def test_trajectory_modes() -> None:
    actual = ["search", {"name": "fetch", "args": {"url": "a", "n": 1}}, "answer"]
    assert trajectory_match(actual, ["search", "answer"], mode="in_order")
    assert not trajectory_match(actual, ["answer", "search"], mode="in_order")
    assert trajectory_match(actual, ["answer", "search"], mode="subset")
    assert not trajectory_match(actual, ["search", "answer"], mode="exact")
    assert trajectory_match(actual, ["search", "fetch", "answer"], mode="exact")
    partial = trajectory_match(actual, ["search", "fetch", "delete"], mode="exact")
    assert partial.value == pytest.approx(2 / 3)
    assert trajectory_match(actual, ["search", "fetch", "answer", "x"], mode="superset")
    assert not trajectory_match(actual + ["rm_rf"], ["search", "fetch", "answer"], mode="superset")
    assert trajectory_match(
        actual, [{"name": "fetch", "args": {"url": "a"}}], mode="subset", check_args=True
    )
    assert not trajectory_match(
        actual, [{"name": "fetch", "args": {"url": "b"}}], mode="subset", check_args=True
    )
    assert trajectory_match([], [], mode="exact")
    assert not trajectory_match(["x"], [], mode="exact")
    assert trajectory_match(["x"], [], mode="in_order")
    assert trajectory_match([], ["x"], mode="superset")
    with pytest.raises(ValueError):
        trajectory_match(["x"], ["x"], mode="bogus")  # type: ignore[arg-type]


def test_huge_text_refused() -> None:
    with pytest.raises(ValueError):
        contains("x" * 1_000_001, "x")
