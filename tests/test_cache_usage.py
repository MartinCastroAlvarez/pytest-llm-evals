from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytest_llm_evals import Budget
from pytest_llm_evals import BudgetExceededError
from pytest_llm_evals import CachedClient
from pytest_llm_evals import CacheMissError
from pytest_llm_evals import OpenAICompatibleClient
from pytest_llm_evals import Price
from pytest_llm_evals import PricingTable
from pytest_llm_evals import ResponseCache
from pytest_llm_evals import TrackedClient
from pytest_llm_evals import UsageLedger
from pytest_llm_evals.cache import current_attempt
from pytest_llm_evals.cache import redact
from pytest_llm_evals.llm import Completion
from pytest_llm_evals.testing import FakeLLM
from pytest_llm_evals.usage import Usage
from pytest_llm_evals.usage import current_usage
from tests.conftest import FakeServer

MSGS = [{"role": "user", "content": "What is the capital of France?"}]


def cached(tmp_path: Path, mode: str = "auto", llm: FakeLLM | None = None) -> CachedClient:
    return CachedClient(llm or FakeLLM(["Paris", "Lyon", "Nice"]), ResponseCache(tmp_path, mode))  # type: ignore[arg-type]


def test_auto_mode_reads_through(tmp_path: Path) -> None:
    client = cached(tmp_path)
    first = client.complete(MSGS, temperature=0)
    again = client.complete(MSGS, temperature=0)
    assert first.text == again.text == "Paris"
    assert not first.cached and again.cached
    assert client.cache.hits == 1 and client.cache.misses == 1 and client.cache.writes == 1
    other = client.complete(MSGS, temperature=0.5)
    assert other.text == "Lyon" and not other.cached
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 2 and all(f.parent.name == f.stem[:2] for f in files)


def test_key_covers_model_params_sample_and_attempt(tmp_path: Path) -> None:
    key = ResponseCache.key
    base = key("ollama", "m", MSGS, {"temperature": 0})
    assert base == key("ollama", "m", MSGS, {"temperature": 0, "seed": None})
    assert base != key("ollama", "m2", MSGS, {"temperature": 0})
    assert base != key("openai", "m", MSGS, {"temperature": 0})
    assert base != key("ollama", "m", MSGS, {"temperature": 0, "sample": 1})
    assert base == key("ollama", "m", MSGS, {"temperature": 0, "api_key": "sk-123"})
    client = cached(tmp_path)
    client.complete(MSGS)
    token = current_attempt.set(1)
    try:
        assert client.complete(MSGS).text == "Lyon"
    finally:
        current_attempt.reset(token)
    assert client.complete(MSGS).cached


def test_record_and_replay(tmp_path: Path) -> None:
    rec = cached(tmp_path, "record")
    rec.complete(MSGS)
    rec.complete(MSGS)
    assert rec.cache.writes == 2 and rec.cache.hits == 0
    replay = cached(tmp_path, "replay", FakeLLM(["SHOULD NOT BE CALLED"]))
    assert replay.complete(MSGS).text == "Lyon"
    assert replay.inner.calls == []  # type: ignore[attr-defined]
    with pytest.raises(CacheMissError, match="no recorded response"):
        replay.complete([{"role": "user", "content": "new"}])


def test_off_mode(tmp_path: Path) -> None:
    client = cached(tmp_path, "off")
    assert client.complete(MSGS).text == "Paris"
    assert client.complete(MSGS).text == "Lyon"
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        ResponseCache(tmp_path, "sometimes")  # type: ignore[arg-type]


def test_corrupt_or_tampered_entries_are_misses(tmp_path: Path) -> None:
    client = cached(tmp_path)
    client.complete(MSGS)
    (entry,) = tmp_path.rglob("*.json")
    entry.write_text("{not json")
    assert not client.complete(MSGS).cached
    (entry,) = tmp_path.rglob("*.json")
    data = json.loads(entry.read_text())
    data["key"] = "0" * 64
    entry.write_text(json.dumps(data))
    assert not client.complete(MSGS).cached
    with pytest.raises(ValueError):
        client.cache.get("../../etc/passwd")


def test_symlinked_entry_is_ignored(tmp_path: Path) -> None:
    client = cached(tmp_path)
    client.complete(MSGS)
    (entry,) = tmp_path.rglob("*.json")
    target = tmp_path / "elsewhere.txt"
    target.write_text(entry.read_text())
    entry.unlink()
    entry.symlink_to(target)
    assert not client.complete(MSGS).cached


def test_cache_never_stores_api_keys(
    tmp_path: Path, fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-live-ABCDEFGHIJKLMNOPQRSTUV0123"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    fake_server.responder = lambda path, body: (
        200,
        {"choices": [{"message": {"content": f"echo {secret}"}}], "usage": {}},
    )
    inner = OpenAICompatibleClient("m", base_url=fake_server.url)
    client = CachedClient(inner, ResponseCache(tmp_path))
    prompt = [{"role": "user", "content": f"my key is {secret} and ghp_abcdefghijklmnopqrstuvwxyz"}]
    client.complete(prompt, api_key=secret, authorization=secret)
    assert fake_server.requests[0]["headers"]["Authorization"] == f"Bearer {secret}"
    blob = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert secret not in blob and "ghp_" not in blob and "[REDACTED]" in blob
    assert "api_key" not in blob and "authorization" not in blob


def test_redact() -> None:
    assert redact({"a": ["AKIAABCDEFGHIJKLMNOP", 3]}) == {"a": ["[REDACTED]", 3]}
    assert redact("Bearer abcdefghijklmnopqrstuvwxyz") == "[REDACTED]"


def test_pricing_table(tmp_path: Path) -> None:
    table = PricingTable()
    assert table.cost("ollama", "qwen3", 10**6, 10**6) == (0.0, True)
    assert table.cost("anthropic", "claude-sonnet-5-5", 10**6, 10**6) == (12.0, True)
    assert table.cost("anthropic", "claude-haiku-4-5", 10**6, 0) == (1.0, True)
    assert table.cost("openai", "mystery", 100, 100) == (0.0, False)
    path = tmp_path / "prices.yaml"
    path.write_text("free_providers: [ollama]\nmodels:\n  'team-*': {input: 1.5, output: 3}\n")
    custom = PricingTable.from_file(path)
    assert custom.cost("openai", "team-large", 2 * 10**6, 10**6) == (6.0, True)
    assert custom.lookup("vllm", "x") is None
    js = tmp_path / "p.json"
    js.write_text(json.dumps({"models": {"m": {"input": 1, "output": 1}}}))
    assert PricingTable.from_file(js).lookup("openai", "m") == Price(1, 1)


@pytest.mark.parametrize(
    "text",
    ["[1]", "models: [1]", "models: {m: {input: 1}}", "models: {m: {input: -1, output: 1}}"],
)
def test_pricing_validation(tmp_path: Path, text: str) -> None:
    path = tmp_path / "p.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        PricingTable.from_file(path)


def hosted(text: str = "ok", tokens: int = 1000) -> FakeLLM:
    llm = FakeLLM(lambda m: text, provider="anthropic", model="claude-sonnet-5-5")
    original = llm.complete

    def complete(*args, **kwargs):  # type: ignore[no-untyped-def]
        out = original(*args, **kwargs)
        out.input_tokens, out.output_tokens = tokens, tokens
        return out

    llm.complete = complete  # type: ignore[method-assign]
    return llm


def test_usd_budget_refuses_further_calls() -> None:
    ledger = UsageLedger(Budget(max_usd=0.02))
    client = TrackedClient(hosted(), ledger)  # type: ignore[arg-type]
    client.complete(MSGS)  # $0.012
    with pytest.raises(BudgetExceededError, match="cost"):
        client.complete(MSGS)  # $0.024 > $0.02
    llm = client.inner
    calls = len(llm.calls)  # type: ignore[attr-defined]
    with pytest.raises(BudgetExceededError, match="call refused"):
        client.complete(MSGS)
    assert len(llm.calls) == calls  # type: ignore[attr-defined]  # never reached the provider
    assert ledger.exceeded and ledger.total.cost_usd == pytest.approx(0.024)


def test_token_budget_counts_live_tokens_only(tmp_path: Path) -> None:
    ledger = UsageLedger(Budget(max_tokens=2500))
    client = TrackedClient(CachedClient(hosted(), ResponseCache(tmp_path)), ledger)  # type: ignore[arg-type]
    for _ in range(5):
        client.complete(MSGS)  # one live call (2000 tokens), four cached
    assert ledger.total.live_tokens == 2000 and ledger.total.total_tokens == 10_000
    assert ledger.total.cached_calls == 4 and ledger.total.cost_usd == pytest.approx(0.012)
    with pytest.raises(BudgetExceededError, match="tokens"):
        client.complete([{"role": "user", "content": "new"}])


def test_latency_budget_and_shortcut() -> None:
    ledger = UsageLedger(Budget(max_latency_s=0.5))
    slow = FakeLLM(lambda m: "late", latency_s=1.0)
    client = TrackedClient(slow, ledger, role="judge")
    with pytest.raises(BudgetExceededError, match="took 1.00s"):
        client("hello", system="sys")
    assert slow.calls[0]["messages"][0] == {"role": "system", "content": "sys"}
    assert ledger.total.by_role == {"judge": 1}
    assert "judge" in repr(client)
    with pytest.raises(ValueError):
        Budget(max_usd=-1)


def test_unpriced_hosted_calls_are_counted_and_bucket_tracks() -> None:
    ledger = UsageLedger()
    llm = FakeLLM(lambda m: "x", provider="openai", model="unknown")
    bucket = Usage()
    token = current_usage.set(bucket)
    try:
        assert TrackedClient(llm, ledger)("hi") == "x"
    finally:
        current_usage.reset(token)
    assert ledger.total.unpriced_calls == 1 and bucket.calls == 1
    assert ledger.total.to_dict()["total_tokens"] == ledger.total.total_tokens
    ledger.record(Completion("", "m", "fake", 1, 1, cached=True))
    assert ledger.total.cached_calls == 1
