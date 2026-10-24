from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from pytest_llm_evals import AnthropicClient
from pytest_llm_evals import LLMError
from pytest_llm_evals import OllamaClient
from pytest_llm_evals import OpenAICompatibleClient
from pytest_llm_evals import make_client
from pytest_llm_evals.llm import strip_thinking
from tests.conftest import FakeServer

MSGS = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]


def test_ollama_client(fake_server: FakeServer) -> None:
    fake_server.responder = lambda path, body: (
        200,
        {
            "model": body["model"],
            "message": {"role": "assistant", "content": "<think>hmm</think>Hello!"},
            "prompt_eval_count": 7,
            "eval_count": 2,
        },
    )
    client = OllamaClient("llama3.2", host=fake_server.url)
    out = client.complete(MSGS, temperature=0.0, max_tokens=50, seed=3, format="json")
    assert out.text == "Hello!" and out.model == "llama3.2" and out.provider == "ollama"
    assert (out.input_tokens, out.output_tokens, out.total_tokens) == (7, 2, 9)
    req = fake_server.requests[0]
    assert req["path"] == "/api/chat"
    assert req["body"]["options"] == {"temperature": 0.0, "num_predict": 50, "seed": 3}
    assert req["body"]["stream"] is False and req["body"]["format"] == "json"
    assert "llama3.2" in repr(client)


def test_ollama_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EVAL_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    client = OllamaClient()
    assert client.model == "qwen3" and client.host == "http://localhost:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434/")
    assert OllamaClient().host == "http://gpu-box:11434"


def test_openai_compatible_client_sends_key_only_in_header(
    fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_KEY", "sk-test-0123456789abcdefghij")
    fake_server.responder = lambda path, body: (
        200,
        {
            "model": "served-model",
            "choices": [{"message": {"content": "pong"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        },
    )
    client = OpenAICompatibleClient(
        "m", base_url=fake_server.url + "/v1", api_key_env="MY_KEY", provider="vllm"
    )
    out = client.complete(
        MSGS, temperature=0.2, max_tokens=8, seed=1, response_format={"type": "json_object"}
    )
    assert out.text == "pong" and out.model == "served-model" and out.provider == "vllm"
    assert (out.input_tokens, out.output_tokens) == (5, 1)
    req = fake_server.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["headers"]["Authorization"] == "Bearer sk-test-0123456789abcdefghij"
    assert req["body"]["temperature"] == 0.2 and req["body"]["seed"] == 1
    assert "sk-test" not in repr(client)


def test_openai_without_key_sends_no_auth(
    fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    fake_server.responder = lambda path, body: (200, {"choices": [{"message": {"content": "x"}}]})
    OpenAICompatibleClient("m", base_url=fake_server.url).complete(MSGS)
    assert "Authorization" not in fake_server.requests[0]["headers"]


@pytest.mark.parametrize(
    ("status", "payload", "message"),
    [
        (500, {"error": "boom"}, "HTTP 500"),
        (200, b"not json", "invalid JSON"),
        (200, [1, 2], "non-object"),
        (200, {"choices": []}, "unexpected chat/completions"),
    ],
)
def test_openai_errors(
    fake_server: FakeServer,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: Any,
    message: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-0123456789abcdef")
    fake_server.responder = lambda path, body: (status, payload)
    with pytest.raises(LLMError, match=message) as info:
        OpenAICompatibleClient("m", base_url=fake_server.url).complete(MSGS)
    assert "sk-secret" not in str(info.value)


def test_unreachable_and_bad_scheme() -> None:
    with pytest.raises(LLMError, match="cannot reach"):
        OllamaClient(host="http://127.0.0.1:9", timeout=2).complete(MSGS)
    with pytest.raises(LLMError, match="non-http"):
        OllamaClient(host="file:///etc/passwd")
    with pytest.raises(LLMError, match="non-http"):
        OpenAICompatibleClient(base_url="ftp://x")


def test_response_size_limit(fake_server: FakeServer, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pytest_llm_evals.llm.MAX_RESPONSE_BYTES", 10)
    fake_server.responder = lambda path, body: (200, {"message": {"content": "x" * 100}})
    with pytest.raises(LLMError, match="larger than"):
        OllamaClient(host=fake_server.url).complete(MSGS)


def test_strip_thinking() -> None:
    assert strip_thinking("<thinking>a\nb</thinking> ok ") == "ok"
    assert strip_thinking("<THINK>x</THINK>y") == "y"


class _FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _install_fake_anthropic(monkeypatch: pytest.MonkeyPatch, response: Any) -> _FakeMessages:
    messages = _FakeMessages(response)
    module = types.ModuleType("anthropic")

    class Anthropic:
        def __init__(self) -> None:
            self.messages = messages

    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return messages


def _resp(stop: str = "end_turn") -> Any:
    block = types.SimpleNamespace(type="text", text="Bonjour")
    usage = types.SimpleNamespace(input_tokens=11, output_tokens=3)
    return types.SimpleNamespace(
        content=[block], usage=usage, model="claude-sonnet-5-5", stop_reason=stop
    )


def test_anthropic_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_EVAL_MODEL", raising=False)
    fake = _install_fake_anthropic(monkeypatch, _resp())
    client = AnthropicClient()
    out = client.complete(MSGS, temperature=0.9, seed=4)
    assert out.text == "Bonjour" and (out.input_tokens, out.output_tokens) == (11, 3)
    assert fake.kwargs["system"] == "be brief"
    assert fake.kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert fake.kwargs["model"] == "claude-sonnet-5-5"
    assert "temperature" not in fake.kwargs and "seed" not in fake.kwargs
    assert "claude" in repr(client)


def test_anthropic_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_anthropic(monkeypatch, _resp("refusal"))
    with pytest.raises(LLMError, match="refusal"):
        AnthropicClient("claude-haiku-4-5").complete(MSGS)
    _install_fake_anthropic(monkeypatch, RuntimeError("x-api-key: sk-ant-secret"))
    with pytest.raises(LLMError) as info:
        AnthropicClient().complete(MSGS)
    assert "sk-ant" not in str(info.value)


def test_anthropic_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMError, match="pip install"):
        AnthropicClient()


def test_make_client(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(make_client("ollama", "m"), OllamaClient)
    assert isinstance(make_client("OpenAI", "m", "http://localhost:1/v1"), OpenAICompatibleClient)
    _install_fake_anthropic(monkeypatch, _resp())
    assert isinstance(make_client("anthropic"), AnthropicClient)
    with pytest.raises(LLMError, match="unknown LLM provider"):
        make_client("gemini")
