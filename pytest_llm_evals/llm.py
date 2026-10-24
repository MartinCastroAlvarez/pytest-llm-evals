"""LLM clients: a tiny protocol plus local-first adapters.

* :class:`OllamaClient` (default): a local open-weight model, cost 0.
* :class:`OpenAICompatibleClient`: any ``/v1/chat/completions`` server: llama.cpp,
  vLLM, LM Studio, LocalAI, or a hosted provider with your own key.
* :class:`AnthropicClient`: optional, ``pip install pytest-llm-evals[anthropic]``.

HTTP adapters use the standard library. API keys are read from the
environment at construction, sent only in request headers, and never appear
in exceptions, reprs, cache files or reports.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import time
import urllib.parse
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Protocol
from typing import runtime_checkable

__all__ = [
    "AnthropicClient",
    "Completion",
    "LLMClient",
    "LLMError",
    "Message",
    "OllamaClient",
    "OpenAICompatibleClient",
    "make_client",
    "strip_thinking",
]

Message = Mapping[str, str]
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class LLMError(RuntimeError):
    """A provider call failed. The message never contains credentials."""


@dataclass
class Completion:
    """One model response with the numbers the budget and reports need."""

    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False
    cost_usd: float = 0.0
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMClient(Protocol):
    """Anything with ``provider``, ``model`` and ``complete``."""

    provider: str
    model: str

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion: ...


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks some local reasoning models emit."""
    return _THINK_RE.sub("", text).strip()


def _check_url(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise LLMError(f"refusing non-http(s) URL scheme {parts.scheme!r}")
    return url.rstrip("/")


def _post_json(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> dict[str, Any]:
    parts = urllib.parse.urlsplit(_check_url(url))
    conn_cls = (
        http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    )
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    body = json.dumps(payload).encode("utf-8")
    conn = conn_cls(parts.netloc, timeout=timeout)
    try:
        conn.request("POST", path, body, {"Content-Type": "application/json", **headers})
        resp = conn.getresponse()
        raw = resp.read(MAX_RESPONSE_BYTES + 1)
    except (OSError, http.client.HTTPException) as exc:
        raise LLMError(f"cannot reach {parts.netloc}: {type(exc).__name__}: {exc}") from None
    finally:
        conn.close()
    if resp.status >= 400:
        detail = raw[:2000].decode("utf-8", "replace")
        raise LLMError(f"HTTP {resp.status} from {parts.netloc}: {detail}")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise LLMError("response larger than 8 MiB")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise LLMError("provider returned invalid JSON") from None
    if not isinstance(data, dict):
        raise LLMError("provider returned a non-object JSON response")
    return data


class OllamaClient:
    """Local model through ollama's ``/api/chat``. Defaults: ``qwen3`` on localhost."""

    provider = "ollama"

    def __init__(
        self, model: str | None = None, *, host: str | None = None, timeout: float = 300.0
    ) -> None:
        self.model = model or os.environ.get("LLM_EVAL_MODEL") or "qwen3"
        self.host = _check_url(host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434")
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"OllamaClient(model={self.model!r}, host={self.host!r})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if seed is not None:
            options["seed"] = seed
        payload: dict[str, Any] = {
            "model": params.pop("model", None) or self.model,
            "messages": [dict(m) for m in messages],
            "stream": False,
            "options": options,
        }
        if "format" in params:
            payload["format"] = params.pop("format")
        start = time.perf_counter()
        data = _post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        latency = time.perf_counter() - start
        message = data.get("message") or {}
        return Completion(
            text=strip_thinking(str(message.get("content", ""))),
            model=str(data.get("model", payload["model"])),
            provider=self.provider,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            latency_s=latency,
        )


class OpenAICompatibleClient:
    """Any OpenAI-compatible ``/chat/completions`` endpoint.

    ``base_url`` defaults to ``$OPENAI_BASE_URL`` or a local server on port
    8080 (llama.cpp's ``llama-server``). The key comes from ``api_key_env``
    (default ``OPENAI_API_KEY``) and is optional for local servers.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        provider: str = "openai",
        timeout: float = 300.0,
    ) -> None:
        self.provider = provider
        self.model = model or os.environ.get("LLM_EVAL_MODEL") or "local-model"
        self.base_url = _check_url(
            base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8080/v1"
        )
        self._api_key = os.environ.get(api_key_env) or ""
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(model={self.model!r}, base_url={self.base_url!r})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": params.pop("model", None) or self.model,
            "messages": [dict(m) for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if "response_format" in params:
            payload["response_format"] = params.pop("response_format")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        start = time.perf_counter()
        data = _post_json(f"{self.base_url}/chat/completions", payload, headers, self.timeout)
        latency = time.perf_counter() - start
        try:
            text = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            raise LLMError("unexpected chat/completions response shape") from None
        usage = data.get("usage") or {}
        return Completion(
            text=strip_thinking(str(text)),
            model=str(data.get("model", payload["model"])),
            provider=self.provider,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=latency,
        )


class AnthropicClient:
    """Claude through the official ``anthropic`` SDK (optional extra).

    The SDK reads ``ANTHROPIC_API_KEY`` (or a profile) itself. Sampling
    parameters are not sent: current Claude models fix them server-side.
    """

    provider = "anthropic"

    def __init__(self, model: str | None = None, *, max_tokens: int = 2048) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "the anthropic provider needs `pip install pytest-llm-evals[anthropic]`"
            ) from exc
        self.model = model or os.environ.get("LLM_EVAL_MODEL") or "claude-sonnet-5-5"
        self.max_tokens = max_tokens
        self._client: Any = anthropic.Anthropic()

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        kwargs: dict[str, Any] = {
            "model": params.pop("model", None) or self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": chat,
        }
        if system:
            kwargs["system"] = system
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # the SDK's typed errors; never echo request headers
            raise LLMError(f"anthropic call failed: {type(exc).__name__}") from None
        latency = time.perf_counter() - start
        if getattr(resp, "stop_reason", None) == "refusal":
            raise LLMError("anthropic model declined the request (stop_reason=refusal)")
        text = "".join(
            getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
        )
        return Completion(
            text=text,
            model=str(getattr(resp, "model", kwargs["model"])),
            provider=self.provider,
            input_tokens=int(resp.usage.input_tokens or 0),
            output_tokens=int(resp.usage.output_tokens or 0),
            latency_s=latency,
        )


def make_client(provider: str, model: str | None = None, base_url: str | None = None) -> LLMClient:
    """Build a client by provider name: ``ollama``, ``openai`` or ``anthropic``."""
    name = provider.strip().lower()
    if name == "ollama":
        return OllamaClient(model, host=base_url)
    if name in {"openai", "openai-compatible"}:
        return OpenAICompatibleClient(model, base_url=base_url)
    if name == "anthropic":
        return AnthropicClient(model)
    raise LLMError(f"unknown LLM provider {provider!r} (use ollama, openai or anthropic)")
