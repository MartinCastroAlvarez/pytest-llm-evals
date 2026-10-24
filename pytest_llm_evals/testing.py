"""Test doubles: a scripted LLM and a bag-of-words embedder. No network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from pytest_llm_evals.llm import Completion
from pytest_llm_evals.llm import Message

__all__ = ["FakeEmbedder", "FakeLLM"]

Responder = Callable[[Sequence[Message]], str]


class FakeLLM:
    """A deterministic stand-in for a model.

    ``responses`` is a callable ``messages -> text``, a mapping from a
    substring of the last user message to a reply (first match wins), or a
    list of replies returned in turn. Tokens are counted as whitespace words.

    >>> llm = FakeLLM({"France": "Paris"})
    >>> llm.complete([{"role": "user", "content": "Capital of France?"}]).text
    'Paris'
    """

    def __init__(
        self,
        responses: Responder | Mapping[str, str] | Sequence[str] = (),
        *,
        model: str = "fake-model",
        provider: str = "fake",
        default: str = "",
        latency_s: float = 0.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.default = default
        self.latency_s = latency_s
        self.calls: list[dict[str, Any]] = []
        self._responses = responses
        self._turn = 0

    def _reply(self, messages: Sequence[Message]) -> str:
        if callable(self._responses):
            return self._responses(messages)
        if isinstance(self._responses, Mapping):
            last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            for needle, reply in self._responses.items():
                if needle in last:
                    return reply
            return self.default
        if self._turn < len(self._responses):
            reply = self._responses[self._turn]
            self._turn += 1
            return reply
        return self.default

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "temperature": temperature,
                "seed": seed,
                **params,
            }
        )
        text = self._reply(messages)
        return Completion(
            text=text,
            model=str(params.get("model") or self.model),
            provider=self.provider,
            input_tokens=sum(len(m["content"].split()) for m in messages),
            output_tokens=len(text.split()),
            latency_s=self.latency_s,
        )


class FakeEmbedder:
    """Hashing bag-of-words embedder (deterministic, 256 dims)."""

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"\w+", text.lower()):
                h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                vec[h % self.dims] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out
