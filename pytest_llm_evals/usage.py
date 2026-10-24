"""Token, cost and latency accounting, and budgets that fail the run.

:class:`TrackedClient` wraps a client (usually a :class:`~pytest_llm_evals.cache.CachedClient`),
prices every completion, adds it to the session :class:`UsageLedger` and to
the running test's record, and raises :class:`BudgetExceededError` once a
budget is crossed. After that, every further call is refused *before* it
reaches the provider, so a runaway suite stops spending.

Cached responses count their tokens in reports but cost 0 and do not count
against ``max_tokens`` (only live tokens are spent).
"""

from __future__ import annotations

import contextvars
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from pytest_llm_evals.llm import Completion
from pytest_llm_evals.llm import LLMClient
from pytest_llm_evals.llm import Message
from pytest_llm_evals.pricing import PricingTable

__all__ = [
    "Budget",
    "BudgetExceededError",
    "TrackedClient",
    "Usage",
    "UsageLedger",
    "current_usage",
]


class BudgetExceededError(RuntimeError):
    """A cost, token or latency budget was exceeded."""


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    live_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    max_latency_s: float = 0.0
    unpriced_calls: int = 0
    by_role: dict[str, int] = field(default_factory=dict)

    def add(self, completion: Completion, role: str, priced: bool) -> None:
        self.calls += 1
        self.by_role[role] = self.by_role.get(role, 0) + 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.latency_s += completion.latency_s
        self.max_latency_s = max(self.max_latency_s, completion.latency_s)
        if completion.cached:
            self.cached_calls += 1
        else:
            self.live_tokens += completion.total_tokens
            self.cost_usd += completion.cost_usd
            if not priced:
                self.unpriced_calls += 1

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_tokens"] = self.total_tokens
        data["cost_usd"] = round(self.cost_usd, 6)
        data["latency_s"] = round(self.latency_s, 4)
        data["max_latency_s"] = round(self.max_latency_s, 4)
        return data


@dataclass(frozen=True)
class Budget:
    max_usd: float | None = None
    max_tokens: int | None = None
    max_latency_s: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_usd", "max_tokens", "max_latency_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must not be negative")


#: The usage bucket of the running test (set by the plugin).
current_usage: contextvars.ContextVar[Usage | None] = contextvars.ContextVar(
    "llm_eval_usage", default=None
)


class UsageLedger:
    """Session-wide usage, thread-safe, with budget checks."""

    def __init__(self, budget: Budget | None = None, pricing: PricingTable | None = None) -> None:
        self.budget = budget or Budget()
        self.pricing = pricing or PricingTable()
        self.total = Usage()
        self.violations: list[str] = []
        self._lock = threading.Lock()

    @property
    def exceeded(self) -> bool:
        return bool(self.violations)

    def check_open(self) -> None:
        if self.violations:
            raise BudgetExceededError(f"LLM budget exhausted: {self.violations[0]}; call refused")

    def record(self, completion: Completion, *, role: str = "app") -> None:
        cost, priced = (
            (0.0, True)
            if completion.cached
            else self.pricing.cost(
                completion.provider,
                completion.model,
                completion.input_tokens,
                completion.output_tokens,
            )
        )
        completion.cost_usd = cost
        new: list[str] = []
        with self._lock:
            self.total.add(completion, role, priced)
            bucket = current_usage.get()
            if bucket is not None:
                bucket.add(completion, role, priced)
            b = self.budget
            if b.max_usd is not None and self.total.cost_usd > b.max_usd:
                new.append(f"cost ${self.total.cost_usd:.4f} > budget ${b.max_usd:.4f}")
            if b.max_tokens is not None and self.total.live_tokens > b.max_tokens:
                new.append(f"tokens {self.total.live_tokens} > budget {b.max_tokens}")
            if b.max_latency_s is not None and completion.latency_s > b.max_latency_s:
                new.append(
                    f"{completion.provider}:{completion.model} call took "
                    f"{completion.latency_s:.2f}s > budget {b.max_latency_s:.2f}s"
                )
            self.violations.extend(v for v in new if v not in self.violations)
        if new:
            raise BudgetExceededError("LLM budget exceeded: " + "; ".join(new))


class TrackedClient:
    """Price, time and budget-check every call of ``inner``."""

    def __init__(self, inner: LLMClient, ledger: UsageLedger, *, role: str = "app") -> None:
        self.inner = inner
        self.ledger = ledger
        self.role = role
        self.provider = inner.provider
        self.model = inner.model

    def __repr__(self) -> str:
        return f"TrackedClient({self.inner!r}, role={self.role!r})"

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        self.ledger.check_open()
        start = time.perf_counter()
        completion = self.inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed, **params
        )
        if not completion.latency_s and not completion.cached:
            completion.latency_s = time.perf_counter() - start
        self.ledger.record(completion, role=self.role)
        return completion

    def __call__(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        """Shortcut: ``llm("question")`` returns the text."""
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(messages, **kwargs).text
