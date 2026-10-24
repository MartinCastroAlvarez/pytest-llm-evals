"""Config-driven pricing: USD per million input/output tokens.

Local providers (ollama, fake, local servers) always cost 0. A pricing file
(YAML or JSON) overrides or extends the built-in table::

    free_providers: [ollama, fake, llamacpp]
    models:
      claude-sonnet-5-5: {input: 2.00, output: 10.00}
      "my-hosted-*":     {input: 0.50, output: 1.50}   # glob patterns allowed

Unknown hosted models cost 0 and are reported as *unpriced*, so a missing
price is visible instead of silently passing a USD budget.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULT_PRICES", "Price", "PricingTable"]

MAX_PRICING_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


# First-party list prices (USD / 1M tokens) for the optional Anthropic adapter.
DEFAULT_PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.0, 20.0),
    "claude-sonnet-5-5": Price(2.0, 10.0),
    "claude-haiku-4-5*": Price(1.0, 5.0),
}
DEFAULT_FREE_PROVIDERS = ("ollama", "fake", "local", "llamacpp", "vllm", "lmstudio")


class PricingTable:
    def __init__(
        self,
        prices: Mapping[str, Price] | None = None,
        *,
        free_providers: Iterable[str] = DEFAULT_FREE_PROVIDERS,
    ) -> None:
        self.prices = dict(DEFAULT_PRICES if prices is None else prices)
        self.free_providers = frozenset(p.lower() for p in free_providers)

    @classmethod
    def from_file(cls, path: str | Path) -> PricingTable:
        path = Path(path)
        if path.stat().st_size > MAX_PRICING_BYTES:
            raise ValueError(f"pricing file {path} is larger than 1 MiB")
        text = path.read_text(encoding="utf-8")
        data: Any = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(data, Mapping):
            raise ValueError(f"pricing file {path} must be a mapping")
        prices = dict(DEFAULT_PRICES)
        models = data.get("models") or {}
        if not isinstance(models, Mapping):
            raise ValueError("pricing 'models' must be a mapping")
        for name, entry in models.items():
            if not isinstance(entry, Mapping) or not {"input", "output"} <= set(entry):
                raise ValueError(f"pricing for {name!r} needs 'input' and 'output'")
            inp, out = float(entry["input"]), float(entry["output"])
            if inp < 0 or out < 0:
                raise ValueError(f"pricing for {name!r} must not be negative")
            prices[str(name)] = Price(inp, out)
        free = data.get("free_providers", DEFAULT_FREE_PROVIDERS)
        return cls(prices, free_providers=[str(p) for p in free])

    def lookup(self, provider: str, model: str) -> Price | None:
        if provider.lower() in self.free_providers:
            return Price(0.0, 0.0)
        if model in self.prices:
            return self.prices[model]
        for pattern, price in self.prices.items():
            if fnmatch.fnmatchcase(model, pattern):
                return price
        return None

    def cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> tuple[float, bool]:
        """Return ``(usd, priced)``; ``priced`` is False for an unknown hosted model."""
        price = self.lookup(provider, model)
        if price is None:
            return 0.0, False
        return price.cost(input_tokens, output_tokens), True
