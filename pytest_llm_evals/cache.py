"""Response cache: re-runs are free and deterministic.

Entries live at ``<dir>/<key[:2]>/<key>.json`` where ``key`` is the SHA-256 of
the canonical request: provider, model, messages and sampling parameters
(temperature, max_tokens, seed, the judge sample and the repeat attempt).
Credentials are never part of a request, and every stored string goes through
:func:`redact` so a token pasted into a prompt does not land on disk.

Modes:

* ``auto``: read a hit, otherwise call and store (default).
* ``record``: always call and overwrite.
* ``replay``: read only; a miss raises :class:`CacheMissError` (CI without a model).
* ``off``: no cache.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from collections.abc import Sequence
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

from pytest_llm_evals.llm import Completion
from pytest_llm_evals.llm import LLMClient
from pytest_llm_evals.llm import Message

__all__ = ["CacheMissError", "CachedClient", "ResponseCache", "current_attempt", "redact"]

CacheMode = Literal["auto", "record", "replay", "off"]
CACHE_MODES: tuple[CacheMode, ...] = ("auto", "record", "replay", "off")
SCHEMA = 1
MAX_ENTRY_BYTES = 8 * 1024 * 1024
_KEY_RE = re.compile(r"^[0-9a-f]{64}$")
_SECRET_PARAM_RE = re.compile(r"key|token|secret|password|authorization|credential", re.I)
_SECRET_VALUE_RE = re.compile(
    r"(sk-ant-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[0-9A-Za-z_\-]{30,}|Bearer\s+[A-Za-z0-9._\-]{16,})"
)

#: The repeat attempt of the running test (0 for the first). Part of the key,
#: so ``repeat=5`` replays five distinct recorded samples.
current_attempt: contextvars.ContextVar[int] = contextvars.ContextVar("llm_eval_attempt", default=0)


class CacheMissError(LookupError):
    """``replay`` mode found no recorded response for a request."""


def redact(value: Any) -> Any:
    """Replace token-shaped substrings with ``[REDACTED]``, recursively."""
    if isinstance(value, str):
        return _SECRET_VALUE_RE.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {str(k): redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        k: v for k, v in sorted(params.items()) if v is not None and not _SECRET_PARAM_RE.search(k)
    }


class ResponseCache:
    def __init__(self, directory: str | Path, mode: CacheMode = "auto") -> None:
        if mode not in CACHE_MODES:
            raise ValueError(f"cache mode must be one of {CACHE_MODES}, got {mode!r}")
        self.directory = Path(directory)
        self.mode: CacheMode = mode
        self.hits = 0
        self.misses = 0
        self.writes = 0

    @staticmethod
    def key(
        provider: str, model: str, messages: Sequence[Message], params: Mapping[str, Any]
    ) -> str:
        canonical = json.dumps(
            {
                "v": SCHEMA,
                "provider": provider,
                "model": model,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "params": _clean_params(params),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        if not _KEY_RE.match(key):
            raise ValueError("invalid cache key")
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        try:
            if path.is_symlink() or path.stat().st_size > MAX_ENTRY_BYTES:
                return None
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(entry, dict) or entry.get("key") != key or entry.get("schema") != SCHEMA:
            return None
        response = entry.get("response")
        return response if isinstance(response, dict) else None

    def put(self, key: str, request: Mapping[str, Any], response: Mapping[str, Any]) -> Path:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "schema": SCHEMA,
            "key": key,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "request": redact(dict(request)),
            "response": redact(dict(response)),
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, indent=1, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        self.writes += 1
        return path


class CachedClient:
    """Wrap any :class:`LLMClient` with a :class:`ResponseCache`."""

    def __init__(self, inner: LLMClient, cache: ResponseCache) -> None:
        self.inner = inner
        self.cache = cache
        self.provider = inner.provider
        self.model = inner.model

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        **params: Any,
    ) -> Completion:
        model = str(params.get("model") or self.inner.model)
        sample = params.pop("sample", None)
        key_params = {
            **params,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed,
            "sample": sample,
            "attempt": current_attempt.get() or None,
        }
        key_params.pop("model", None)
        mode = self.cache.mode
        if mode == "off":
            return self.inner.complete(
                messages, temperature=temperature, max_tokens=max_tokens, seed=seed, **params
            )
        key = self.cache.key(self.provider, model, messages, key_params)
        if mode in {"auto", "replay"}:
            hit = self.cache.get(key)
            if hit is not None:
                self.cache.hits += 1
                return Completion(
                    text=str(hit.get("text", "")),
                    model=str(hit.get("model", model)),
                    provider=self.provider,
                    input_tokens=int(hit.get("input_tokens", 0)),
                    output_tokens=int(hit.get("output_tokens", 0)),
                    latency_s=float(hit.get("latency_s", 0.0)),
                    cached=True,
                )
            self.cache.misses += 1
            if mode == "replay":
                raise CacheMissError(
                    f"no recorded response for {self.provider}:{model} (key {key[:12]}); "
                    "run once with --llm-cache=auto or record"
                )
        completion = self.inner.complete(
            messages, temperature=temperature, max_tokens=max_tokens, seed=seed, **params
        )
        self.cache.put(
            key,
            {
                "provider": self.provider,
                "model": model,
                "messages": [dict(m) for m in messages],
                "params": _clean_params(key_params),
            },
            {
                "text": completion.text,
                "model": completion.model,
                "input_tokens": completion.input_tokens,
                "output_tokens": completion.output_tokens,
                "latency_s": round(completion.latency_s, 4),
            },
        )
        return completion
