"""Scorers: pure functions that turn an output into a :class:`Score` in [0, 1].

Every scorer takes a ``threshold``; ``Score.passed`` is ``value >= threshold``
and ``bool(score)`` is ``passed``, so ``assert contains(out, "Paris")`` reads
like a unit test while ``llm_eval.add(score)`` keeps the number for reports.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Protocol

import jsonschema

__all__ = [
    "Embedder",
    "Score",
    "ToolCall",
    "bleu",
    "contains",
    "cosine",
    "embedding_similarity",
    "exact_match",
    "json_valid",
    "normalize_text",
    "not_contains",
    "numeric_close",
    "regex_match",
    "rouge_l",
    "rouge_n",
    "tokens",
    "trajectory_match",
]

MAX_TEXT = 1_000_000
MAX_PATTERN = 2_000
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class Score:
    """A named score in [0, 1] and whether it met its threshold."""

    name: str
    value: float
    threshold: float = 1.0
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if math.isnan(self.value):
            object.__setattr__(self, "value", 0.0)
        object.__setattr__(self, "value", min(1.0, max(0.0, float(self.value))))

    @property
    def passed(self) -> bool:
        return self.value >= self.threshold

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return f"{self.name}={self.value:.3f} (threshold {self.threshold:.2f}, {mark})"


def _text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    if len(text) > MAX_TEXT:
        raise ValueError(f"text longer than {MAX_TEXT} characters")
    return text


def normalize_text(text: str) -> str:
    """NFKC, casefold, collapse whitespace, strip surrounding punctuation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(text.split())
    return text.strip(" .,;:!?\"'`")


def tokens(text: str) -> list[str]:
    """Lower-cased word tokens used by ROUGE and BLEU."""
    return _WORD_RE.findall(unicodedata.normalize("NFKC", _text(text)).casefold())


# --------------------------------------------------------------------------- #
# String checks
# --------------------------------------------------------------------------- #
def exact_match(
    output: Any, expected: Any, *, normalize: bool = True, threshold: float = 1.0
) -> Score:
    """1.0 when ``output`` equals ``expected`` (or any of them, for a list)."""
    options = expected if isinstance(expected, list | tuple) else [expected]
    out = _text(output)
    norm: Callable[[str], str] = normalize_text if normalize else (lambda s: s)
    hit = any(norm(out) == norm(_text(o)) for o in options)
    return Score("exact_match", 1.0 if hit else 0.0, threshold)


def contains(
    output: Any,
    needles: str | Sequence[str],
    *,
    mode: Literal["all", "any"] = "all",
    case_sensitive: bool = False,
    threshold: float | None = None,
) -> Score:
    """Fraction of ``needles`` found in ``output``. ``mode="any"`` needs one."""
    items = [needles] if isinstance(needles, str) else list(needles)
    if not items:
        raise ValueError("contains() needs at least one needle")
    out = _text(output)
    hay = out if case_sensitive else out.casefold()
    found = [n for n in items if (n if case_sensitive else n.casefold()) in hay]
    missing = [n for n in items if n not in found]
    if mode == "any":
        return Score("contains_any", 1.0 if found else 0.0, threshold or 1.0, {"missing": missing})
    return Score(
        "contains",
        len(found) / len(items),
        1.0 if threshold is None else threshold,
        {"missing": missing},
    )


def not_contains(
    output: Any, needles: str | Sequence[str], *, case_sensitive: bool = False
) -> Score:
    """1.0 when none of ``needles`` occur (refusal leaks, banned phrases, PII markers)."""
    items = [needles] if isinstance(needles, str) else list(needles)
    out = _text(output)
    hay = out if case_sensitive else out.casefold()
    present = [n for n in items if (n if case_sensitive else n.casefold()) in hay]
    return Score("not_contains", 0.0 if present else 1.0, 1.0, {"present": present})


def regex_match(output: Any, pattern: str, *, flags: int = 0, fullmatch: bool = False) -> Score:
    """1.0 when ``pattern`` matches (``re.search``, or ``re.fullmatch``)."""
    if len(pattern) > MAX_PATTERN:
        raise ValueError(f"regex longer than {MAX_PATTERN} characters")
    compiled = re.compile(pattern, flags)
    out = _text(output)
    match = compiled.fullmatch(out) if fullmatch else compiled.search(out)
    return Score("regex_match", 1.0 if match else 0.0, 1.0, {"pattern": pattern})


# --------------------------------------------------------------------------- #
# Structured output
# --------------------------------------------------------------------------- #
def json_valid(output: Any, schema: Mapping[str, Any] | None = None) -> Score:
    """1.0 when ``output`` is JSON (code fences allowed) and matches ``schema``.

    The parsed value is in ``details["value"]``; schema errors list the JSON
    path and the validator message (at most 10).
    """
    if isinstance(output, str):
        text = output
        fenced = _FENCE_RE.match(text)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(_text(text))
        except json.JSONDecodeError as exc:
            return Score("json_valid", 0.0, 1.0, {"error": f"invalid JSON: {exc.msg}"})
    else:
        value = output
    if schema is None:
        return Score("json_valid", 1.0, 1.0, {"value": value})
    validator_cls = jsonschema.validators.validator_for(schema)
    schema_dict = dict(schema)
    validator_cls.check_schema(schema_dict)
    validator = validator_cls(schema_dict)
    errors = [
        f"{'/'.join(str(p) for p in e.absolute_path) or '$'}: {e.message[:200]}"
        for e in sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))[:10]
    ]
    return Score("json_valid", 0.0 if errors else 1.0, 1.0, {"errors": errors, "value": value})


def numeric_close(
    output: Any,
    expected: float,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 0.0,
) -> Score:
    """1.0 when the first number in ``output`` is within tolerance of ``expected``.

    Strings are scanned for a number (``"about 1,234.5 km"`` -> 1234.5).
    """
    if isinstance(output, bool):
        return Score("numeric_close", 0.0, 1.0, {"error": "boolean is not a number"})
    if isinstance(output, int | float):
        actual: float | None = float(output)
    else:
        found = _NUMBER_RE.search(_text(output))
        actual = float(found.group(0).replace(",", "")) if found else None
    if actual is None:
        return Score("numeric_close", 0.0, 1.0, {"error": "no number found"})
    ok = math.isclose(actual, float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
    return Score("numeric_close", 1.0 if ok else 0.0, 1.0, {"actual": actual})


# --------------------------------------------------------------------------- #
# Overlap metrics (no heavy dependencies)
# --------------------------------------------------------------------------- #
def _ngrams(toks: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def _f1(overlap: float, n_cand: int, n_ref: int) -> float:
    if not n_cand or not n_ref or not overlap:
        return 0.0
    precision, recall = overlap / n_cand, overlap / n_ref
    return 2 * precision * recall / (precision + recall)


def rouge_n(output: Any, reference: Any, *, n: int = 1, threshold: float = 0.5) -> Score:
    """ROUGE-N F1 over word n-grams.

    >>> rouge_n("the cat sat", "the cat sat down").value
    0.857...
    """
    cand, ref = _ngrams(tokens(output), n), _ngrams(tokens(reference), n)
    overlap = sum((cand & ref).values())
    return Score(f"rouge{n}", _f1(overlap, sum(cand.values()), sum(ref.values())), threshold)


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(output: Any, reference: Any, *, threshold: float = 0.5) -> Score:
    """ROUGE-L F1 (longest common subsequence of words)."""
    cand, ref = tokens(output), tokens(reference)
    return Score("rougeL", _f1(_lcs(cand, ref), len(cand), len(ref)), threshold)


def bleu(
    output: Any,
    references: Any | Sequence[Any],
    *,
    max_n: int = 4,
    threshold: float = 0.3,
) -> Score:
    """Sentence BLEU with add-one smoothing (Lin & Och, 2004) and brevity penalty.

    ``references`` may be one string or a list of acceptable references.
    """
    refs = [references] if isinstance(references, str) else list(references)
    cand = tokens(output)
    ref_toks = [tokens(r) for r in refs]
    if not cand or not any(ref_toks):
        return Score("bleu", 0.0, threshold)
    order = min(max_n, len(cand))
    log_p = 0.0
    for n in range(1, order + 1):
        c = _ngrams(cand, n)
        max_ref: Counter[tuple[str, ...]] = Counter()
        for r in ref_toks:
            max_ref |= _ngrams(r, n)
        clipped = sum((c & max_ref).values())
        total = sum(c.values())
        if n == 1:
            if clipped == 0:
                return Score("bleu", 0.0, threshold)
            log_p += math.log(clipped / total)
        else:
            log_p += math.log((clipped + 1) / (total + 1))
    closest = min((abs(len(r) - len(cand)), len(r)) for r in ref_toks)[1]
    bp = 1.0 if len(cand) > closest else math.exp(1 - closest / len(cand))
    return Score("bleu", bp * math.exp(log_p / order), threshold)


# --------------------------------------------------------------------------- #
# Embedding similarity (bring a local embedder; [embeddings] extra = fastembed)
# --------------------------------------------------------------------------- #
class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors have different dimensions")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def embedding_similarity(
    output: Any, reference: Any, *, embedder: Embedder, threshold: float = 0.8
) -> Score:
    """Cosine similarity of the two texts' embeddings, clipped to [0, 1]."""
    va, vb = embedder.embed([_text(output), _text(reference)])
    return Score("embedding_similarity", max(0.0, cosine(va, vb)), threshold)


class FastEmbedEmbedder:
    """Local ONNX embeddings via ``fastembed`` (``pip install pytest-llm-evals[embeddings]``)."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise ImportError(
                "embedding_similarity with FastEmbedEmbedder needs "
                "`pip install pytest-llm-evals[embeddings]`"
            ) from exc
        self._model = TextEmbedding(model_name=model)  # pragma: no cover

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:  # pragma: no cover
        return [list(map(float, v)) for v in self._model.embed(list(texts))]


# --------------------------------------------------------------------------- #
# Agent trajectories
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def coerce(cls, raw: Any) -> ToolCall:
        """Accept ``"search"``, ``{"name": ..., "args"|"arguments"|"input": ...}``,
        OpenAI/Anthropic-style dicts (``{"function": {"name", "arguments"}}``) or
        any object with ``name`` and ``args``."""
        if isinstance(raw, ToolCall):
            return raw
        if isinstance(raw, str):
            return cls(raw)
        if isinstance(raw, Mapping):
            if "function" in raw and isinstance(raw["function"], Mapping):
                raw = raw["function"]
            args = raw.get("args", raw.get("arguments", raw.get("input", {}))) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            return cls(str(raw["name"]), dict(args))
        name = getattr(raw, "name", None)
        if name is None:
            raise TypeError(f"cannot read a tool call from {type(raw).__name__}")
        args = getattr(raw, "args", None) or getattr(raw, "arguments", None) or {}
        return cls(str(name), dict(args))


def _call_matches(actual: ToolCall, expected: ToolCall, check_args: bool) -> bool:
    if actual.name != expected.name:
        return False
    if not check_args:
        return True
    return all(actual.args.get(k) == v for k, v in expected.args.items())


def trajectory_match(
    actual: Iterable[Any],
    expected: Iterable[Any],
    *,
    mode: Literal["exact", "in_order", "subset", "superset"] = "in_order",
    check_args: bool = False,
    threshold: float = 1.0,
) -> Score:
    """Compare an agent's tool calls with the expected ones.

    * ``exact``: same calls, same order, nothing extra.
    * ``in_order``: expected calls appear in this order; extra calls allowed.
    * ``subset``: every expected call appears, any order (extras allowed).
    * ``superset``: the agent called nothing outside ``expected``.

    With ``check_args`` an expected call's args must be a subset of the actual
    call's args. The value is the fraction of expected calls matched (for
    ``superset``: the fraction of actual calls that were allowed).
    """
    act = [ToolCall.coerce(a) for a in actual]
    exp = [ToolCall.coerce(e) for e in expected]
    names = {"actual": [a.name for a in act], "expected": [e.name for e in exp]}
    if mode == "superset":
        allowed = [a for a in act if any(_call_matches(a, e, check_args) for e in exp)]
        value = len(allowed) / len(act) if act else 1.0
        return Score("trajectory_superset", value, threshold, names)
    if not exp:
        ok = not act if mode == "exact" else True
        return Score(f"trajectory_{mode}", 1.0 if ok else 0.0, threshold, names)
    if mode == "subset":
        pool = list(act)
        hits = 0
        for e in exp:
            for i, a in enumerate(pool):
                if _call_matches(a, e, check_args):
                    hits += 1
                    del pool[i]
                    break
        return Score("trajectory_subset", hits / len(exp), threshold, names)
    if mode == "exact":
        same = len(act) == len(exp) and all(
            _call_matches(a, e, check_args) for a, e in zip(act, exp, strict=True)
        )
        if same:
            return Score("trajectory_exact", 1.0, threshold, names)
        prefix = 0
        for a, e in zip(act, exp, strict=False):
            if not _call_matches(a, e, check_args):
                break
            prefix += 1
        return Score("trajectory_exact", prefix / max(len(exp), len(act)), threshold, names)
    if mode != "in_order":
        raise ValueError(f"unknown trajectory mode {mode!r}")
    j = 0
    for a in act:
        if j < len(exp) and _call_matches(a, exp[j], check_args):
            j += 1
    return Score("trajectory_in_order", j / len(exp), threshold, names)
