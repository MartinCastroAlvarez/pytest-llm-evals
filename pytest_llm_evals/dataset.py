"""Golden datasets: YAML, JSON or JSONL files of eval cases.

A dataset is a list of cases. YAML and JSON files hold either a list, or a
mapping with ``cases`` (and optional ``defaults`` merged into every case).
JSONL files hold one case object per line.

>>> case = EvalCase.from_mapping({"id": "paris", "input": "France?", "expected": "Paris"})
>>> case.id, case.expected
('paris', 'Paris')
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DatasetError", "EvalCase", "load_dataset", "resolve_dataset_path"]

MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_CASES = 50_000
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+=-]{0,127}$")
_KNOWN_FIELDS = frozenset(
    {"id", "input", "expected", "metadata", "min_score", "threshold", "tags", "context"}
)


class DatasetError(ValueError):
    """A dataset file is missing, malformed, too large or outside the project."""


@dataclass(frozen=True)
class EvalCase:
    """One golden case. ``input`` and ``expected`` can be any JSON-like value."""

    id: str
    input: Any
    expected: Any = None
    context: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    min_score: float | None = None
    dataset: str = ""

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, index: int = 0, dataset: str = "") -> EvalCase:
        if not isinstance(raw, Mapping):
            raise DatasetError(f"case #{index} must be a mapping, got {type(raw).__name__}")
        unknown = set(raw) - _KNOWN_FIELDS
        if unknown:
            raise DatasetError(f"case #{index}: unknown field(s) {sorted(unknown)}")
        if "input" not in raw:
            raise DatasetError(f"case #{index}: missing required field 'input'")
        case_id = str(raw.get("id", f"case-{index}"))
        if not _ID_RE.match(case_id):
            raise DatasetError(
                f"case #{index}: id {case_id[:40]!r} must match {_ID_RE.pattern} "
                "(it becomes part of the pytest node id)"
            )
        tags = raw.get("tags") or ()
        if isinstance(tags, str) or not isinstance(tags, Iterable):
            raise DatasetError(f"case {case_id}: tags must be a list of strings")
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise DatasetError(f"case {case_id}: metadata must be a mapping")
        threshold = raw.get("min_score", raw.get("threshold"))
        min_score: float | None = None
        if threshold is not None:
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise DatasetError(f"case {case_id}: min_score must be a number")
            min_score = float(threshold)
            if not 0.0 <= min_score <= 1.0:
                raise DatasetError(f"case {case_id}: min_score must be within [0, 1]")
        return cls(
            id=case_id,
            input=raw["input"],
            expected=raw.get("expected"),
            context=raw.get("context"),
            metadata=dict(metadata),
            tags=tuple(str(t) for t in tags),
            min_score=min_score,
            dataset=dataset,
        )


def resolve_dataset_path(spec: str | Path, *, base: Path, root: Path) -> Path:
    """Resolve ``spec`` against ``base`` (the test file's folder), then ``root``.

    The result must stay inside ``root`` (the pytest rootdir): a marker cannot
    make the plugin read ``../../etc/passwd``.
    """
    spec_path = Path(spec)
    root = root.resolve()
    candidates = [spec_path] if spec_path.is_absolute() else [base / spec_path, root / spec_path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if not resolved.is_relative_to(root):
                raise DatasetError(f"dataset {spec} resolves outside the project root")
            return resolved
    raise DatasetError(f"dataset {spec} not found (looked in {base} and {root})")


def _canon(raw: Mapping[str, Any]) -> dict[str, Any]:
    """``threshold`` is an alias of ``min_score``; a case's own value beats a default."""
    out = dict(raw)
    if "threshold" in out and "min_score" not in out:
        out["min_score"] = out.pop("threshold")
    return out


def _parse(path: Path, text: str) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[Any] = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{lineno}: invalid JSON ({exc.msg})") from None
        return rows
    if suffix == ".json":
        try:
            data: Any = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}: invalid JSON ({exc.msg})") from None
    elif suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise DatasetError(f"{path.name}: invalid YAML ({exc})") from None
    else:
        raise DatasetError(f"{path.name}: unsupported dataset type (use .yaml, .json or .jsonl)")
    if data is None:
        return []
    if isinstance(data, Mapping):
        if "cases" not in data:
            raise DatasetError(f"{path.name}: a mapping dataset needs a 'cases' list")
        defaults = data.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            raise DatasetError(f"{path.name}: 'defaults' must be a mapping")
        cases = data["cases"] or []
        if not isinstance(cases, list):
            raise DatasetError(f"{path.name}: 'cases' must be a list")
        return [{**_canon(defaults), **_canon(c)} if isinstance(c, Mapping) else c for c in cases]
    if not isinstance(data, list):
        raise DatasetError(f"{path.name}: a dataset must be a list of cases")
    return data


def load_dataset(path: str | Path, *, tags: Iterable[str] = ()) -> list[EvalCase]:
    """Load and validate a dataset file. ``tags`` keeps only cases with any of them."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError:
        raise DatasetError(f"dataset {path} not found") from None
    if size > MAX_DATASET_BYTES:
        raise DatasetError(f"{path.name} is larger than {MAX_DATASET_BYTES} bytes")
    rows = _parse(path, path.read_text(encoding="utf-8"))
    if len(rows) > MAX_CASES:
        raise DatasetError(f"{path.name} has more than {MAX_CASES} cases")
    cases = [EvalCase.from_mapping(row, index=i, dataset=path.name) for i, row in enumerate(rows)]
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise DatasetError(f"{path.name}: duplicate case id {case.id!r}")
        seen.add(case.id)
    wanted = set(tags)
    if wanted:
        cases = [c for c in cases if wanted.intersection(c.tags)]
    return cases
