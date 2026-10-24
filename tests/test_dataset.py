from __future__ import annotations

import json
from pathlib import Path

import pytest

from pytest_llm_evals import DatasetError
from pytest_llm_evals import EvalCase
from pytest_llm_evals import load_dataset
from pytest_llm_evals.dataset import resolve_dataset_path


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_list_and_mapping_with_defaults(tmp_path: Path) -> None:
    listed = write(tmp_path / "a.yaml", "- input: hi\n  expected: hello\n- id: two\n  input: x\n")
    cases = load_dataset(listed)
    assert [c.id for c in cases] == ["case-0", "two"]
    assert cases[0].expected == "hello" and cases[0].dataset == "a.yaml"
    mapped = write(
        tmp_path / "b.yml",
        "defaults:\n  tags: [smoke]\n  min_score: 0.5\ncases:\n  - id: c1\n    input: q\n"
        "  - id: c2\n    input: q2\n    threshold: 0.9\n    metadata: {lang: en}\n",
    )
    cases = load_dataset(mapped)
    assert cases[0].tags == ("smoke",) and cases[0].min_score == 0.5
    assert cases[1].min_score == 0.9 and cases[1].metadata == {"lang": "en"}


def test_jsonl_and_json(tmp_path: Path) -> None:
    rows = [{"id": "a", "input": {"q": 1}, "expected": [1]}, {"id": "b", "input": "x"}]
    jsonl = write(tmp_path / "d.jsonl", "\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert [c.input for c in load_dataset(jsonl)] == [{"q": 1}, "x"]
    js = write(tmp_path / "d.json", json.dumps(rows))
    assert len(load_dataset(js)) == 2
    assert load_dataset(write(tmp_path / "empty.yaml", "")) == []


def test_tag_filter(tmp_path: Path) -> None:
    path = write(
        tmp_path / "t.yaml",
        "- {id: a, input: x, tags: [fast]}\n- {id: b, input: y, tags: [slow]}\n- {id: c, input: z}\n",
    )
    assert [c.id for c in load_dataset(path, tags=["slow", "fast"])] == ["a", "b"]


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("bad.yaml", "- {id: a}\n", "missing required field 'input'"),
        ("bad.yaml", "- {input: a, nope: 1}\n", "unknown field"),
        ("bad.yaml", "- {id: '../x', input: a}\n", "must match"),
        ("bad.yaml", "- {id: a, input: x}\n- {id: a, input: y}\n", "duplicate case id"),
        ("bad.yaml", "- {input: a, tags: oops}\n", "tags must be a list"),
        ("bad.yaml", "- {input: a, metadata: [1]}\n", "metadata must be a mapping"),
        ("bad.yaml", "- {input: a, min_score: high}\n", "must be a number"),
        ("bad.yaml", "- {input: a, min_score: 2}\n", "within [0, 1]"),
        ("bad.yaml", "- just a string\n", "must be a mapping"),
        ("bad.yaml", "key: value\n", "needs a 'cases' list"),
        ("bad.yaml", "cases: {a: 1}\n", "'cases' must be a list"),
        ("bad.yaml", "defaults: [1]\ncases: []\n", "'defaults' must be a mapping"),
        ("bad.yaml", "42\n", "must be a list"),
        ("bad.yaml", "a: [unclosed\n", "invalid YAML"),
        ("bad.json", "{oops", "invalid JSON"),
        ("bad.jsonl", '{"input": 1}\n{bad\n', "bad.jsonl:2"),
        ("bad.csv", "input\nx\n", "unsupported dataset type"),
    ],
)
def test_invalid_datasets(tmp_path: Path, name: str, text: str, message: str) -> None:
    with pytest.raises(DatasetError, match=None) as info:
        load_dataset(write(tmp_path / name, text))
    assert message in str(info.value)


def test_yaml_is_safe_loaded(tmp_path: Path) -> None:
    path = write(
        tmp_path / "evil.yaml", "- input: !!python/object/apply:os.system ['echo pwned']\n"
    )
    with pytest.raises(DatasetError, match="invalid YAML"):
        load_dataset(path)


def test_size_limit_and_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pytest_llm_evals.dataset.MAX_DATASET_BYTES", 10)
    with pytest.raises(DatasetError, match="larger than"):
        load_dataset(write(tmp_path / "big.yaml", "- input: " + "x" * 50))
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "missing.yaml")
    monkeypatch.setattr("pytest_llm_evals.dataset.MAX_CASES", 1)
    monkeypatch.setattr("pytest_llm_evals.dataset.MAX_DATASET_BYTES", 10_000)
    with pytest.raises(DatasetError, match="more than 1 cases"):
        load_dataset(write(tmp_path / "many.yaml", "- {input: a}\n- {input: b}\n"))


def test_resolution_is_confined_to_root(tmp_path: Path) -> None:
    root = tmp_path / "project"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    write(tests_dir / "local.yaml", "- {input: a}\n")
    (root / "evals").mkdir()
    write(root / "evals" / "qa.yaml", "- {input: a}\n")
    outside = write(tmp_path / "secret.yaml", "- {input: s}\n")
    assert resolve_dataset_path("local.yaml", base=tests_dir, root=root).name == "local.yaml"
    assert resolve_dataset_path("evals/qa.yaml", base=tests_dir, root=root).parent.name == "evals"
    with pytest.raises(DatasetError, match="outside the project root"):
        resolve_dataset_path("../../secret.yaml", base=tests_dir, root=root)
    with pytest.raises(DatasetError, match="outside the project root"):
        resolve_dataset_path(str(outside), base=tests_dir, root=root)
    link = root / "evals" / "link.yaml"
    link.symlink_to(outside)
    with pytest.raises(DatasetError, match="outside the project root"):
        resolve_dataset_path("evals/link.yaml", base=tests_dir, root=root)
    with pytest.raises(DatasetError, match="not found"):
        resolve_dataset_path("nope.yaml", base=tests_dir, root=root)


def test_case_from_mapping_rejects_non_mapping() -> None:
    with pytest.raises(DatasetError):
        EvalCase.from_mapping([1, 2])  # type: ignore[arg-type]
