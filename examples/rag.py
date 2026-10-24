"""A tiny local RAG function: keyword retrieval over a handful of documents,
then generation by any LLM client (a local ollama model by default)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

DOCS: dict[str, str] = {
    "mercury": "Mercury is the smallest planet in the Solar System and the closest to the Sun.",
    "venus": "Venus is the hottest planet, with a surface temperature of about 465 degrees "
    "Celsius.",
    "mars": "Mars has two small moons, Phobos and Deimos. Its red colour comes from iron oxide.",
    "jupiter": "Jupiter is the largest planet. Its Great Red Spot is a storm bigger than Earth.",
    "saturn": "Saturn's rings are made mostly of ice particles with some rock and dust.",
    "neptune": "Neptune is the farthest planet from the Sun and has the strongest winds measured.",
}

SYSTEM = (
    "Answer the question using only the context. Cite the document id in square "
    "brackets, like [mars]. If the context does not contain the answer, reply exactly: "
    "I don't know."
)
_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "of",
        "in",
        "to",
        "and",
        "what",
        "which",
        "how",
        "does",
        "do",
        "has",
        "its",
        "with",
    }
)


def words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


@dataclass
class RagAnswer:
    text: str
    sources: list[str]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def retrieve(question: str, k: int = 2) -> list[str]:
    q = words(question)
    scored = sorted(((len(q & words(text)), doc_id) for doc_id, text in DOCS.items()), reverse=True)
    return [doc_id for score, doc_id in scored[:k] if score > 0]


def build_prompt(question: str, doc_ids: list[str]) -> list[dict[str, str]]:
    context = "\n".join(f"[{d}] {DOCS[d]}" for d in doc_ids) or "(no documents found)"
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def answer(question: str, llm: Any) -> RagAnswer:
    """Retrieve, then generate. ``llm`` is any client with ``complete(messages)``."""
    doc_ids = retrieve(question)
    calls: list[dict[str, Any]] = [{"name": "retrieve", "args": {"query": question}}]
    if not doc_ids:
        return RagAnswer("I don't know.", [], calls)
    calls.append({"name": "generate", "args": {"docs": doc_ids}})
    text = llm.complete(build_prompt(question, doc_ids), temperature=0.0).text.strip()
    cited = [d for d in doc_ids if f"[{d}]" in text]
    return RagAnswer(text, cited, calls)


def answer_json(question: str, llm: Any) -> str:
    """The same pipeline behind a JSON API contract."""
    result = answer(question, llm)
    return json.dumps({"answer": result.text, "sources": result.sources})
