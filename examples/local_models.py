"""Deterministic stand-ins for the local models, so the example runs anywhere.

Set ``LLM_EVAL_EXAMPLES_LIVE=1`` to use the real local models instead
(``ollama pull qwen3`` first): the app model and the judge then come from
``--llm-provider`` / ``--llm-model`` / ``--llm-judge-model``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from collections.abc import Sequence

from rag import words


def extractive_reader(messages: Sequence[Mapping[str, str]]) -> str:
    """Answer with the context sentence that best overlaps the question, cited."""
    user = messages[-1]["content"]
    question = user.rsplit("Question:", 1)[-1]
    best, best_score = None, 0
    for doc_id, text in re.findall(r"^\[(\w+)\] (.+)$", user, re.MULTILINE):
        for sentence in re.split(r"(?<=\.)\s+", text):
            score = len(words(question) & words(sentence))
            if score > best_score:
                best, best_score = f"{sentence} [{doc_id}]", score
    return best if best and best_score >= 2 else "I don't know."


def overlap_judge(messages: Sequence[Mapping[str, str]]) -> str:
    """Grade the candidate by word overlap with the reference answer (1-5)."""
    user = messages[-1]["content"]

    def block(label: str) -> str:
        m = re.search(rf"<<<{label} (\w+)>>>\n(.*?)\n<<<END {label} \1>>>", user, re.DOTALL)
        return m.group(2) if m else ""

    if "RESPONSE A" in user:
        a, b = block("RESPONSE A"), block("RESPONSE B")
        ref = words(block("REQUEST"))
        wa, wb = len(words(a) & ref), len(words(b) & ref)
        return json.dumps({"winner": "A" if wa > wb else "B" if wb > wa else "tie"})
    reference, candidate = words(block("REFERENCE ANSWER")), words(block("CANDIDATE"))
    recall = len(reference & candidate) / len(reference) if reference else 0.0
    return json.dumps({"score": 1 + round(4 * recall), "reason": "overlap with the reference"})
