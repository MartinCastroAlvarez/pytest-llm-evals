"""Shared fixtures: a fake local HTTP model server and pytester helpers. No network."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

# A conftest for pytester runs: replaces the provider clients with scripted fakes.
FAKE_CONFTEST = """
import json
import pytest
from pytest_llm_evals.testing import FakeLLM

ANSWERS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Milan is the capital", "Peru": "Lima"}

def app_reply(messages):
    text = messages[-1]["content"]
    for key, value in ANSWERS.items():
        if key in text:
            return value
    return "I do not know"

def judge_reply(messages):
    text = messages[-1]["content"]
    if "RESPONSE A" in text:
        a = text.split("<<<RESPONSE A", 1)[1].split("<<<END RESPONSE A", 1)[0]
        return json.dumps({"winner": "A" if "Paris" in a else "B", "reason": "content"})
    cand = text.split("<<<CANDIDATE", 1)[1].split("<<<END CANDIDATE", 1)[0]
    return json.dumps({"score": 5 if ("Paris" in cand or "Tokyo" in cand) else 2, "reason": "ok"})

@pytest.fixture(scope="session")
def llm_client():
    return FakeLLM(app_reply, model="fake-app")

@pytest.fixture(scope="session")
def llm_judge_client():
    return FakeLLM(judge_reply, model="fake-judge")
"""


class FakeServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responder: Callable[[str, dict[str, Any]], tuple[int, Any]] = lambda path, body: (
            200,
            {},
        )
        self.url = ""


@pytest.fixture
def fake_server() -> Iterator[FakeServer]:
    """A local HTTP server standing in for ollama / an OpenAI-compatible server."""
    state = FakeServer()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
            status, payload = state.responder(self.path, body)
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def evals(pytester: pytest.Pytester) -> pytest.Pytester:
    """A pytester project with scripted app and judge models."""
    pytester.makeconftest(FAKE_CONFTEST)
    pytester.makefile(
        ".yaml",
        qa="""
cases:
  - id: france
    input: "What is the capital of France?"
    expected: Paris
    tags: [europe]
  - id: japan
    input: "What is the capital of Japan?"
    expected: Tokyo
    tags: [asia]
  - id: italy
    input: "What is the capital of Italy?"
    expected: Rome
    tags: [europe]
""",
    )
    return pytester
