"""Example wiring: real local models when asked, deterministic stand-ins otherwise."""

from __future__ import annotations

import os

import pytest
from local_models import extractive_reader
from local_models import overlap_judge

from pytest_llm_evals import make_client
from pytest_llm_evals.testing import FakeLLM

LIVE = os.environ.get("LLM_EVAL_EXAMPLES_LIVE") == "1"


@pytest.fixture(scope="session")
def llm_client(llm_eval_session):  # type: ignore[no-untyped-def]
    if LIVE:
        s = llm_eval_session
        return make_client(
            s.option("llm_provider"), s.option("llm_model"), s.option("llm_base_url")
        )
    return FakeLLM(extractive_reader, provider="local", model="extractive-reader")


@pytest.fixture(scope="session")
def llm_judge_client(llm_eval_session):  # type: ignore[no-untyped-def]
    if LIVE:
        s = llm_eval_session
        return make_client(
            s.option("llm_judge_provider") or s.option("llm_provider"), s.option("llm_judge_model")
        )
    return FakeLLM(overlap_judge, provider="local", model="overlap-judge")
