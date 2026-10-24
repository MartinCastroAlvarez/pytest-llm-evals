"""pytest-llm-evals: evaluate LLM apps and agents like unit tests.

Runs 100% locally: the default model is a local ollama one, responses are
cached, and nothing here needs a paid API.
"""

from pytest_llm_evals.cache import CachedClient
from pytest_llm_evals.cache import CacheMissError
from pytest_llm_evals.cache import ResponseCache
from pytest_llm_evals.dataset import DatasetError
from pytest_llm_evals.dataset import EvalCase
from pytest_llm_evals.dataset import load_dataset
from pytest_llm_evals.judge import Judge
from pytest_llm_evals.judge import PairwiseVerdict
from pytest_llm_evals.judge import Rubric
from pytest_llm_evals.judge import Verdict
from pytest_llm_evals.llm import AnthropicClient
from pytest_llm_evals.llm import Completion
from pytest_llm_evals.llm import LLMClient
from pytest_llm_evals.llm import LLMError
from pytest_llm_evals.llm import OllamaClient
from pytest_llm_evals.llm import OpenAICompatibleClient
from pytest_llm_evals.llm import make_client
from pytest_llm_evals.pricing import Price
from pytest_llm_evals.pricing import PricingTable
from pytest_llm_evals.results import EvalRecorder
from pytest_llm_evals.results import pass_at_k
from pytest_llm_evals.scores import Score
from pytest_llm_evals.scores import ToolCall
from pytest_llm_evals.scores import bleu
from pytest_llm_evals.scores import contains
from pytest_llm_evals.scores import embedding_similarity
from pytest_llm_evals.scores import exact_match
from pytest_llm_evals.scores import json_valid
from pytest_llm_evals.scores import not_contains
from pytest_llm_evals.scores import numeric_close
from pytest_llm_evals.scores import regex_match
from pytest_llm_evals.scores import rouge_l
from pytest_llm_evals.scores import rouge_n
from pytest_llm_evals.scores import trajectory_match
from pytest_llm_evals.usage import Budget
from pytest_llm_evals.usage import BudgetExceededError
from pytest_llm_evals.usage import TrackedClient
from pytest_llm_evals.usage import UsageLedger

__version__ = "0.1.0"

__all__ = [
    "AnthropicClient",
    "Budget",
    "BudgetExceededError",
    "CacheMissError",
    "CachedClient",
    "Completion",
    "DatasetError",
    "EvalCase",
    "EvalRecorder",
    "Judge",
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "OpenAICompatibleClient",
    "PairwiseVerdict",
    "Price",
    "PricingTable",
    "ResponseCache",
    "Rubric",
    "Score",
    "ToolCall",
    "TrackedClient",
    "UsageLedger",
    "Verdict",
    "__version__",
    "bleu",
    "contains",
    "embedding_similarity",
    "exact_match",
    "json_valid",
    "load_dataset",
    "make_client",
    "not_contains",
    "numeric_close",
    "pass_at_k",
    "regex_match",
    "rouge_l",
    "rouge_n",
    "trajectory_match",
]
