"""End-to-end plugin tests with pytester and scripted models (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

QA_TEST = """
import pytest
from pytest_llm_evals import contains, exact_match

@pytest.mark.llm_eval(dataset="qa.yaml")
def test_qa(eval_case, llm, llm_eval):
    answer = llm(eval_case.input)
    llm_eval.output = answer
    llm_eval.add(contains(answer, eval_case.expected))
"""


def report(pytester: pytest.Pytester, name: str = "out.json") -> dict:  # type: ignore[type-arg]
    return json.loads((pytester.path / name).read_text())


def test_dataset_parametrizes_cases_and_soft_scores_fail(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    result = evals.runpytest("-v", "--llm-eval-json=out.json")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*test_qa[[]france[]] PASSED*",
            "*test_qa[[]italy[]] FAILED*",
            "*LLM eval below threshold:*",
            "*contains=0.000*",
            "*llm-evals summary*",
            "*cases: 3 (2 passed, 1 failed), mean score 0.667*",
            "*llm calls: 3 (0 cached, cache mode auto)*",
        ]
    )
    data = report(evals)
    assert data["summary"]["cases"] == 3 and data["summary"]["mean_score"] == pytest.approx(0.6667)
    italy = next(c for c in data["cases"] if c["case_id"] == "italy")
    assert italy["output"] == "Milan is the capital" and not italy["passed"]
    assert italy["scores"][0]["details"]["missing"] == ["Rome"] or italy["scores"][0]["value"] == 0
    assert italy["dataset"] == "qa.yaml" and italy["usage"]["calls"] == 1


def test_second_run_is_served_from_cache_and_replay_works(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    evals.runpytest().assert_outcomes(passed=2, failed=1)
    assert list((evals.path / ".llm-eval-cache").rglob("*.json"))
    again = evals.runpytest("--llm-cache=replay")
    again.assert_outcomes(passed=2, failed=1)
    again.stdout.fnmatch_lines(["*llm calls: 3 (3 cached, cache mode replay)*"])


def test_replay_miss_fails_clearly(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    result = evals.runpytest("--llm-cache=replay")
    result.assert_outcomes(failed=3)
    result.stdout.fnmatch_lines(["*CacheMissError*no recorded response*"])


def test_tags_filter_cases(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        @pytest.mark.llm_eval(dataset="qa.yaml", tags=["asia"])
        def test_asia(eval_case):
            assert eval_case.id == "japan"
        """
    )
    evals.runpytest().assert_outcomes(passed=1)


def test_case_min_score_and_marker_min_score(evals: pytest.Pytester) -> None:
    evals.makefile(".yaml", strict="- {id: a, input: x, min_score: 0.9}\n- {id: b, input: y}\n")
    evals.makepyfile(
        """
        import pytest
        @pytest.mark.llm_eval(dataset="strict.yaml", min_score=0.4)
        def test_mean(eval_case, llm_eval):
            llm_eval.score("half", 0.5, threshold=0.0)
        """
    )
    result = evals.runpytest("-v")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        ["*test_mean[[]a[]] FAILED*", "*mean score 0.500 < case min_score 0.90*"]
    )


def test_aggregate_min_score_fails_the_run(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        @pytest.mark.llm_eval(dataset="qa.yaml")
        def test_soft(eval_case, llm_eval):
            llm_eval.score("quality", 0.6 if eval_case.id == "italy" else 0.9, threshold=0.5)
        """
    )
    ok = evals.runpytest("--llm-eval-min-score=0.7")
    ok.assert_outcomes(passed=3)
    assert ok.ret == 0
    failing = evals.runpytest("--llm-eval-min-score=0.85")
    failing.assert_outcomes(passed=3)
    assert failing.ret == pytest.ExitCode.TESTS_FAILED
    failing.stdout.fnmatch_lines(["*FAILED gate: mean score 0.800 < --llm-eval-min-score 0.85*"])


def test_ini_configuration(evals: pytest.Pytester) -> None:
    evals.makeini("[pytest]\nllm_eval_min_score = 0.99\nllm_cache = off\n")
    evals.makepyfile(QA_TEST)
    result = evals.runpytest()
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(["*cache mode off*", "*FAILED gate: mean score*"])
    assert not (evals.path / ".llm-eval-cache").exists()


def test_bad_option_value_is_a_usage_error(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    result = evals.runpytest("--llm-cache=sometimes")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--llm-cache='sometimes': must be one of*"])
    assert evals.runpytest("--llm-eval-min-score=2").ret == pytest.ExitCode.USAGE_ERROR
    assert evals.runpytest("--llm-budget-tokens=0").ret == pytest.ExitCode.USAGE_ERROR


def test_repeat_with_pass_at_k(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        from pytest_llm_evals.cache import current_attempt

        @pytest.mark.llm_eval(repeat=4, k=2)
        def test_flaky(llm_eval):
            llm_eval.score("ok", 1.0 if current_attempt.get() == 2 else 0.0)

        @pytest.mark.llm_eval(repeat=3, min_passes=2)
        def test_needs_two(llm_eval):
            assert current_attempt.get() == 0

        @pytest.mark.llm_eval(repeat=3)
        def test_never(llm_eval):
            llm_eval.score("ok", 0.0)
        """
    )
    result = evals.runpytest("--llm-eval-json=out.json")
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(["*passes 0/3 < min_passes 1*"])
    cases = {c["nodeid"].split("::")[1]: c for c in report(evals)["cases"]}
    flaky = cases["test_flaky"]
    assert flaky["passed"] and flaky["passes"] == 1 and flaky["attempts"] == 4
    assert flaky["pass_at_1"] == 0.25 and flaky["pass_at_k"] == 0.5 and flaky["k"] == 2
    assert flaky["score"] == 0.25
    assert not cases["test_needs_two"]["passed"] and cases["test_needs_two"]["passes"] == 1


def test_repeat_attempts_get_distinct_cached_samples(evals: pytest.Pytester) -> None:
    evals.makeconftest(
        """
        import itertools, pytest
        from pytest_llm_evals.testing import FakeLLM
        counter = itertools.count()
        @pytest.fixture(scope="session")
        def llm_client():
            return FakeLLM(lambda m: f"sample-{next(counter)}")
        """
    )
    evals.makepyfile(
        """
        import pytest
        SEEN = []
        @pytest.mark.llm_eval(repeat=3)
        def test_samples(llm, llm_eval):
            SEEN.append(llm("q"))
            llm_eval.score("ok", 1.0)

        def test_check():
            assert SEEN in (["sample-0", "sample-1", "sample-2"],)
        """
    )
    evals.runpytest().assert_outcomes(passed=2)
    replay = evals.runpytest("--llm-cache=replay")
    replay.assert_outcomes(passed=2)


def test_token_budget_fails_the_run_and_stops_calls(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    result = evals.runpytest("--llm-budget-tokens=12", "-p", "no:randomly")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        ["*BudgetExceededError*", "*FAILED gate: budget: tokens * > budget 12*"]
    )
    outcomes = result.parseoutcomes()
    assert outcomes.get("failed", 0) >= 1


def test_usd_budget_with_priced_hosted_model(evals: pytest.Pytester) -> None:
    evals.makeconftest(
        """
        import pytest
        from pytest_llm_evals.testing import FakeLLM
        @pytest.fixture(scope="session")
        def llm_client():
            return FakeLLM(lambda m: "word " * 100, provider="openai", model="team-model")
        """
    )
    evals.makefile(".yaml", prices="models:\n  team-model: {input: 1000, output: 1000}\n")
    evals.makepyfile(
        """
        def test_spend(llm, llm_eval):
            llm("hello there")
            llm("hello again")
        """
    )
    result = evals.runpytest("--llm-pricing=prices.yaml", "--llm-budget-usd=0.15")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*cost $0.2040 > budget $0.1500*"])
    unpriced = evals.runpytest("--llm-cache=off")
    unpriced.stdout.fnmatch_lines(["*warning: 2 live call(s) to models with no price*"])
    assert evals.runpytest("--llm-pricing=missing.yaml").ret == pytest.ExitCode.USAGE_ERROR


def test_record_usage_for_own_client(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        def test_own(llm_eval):
            llm_eval.record_usage(model="mine", provider="ollama", input_tokens=40, output_tokens=2)
            llm_eval.score("ok", 1.0)
        """
    )
    result = evals.runpytest("--llm-budget-tokens=10")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*tokens 42 > budget 10*"])


def test_latency_budget(evals: pytest.Pytester) -> None:
    evals.makeconftest(
        """
        import pytest
        from pytest_llm_evals.testing import FakeLLM
        @pytest.fixture(scope="session")
        def llm_client():
            return FakeLLM(lambda m: "slow", latency_s=3.0)
        """
    )
    evals.makepyfile("def test_slow(llm, llm_eval):\n    llm('x')\n")
    result = evals.runpytest("--llm-budget-latency=1")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*took 3.00s > budget 1.00s*"])


def test_judge_fixture_end_to_end(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        from pytest_llm_evals import Rubric

        RUBRIC = Rubric("Names the correct capital city.", name="correct")

        @pytest.mark.llm_eval(dataset="qa.yaml")
        def test_judged(eval_case, llm, llm_judge, llm_eval):
            answer = llm(eval_case.input)
            llm_eval.add(llm_judge.score(answer, rubric=RUBRIC, input=eval_case.input))

        def test_pairwise(llm_judge, llm_eval):
            llm_eval.add(llm_judge.compare("Paris", "Lyon", rubric=RUBRIC).as_score())
        """
    )
    result = evals.runpytest("-v", "--llm-judge-samples=3", "--llm-eval-json=out.json")
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(["*test_judged[[]italy[]] FAILED*", "*judge:correct=0.250*"])
    data = report(evals)
    assert data["summary"]["usage"]["by_role"]["judge"] == 3 * 3 + 3 * 2
    france = next(c for c in data["cases"] if c["case_id"] == "france")
    assert france["scores"][0]["details"]["samples"] == [5.0, 5.0, 5.0]


def test_html_and_junit_reports(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        QA_TEST
        + """
def test_markup(llm_eval):
    llm_eval.score("<script>alert(1)</script>", 1.0)
"""
    )
    result = evals.runpytest("--llm-eval-html=reports/r.html", "--junitxml=junit.xml")
    result.stdout.fnmatch_lines(["*report: */reports/r.html*"])
    page = (evals.path / "reports" / "r.html").read_text()
    assert "<table>" in page and "test_qa[france]" in page
    assert "<script>" not in page and "&lt;script&gt;" in page
    junit = (evals.path / "junit.xml").read_text()
    assert 'name="llm_eval_score"' in junit and 'name="llm_eval_tokens"' in junit


def test_baseline_regression(evals: pytest.Pytester) -> None:
    evals.makepyfile(QA_TEST)
    evals.runpytest("--llm-eval-json=baseline.json")
    base = json.loads((evals.path / "baseline.json").read_text())
    for case in base["cases"]:
        case["passed"], case["score"] = True, 1.0
    (evals.path / "baseline.json").write_text(json.dumps(base))
    result = evals.runpytest("--llm-eval-baseline=baseline.json", "--llm-cache=replay")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    result.stdout.fnmatch_lines(
        [
            "*FAILED gate: regression: *test_qa[[]italy[]]: passed in baseline, fails now*",
            "*FAILED gate: regression: mean score 0.667 < baseline 1.000*",
        ]
    )
    evals.runpytest("--llm-eval-json=current.json", "--llm-cache=replay")
    same = evals.runpytest("--llm-eval-baseline=current.json", "--llm-cache=replay")
    same.assert_outcomes(passed=2, failed=1)
    assert "FAILED gate" not in same.stdout.str()
    (evals.path / "bad.json").write_text("{}")
    broken = evals.runpytest("--llm-eval-baseline=bad.json", "--llm-cache=replay")
    broken.stdout.fnmatch_lines(["*FAILED gate: baseline bad.json:*not a pytest-llm-evals report*"])


def test_marker_without_eval_case_argument_is_a_collection_error(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        @pytest.mark.llm_eval(dataset="qa.yaml")
        def test_x():
            pass
        """
    )
    result = evals.runpytest()
    result.stdout.fnmatch_lines(["*needs an `eval_case` argument*"])
    assert result.ret == pytest.ExitCode.INTERRUPTED


def test_dataset_outside_root_is_refused(
    evals: pytest.Pytester, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside") / "steal.yaml"
    outside.write_text("- {input: x}\n")
    evals.makepyfile(
        f"""
        import pytest
        @pytest.mark.llm_eval(dataset={str(outside)!r})
        def test_x(eval_case):
            pass
        """
    )
    result = evals.runpytest()
    result.stdout.fnmatch_lines(["*resolves outside the project root*"])


def test_eval_case_without_marker(evals: pytest.Pytester) -> None:
    evals.makepyfile("def test_x(eval_case):\n    pass\n")
    result = evals.runpytest()
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*uses `eval_case` without @pytest.mark.llm_eval*"])


def test_skip_xfail_and_plain_tests_are_untouched(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        def test_plain():
            assert 1

        @pytest.mark.llm_eval
        def test_skips(llm_eval):
            pytest.skip("not today")

        @pytest.mark.llm_eval
        def test_xfails(llm_eval):
            pytest.xfail("known")

        @pytest.mark.llm_eval
        def test_raises(llm_eval):
            raise RuntimeError("boom")

        @pytest.mark.llm_eval
        async def test_async(llm_eval):
            pass
        """
    )
    result = evals.runpytest("-p", "no:anyio")
    # async tests are left to an async plugin: without one, pytest itself fails them
    result.assert_outcomes(passed=1, skipped=1, xfailed=1, failed=2)
    result.stdout.fnmatch_lines(["*RuntimeError: boom*"])
    assert "cases: 1 (0 passed, 1 failed)" in result.stdout.str()


def test_no_summary_without_evals(pytester: pytest.Pytester) -> None:
    pytester.makepyfile("def test_x():\n    assert True\n")
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    assert "llm-evals summary" not in result.stdout.str()


def test_default_clients_are_built_from_options(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        """
        from pytest_llm_evals import OllamaClient, OpenAICompatibleClient

        def test_clients(llm_client, llm_judge_client, llm_eval_session):
            assert isinstance(llm_client, OpenAICompatibleClient)
            assert llm_client.model == "app-model"
            assert llm_client.base_url == "http://127.0.0.1:1/v1"
            assert isinstance(llm_judge_client, OllamaClient)
            assert llm_judge_client.model == "qwen3"
            assert llm_eval_session.cache.mode == "auto"
        """
    )
    result = pytester.runpytest(
        "--llm-provider=openai",
        "--llm-model=app-model",
        "--llm-base-url=http://127.0.0.1:1/v1",
        "--llm-judge-provider=ollama",
        "--llm-judge-base-url=http://127.0.0.1:2",
    )
    result.assert_outcomes(passed=1)


def test_invalid_repeat_marker(evals: pytest.Pytester) -> None:
    evals.makepyfile(
        """
        import pytest
        @pytest.mark.llm_eval(repeat=2, min_passes=3)
        def test_x(llm_eval):
            pass
        """
    )
    result = evals.runpytest()
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*min_passes <= repeat*"])


def test_plugin_is_registered_by_entry_point(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(["*LLM evals (pytest-llm-evals)*", "*--llm-budget-usd*"])
    markers = pytester.runpytest("--markers")
    markers.stdout.fnmatch_lines(["*@pytest.mark.llm_eval(dataset=None*"])


def test_absolute_report_paths(evals: pytest.Pytester, tmp_path: Path) -> None:
    evals.makepyfile(QA_TEST)
    target = tmp_path / "abs.json"
    evals.runpytest(f"--llm-eval-json={target}", f"--llm-cache-dir={tmp_path / 'cache'}")
    assert json.loads(target.read_text())["summary"]["cases"] == 3
    assert list((tmp_path / "cache").rglob("*.json"))
