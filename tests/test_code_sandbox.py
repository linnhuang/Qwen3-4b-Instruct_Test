from qwen3_post_training.rewards.code_sandbox import run_python_tests


def test_code_sandbox_passes_assertions() -> None:
    result = run_python_tests("def add(a, b): return a + b", ["assert add(1, 2) == 3"])
    assert result.passed


def test_code_sandbox_reports_failure() -> None:
    result = run_python_tests("def add(a, b): return 0", ["assert add(1, 2) == 3"])
    assert not result.passed
