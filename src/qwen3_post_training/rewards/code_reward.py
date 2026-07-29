from __future__ import annotations

import ast

from qwen3_post_training.evaluation.answer_parser import extract_python_code
from qwen3_post_training.rewards.code_sandbox import run_python_tests


def syntax_reward(completions, **_kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        try:
            ast.parse(extract_python_code(str(completion)))
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


def public_test_reward(completions, test_list, timeout_seconds: float = 5.0, **_kwargs):
    return [
        float(
            run_python_tests(
                extract_python_code(str(completion)), tests, timeout_seconds=timeout_seconds
            ).passed
        )
        for completion, tests in zip(completions, test_list, strict=True)
    ]
