from __future__ import annotations

from typing import Any

from qwen3_post_training.evaluation.answer_parser import (
    answers_equal,
    extract_final_answer,
    extract_gsm8k_reference,
    has_final_answer_format,
)


def completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def correctness_reward(completions, answer, **_kwargs) -> list[float]:
    references = [extract_gsm8k_reference(value) for value in answer]
    return [
        float(answers_equal(extract_final_answer(completion_text(completion)), reference))
        for completion, reference in zip(completions, references, strict=True)
    ]


def format_reward(completions, **_kwargs) -> list[float]:
    return [float(has_final_answer_format(completion_text(completion))) for completion in completions]


def build_weighted_rewards(
    correctness_weight: float = 1.0, format_weight: float = 0.1
):
    def weighted_correctness(completions, answer, **kwargs) -> list[float]:
        return [
            correctness_weight * value
            for value in correctness_reward(completions, answer, **kwargs)
        ]

    def weighted_format(completions, **kwargs) -> list[float]:
        return [format_weight * value for value in format_reward(completions, **kwargs)]

    weighted_correctness.__name__ = "correctness_reward"
    weighted_format.__name__ = "format_reward"
    return [weighted_correctness, weighted_format]
