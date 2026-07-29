from qwen3_post_training.rewards.math_reward import correctness_reward, format_reward


def test_correctness_reward() -> None:
    rewards = correctness_reward(
        completions=["Final answer: 42", "Final answer: 1"],
        answer=["reasoning #### 42", "reasoning #### 2"],
    )
    assert rewards == [1.0, 0.0]


def test_format_reward() -> None:
    assert format_reward(["Final answer: 42", "42"]) == [1.0, 0.0]
