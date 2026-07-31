from qwen3_post_training.rewards.math_reward import correctness_reward, format_reward
from qwen3_post_training.train_grpo import build_reward_functions


def test_correctness_reward() -> None:
    rewards = correctness_reward(
        completions=["Final answer: 42", "Final answer: 1"],
        answer=["reasoning #### 42", "reasoning #### 2"],
    )
    assert rewards == [1.0, 0.0]


def test_format_reward() -> None:
    assert format_reward(["Final answer: 42", "42"]) == [1.0, 0.0]


def test_mbpp_reward_selection() -> None:
    rewards = build_reward_functions(
        {
            "data": {"dataset": "mbpp"},
            "reward": {"public_test_weight": 1.0, "syntax_weight": 0.1},
        }
    )
    assert [reward.__name__ for reward in rewards] == ["public_test_reward", "syntax_reward"]
