from qwen3_post_training.evaluation.generate_evalplus import task_messages


def test_evalplus_prompt_uses_code_contract() -> None:
    messages = task_messages({"prompt": "def add(a, b):\n    pass"})
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "def add(a, b):\n    pass"}