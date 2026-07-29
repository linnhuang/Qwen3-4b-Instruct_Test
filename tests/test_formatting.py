from qwen3_post_training.data.formatting import (
    FINAL_ANSWER_TAG,
    code_messages,
    gsm8k_reference,
    math_messages,
)


def test_math_messages_have_output_contract() -> None:
    messages = math_messages("What is 1 + 1?")
    assert FINAL_ANSWER_TAG in messages[0]["content"]
    assert messages[1]["content"] == "What is 1 + 1?"


def test_gsm8k_reference_rewrites_delimiter() -> None:
    reference = gsm8k_reference("One plus one is two.\n#### 2")
    assert reference.endswith("Final answer: 2")


def test_code_messages_include_public_examples() -> None:
    messages = code_messages("Return one.", ["assert f() == 1"])
    assert "assert f() == 1" in messages[1]["content"]
