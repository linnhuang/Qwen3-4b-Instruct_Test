from decimal import Decimal

from qwen3_post_training.evaluation.answer_parser import (
    answers_equal,
    extract_final_answer,
    extract_gsm8k_reference,
    extract_python_code,
    has_final_answer_format,
    normalize_numeric_answer,
)


def test_extract_marked_answer() -> None:
    assert extract_final_answer("Reasoning\nFinal answer: 1,234") == "1,234"


def test_extract_boxed_answer() -> None:
    assert extract_final_answer(r"Therefore, \boxed{42}.") == "42"


def test_extract_gsm8k_reference() -> None:
    assert extract_gsm8k_reference("work\n#### 17") == "17"


def test_numeric_normalization() -> None:
    assert normalize_numeric_answer("1,234.50") == Decimal("1234.50")
    assert normalize_numeric_answer("25%") == Decimal("0.25")


def test_answers_equal() -> None:
    assert answers_equal("1,000", "1000")
    assert not answers_equal("1001", "1000")


def test_format_requires_one_marker() -> None:
    assert has_final_answer_format("Final answer: 2")
    assert not has_final_answer_format("Final answer: 2\nFinal answer: 3")


def test_extract_python_code() -> None:
    assert extract_python_code("```python\ndef f():\n    return 1\n```").startswith("def f")
