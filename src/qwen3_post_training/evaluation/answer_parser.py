from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from qwen3_post_training.data.formatting import FINAL_ANSWER_TAG

NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?%?")
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")
CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_gsm8k_reference(answer: str) -> str | None:
    _, separator, final = answer.rpartition("####")
    return final.strip() if separator else extract_final_answer(answer)


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None

    marker_matches = re.findall(
        rf"{re.escape(FINAL_ANSWER_TAG)}\s*(.+)", text, flags=re.IGNORECASE
    )
    if marker_matches:
        lines = marker_matches[-1].strip().splitlines()
        if not lines:
            return None       # 或 continue 到 fallback 逻辑
        candidate = lines[0]
        number = _last_number(candidate)
        return number or candidate.strip()

    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()

    return _last_number(text)


def normalize_numeric_answer(value: str | None) -> Decimal | None:
    if value is None:
        return None
    cleaned = value.strip().replace("$", "").replace(",", "")
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]
    cleaned = cleaned.rstrip(". ")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    return number / Decimal(100) if is_percent else number


def answers_equal(prediction: str | None, reference: str | None) -> bool:
    predicted_number = normalize_numeric_answer(prediction)
    reference_number = normalize_numeric_answer(reference)
    if predicted_number is not None and reference_number is not None:
        return predicted_number == reference_number
    if prediction is None or reference is None:
        return False
    return prediction.strip().casefold() == reference.strip().casefold()


def extract_python_code(text: str) -> str:
    blocks = CODE_BLOCK_PATTERN.findall(text or "")
    return (blocks[-1] if blocks else text or "").strip()


def has_final_answer_format(text: str) -> bool:
    return len(re.findall(re.escape(FINAL_ANSWER_TAG), text or "", flags=re.IGNORECASE)) == 1


def _last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text)
    return matches[-1].strip() if matches else None
