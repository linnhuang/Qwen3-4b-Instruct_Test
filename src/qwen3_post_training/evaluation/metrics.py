from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def accuracy(rows: Iterable[dict[str, Any]], key: str = "correct") -> float:
    values = [bool(row.get(key, False)) for row in rows]
    return sum(values) / len(values) if values else 0.0


def summarize_boolean_metric(
    rows: list[dict[str, Any]], key: str = "correct"
) -> dict[str, int | float]:
    correct = sum(bool(row.get(key, False)) for row in rows)
    total = len(rows)
    return {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total if total else 0.0,
    }
