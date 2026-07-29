from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen3_post_training.common import read_jsonl
from qwen3_post_training.evaluation.answer_parser import (
    answers_equal,
    extract_final_answer,
    extract_gsm8k_reference,
)
from qwen3_post_training.evaluation.metrics import summarize_boolean_metric


def evaluate_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    evaluated = []
    for row in rows:
        prediction = extract_final_answer(row.get("completion", ""))
        reference = extract_gsm8k_reference(row.get("answer", row.get("reference", "")))
        evaluated.append(
            {
                **row,
                "parsed_prediction": prediction,
                "parsed_reference": reference,
                "correct": answers_equal(prediction, reference),
            }
        )
    summary = summarize_boolean_metric(evaluated)
    summary["parse_failures"] = sum(row["parsed_prediction"] is None for row in evaluated)
    return evaluated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GSM8K JSONL predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    output_path = Path(args.output) if args.output else predictions_path.with_suffix(".summary.json")
    _, summary = evaluate_rows(read_jsonl(predictions_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
