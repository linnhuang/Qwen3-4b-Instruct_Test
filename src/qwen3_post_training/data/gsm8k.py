from __future__ import annotations

from datasets import Dataset, load_dataset

from .formatting import gsm8k_reference, math_messages


def load_gsm8k(
    split: str,
    subset: str = "main",
    revision: str = "main",
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_dataset("openai/gsm8k", subset, split=split, revision=revision)
    dataset = dataset.map(
        lambda row, index: {
            "id": f"gsm8k-{split}-{index}",
            "question": row["question"],
            "answer": row["answer"],
            "messages": math_messages(row["question"]),
            "reference": gsm8k_reference(row["answer"]),
        },
        with_indices=True,
    )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def prepare_gsm8k_sft(
    split: str = "train",
    subset: str = "main",
    revision: str = "main",
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_gsm8k(split, subset, revision, max_samples)
    dataset = dataset.map(
        lambda row: {
            "messages": row["messages"]
            + [{"role": "assistant", "content": row["reference"]}]
        }
    )
    return dataset.select_columns(["messages"])
