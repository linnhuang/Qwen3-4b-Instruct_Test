from __future__ import annotations

from datasets import Dataset, load_dataset

from .formatting import code_messages, fenced_python


def load_mbpp(
    split: str = "train",
    revision: str = "main",
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split, revision=revision)
    dataset = dataset.map(
        lambda row: {
            "id": f"mbpp-{row['task_id']}",
            "messages": code_messages(row["prompt"], row.get("test_list")),
            "reference": fenced_python(row["code"]),
        }
    )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def prepare_mbpp_sft(
    split: str = "train",
    revision: str = "main",
    max_samples: int | None = None,
) -> Dataset:
    dataset = load_mbpp(split, revision, max_samples)
    dataset = dataset.map(
        lambda row: {
            "messages": row["messages"]
            + [{"role": "assistant", "content": row["reference"]}]
        }
    )
    # The source MBPP schema already contains a string column named ``prompt``. TRL treats any
    # dataset containing that column as prompt-completion data and then requires ``completion``.
    # Keep only the conversational field so SFTTrainer applies the chat template instead.
    return dataset.select_columns(["messages"])
