from datasets import Dataset


def test_conversational_sft_schema_has_no_prompt_column() -> None:
    dataset = Dataset.from_dict(
        {
            "prompt": ["source prompt"],
            "messages": [
                [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            ],
            "reference": ["answer"],
        }
    )

    prepared = dataset.select_columns(["messages"])

    assert prepared.column_names == ["messages"]
    assert prepared[0]["messages"][-1]["role"] == "assistant"