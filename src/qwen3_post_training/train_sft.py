from __future__ import annotations

import argparse
from typing import Any

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from qwen3_post_training.common import load_yaml, resolve_dtype, set_seed
from qwen3_post_training.data.gsm8k import prepare_gsm8k_sft
from qwen3_post_training.data.mbpp import prepare_mbpp_sft


def prepare_dataset(config: dict[str, Any]):
    data = config["data"]
    common = {
        "split": data.get("split", "train"),
        "revision": data.get("revision", "main"),
        "max_samples": data.get("max_train_samples"),
    }
    if data["dataset"] == "gsm8k":
        dataset = prepare_gsm8k_sft(subset=data.get("subset", "main"), **common)
    elif data["dataset"] == "mbpp":
        dataset = prepare_mbpp_sft(**common)
    else:
        raise ValueError(f"Unsupported SFT dataset: {data['dataset']}")

    validation_size = min(data.get("validation_size", 0), max(0, len(dataset) - 1))
    if validation_size == 0:
        return dataset, None
    split = dataset.train_test_split(test_size=validation_size, seed=config["training"]["seed"])
    return split["train"], split["test"]


def build_model(config: dict[str, Any]):
    model_config = config["model"]
    quantization_config = None
    if model_config.get("use_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision", "main"),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        quantization_config=quantization_config,
        device_map="auto" if quantization_config else None,
    )
    model.config.use_cache = False
    if model_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    return model


def run(config: dict[str, Any]) -> None:
    training = config["training"]
    set_seed(training.get("seed", 42))
    train_dataset, eval_dataset = prepare_dataset(config)
    model = build_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        revision=config["model"].get("revision", "main"),
        padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=training["output_dir"],
        num_train_epochs=training["num_train_epochs"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        max_length=training["max_length"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        logging_steps=training["logging_steps"],
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=training["eval_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        bf16=config["model"].get("dtype") == "bfloat16",
        gradient_checkpointing=config["model"].get("gradient_checkpointing", True),
        seed=training["seed"],
        report_to=training.get("report_to", "none"),
        dataset_num_proc=4,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training["output_dir"])
    tokenizer.save_pretrained(training["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA/QLoRA SFT for Qwen3.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    run(config)


if __name__ == "__main__":
    main()
