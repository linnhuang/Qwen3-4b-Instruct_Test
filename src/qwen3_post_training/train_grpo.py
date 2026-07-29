from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from qwen3_post_training.common import load_yaml, resolve_dtype, set_seed
from qwen3_post_training.data.gsm8k import load_gsm8k
from qwen3_post_training.rewards.math_reward import build_weighted_rewards


def prepare_dataset(config: dict[str, Any]):
    data = config["data"]
    if data["dataset"] != "gsm8k":
        raise ValueError("The initial GRPO implementation supports GSM8K only.")
    dataset = load_gsm8k(
        split=data.get("split", "train"),
        subset=data.get("subset", "main"),
        revision=data.get("revision", "main"),
        max_samples=data.get("max_train_samples"),
    )
    return dataset.rename_column("messages", "prompt")


def build_model_and_peft(config: dict[str, Any]):
    model_config = config["model"]
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision", "main"),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    adapter_path = model_config.get("init_adapter_path")
    if adapter_path and Path(adapter_path).exists():
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True), None

    lora = config["lora"]
    peft_config = LoraConfig(
        r=lora["r"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora.get("dropout", 0.0),
        target_modules=lora["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, peft_config


def run(config: dict[str, Any]) -> None:
    training = config["training"]
    set_seed(training.get("seed", 42))
    dataset = prepare_dataset(config)
    model, peft_config = build_model_and_peft(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name_or_path"],
        revision=config["model"].get("revision", "main"),
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reward = config["reward"]
    reward_functions = build_weighted_rewards(
        correctness_weight=reward.get("correctness_weight", 1.0),
        format_weight=reward.get("format_weight", 0.1),
    )
    args = GRPOConfig(
        output_dir=training["output_dir"],
        num_train_epochs=training["num_train_epochs"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_generations=training["num_generations"],
        max_prompt_length=training["max_prompt_length"],
        max_completion_length=training["max_completion_length"],
        beta=training["beta"],
        logging_steps=training["logging_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        bf16=config["model"].get("dtype") == "bfloat16",
        gradient_checkpointing=True,
        seed=training["seed"],
        report_to=training.get("report_to", "none"),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(training["output_dir"])
    tokenizer.save_pretrained(training["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for verifiable GSM8K rewards.")
    parser.add_argument("--config", default="configs/grpo_gsm8k.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.max_train_samples is not None:
        config["data"]["max_train_samples"] = args.max_train_samples
    run(config)


if __name__ == "__main__":
    main()
