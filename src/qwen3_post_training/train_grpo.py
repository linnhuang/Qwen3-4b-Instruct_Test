from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from qwen3_post_training.common import load_yaml, resolve_dtype, set_seed
from qwen3_post_training.data.gsm8k import load_gsm8k
from qwen3_post_training.data.mbpp import load_mbpp
from qwen3_post_training.rewards.code_reward import public_test_reward, syntax_reward
from qwen3_post_training.rewards.math_reward import build_weighted_rewards


def prepare_dataset(config: dict[str, Any]):
    data = config["data"]
    if data["dataset"] == "gsm8k":
        dataset = load_gsm8k(
            split=data.get("split", "train"),
            subset=data.get("subset", "main"),
            revision=data.get("revision", "main"),
            max_samples=data.get("max_train_samples"),
        )
    elif data["dataset"] == "mbpp":
        dataset = load_mbpp(
            split=data.get("split", "train"),
            revision=data.get("revision", "main"),
            max_samples=data.get("max_train_samples"),
        )
    else:
        raise ValueError(f"Unsupported GRPO dataset: {data['dataset']}")
    return dataset.rename_column("messages", "prompt")


def build_reward_functions(config: dict[str, Any]):
    reward = config["reward"]
    if config["data"]["dataset"] == "gsm8k":
        return build_weighted_rewards(
            correctness_weight=reward.get("correctness_weight", 1.0),
            format_weight=reward.get("format_weight", 0.1),
        )

    syntax_weight = float(reward.get("syntax_weight", 0.1))
    public_test_weight = float(reward.get("public_test_weight", 1.0))
    timeout_seconds = float(reward.get("timeout_seconds", 5.0))

    def weighted_syntax(completions, **kwargs):
        return [syntax_weight * value for value in syntax_reward(completions, **kwargs)]

    def weighted_public_tests(completions, test_list, **kwargs):
        return [
            public_test_weight * value
            for value in public_test_reward(
                completions,
                test_list,
                timeout_seconds=timeout_seconds,
                **kwargs,
            )
        ]

    weighted_syntax.__name__ = "syntax_reward"
    weighted_public_tests.__name__ = "public_test_reward"
    return [weighted_public_tests, weighted_syntax]


def build_model_and_peft(config: dict[str, Any]):
    model_config = config["model"]
    init_mode = model_config.get("init_mode")
    adapter_path = model_config.get("init_adapter_path")
    if init_mode not in {"base", "adapter"}:
        raise ValueError("model.init_mode must be explicitly set to 'base' or 'adapter'.")
    if init_mode == "base" and adapter_path:
        raise ValueError("model.init_adapter_path must be null when init_mode='base'.")
    if init_mode == "adapter" and not adapter_path:
        raise ValueError("model.init_adapter_path is required when init_mode='adapter'.")
    if init_mode == "adapter" and not Path(adapter_path).is_dir():
        raise FileNotFoundError(f"Initial adapter directory does not exist: {adapter_path}")

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision", "main"),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
    )
    model.config.use_cache = False
    if init_mode == "adapter":
        print(f"GRPO initialization: adapter={adapter_path}")
        return PeftModel.from_pretrained(model, adapter_path, is_trainable=True), None

    print(f"GRPO initialization: base={model_config['name_or_path']}")
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
    output_dir = Path(training["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty GRPO output directory: {output_dir}. "
            "Choose a new output_dir for a new experiment."
        )
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

    reward_functions = build_reward_functions(config)
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
        epsilon=training.get("epsilon", 0.2),
        temperature=training.get("temperature", 1.0),
        top_p=training.get("top_p", 1.0),
        logging_steps=training["logging_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training.get("save_total_limit", 3),
        bf16=config["model"].get("dtype") == "bfloat16",
        gradient_checkpointing=True,
        seed=training.get("seed", 42),
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
