from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_post_training.common import (
    append_jsonl,
    load_yaml,
    read_jsonl,
    resolve_dtype,
    set_seed,
)
from qwen3_post_training.data.gsm8k import load_gsm8k


def load_eval_dataset(config: dict[str, Any]):
    data = config["data"]
    if data["dataset"] != "gsm8k":
        raise ValueError("The built-in inference runner currently supports dataset=gsm8k.")
    return load_gsm8k(
        split=data["split"],
        subset=data.get("subset", "main"),
        revision=data.get("revision", "main"),
        max_samples=data.get("max_samples"),
    )


def load_model_and_tokenizer(config: dict[str, Any]):
    model_config = config["model"]
    device_map = model_config.get("device_map", "cuda:0" if torch.cuda.is_available() else "cpu")
    if device_map == "cuda:0" and not torch.cuda.is_available():
        device_map = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision", "main"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name_or_path"],
        revision=model_config.get("revision", "main"),
        torch_dtype=resolve_dtype(model_config.get("dtype", "bfloat16")),
        attn_implementation=model_config.get("attn_implementation", "sdpa"),
        trust_remote_code=model_config.get("trust_remote_code", False),
        device_map=device_map,
    )
    adapter_path = model_config.get("adapter_path")
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    vector_config = model_config.get("reasoning_vector")
    vector_metadata = None
    if vector_config:
        from qwen3_post_training.reasoning_vectors import (
            apply_reasoning_vector,
            merge_target_adapter,
        )

        model = merge_target_adapter(model)
        vector_metadata = apply_reasoning_vector(
            model,
            sft_adapter_path=vector_config["sft_adapter_path"],
            grpo_adapter_path=vector_config["grpo_adapter_path"],
            alpha=float(vector_config.get("alpha", 1.0)),
        )
        print(f"Applied reasoning vector: {vector_metadata}")
    task_vector_config = model_config.get("lora_task_vector")
    if task_vector_config:
        if vector_config:
            raise ValueError("Configure only one of reasoning_vector and lora_task_vector.")
        from qwen3_post_training.reasoning_vectors import (
            apply_lora_task_vector,
            merge_target_adapter,
        )

        model = merge_target_adapter(model)
        vector_metadata = apply_lora_task_vector(
            model,
            adapter_path=task_vector_config["adapter_path"],
            alpha=float(task_vector_config.get("alpha", 1.0)),
        )
        print(f"Applied LoRA task vector: {vector_metadata}")
    model.eval()
    return model, tokenizer, vector_metadata


def render_prompts(tokenizer, messages_batch: list[list[dict[str, str]]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]


@torch.inference_mode()
def generate_batch(model, tokenizer, messages_batch, generation: dict[str, Any]):
    prompts = render_prompts(tokenizer, messages_batch)
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_args = {
        "max_new_tokens": generation.get("max_new_tokens", 1024),
        "do_sample": generation.get("do_sample", False),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_args["do_sample"]:
        generation_args["temperature"] = generation.get("temperature", 0.7)
        generation_args["top_p"] = generation.get("top_p", 0.8)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output_ids = model.generate(**encoded, **generation_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    prompt_width = encoded["input_ids"].shape[1]
    completion_ids = output_ids[:, prompt_width:]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    return prompts, completions, elapsed


def run(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    set_seed(runtime.get("seed", 42))
    dataset = load_eval_dataset(config)
    model, tokenizer, vector_metadata = load_model_and_tokenizer(config)

    output_path = Path(runtime["output_path"])
    existing = read_jsonl(output_path) if runtime.get("resume", True) else []
    completed_ids = {row["id"] for row in existing}
    pending = [row for row in dataset if row["id"] not in completed_ids]
    batch_size = config["generation"].get("batch_size", 8)

    for offset in range(0, len(pending), batch_size):
        rows = pending[offset : offset + batch_size]
        prompts, completions, elapsed = generate_batch(
            model, tokenizer, [row["messages"] for row in rows], config["generation"]
        )
        per_sample_latency = elapsed / len(rows)
        records = []
        for row, prompt, completion in zip(rows, prompts, completions, strict=True):
            records.append(
                {
                    "id": row["id"],
                    "dataset": config["data"]["dataset"],
                    "split": config["data"]["split"],
                    "model": config["model"]["name_or_path"],
                    "adapter": config["model"].get("adapter_path"),
                    "reasoning_vector": vector_metadata,
                    "prompt": prompt,
                    "completion": completion,
                    "answer": row["answer"],
                    "latency_seconds": per_sample_latency,
                    "generation": config["generation"],
                    "seed": runtime.get("seed", 42),
                }
            )
        append_jsonl(output_path, records)
        print(f"Generated {min(offset + len(rows), len(pending))}/{len(pending)} pending samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible Qwen3 benchmark inference.")
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--reasoning-vector-alpha", type=float, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    if args.output is not None:
        config["runtime"]["output_path"] = args.output
    if args.reasoning_vector_alpha is not None:
        vector_config = config["model"].get("reasoning_vector")
        task_vector_config = config["model"].get("lora_task_vector")
        active_config = vector_config or task_vector_config
        if active_config is None:
            raise ValueError(
                "--reasoning-vector-alpha requires model.reasoning_vector or "
                "model.lora_task_vector config."
            )
        active_config["alpha"] = args.reasoning_vector_alpha
    run(config)


if __name__ == "__main__":
    main()
