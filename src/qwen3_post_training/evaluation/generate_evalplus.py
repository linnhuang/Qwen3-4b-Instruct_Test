from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from evalplus.data import get_human_eval_plus, get_mbpp_plus

from qwen3_post_training.common import append_jsonl, load_yaml, read_jsonl, set_seed
from qwen3_post_training.data.formatting import CODE_SYSTEM_PROMPT
from qwen3_post_training.evaluation.answer_parser import extract_python_code
from qwen3_post_training.inference import load_model_and_tokenizer


def load_evalplus_tasks(dataset: str) -> list[tuple[str, dict[str, Any]]]:
    if dataset == "mbpp":
        tasks = get_mbpp_plus()
    elif dataset == "humaneval":
        tasks = get_human_eval_plus()
    else:
        raise ValueError("dataset must be 'mbpp' or 'humaneval'.")
    return list(tasks.items())


def task_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CODE_SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"].strip()},
    ]


@torch.inference_mode()
def generate_batch(model, tokenizer, messages_batch, generation: dict[str, Any]):
    prompts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    generation_args = {
        "max_new_tokens": generation.get("max_new_tokens", 512),
        "do_sample": generation.get("do_sample", False),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if generation_args["do_sample"]:
        generation_args["temperature"] = generation.get("temperature", 0.5)
        generation_args["top_p"] = generation.get("top_p", 0.95)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    output_ids = model.generate(**encoded, **generation_args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    prompt_width = encoded["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        output_ids[:, prompt_width:], skip_special_tokens=True
    )
    return completions, elapsed


def run(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    set_seed(runtime.get("seed", 42))
    tasks = load_evalplus_tasks(config["data"]["dataset"])
    max_samples = config["data"].get("max_samples")
    if max_samples is not None:
        tasks = tasks[:max_samples]

    model, tokenizer, vector_metadata = load_model_and_tokenizer(config)
    output_path = Path(runtime["output_path"])
    existing = read_jsonl(output_path) if runtime.get("resume", True) else []
    completed_ids = {row["task_id"] for row in existing}
    pending = [(task_id, task) for task_id, task in tasks if task_id not in completed_ids]
    batch_size = config["generation"].get("batch_size", 8)

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        completions, elapsed = generate_batch(
            model,
            tokenizer,
            [task_messages(task) for _, task in batch],
            config["generation"],
        )
        records = []
        for (task_id, _task), completion in zip(batch, completions, strict=True):
            records.append(
                {
                    "task_id": task_id,
                    "solution": extract_python_code(completion),
                    "raw_completion": completion,
                    "model": config["model"]["name_or_path"],
                    "adapter": config["model"].get("adapter_path"),
                    "reasoning_vector": vector_metadata,
                    "generation": config["generation"],
                    "seed": runtime.get("seed", 42),
                    "latency_seconds": elapsed / len(batch),
                }
            )
        append_jsonl(output_path, records)
        print(f"Generated {min(offset + len(batch), len(pending))}/{len(pending)} pending tasks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EvalPlus MBPP+/HumanEval+ solutions.")
    parser.add_argument("--config", required=True)
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
            raise ValueError("Alpha override requires a configured parameter vector.")
        active_config["alpha"] = args.reasoning_vector_alpha
    run(config)


if __name__ == "__main__":
    main()
