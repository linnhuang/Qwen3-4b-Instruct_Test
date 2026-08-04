#!/usr/bin/env python3
"""Retroactively upload training metrics to Weights & Biases.

Reads from trainer_state.json if available, falls back to parsing log files.
Supports SFT (loss, token_accuracy, lr, eval) and GRPO (loss, reward, KL, lr).

Usage:
    python scripts/upload_to_wandb.py
    python scripts/upload_to_wandb.py --project my-project --entity my-entity
    python scripts/upload_to_wandb.py --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("Please install wandb: pip install wandb")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent

# ── experiment definitions ──────────────────────────────────────────────
EXPERIMENTS = {
    "sft_gsm8k": {
        "checkpoint": ROOT / "checkpoints" / "sft_gsm8k" / "checkpoint-450",
        "log": ROOT / "logs" / "sft_gsm8k.log",
        "type": "SFT",
        "dataset": "GSM8K",
        "notes": "LoRA r=32, lr=1e-4, 2 epochs",
    },
    "sft_mbpp_full": {
        "checkpoint": ROOT / "checkpoints" / "sft_mbpp_full" / "checkpoint-15",
        "log": ROOT / "logs" / "sft_mbpp_full.log",
        "type": "SFT",
        "dataset": "MBPP",
        "notes": "LoRA r=32, lr=1e-4, 5 epochs",
    },
    "grpo_gsm8k": {
        "checkpoint": ROOT / "checkpoints" / "grpo_gsm8k" / "checkpoint-1868",
        "log": ROOT / "logs" / "grpo_gsm8k.log",
        "type": "GRPO",
        "dataset": "GSM8K",
        "notes": "From SFT adapter, beta=0.01, lr=1e-6",
    },
    "grpo_gsm8k_from_base": {
        "checkpoint": ROOT / "checkpoints" / "grpo_gsm8k_from_base" / "checkpoint-1800",
        "log": ROOT / "logs" / "grpo_gsm8k_from_base.log",
        "type": "GRPO",
        "dataset": "GSM8K",
        "notes": "From base model, beta=0.01, lr=1e-6",
    },
}

# ── data source helpers ─────────────────────────────────────────────────

def from_trainer_state(checkpoint_dir: Path) -> list[dict] | None:
    """Read log_history from trainer_state.json. Returns None if missing."""
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    with open(state_path) as f:
        return json.load(f)["log_history"]


def from_log_file(log_path: Path) -> list[dict] | None:
    """Parse loss/metrics dicts from a training log file. Returns None if missing."""
    if not log_path.exists():
        return None

    with open(log_path) as f:
        text = f.read()

    # Match dict literals on their own line, e.g. {'loss': 0.5962, ...}
    # Use ast.literal_eval for robust parsing
    pattern = re.compile(r"^\{'loss':.+", re.MULTILINE)
    entries = []
    for match in pattern.finditer(text):
        try:
            entry = ast.literal_eval(match.group())
            entries.append(entry)
        except (ValueError, SyntaxError):
            continue
    return entries if entries else None


def load_history(exp_info: dict) -> list[dict]:
    """Try trainer_state.json first, fall back to log file."""
    history = from_trainer_state(exp_info["checkpoint"])
    if history:
        print(f"   📁 Source: trainer_state.json  ({len(history)} entries)")
        return history

    history = from_log_file(exp_info["log"])
    if history:
        print(f"   📄 Source: log file  ({len(history)} entries)")
        return history

    raise FileNotFoundError(
        f"No data found for checkpoint={exp_info['checkpoint']} or log={exp_info['log']}"
    )


# ── filtering helpers ───────────────────────────────────────────────────

def is_train(entry: dict) -> bool:
    return "loss" in entry and "eval_loss" not in entry


def is_eval(entry: dict) -> bool:
    return "eval_loss" in entry


# ── main ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Upload training logs to wandb")
    parser.add_argument("--project", default="qwen3-post-training")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print without uploading")
    args = parser.parse_args()

    if args.dry_run:
        wandb.init(mode="disabled")
    else:
        wandb.login()

    for exp_name, exp_info in EXPERIMENTS.items():
        log_path = exp_info.get("log")

        try:
            history = load_history(exp_info)
        except FileNotFoundError as e:
            print(f"⚠  Skipping {exp_name}: {e}")
            continue

        run = wandb.init(
            project=args.project,
            entity=args.entity,
            name=exp_name,
            config={
                "type": exp_info["type"],
                "dataset": exp_info["dataset"],
                "notes": exp_info["notes"],
            },
            reinit=True,
        )

        train_entries = [e for e in history if is_train(e)]
        eval_entries = [e for e in history if is_eval(e)]
        exp_type = exp_info["type"]

        for entry in train_entries:
            step = entry.get("step", 0)
            metrics = {}

            if exp_type == "SFT":
                metrics.update({
                    "train/loss": entry.get("loss"),
                    "train/token_accuracy": entry.get("mean_token_accuracy"),
                    "train/grad_norm": entry.get("grad_norm"),
                    "train/learning_rate": entry.get("learning_rate"),
                })
            elif exp_type == "GRPO":
                metrics.update({
                    "train/loss": entry.get("loss"),
                    "train/kl": entry.get("kl"),
                    "train/reward": entry.get("reward"),
                    "train/reward_std": entry.get("reward_std"),
                    "train/grad_norm": entry.get("grad_norm"),
                    "train/learning_rate": entry.get("learning_rate"),
                })
                correctness = entry.get("rewards/correctness_reward/mean")
                format_r = entry.get("rewards/format_reward/mean")
                if correctness is not None:
                    metrics["train/correctness_reward"] = correctness
                if format_r is not None:
                    metrics["train/format_reward"] = format_r

            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=step)

        for entry in eval_entries:
            step = entry.get("step", 0)
            metrics = {
                "eval/loss": entry.get("eval_loss"),
                "eval/token_accuracy": entry.get("eval_mean_token_accuracy"),
            }
            metrics = {k: v for k, v in metrics.items() if v is not None}
            if metrics:
                wandb.log(metrics, step=step)

        # ── attach final benchmark results as run summary ────────────
        _attach_benchmark_summary(exp_name)

        steps_str = f"{len(train_entries)} train"
        if eval_entries:
            steps_str += f" + {len(eval_entries)} eval"
        print(f"✅ {exp_name}: {steps_str} steps uploaded")
        run.finish()

    print("\nDone! All experiments uploaded to wandb.")


def _attach_benchmark_summary(exp_name: str) -> None:
    """Look up final benchmark accuracy and attach to wandb summary."""
    summary_map = {
        "sft_gsm8k":        "sft_gsm8k_gsm8k.summary.json",
        "grpo_gsm8k":        "grpo_gsm8k_gsm8k.summary.json",
        "grpo_gsm8k_from_base": "grpo_gsm8k_from_base_gsm8k.summary.json",
        # sft_mbpp_full has evalplus results instead
    }
    filename = summary_map.get(exp_name)
    if not filename:
        return
    summary_path = ROOT / "predictions" / filename
    if summary_path.exists():
        with open(summary_path) as f:
            data = json.load(f)
        for key in ["accuracy", "correct", "incorrect", "total", "parse_failures"]:
            if key in data:
                wandb.summary[f"gsm8k_{key}"] = data[key]


if __name__ == "__main__":
    main()

