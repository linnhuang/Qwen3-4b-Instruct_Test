#!/usr/bin/env bash
set -euo pipefail

python -m qwen3_post_training.train_grpo --config configs/grpo_gsm8k.yaml
