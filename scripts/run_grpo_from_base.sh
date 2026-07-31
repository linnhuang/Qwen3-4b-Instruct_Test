#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results
cp configs/grpo_gsm8k_from_base.yaml results/grpo_gsm8k_from_base.config.yaml
python -m qwen3_post_training.train_grpo \
  --config configs/grpo_gsm8k_from_base.yaml \
  2>&1 | tee logs/grpo_gsm8k_from_base.log