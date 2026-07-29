#!/usr/bin/env bash
set -euo pipefail

python -m qwen3_post_training.inference \
  --config configs/inference.yaml \
  --max-samples 2 \
  --output predictions/smoke_gsm8k.jsonl

python -m qwen3_post_training.evaluation.evaluate_gsm8k \
  --predictions predictions/smoke_gsm8k.jsonl
