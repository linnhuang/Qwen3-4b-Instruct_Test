#!/usr/bin/env bash
set -euo pipefail

python -m qwen3_post_training.inference --config configs/inference.yaml
python -m qwen3_post_training.evaluation.evaluate_gsm8k \
  --predictions predictions/base_gsm8k.jsonl
