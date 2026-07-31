#!/usr/bin/env bash
set -euo pipefail

output="predictions/reasoning_vector_smoke.jsonl"
rm -f "${output}" "${output%.jsonl}.summary.json"

python -m qwen3_post_training.inference \
  --config configs/inference_reasoning_vector.yaml \
  --max-samples 2 \
  --output "${output}"

python -m qwen3_post_training.evaluation.evaluate_gsm8k \
  --predictions "${output}"