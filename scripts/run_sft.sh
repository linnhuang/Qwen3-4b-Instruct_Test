#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/sft_gsm8k.yaml}"
python -m qwen3_post_training.train_sft --config "${config}"
