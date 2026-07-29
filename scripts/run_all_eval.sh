#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_base_eval.sh

for adapter in checkpoints/sft_gsm8k checkpoints/grpo_gsm8k; do
  if [[ -d "${adapter}" ]]; then
    name="$(basename "${adapter}")"
    python - "${adapter}" "${name}" <<'PY'
import sys
from pathlib import Path
import yaml

adapter, name = sys.argv[1:]
config = yaml.safe_load(Path("configs/inference.yaml").read_text())
config["model"]["adapter_path"] = adapter
config["runtime"]["output_path"] = f"predictions/{name}_gsm8k.jsonl"
Path(f"configs/.inference_{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
PY
    python -m qwen3_post_training.inference --config "configs/.inference_${name}.yaml"
    python -m qwen3_post_training.evaluation.evaluate_gsm8k \
      --predictions "predictions/${name}_gsm8k.jsonl"
    rm -f "configs/.inference_${name}.yaml"
  fi
done
