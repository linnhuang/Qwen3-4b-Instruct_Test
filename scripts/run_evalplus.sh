#!/usr/bin/env bash
set -euo pipefail

config="${1:?Usage: bash scripts/run_evalplus.sh CONFIG_YAML}"
python -m qwen3_post_training.evaluation.generate_evalplus --config "${config}"

dataset="$(python - "${config}" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config["data"]["dataset"])
PY
)"
samples="$(python - "${config}" <<'PY'
import sys
from pathlib import Path
import yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config["runtime"]["output_path"])
PY
)"

python -m evalplus.evaluate \
  --dataset "${dataset}" \
  --samples "${samples}" \
  --parallel 8
