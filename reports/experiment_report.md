# Experiment report

## Environment

- GPU: NVIDIA H100 80GB HBM3
- Python: 3.11
- Base model: `Qwen/Qwen3-4B-Instruct-2507`
- Dataset revisions and package versions: fill from the actual run metadata.

## Results

| Checkpoint | GSM8K exact match | MBPP+ pass@1 | HumanEval+ pass@1 |
|---|---:|---:|---:|
| Base | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD |
| GRPO | TBD | TBD | TBD |

## Notes

- Do not tune against benchmark test sets.
- MBPP+ and HumanEval+ hidden tests must never be used as training rewards.
- Record parser failures, code timeouts, generation settings, runtime, and peak GPU memory.
