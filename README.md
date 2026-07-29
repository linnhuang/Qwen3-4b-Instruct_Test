# Qwen3-4B Post-training Experiments

在固定评测协议下比较 Qwen3-4B-Instruct 的 Base inference、LoRA/QLoRA SFT 和
GSM8K GRPO。当前核心闭环为：

- GSM8K：Base → SFT → GRPO，指标为最终答案 exact match。
- MBPP：用于代码 SFT。
- MBPP+、HumanEval+：用于后续 EvalPlus 泛化评测，不参与训练或 reward。

## 当前环境

已检测到 NVIDIA H100 80GB HBM3。默认配置使用 BF16 LoRA；如迁移到较小显存 GPU，
可在 SFT 配置中开启 `use_4bit` 并安装 `bitsandbytes`。

## 安装

```bash
cd /home/huangying/CTIC/Qwen3-4b-Instruct_Test
pip install -e '.[dev,code-eval]'
```

如果计算节点无法访问 PyPI，请先在可联网节点下载 wheel/创建 Conda 环境，再在计算节点使用
本地 wheelhouse 安装。仅安装本项目源码可使用 `pip install --no-build-isolation --no-deps -e .`，
但训练前仍必须确保 PyTorch、Transformers、Datasets、TRL、PEFT 和 Accelerate 已安装。

默认模型为 `Qwen/Qwen3-4B-Instruct-2507`。首次运行需要能够访问模型和 Hugging Face
数据集，或提前将它们放入本地缓存。

## 快速验证

```bash
pytest
bash scripts/run_smoke_test.sh
```

## Base inference 和评测

```bash
bash scripts/run_base_eval.sh
```

预测写入 `predictions/base_gsm8k.jsonl`，汇总写入对应的 `.summary.json`。推理支持根据样本
ID 断点续跑。修改模型、batch size 和生成参数时应新建输出文件，避免混合实验结果。

默认 `device_map: cuda:0` 明确使用单张 GPU，避免容器注入 `WORLD_SIZE=1`、`RANK=0` 但没有
`LOCAL_RANK` 时，Transformers 将 `device_map: auto` 误判为 torchrun 张量并行。

## SFT

GSM8K：

```bash
bash scripts/run_sft.sh configs/sft_gsm8k.yaml
```

MBPP：

```bash
bash scripts/run_sft.sh configs/sft_mbpp.yaml
```

建议正式训练前直接运行 Python 入口并增加 `--max-train-samples 100` 进行小样本 smoke run。

## GRPO

`configs/grpo_gsm8k.yaml` 默认从 `checkpoints/sft_gsm8k` adapter 继续训练；如果该目录不存在，
则从基础模型创建新的 LoRA adapter。

```bash
bash scripts/run_grpo.sh
```

reward 包括：

- 最终数值答案正确性：权重 `1.0`。
- 唯一 `Final answer:` 输出格式：权重 `0.1`。

## 统一复评

```bash
bash scripts/run_all_eval.sh
```

该脚本会评测 Base，并在 adapter 存在时评测 `sft_gsm8k` 和 `grpo_gsm8k`。

## 重要限制

- 当前内置批量推理和 scorer 先覆盖 GSM8K。
- `code_sandbox.py` 只是带超时的本地子进程，不是强安全边界；正式代码评测需使用容器或专用沙箱。
- 不得将 GSM8K test、MBPP+ 或 HumanEval+ 隐藏测试用于训练或 reward。
- 每次正式实验应记录模型/data revision、完整配置、依赖版本、seed、运行时间和峰值显存。
