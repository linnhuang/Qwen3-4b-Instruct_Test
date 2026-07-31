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

原实验是 `SFT → GRPO`，配置通过 `init_mode: adapter` 显式加载
`checkpoints/sft_gsm8k`。新建的纯 GRPO 实验完全独立：

```bash
bash scripts/run_grpo_from_base.sh
```

它使用 `configs/grpo_gsm8k_from_base.yaml`，通过 `init_mode: base` 从原始 Instruct
模型创建全新 LoRA，输出到 `checkpoints/grpo_gsm8k_from_base`，不会覆盖原有 SFT 或
SFT→GRPO 产物。训练入口会拒绝写入任何非空 output directory；如需重跑，应使用新的
实验名/目录，而不是删除已有正式实验。

reward 包括：

- 最终数值答案正确性：权重 `1.0`。
- 唯一 `Final answer:` 输出格式：权重 `0.1`。

## 统一复评

```bash
bash scripts/run_all_eval.sh
```

该脚本会评测 Base，并在 adapter 存在时评测 `sft_gsm8k` 和 `grpo_gsm8k`。

## Reasoning Vector（参数空间）

这里复现论文 *Reasoning Vectors: Transferring Chain-of-Thought Capabilities via Task
Arithmetic* 的定义，而不是 hidden-state steering vector：

$$
v_{reason}=\theta_{GRPO}-\theta_{SFT},\qquad
	heta_{enhanced}=\theta_{target}+\alpha v_{reason}.
$$

本项目的 donor 是 LoRA adapter。实现会先对每个模块精确重建
`ΔW = (lora_alpha / r) * B @ A`，再计算 `ΔW_GRPO - ΔW_SFT`；不能直接相减
LoRA 的 A/B 因子。

先验证 adapter 结构和 2 条 GSM8K：

```bash
pytest -q tests/test_reasoning_vectors.py
python -m qwen3_post_training.inference \
	--config configs/inference_reasoning_vector.yaml \
	--max-samples 2 \
	--output predictions/reasoning_vector_smoke.jsonl
python -m qwen3_post_training.evaluation.evaluate_gsm8k \
	--predictions predictions/reasoning_vector_smoke.jsonl
```

随后建议对 `alpha ∈ {-1, 0, 0.5, 1.0, 1.5}` 使用独立输出文件：

```bash
python -m qwen3_post_training.inference \
	--config configs/inference_reasoning_vector.yaml \
	--reasoning-vector-alpha 0 \
	--max-samples 20 \
	--output predictions/reasoning_vector_alpha0_smoke.jsonl
```

`alpha=-1` 对应论文的向量移除消融；`alpha=0` 必须与 Base 一致，是重要的实现 sanity check。

跨任务迁移配置 `configs/inference_mbpp_transfer_vector.yaml` 会先把 MBPP SFT adapter
安全合并到 dense target，再加入 GSM8K 的 Reasoning Vector。当前内置 runner 仍只评测 GSM8K；
要验证向代码域迁移，需接入 MBPP+/HumanEval+ evaluator 后复用同一加载路径。

严格限制：SFT 与 GRPO donor 必须来自同一 Base，且 LoRA rank、alpha、target modules 和
张量形状完全一致；目标模型也必须具有相同架构、参数命名、形状和 tokenizer。

### 纯 GRPO 向量

Base→GRPO 没有 SFT donor，因此其向量定义与论文的 `GRPO−SFT` 不同，应明确称为
**GRPO task vector**：

$$
v_{GRPO}=\theta_{GRPO-from-Base}-\theta_{Base}.
$$

对应配置为 `configs/inference_grpo_from_base_vector.yaml`。训练完成后可先运行：

```bash
python -m qwen3_post_training.inference \
	--config configs/inference_grpo_from_base_vector.yaml \
	--max-samples 20 \
	--output predictions/grpo_from_base_vector_smoke.jsonl
```

跨任务注入 MBPP target 的配置是
`configs/inference_mbpp_plus_grpo_from_base_vector.yaml`。两种向量定义必须分别报告：

- 原论文式：`GRPO-after-SFT − SFT`；
- 新对照式：`GRPO-from-Base − Base`。

不能把后者描述为复现论文的受控 Reasoning Vector，因为它没有利用 SFT subtraction
消除共享的任务知识。

### 纯 GRPO 减 SFT 的跨路径向量

另一个独立消融定义为：

$$
v_{pureGRPO-SFT}=\theta_{GRPO-from-Base}-\theta_{SFT}.
$$

配置为 `configs/inference_pure_grpo_minus_sft_vector.yaml`。它使用彼此独立训练的
`checkpoints/grpo_gsm8k_from_base` 和 `checkpoints/sft_gsm8k`，不同于论文式的
`GRPO-after-SFT − SFT`。因此应称为“跨路径差分向量”，并单独报告，不能与论文式
Reasoning Vector 混为一谈。

### SFT 退化说明

当前 GSM8K Base/SFT/SFT→GRPO 分别约为 `91.74% / 80.06% / 93.56%`。逐样本检查表明
SFT 下降是真实退化，不是 prompt、split 或 parser 差异。主要风险是对已经很强的 Instruct
模型使用全 attention+MLP、`r=32`、`1e-4`、2 epoch LoRA，强制模仿未经筛选的 GSM8K
rationale；teacher-forced validation loss 改善不等于生成式 exact match 改善。

不建议原参数原样重跑。若要修正 SFT，应建立全新实验目录并优先测试：assistant-only loss、
`r=8/16`、attention-only LoRA、学习率 `1e-5/2e-5`、`0.25/0.5` epoch，以及按独立 train
validation 的生成式 exact match 选择 checkpoint。原有 SFT 仍应保留，因为它与现有
SFT→GRPO 构成论文定义所需的受控 donor pair。

## 重要限制

- 当前内置批量推理和 scorer 先覆盖 GSM8K。
- `code_sandbox.py` 只是带超时的本地子进程，不是强安全边界；正式代码评测需使用容器或专用沙箱。
- 不得将 GSM8K test、MBPP+ 或 HumanEval+ 隐藏测试用于训练或 reward。
- 每次正式实验应记录模型/data revision、完整配置、依赖版本、seed、运行时间和峰值显存。
