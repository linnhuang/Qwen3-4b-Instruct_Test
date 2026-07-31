from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


@dataclass(frozen=True)
class LoraMetadata:
    rank: int
    alpha: float
    use_rslora: bool
    target_modules: frozenset[str]

    @property
    def scaling(self) -> float:
        denominator = self.rank**0.5 if self.use_rslora else self.rank
        return self.alpha / denominator


def load_lora_metadata(adapter_path: str | Path) -> LoraMetadata:
    config_path = Path(adapter_path) / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise ValueError(f"Only LoRA adapters are supported: {config_path}")
    if config.get("fan_in_fan_out", False):
        raise ValueError("fan_in_fan_out=True adapters are not supported by this experiment.")
    rank_pattern = config.get("rank_pattern") or {}
    alpha_pattern = config.get("alpha_pattern") or {}
    if rank_pattern or alpha_pattern:
        raise ValueError("Per-module rank/alpha patterns are not supported yet.")
    return LoraMetadata(
        rank=int(config["r"]),
        alpha=float(config["lora_alpha"]),
        use_rslora=bool(config.get("use_rslora", False)),
        target_modules=frozenset(config["target_modules"]),
    )


def adapter_weight_path(adapter_path: str | Path) -> Path:
    path = Path(adapter_path) / "adapter_model.safetensors"
    if not path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {path}")
    return path


def lora_module_pairs(weight_path: str | Path) -> dict[str, dict[str, str]]:
    pairs: dict[str, dict[str, str]] = {}
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        # ``safe_open`` exposes ``keys()`` but is not itself iterable. Store the returned view
        # first to avoid Ruff's mapping-specific SIM118 rewrite, which is invalid for this API.
        tensor_keys = handle.keys()
        for key in tensor_keys:
            for kind in ("A", "B"):
                marker = f".lora_{kind}.weight"
                if key.endswith(marker):
                    module_name = normalize_module_name(key[: -len(marker)])
                    pairs.setdefault(module_name, {})[kind] = key
                    break
    incomplete = {name: pair for name, pair in pairs.items() if set(pair) != {"A", "B"}}
    if incomplete:
        raise ValueError(f"Incomplete LoRA A/B pairs: {sorted(incomplete)}")
    return pairs


def normalize_module_name(name: str) -> str:
    prefixes = ("base_model.model.", "base_model.")
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def compute_lora_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    return torch.matmul(lora_b.float(), lora_a.float()) * scaling


def validate_donor_compatibility(
    sft_adapter_path: str | Path,
    grpo_adapter_path: str | Path,
) -> tuple[LoraMetadata, LoraMetadata]:
    sft = load_lora_metadata(sft_adapter_path)
    grpo = load_lora_metadata(grpo_adapter_path)
    if sft != grpo:
        raise ValueError(f"Donor LoRA configurations differ: SFT={sft}, GRPO={grpo}")
    sft_pairs = lora_module_pairs(adapter_weight_path(sft_adapter_path))
    grpo_pairs = lora_module_pairs(adapter_weight_path(grpo_adapter_path))
    if sft_pairs.keys() != grpo_pairs.keys():
        only_sft = sorted(sft_pairs.keys() - grpo_pairs.keys())
        only_grpo = sorted(grpo_pairs.keys() - sft_pairs.keys())
        raise ValueError(f"Donor module sets differ; only SFT={only_sft}, only GRPO={only_grpo}")
    return sft, grpo


@torch.no_grad()
def apply_lora_task_vector(
    model: torch.nn.Module,
    adapter_path: str | Path,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Apply θ_target + alpha * (θ_adapter - θ_base) to a compatible dense model."""
    metadata = load_lora_metadata(adapter_path)
    weight_path = adapter_weight_path(adapter_path)
    pairs = lora_module_pairs(weight_path)
    modules = dict(model.named_modules())
    applied = 0
    vector_norm_sq = 0.0
    missing_modules: list[str] = []

    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for module_name, pair in pairs.items():
            module = modules.get(module_name)
            if module is None or not hasattr(module, "weight"):
                missing_modules.append(module_name)
                continue
            delta = compute_lora_delta(
                handle.get_tensor(pair["A"]),
                handle.get_tensor(pair["B"]),
                metadata.scaling,
            )
            weight = module.weight
            if delta.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for {module_name}: vector={delta.shape}, target={weight.shape}"
                )
            vector_norm_sq += delta.square().sum().item()
            weight.add_(delta.to(device=weight.device, dtype=weight.dtype), alpha=alpha)
            applied += 1

    if missing_modules:
        raise ValueError(f"Target model is missing donor modules: {missing_modules[:10]}")
    return {
        "definition": "theta_target + alpha * (theta_adapter - theta_base)",
        "alpha": alpha,
        "adapter": str(adapter_path),
        "applied_modules": applied,
        "vector_l2_norm": vector_norm_sq**0.5,
    }


@torch.no_grad()
def apply_reasoning_vector(
    model: torch.nn.Module,
    sft_adapter_path: str | Path,
    grpo_adapter_path: str | Path,
    alpha: float = 1.0,
) -> dict[str, Any]:
    """Apply θ_target + alpha * (θ_GRPO - θ_SFT) to a compatible dense model.

    The donors are LoRA adapters. Their exact dense updates are reconstructed as
    ``scale * B @ A`` before subtraction; directly subtracting LoRA A/B factors would be wrong.
    The target model must be dense (merge any target adapter before calling this function).
    """
    sft_meta, grpo_meta = validate_donor_compatibility(sft_adapter_path, grpo_adapter_path)
    sft_weights = adapter_weight_path(sft_adapter_path)
    grpo_weights = adapter_weight_path(grpo_adapter_path)
    sft_pairs = lora_module_pairs(sft_weights)
    grpo_pairs = lora_module_pairs(grpo_weights)
    modules = dict(model.named_modules())

    applied = 0
    vector_norm_sq = 0.0
    missing_modules: list[str] = []
    with safe_open(sft_weights, framework="pt", device="cpu") as sft_handle, safe_open(
        grpo_weights, framework="pt", device="cpu"
    ) as grpo_handle:
        for module_name, sft_pair in sft_pairs.items():
            module = modules.get(module_name)
            if module is None or not hasattr(module, "weight"):
                missing_modules.append(module_name)
                continue
            grpo_pair = grpo_pairs[module_name]
            sft_delta = compute_lora_delta(
                sft_handle.get_tensor(sft_pair["A"]),
                sft_handle.get_tensor(sft_pair["B"]),
                sft_meta.scaling,
            )
            grpo_delta = compute_lora_delta(
                grpo_handle.get_tensor(grpo_pair["A"]),
                grpo_handle.get_tensor(grpo_pair["B"]),
                grpo_meta.scaling,
            )
            reasoning_delta = grpo_delta - sft_delta
            weight = module.weight
            if reasoning_delta.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for {module_name}: vector={reasoning_delta.shape}, "
                    f"target={weight.shape}"
                )
            vector_norm_sq += reasoning_delta.square().sum().item()
            weight.add_(reasoning_delta.to(device=weight.device, dtype=weight.dtype), alpha=alpha)
            applied += 1

    if missing_modules:
        raise ValueError(f"Target model is missing donor modules: {missing_modules[:10]}")
    return {
        "definition": "theta_target + alpha * (theta_grpo - theta_sft)",
        "alpha": alpha,
        "sft_adapter": str(sft_adapter_path),
        "grpo_adapter": str(grpo_adapter_path),
        "applied_modules": applied,
        "vector_l2_norm": vector_norm_sq**0.5,
    }


def merge_target_adapter(model):
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload(safe_merge=True)
    return model
