import torch

from qwen3_post_training.reasoning_vectors import (
    apply_lora_task_vector,
    compute_lora_delta,
    normalize_module_name,
)


def test_compute_lora_delta_reconstructs_dense_update() -> None:
    lora_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    lora_b = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

    delta = compute_lora_delta(lora_a, lora_b, scaling=0.5)

    assert delta.shape == (3, 2)
    assert torch.equal(delta, (lora_b @ lora_a) * 0.5)


def test_normalize_peft_module_name() -> None:
    name = "base_model.model.model.layers.0.self_attn.q_proj"
    assert normalize_module_name(name) == "model.layers.0.self_attn.q_proj"


def test_apply_lora_task_vector_symbol_is_available() -> None:
    assert callable(apply_lora_task_vector)
