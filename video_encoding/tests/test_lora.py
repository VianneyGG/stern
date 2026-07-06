import torch
from torch import nn

from video_encoding.encoders.lora import attach_lora


class _ToyAttention(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.query(x) + self.key(x) + self.value(x)


def test_attach_lora_freezes_base_params():
    model = _ToyAttention()

    wrapped = attach_lora(model, rank=4, target_modules=["query", "value"])

    base_param_names = {"query.weight", "query.bias", "key.weight", "key.bias", "value.weight", "value.bias"}
    for name, param in wrapped.named_parameters():
        if any(name.endswith(base) for base in base_param_names) and "lora_" not in name:
            assert not param.requires_grad, f"base param {name} should be frozen"


def test_attach_lora_makes_lora_params_trainable():
    model = _ToyAttention()

    wrapped = attach_lora(model, rank=4, target_modules=["query", "value"])

    lora_params = [p for name, p in wrapped.named_parameters() if "lora_" in name]
    assert len(lora_params) > 0
    assert all(p.requires_grad for p in lora_params)


def test_attach_lora_does_not_adapt_untargeted_modules():
    model = _ToyAttention()

    wrapped = attach_lora(model, rank=4, target_modules=["query", "value"])

    lora_module_names = [name for name, _ in wrapped.named_modules() if "lora_" in name]
    assert not any("key" in name for name in lora_module_names)


def test_attach_lora_preserves_forward_output_shape():
    model = _ToyAttention()
    x = torch.randn(2, 8)

    wrapped = attach_lora(model, rank=4, target_modules=["query", "value"])
    out = wrapped(x)

    assert out.shape == (2, 8)


def test_attach_lora_defaults_target_modules_to_query_and_value():
    model = _ToyAttention()

    wrapped = attach_lora(model, rank=4)

    lora_module_names = [name for name, _ in wrapped.named_modules() if "lora_" in name]
    assert any("query" in name for name in lora_module_names)
    assert any("value" in name for name in lora_module_names)
    assert not any("key" in name for name in lora_module_names)
