"""LoRA adapters on the frozen encoder's attention projection matrices
(docs/superpowers/specs/2026-07-01-training-pipeline-design.md §4.4). Never
full-finetune the ViT -- catastrophic overfit on available label volume.
"""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, get_peft_model


def attach_lora(encoder: Any, rank: int = 8, target_modules: list[str] | None = None):
    """Attach LoRA adapters to `target_modules` on `encoder` via peft, return the wrapped model."""
    if target_modules is None:
        target_modules = ["query", "value"]
    config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )
    return get_peft_model(encoder, config)
