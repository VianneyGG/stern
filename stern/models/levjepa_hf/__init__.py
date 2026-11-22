"""Released LeVJEPA ViT-L from HuggingFace, wrapped as a VA regressor."""

from stern.models.levjepa_hf.regressor import (
    HFLeVJEPAVARegressor,
    load_hf_levjepa,
)

__all__ = ["HFLeVJEPAVARegressor", "load_hf_levjepa"]
