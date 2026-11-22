"""Public model entrypoints: the VideoMAE VA regressor and its linear head."""

from stern.models.videomae.regressor import (
    LinearVAHead,
    VideoMAEVARegressor,
)

__all__ = ["VideoMAEVARegressor", "LinearVAHead"]
