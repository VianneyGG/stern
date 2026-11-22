import torch
from omegaconf import OmegaConf

from stern.models.videomae.regressor import VideoMAEVARegressor

D = 24  # divisible by the head's 12 attention heads


class _Tokens(torch.nn.Module):
    """Backbone stub: returns whatever tokens were set, ignoring the pixels."""

    def __init__(self):
        super().__init__()
        self.tokens = None

    def forward(self, pixel_values):
        return self.tokens


def _model(**flags) -> tuple[VideoMAEVARegressor, _Tokens]:
    torch.manual_seed(0)
    backbone = _Tokens()
    cfg = OmegaConf.create({"embed_dim": D, "head_bounded": False, **flags})
    return VideoMAEVARegressor(cfg, backbone=backbone).train(), backbone


def _outputs_with_and_without_offset(model, backbone):
    g = torch.Generator().manual_seed(1)
    tokens = torch.randn(8, 10, D, generator=g)
    offset = 30.0 * torch.randn(D, generator=g)  # one vector shared by every token
    backbone.tokens = tokens
    plain = model(None)
    backbone.tokens = tokens + offset
    return plain, model(None)


def test_token_norm_makes_the_head_blind_to_a_shared_offset():
    model, backbone = _model(head_token_norm=True)
    plain, shifted = _outputs_with_and_without_offset(model, backbone)
    torch.testing.assert_close(plain, shifted, atol=1e-4, rtol=1e-4)


def test_without_token_norm_the_shared_offset_changes_the_output():
    model, backbone = _model()
    plain, shifted = _outputs_with_and_without_offset(model, backbone)
    assert not torch.allclose(plain, shifted, atol=1e-3)


def test_flag_off_keeps_the_checkpoint_layout():
    model, _ = _model()
    assert model.token_norm is None
    assert not any(k.startswith("token_norm.") for k in model.state_dict())
