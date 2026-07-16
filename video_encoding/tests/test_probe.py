import torch

from eval.ccc import ccc_loss
from video_encoding.models.videomaev2.regressor import LinearVAHead


def test_linear_va_head_output_shape():
    head = LinearVAHead(embed_dim=4)
    x = torch.randn(5, 4)

    out = head(x)

    assert out.shape == (5, 2)


def test_linear_va_head_output_bounded_by_tanh():
    head = LinearVAHead(embed_dim=4)
    x = torch.randn(20, 4) * 1000  # large magnitude to stress the bound

    out = head(x)

    assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


def test_ccc_loss_near_minimum_for_perfect_prediction():
    torch.manual_seed(0)
    target = torch.randn(50, 2)
    pred = target.clone()

    loss = ccc_loss(pred, target)

    assert loss.item() < -0.99  # -(CCC_v + CCC_a)/2 -> -1 when CCC=1 on both dims


def test_ccc_loss_decreases_with_gradient_descent():
    torch.manual_seed(0)
    embed_dim = 4
    head = LinearVAHead(embed_dim=embed_dim)
    x = torch.randn(30, embed_dim)
    target = torch.tanh(torch.randn(30, 2))
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)

    initial_loss = ccc_loss(head(x), target).item()
    for _ in range(50):
        optimizer.zero_grad()
        loss = ccc_loss(head(x), target)
        loss.backward()
        optimizer.step()
    final_loss = ccc_loss(head(x), target).item()

    assert final_loss < initial_loss
