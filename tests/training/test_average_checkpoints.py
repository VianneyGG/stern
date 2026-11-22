import pytest
import torch

from stern.training.average_checkpoints import average_state_dicts


def test_float_tensors_are_the_elementwise_mean_in_their_own_dtype():
    a = {"w": torch.tensor([1.0, 2.0], dtype=torch.float16), "b": torch.tensor([0.0])}
    b = {"w": torch.tensor([3.0, 6.0], dtype=torch.float16), "b": torch.tensor([1.0])}
    c = {"w": torch.tensor([5.0, 1.0], dtype=torch.float16), "b": torch.tensor([2.0])}

    avg = average_state_dicts([a, b, c])

    assert avg["w"].dtype == torch.float16
    torch.testing.assert_close(avg["w"], torch.tensor([3.0, 3.0], dtype=torch.float16))
    torch.testing.assert_close(avg["b"], torch.tensor([1.0]))


def test_fp16_average_is_accumulated_in_fp32():
    # 60000 + 60000 overflows fp16 (max 65504); an fp16 running sum would give inf.
    big = torch.tensor([60000.0], dtype=torch.float16)
    avg = average_state_dicts([{"w": big}, {"w": big}])
    assert torch.isfinite(avg["w"]).all()
    assert avg["w"].item() == 60000.0


def test_non_float_tensors_come_from_the_last_snapshot():
    sds = [{"n": torch.tensor(3)}, {"n": torch.tensor(7)}]
    avg = average_state_dicts(sds)
    assert avg["n"].dtype == torch.int64 and avg["n"].item() == 7


def test_mismatched_keys_raise():
    with pytest.raises(ValueError, match="different keys"):
        average_state_dicts([{"a": torch.zeros(1)}, {"b": torch.zeros(1)}])


def test_inputs_are_not_modified():
    a = {"w": torch.tensor([1.0])}
    b = {"w": torch.tensor([3.0])}
    average_state_dicts([a, b])
    assert a["w"].item() == 1.0 and b["w"].item() == 3.0
