"""Unit tests for the assertion-strength scanner itself.

These fixtures encode the exact false positives the ad-hoc scanner hit
during the 2026-09 gap exploration (torch.allclose, torch.equal, bound
comparisons like `>`/`<`, np.testing.assert_allclose) so the detector
never regresses to flagging them again.
"""

from __future__ import annotations

from tests.meta.assertion_strength import Kind, scan_source


def _kinds(source: str) -> set[Kind]:
    return {f.kind for f in scan_source(source)}


def test_flags_a_test_with_no_assertion_at_all():
    source = "def test_it():\n    do_thing()\n"
    assert _kinds(source) == {Kind.NO_ASSERT}


def test_flags_bare_truthiness_on_a_name():
    source = "def test_it():\n    result = f()\n    assert result\n"
    assert _kinds(source) == {Kind.BARE_TRUTHY}


def test_flags_bare_truthiness_on_an_attribute():
    source = "def test_it():\n    assert mock_open.called\n"
    assert _kinds(source) == {Kind.BARE_TRUTHY}


def test_flags_raises_without_match():
    source = "def test_it():\n    with pytest.raises(ValueError):\n        f()\n"
    assert _kinds(source) == {Kind.RAISES_NO_MATCH}


def test_does_not_flag_raises_with_match():
    source = (
        "def test_it():\n"
        "    with pytest.raises(ValueError, match='bad shape'):\n"
        "        f()\n"
    )
    assert _kinds(source) == set()


def test_does_not_flag_a_bound_comparison():
    source = "def test_it():\n    assert result > 5\n"
    assert _kinds(source) == set()


def test_does_not_flag_an_equality_comparison():
    source = "def test_it():\n    assert result == 42\n"
    assert _kinds(source) == set()


def test_does_not_flag_torch_allclose():
    source = "def test_it():\n    assert torch.allclose(a, b)\n"
    assert _kinds(source) == set()


def test_does_not_flag_torch_equal():
    source = "def test_it():\n    assert torch.equal(a, b)\n"
    assert _kinds(source) == set()


def test_does_not_flag_np_testing_assert_allclose():
    source = "def test_it():\n    np.testing.assert_allclose(actual, expected)\n"
    assert _kinds(source) == set()


def test_does_not_flag_a_subset_property_assertion():
    source = "def test_it():\n    assert set(shards) <= holdings\n"
    assert _kinds(source) == set()


def test_does_not_flag_is_none():
    source = "def test_it():\n    assert result is None\n"
    assert _kinds(source) == set()


def test_flags_is_not_none_as_bare_truthy():
    source = "def test_it():\n    assert result is not None\n"
    assert _kinds(source) == {Kind.BARE_TRUTHY}


def test_does_not_flag_a_boolop_of_two_concrete_comparisons():
    source = "def test_it():\n    assert a.shape == (8, 768) and b == 1\n"
    assert _kinds(source) == set()


def test_flags_a_boolop_of_two_bare_truthy_names():
    source = "def test_it():\n    assert a and b\n"
    assert _kinds(source) == {Kind.BARE_TRUTHY}


def test_does_not_flag_isinstance():
    source = "def test_it():\n    assert isinstance(result, MyType)\n"
    assert _kinds(source) == set()


def test_does_not_flag_isinstance_negated():
    source = "def test_it():\n    assert not isinstance(result, MyType)\n"
    assert _kinds(source) == set()


def test_does_not_flag_bool_wrapped_elementwise_comparison_all():
    source = "def test_it():\n    assert bool((a >= b - 1e-6).all())\n"
    assert _kinds(source) == set()


def test_does_not_flag_torch_all_wrapping_a_comparison():
    source = "def test_it():\n    assert torch.all(x >= 0.0)\n"
    assert _kinds(source) == set()


def test_does_not_flag_math_isclose():
    source = "def test_it():\n    assert math.isclose(result, 8.2)\n"
    assert _kinds(source) == set()


def test_does_not_flag_builtin_all_over_a_generator_of_comparisons():
    source = "def test_it():\n    assert all(not p.flag for p in items)\n"
    assert _kinds(source) == set()


def test_does_not_flag_a_standalone_mock_assert_not_called_call():
    source = "def test_it():\n    f()\n    mock_thing.assert_not_called()\n"
    assert _kinds(source) == set()


def test_does_not_flag_a_standalone_mock_assert_called_once_with_call():
    source = "def test_it():\n    f()\n    mock_thing.assert_called_once_with(1, 2)\n"
    assert _kinds(source) == set()


def test_does_not_flag_a_bare_truthy_assert_alongside_a_concrete_one():
    source = "def test_it():\n    assert result.ok\n    assert result.value == 42\n"
    assert _kinds(source) == set()
