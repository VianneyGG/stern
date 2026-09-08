"""Property tests for mesh/quorum.token_weights.

Complements the example-based tests in test_mesh_quorum.py: instead of a
few hand-picked stats dicts, hypothesis generates many and checks the
invariants the formula must hold for all of them.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from video_encoding.mesh.quorum import NodeStats, token_weights

_node_stats = st.builds(
    NodeStats,
    c_tokens=st.integers(min_value=0, max_value=100_000),
    c_steps=st.integers(min_value=0, max_value=100_000),
)


@st.composite
def _stats_dict_with_at_least_one_positive_node(draw):
    names = draw(
        st.lists(
            st.text(alphabet="abcdefgh", min_size=1, max_size=4),
            min_size=1,
            max_size=8,
            unique=True,
        )
    )
    stats = {name: draw(_node_stats) for name in names}
    # Force at least one node to have positive weight, or token_weights
    # raises by design (see test_mesh_quorum.py::test_token_weights_all_zero_raises).
    forced = draw(st.sampled_from(names))
    stats[forced] = NodeStats(
        c_tokens=draw(st.integers(min_value=1, max_value=100_000)),
        c_steps=draw(st.integers(min_value=1, max_value=100_000)),
    )
    return stats


@given(_stats_dict_with_at_least_one_positive_node())
def test_weights_sum_to_one(stats):
    weights = token_weights(stats)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


@given(_stats_dict_with_at_least_one_positive_node())
def test_weights_are_non_negative(stats):
    weights = token_weights(stats)
    assert all(w >= 0.0 for w in weights.values())


@given(_stats_dict_with_at_least_one_positive_node())
def test_zero_token_or_zero_step_nodes_weigh_exactly_zero(stats):
    weights = token_weights(stats)
    for name, s in stats.items():
        if s.c_tokens <= 0 or s.c_steps <= 0:
            assert weights[name] == 0.0
