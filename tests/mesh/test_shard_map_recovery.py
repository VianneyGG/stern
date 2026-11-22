from __future__ import annotations

import pytest

from stern.mesh.shard_map import Holder
from stern.mesh.shard_map_recovery import build_shard_map_from_holdings

pytestmark = pytest.mark.integration


def test_inverts_holdings_into_shard_map():
    holdings = {
        "node090": ["shard-000000.tar", "shard-000001.tar"],
        "node092": ["shard-000000.tar"],
    }
    node_subnets = {"node090": "/253", "node092": "/253"}

    shard_map = build_shard_map_from_holdings(holdings, node_subnets)

    assert shard_map["shard-000000.tar"] == [
        Holder(host="node090", subnet="/253", port=8000),
        Holder(host="node092", subnet="/253", port=8000),
    ]
    assert shard_map["shard-000001.tar"] == [
        Holder(host="node090", subnet="/253", port=8000)
    ]


def test_missing_subnet_raises_keyerror_naming_host():
    holdings = {"node096": ["shard-000000.tar"]}

    with pytest.raises(KeyError, match="node096"):
        build_shard_map_from_holdings(holdings, {})


def test_holder_order_is_deterministic():
    holdings_a = {
        "node092": ["shard-000000.tar"],
        "node090": ["shard-000000.tar"],
    }
    holdings_b = {
        "node090": ["shard-000000.tar"],
        "node092": ["shard-000000.tar"],
    }
    node_subnets = {"node090": "/253", "node092": "/253"}

    result_a = build_shard_map_from_holdings(holdings_a, node_subnets)
    result_b = build_shard_map_from_holdings(holdings_b, node_subnets)

    expected = [
        Holder(host="node090", subnet="/253", port=8000),
        Holder(host="node092", subnet="/253", port=8000),
    ]
    assert result_a["shard-000000.tar"] == expected
    assert result_b["shard-000000.tar"] == expected


def test_empty_holdings_returns_empty_map():
    assert build_shard_map_from_holdings({}, {}) == {}
