from __future__ import annotations

from unittest.mock import patch

from video_encoding.mesh.scoring import NodeInfo
from video_encoding.mesh.shard_map import Holder
from video_encoding.scripts.seed_mesh import (
    MAX_MESH_NODES,
    NODE_GROUPS,
    _execute_rsync_pull,
    _remote_file_exists,
    _symlink_source_node,
    build_shard_map,
    main,
    probe_nodes,
)


def test_node_groups_cover_the_real_roster():
    assert NODE_GROUPS["node001"] == "fish"
    assert NODE_GROUPS["node024"] == "birds"
    assert NODE_GROUPS["node089"] == "orchids"
    assert NODE_GROUPS["node070"] == "countries"


def test_probe_nodes_parses_df_output():
    outputs = {"node001": "1000G 379G", "node024": "1000G 905G"}

    def fake_run_ssh(host, command):
        return outputs[host]

    nodes = probe_nodes(["node001", "node024"], run_ssh=fake_run_ssh)

    assert nodes[0] == NodeInfo("node001", "fish", "/254", 379.0)
    assert nodes[1].free_gb == 905.0


def test_probe_skips_unreachable_nodes_instead_of_failing():
    def fake_run_ssh(host, command):
        if host == "node089":
            raise OSError("connection refused")
        return "1000G 500G"

    nodes = probe_nodes(["node001", "node089"], run_ssh=fake_run_ssh)

    assert [n.hostname for n in nodes] == ["node001"]


def test_build_shard_map_inverts_placement_into_holders():
    placement = {"node001": ["s0", "s1"], "node024": ["s0"]}
    nodes = [
        NodeInfo("node001", "fish", "/254", 379.0),
        NodeInfo("node024", "birds", "/253", 905.0),
    ]

    shard_map = build_shard_map(placement, nodes, port=8000)

    assert shard_map["s0"] == [
        Holder("node001", "/254", 8000),
        Holder("node024", "/253", 8000),
    ]
    assert len(shard_map["s1"]) == 1


def _completed(returncode=0):
    class _Result:
        pass

    r = _Result()
    r.returncode = returncode
    r.stdout = ""
    r.stderr = ""
    return r


def test_train_shards_land_flat_under_mesh_dir_matching_resolve_shard():
    with patch("video_encoding.scripts.seed_mesh.subprocess.run") as run:
        run.return_value = _completed()
        _execute_rsync_pull(
            "node001",
            "node002",
            ["shard-000000.tar"],
            mesh_dir="/Data/$USER/mesh",
            remote_source_dir="/Data/$USER/shards",
        )

    remote_cmd = run.call_args.args[0][-1]
    assert "mkdir -p /Data/$USER/mesh &&" in remote_cmd
    assert "chmod 700 /Data/$USER/mesh &&" in remote_cmd
    assert remote_cmd.rstrip().endswith("/Data/$USER/mesh/")
    assert "/mesh/shards" not in remote_cmd


def test_val_shards_stay_under_a_val_subdir_isolated_from_train():
    with patch("video_encoding.scripts.seed_mesh.subprocess.run") as run:
        run.return_value = _completed()
        _execute_rsync_pull(
            "node001",
            "node002",
            ["shard-000000.tar"],
            mesh_dir="/Data/$USER/mesh",
            remote_source_dir="/Data/$USER/shards/val",
            subdir="val",
        )

    remote_cmd = run.call_args.args[0][-1]
    assert "/Data/$USER/mesh/val" in remote_cmd
    assert "chmod 700" in remote_cmd
    assert (
        "/Data/$USER/mesh/val"
        in remote_cmd.split("chmod 700", 1)[1].split("&&")[0]
    )


def test_symlink_source_node_links_own_train_and_val_shards_not_copies():
    with patch("video_encoding.scripts.seed_mesh.subprocess.run") as run:
        run.return_value = _completed()
        _symlink_source_node(
            "node089",
            ["shard-000000.tar", "shard-000001.tar"],
            ["shard-000000.tar"],
            mesh_dir="/Data/$USER/mesh",
            remote_source_dir="/Data/$USER/voxceleb2_shards",
        )

    assert run.call_count == 2
    train_call, val_call = run.call_args_list

    train_cmd = train_call.args[0][-1]
    assert "ln -sf" in train_cmd
    assert "rsync" not in train_cmd
    train_stdin = train_call.kwargs["input"]
    assert train_stdin == "shard-000000.tar\nshard-000001.tar\nmanifest.json\n"

    val_cmd = val_call.args[0][-1]
    assert "ln -sf" in val_cmd
    assert "/Data/$USER/mesh/val" in val_cmd
    val_stdin = val_call.kwargs["input"]
    assert val_stdin == "shard-000000.tar\nmanifest.json\n"


def test_remote_file_exists_returns_true_when_test_dash_f_succeeds():
    def fake_run_ssh(host, command):
        assert "test -f" in command
        return ""

    with patch("video_encoding.scripts.seed_mesh._run_ssh", fake_run_ssh):
        assert _remote_file_exists(
            "node001", "/Data/$USER/shards/manifest.json"
        )


def test_remote_file_exists_returns_false_when_test_dash_f_fails():
    def fake_run_ssh(host, command):
        raise OSError("remote command failed")

    with patch("video_encoding.scripts.seed_mesh._run_ssh", fake_run_ssh):
        assert not _remote_file_exists("node001", "/nonexistent/manifest.json")


def test_main_rejects_more_reachable_nodes_than_the_quota_allows(monkeypatch, capsys):
    fake_nodes = [
        NodeInfo(f"n{i}", "fish", "/254", 500.0) for i in range(MAX_MESH_NODES + 1)
    ]
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh.probe_nodes", lambda hostnames: fake_nodes
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "seed_mesh.py",
            "--nodes",
            ",".join(n.hostname for n in fake_nodes),
            "--source-node",
            "n0",
            "--shards-dir",
            "/Data/$USER/shards",
            "--shard-map",
            "/tmp/shard_map.json",
            "--dataset-gb",
            "100",
        ],
    )

    assert main() == 1
    assert "MAX_MESH_NODES" in capsys.readouterr().err


def _base_argv(tmp_path, nodes: list[str], source_node: str) -> list[str]:
    return [
        "seed_mesh.py",
        "--nodes",
        ",".join(nodes),
        "--source-node",
        source_node,
        "--shards-dir",
        "/Data/$USER/shards",
        "--shard-map",
        str(tmp_path / "shard_map.json"),
        "--dataset-gb",
        "1",
    ]


def test_main_rejects_missing_val_manifest_on_source_node(
    monkeypatch, capsys, tmp_path
):
    fake_nodes = [
        NodeInfo("n0", "fish", "/254", 500.0),
        NodeInfo("n1", "fish", "/254", 500.0),
    ]
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh.probe_nodes", lambda hostnames: fake_nodes
    )
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._list_remote_shards",
        lambda host, remote_dir: ["shard-000000.tar"],
    )

    def fake_remote_file_exists(host, remote_path):
        return not remote_path.endswith("/val/manifest.json")

    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._remote_file_exists", fake_remote_file_exists
    )
    monkeypatch.setattr("sys.argv", _base_argv(tmp_path, ["n0", "n1"], "n0"))

    assert main() == 1
    assert "val manifest.json not found" in capsys.readouterr().err


def test_main_replicates_val_manifest_alongside_val_shards_to_every_replica(
    monkeypatch, tmp_path
):
    fake_nodes = [
        NodeInfo("n0", "fish", "/254", 500.0),
        NodeInfo("n1", "fish", "/254", 500.0),
    ]
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh.probe_nodes", lambda hostnames: fake_nodes
    )
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._list_remote_shards",
        lambda host, remote_dir: ["shard-000000.tar"],
    )
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._remote_file_exists", lambda host, path: True
    )
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._symlink_source_node", lambda *a, **k: None
    )
    rsync_calls = []
    monkeypatch.setattr(
        "video_encoding.scripts.seed_mesh._execute_rsync_pull",
        lambda *a, **k: rsync_calls.append((a, k)),
    )
    monkeypatch.setattr("sys.argv", _base_argv(tmp_path, ["n0", "n1"], "n0"))

    assert main() == 0

    val_calls = [(a, k) for a, k in rsync_calls if k.get("subdir") == "val"]
    assert len(val_calls) == 1
    (args, kwargs) = val_calls[0]
    shards_arg = args[2]
    assert "manifest.json" in shards_arg
    assert kwargs["remote_source_dir"] == "/Data/$USER/shards/val"
