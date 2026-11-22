"""The Python and shell readers of cluster settings must apply the same lookup order."""

import shutil
import subprocess
from pathlib import Path

import pytest

from stern import cluster_env

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cluster_dir(tmp_path, monkeypatch):
    (tmp_path / "site.conf.example").write_text(
        'STERN_LOGIN_HOST=example-login   # comment\nSTERN_FACTORIAL_NODES="n1 n2"\n'
    )
    (tmp_path / "site.conf").write_text("STERN_LOGIN_HOST=real-login\n")
    (tmp_path / "tiers.example.tsv").write_text("fish\t/254\tn1\n")
    shutil.copy2(REPO_ROOT / "cluster" / "cluster_env.sh", tmp_path / "cluster_env.sh")
    monkeypatch.setattr(cluster_env, "CLUSTER_DIR", tmp_path)
    for name in ("STERN_LOGIN_HOST", "STERN_FACTORIAL_NODES", "STERN_TIERS_FILE", "STERN_DATA_ROOT"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def _shell(cluster_dir: Path, name: str, **env: str) -> str:
    script = f'source "{cluster_dir / "cluster_env.sh"}"; printf "%s" "${{{name}}}"'
    result = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", "USER": "someone", **env},
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def test_site_conf_beats_the_example(cluster_dir):
    assert cluster_env.setting("STERN_LOGIN_HOST") == "real-login"
    assert _shell(cluster_dir, "STERN_LOGIN_HOST") == "real-login"


def test_example_fills_what_site_conf_lacks_and_quotes_are_parsed(cluster_dir):
    assert cluster_env.setting("STERN_FACTORIAL_NODES") == "n1 n2"
    assert _shell(cluster_dir, "STERN_FACTORIAL_NODES") == "n1 n2"


def test_environment_beats_both_files(cluster_dir, monkeypatch):
    monkeypatch.setenv("STERN_LOGIN_HOST", "from-env")
    assert cluster_env.setting("STERN_LOGIN_HOST") == "from-env"
    assert _shell(cluster_dir, "STERN_LOGIN_HOST", STERN_LOGIN_HOST="from-env") == "from-env"


def test_unknown_setting_raises(cluster_dir):
    with pytest.raises(KeyError, match="STERN_NOPE"):
        cluster_env.setting("STERN_NOPE")


def test_roster_falls_back_to_the_example_until_a_real_one_exists(cluster_dir):
    fallback = cluster_env.tiers_path()
    assert (fallback.parent, fallback.name) == (cluster_dir, "tiers.example.tsv")
    assert Path(_shell(cluster_dir, "STERN_TIERS_FILE")).name == "tiers.example.tsv"
    (cluster_dir / "tiers.tsv").write_text("fish\t/254\treal1\n")
    real = cluster_env.tiers_path()
    assert (real.parent, real.name) == (cluster_dir, "tiers.tsv")
    assert Path(_shell(cluster_dir, "STERN_TIERS_FILE")).name == "tiers.tsv"


def test_data_root_is_per_user_unless_overridden(cluster_dir, monkeypatch):
    monkeypatch.setenv("USER", "someone")
    monkeypatch.setenv("LOGNAME", "someone")
    assert cluster_env.data_root() == Path("/Data/someone")
    assert _shell(cluster_dir, "STERN_DATA_ROOT") == "/Data/someone"
    monkeypatch.setenv("STERN_DATA_ROOT", "/scratch/x")
    assert cluster_env.data_root() == Path("/scratch/x")
