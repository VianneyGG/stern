import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import subprocess
    import sys
    import tempfile
    import threading
    import time
    from pathlib import Path

    import marimo as mo
    import requests
    import torch

    return Path, mo, os, requests, subprocess, sys, tempfile, threading, time, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        # molab streaming SSL pretrain

        Runs `video_encoding/training/pretrain.py`'s SSL pretrain stage on this
        molab GPU while VoxCeleb2 WDS shards stay on login-node, streamed shard-major
        over a token-gated HTTP tunnel.

        **Before running this notebook:** on login-node, start
        `video_encoding/scripts/serve_shards_public.sh <shards-dir> <manifest-csv>`
        and copy the printed `<tunnel-url>/<token>` into the field below.

        See `openspec/changes/molab-streaming-pretrain/design.md` for the full
        design. Outbound network from molab (HTTP for shards, SSH for checkpoint
        push-back) is unverified until the manual verification steps in that
        change's tasks.md are run for real.
        """
    )
    return


@app.cell
def _(mo):
    github_pat = mo.ui.text(
        label="GitHub PAT (STERN is private, read access required)",
        kind="password",
        full_width=True,
    )
    github_pat
    return (github_pat,)


@app.cell
def _(mo):
    repo_ready = mo.ui.run_button(label="Clone + install repo")
    repo_ready
    return (repo_ready,)


@app.cell
def _(Path, github_pat, mo, os, repo_ready, subprocess, sys):
    mo.stop(not repo_ready.value, mo.md("Click the button above to clone the repo."))
    mo.stop(not github_pat.value, mo.md("Paste a GitHub PAT above before cloning."))

    clone_dir = "/tmp/stern"
    if not os.path.isdir(f"{clone_dir}/.git"):
        clone_url = f"https://{github_pat.value}@github.com/VianneyGG/STERN.git"
        subprocess.run(
            ["git", "clone", clone_url, clone_dir],
            check=True,
        )
    # Not `pip install -e .`: the repo's flat layout (data/, eval/,
    # video_encoding/, etc. all at top level) has no explicit `packages`
    # config, so setuptools' auto-discovery refuses to build it. The notebook
    # only imports a handful of modules, so put the clone on sys.path instead
    # and install just what those imports need (torch/torchvision are assumed
    # already present in the molab GPU image).
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "hydra-core",
            "tensorboard",
            "requests",
            "numpy",
            "Pillow",
            "paramiko",
        ],
        check=True,
    )
    sys.path.insert(0, clone_dir)
    clone_path = Path(clone_dir)
    mo.md(f"Repo ready at `{clone_dir}`.")
    return clone_dir, clone_path


@app.cell
def _(mo):
    shards_url = mo.ui.text(
        label="Tunnel base URL (with token)",
        placeholder="https://<name>.trycloudflare.com/<token>",
        full_width=True,
    )
    shards_url
    return (shards_url,)


@app.cell
def _(mo):
    ssh_host = mo.ui.text(label="login-node SSH host/IP", value="login-node", full_width=True)
    ssh_user = mo.ui.text(label="login-node SSH username", full_width=True)
    # mo.ui.text is single-line and mangles pasted newlines, which corrupts
    # the OpenSSH PEM format; text_area is required for a multi-line key
    # (marimo has no password-masked multiline widget, so this is visible
    # in the browser -- use a scoped deploy key, not a personal one).
    ssh_key = mo.ui.text_area(
        label="SSH private key (paste contents, unencrypted)",
        full_width=True,
        rows=10,
    )
    ssh_remote_dir = mo.ui.text(
        label="Remote checkpoint dir",
        value="stern_checkpoints",
        full_width=True,
    )
    mo.vstack([ssh_host, ssh_user, ssh_key, ssh_remote_dir])
    return ssh_host, ssh_key, ssh_remote_dir, ssh_user


@app.cell
def _(Path, mo, os, shards_url):
    mo.stop(
        not shards_url.value,
        mo.md("Paste the tunnel base URL above before continuing."),
    )

    checkpoint_root = Path("/tmp/molab_checkpoints")
    manifest_dir = Path("/tmp/molab_manifest")
    cache_dir = Path("/tmp/molab_shard_cache")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # cfg.train.checkpoint_root resolves via oc.env:STERN_CHECKPOINT_ROOT at
    # hydra.compose time below, so this must be set before that cell runs --
    # without it, training writes to the default /tmp/stern_checkpoints while
    # the rsync push-back cell reads from checkpoint_root, and nothing syncs.
    os.environ["STERN_CHECKPOINT_ROOT"] = str(checkpoint_root)
    return cache_dir, checkpoint_root, manifest_dir


@app.cell
def _(manifest_dir, mo, requests, shards_url):
    mo.stop(
        not shards_url.value,
        mo.md("Paste the tunnel base URL above before continuing."),
    )

    manifest_resp = requests.get(
        f"{shards_url.value.rstrip('/')}/manifest.csv", timeout=30.0
    )
    manifest_resp.raise_for_status()
    manifest_csv_path = manifest_dir / "manifest.csv"
    manifest_csv_path.write_bytes(manifest_resp.content)
    mo.md(f"Fetched manifest to `{manifest_csv_path}`.")
    return (manifest_csv_path,)


@app.cell
def _(clone_path, mo, shards_url):
    mo.stop(
        not shards_url.value,
        mo.md("Paste the tunnel base URL above before continuing."),
    )

    from hydra import compose, initialize_config_dir

    config_dir = str(clone_path / "video_encoding" / "conf")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "experiment=videomae_ssl_pretrain_voxmini",
                "train.num_workers=2",
            ],
        )
    return (cfg,)


@app.cell
def _(cfg):
    # Mirrors run_pretrain_stage's own ckpt_name derivation (pretrain.py) so
    # a pulled-back checkpoint lands exactly where _maybe_resume_pretrain
    # will look for it.
    _run_tag = cfg.get("run_tag", None)
    ckpt_name = f"{cfg.experiment.name}/{_run_tag}" if _run_tag else cfg.experiment.name
    return (ckpt_name,)


@app.cell
def _(mo):
    pull_checkpoint = mo.ui.run_button(
        label="Pull latest checkpoint from login-node (resume after a disconnect)"
    )
    pull_checkpoint
    return (pull_checkpoint,)


@app.cell
def _(
    checkpoint_root,
    ckpt_name,
    mo,
    os,
    pull_checkpoint,
    ssh_host,
    ssh_key,
    ssh_remote_dir,
    ssh_user,
    tempfile,
):
    # Not gated with mo.stop: pulling is optional (fresh runs have nothing to
    # pull), so this cell must still produce checkpoint_pulled on every run --
    # the training cell below takes it as an (unused) parameter purely to
    # force marimo to run this cell first, the same ordering trick used for
    # STERN_CHECKPOINT_ROOT above.
    if pull_checkpoint.value:
        import paramiko
        import stat as _stat

        # Same per-line normalization as push_checkpoints_periodically --
        # see that cell for why a whole-string .strip() is not enough.
        _key_lines = [
            line.strip()
            for line in ssh_key.value.replace("\r\n", "\n").splitlines()
            if line.strip()
        ]
        _normalized_key = "\n".join(_key_lines) + "\n"
        _key_fd, _key_path = tempfile.mkstemp(prefix="molab_ssh_key_")
        os.write(_key_fd, _normalized_key.encode())
        os.close(_key_fd)
        os.chmod(_key_path, 0o600)

        _client = paramiko.SSHClient()
        _client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        _client.connect(
            hostname=ssh_host.value,
            username=ssh_user.value,
            key_filename=_key_path,
        )
        try:
            _sftp = _client.open_sftp()
            try:
                _remote_root = f"{ssh_remote_dir.value}/{ckpt_name}"
                _local_root = checkpoint_root / ckpt_name
                _local_root.mkdir(parents=True, exist_ok=True)

                def _pull_dir(remote_dir: str, local_dir) -> None:
                    for entry in _sftp.listdir_attr(remote_dir):
                        remote_path = f"{remote_dir}/{entry.filename}"
                        local_path = local_dir / entry.filename
                        if _stat.S_ISDIR(entry.st_mode):
                            local_path.mkdir(parents=True, exist_ok=True)
                            _pull_dir(remote_path, local_path)
                        else:
                            _sftp.get(remote_path, str(local_path))

                _pull_dir(_remote_root, _local_root)
            finally:
                _sftp.close()
        finally:
            _client.close()
        checkpoint_pulled = True
        _status = mo.md(f"Pulled checkpoint tree from `{_remote_root}`.")
    else:
        checkpoint_pulled = True
        _status = mo.md(
            "Not pulling a checkpoint -- training will start from scratch "
            "unless a checkpoint already exists locally."
        )
    # _status must be the cell's last statement (not the checkpoint_pulled
    # assignment) -- marimo only renders a cell's final bare expression as
    # its visible output, so an assignment there silently hides this status
    # message.
    _status
    return (checkpoint_pulled,)


@app.cell
def _(
    checkpoint_root,
    os,
    ssh_host,
    ssh_key,
    ssh_remote_dir,
    ssh_user,
    tempfile,
    threading,
    time,
):
    def push_checkpoints_periodically(
        local_root=checkpoint_root,
        host: str = ssh_host.value,
        username: str = ssh_user.value,
        private_key: str = ssh_key.value,
        remote_dir: str = ssh_remote_dir.value,
        interval_s: float = 300.0,
    ) -> threading.Thread:
        """Best-effort periodic SFTP push of local checkpoints to login-node. A
        push failure must never crash training. Uses paramiko rather than the
        system `rsync`/`ssh` binaries -- neither is guaranteed present in the
        molab sandbox image (design doc open question #1; confirmed missing
        rsync in practice)."""
        import paramiko

        # Per-line strip + blank-line drop, not just an overall .strip(): a
        # browser textarea can leave stray leading/trailing whitespace or an
        # inserted blank line on paste, which corrupts the OpenSSH PEM body
        # and makes paramiko reject the whole file as "not a valid OPENSSH
        # private key file" even though it looks fine to the eye.
        key_lines = [
            line.strip()
            for line in private_key.replace("\r\n", "\n").splitlines()
            if line.strip()
        ]
        normalized_key = "\n".join(key_lines) + "\n"
        key_fd, key_path = tempfile.mkstemp(prefix="molab_ssh_key_")
        os.write(key_fd, normalized_key.encode())
        os.close(key_fd)
        os.chmod(key_path, 0o600)

        def push_once() -> None:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=host, username=username, key_filename=key_path)
            try:
                sftp = client.open_sftp()
                try:
                    for local_path in local_root.rglob("*"):
                        if not local_path.is_file():
                            continue
                        if local_path.suffix == ".tmp":
                            # checkpoint.py's atomic save creates these and
                            # renames them away almost immediately -- if this
                            # listing catches one mid-existence, it's gone by
                            # the time sftp.put tries to read it.
                            continue
                        rel = local_path.relative_to(local_root)
                        remote_path = f"{remote_dir}/{rel.as_posix()}"
                        remote_parent = remote_path.rsplit("/", 1)[0]
                        parts = remote_parent.split("/")
                        for i in range(1, len(parts) + 1):
                            partial = "/".join(parts[:i])
                            try:
                                sftp.mkdir(partial)
                            except IOError:
                                pass  # already exists
                        remote_tmp_path = f"{remote_path}.tmp"
                        sftp.put(str(local_path), remote_tmp_path)
                        sftp.posix_rename(remote_tmp_path, remote_path)
                finally:
                    sftp.close()
            finally:
                client.close()

        def run_loop() -> None:
            while True:
                time.sleep(interval_s)
                try:
                    push_once()
                except Exception as exc:  # noqa: BLE001 -- must not crash training
                    print(f"push_checkpoints_periodically: push failed: {exc}")

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        return thread

    return (push_checkpoints_periodically,)


@app.cell
def _(
    cache_dir,
    cfg,
    checkpoint_pulled,
    mo,
    push_checkpoints_periodically,
    shards_url,
    torch,
):
    del checkpoint_pulled  # dependency-ordering only, see that cell
    mo.stop(
        not shards_url.value,
        mo.md("Paste the tunnel base URL above before continuing."),
    )

    from video_encoding.data.shard_stream import StreamingShardDataset
    from video_encoding.training.pretrain import run_pretrain_stage

    stream_dataset = StreamingShardDataset(
        base_url=shards_url.value,
        cache_dir=cache_dir,
        num_frames=int(cfg.model.num_frames),
        image_size=int(cfg.model.image_size),
        overlap=float(cfg.preprocess.window.overlap),
    )
    push_checkpoints_periodically()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = run_pretrain_stage(
        cfg,
        dataset=stream_dataset,
        rank=0,
        world_size=1,
        local_rank=0,
        device=device,
    )
    result
    return (result,)


if __name__ == "__main__":
    app.run()
