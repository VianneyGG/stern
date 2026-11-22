# Cluster operations

Placeholders: `<login-node>` is the SLURM login host, `<results-host>` (`STERN_RESULTS_HOST`) holds
finished results, `$DATA_ROOT` is node-local data storage, and `$REPO` is the
checkout on the shared home directory.

## Prerequisites

- **Python 3.11 through uv** (`$HOME/.local/bin/uv`, absolute, since cron PATH
  lacks it). Each node builds its own torch with `uv sync`.
- **Always `uv run --no-sync`.** The shared venv is about 9.4 GB on NFS. A sync
  on every tick raced concurrent ticks into import crash loops.
- **Use `python -m stern.<package>.<module>`** from the repo root, not file paths.
- **Quota.** The shared home has a 30 GB hard quota. Keep data on `$DATA_ROOT`.
- **No GPU resource in SLURM.** GPUs are used by landing on a node. Worker jobs
  request `--time=02:00:00`.

## Launching a mesh pool

Run on the login node. Roster: `stern/mesh/tiers.tsv` (113 nodes).

```bash
cd $REPO
QUORUM_K=8 WEIGHTING=token GRACE_S=900 \
  cluster/launch_mesh_pool.sh cars <run_id> 40 0.8 0.9
```

- Arguments: tier, run id, rounds, outer lr, outer momentum (defaults 25, 0.7, 0.9).
- `POOL_NODES=<n1,n2,...>` builds a cross-tier pool; the first argument then
  names the pool. `EXTRA_OVERRIDES="model.mask_type=mixed"` adds Hydra overrides.
- `QUORUM_K` defaults to three quarters of the pool. `register_mesh_run.sh` refuses a run with no quorum unless `ALLOW_NO_QUORUM=1`.
- `register_mesh_run.sh` is the only writer of `runs/state/mesh_run_configs.json`.
  Remove a run with `register_mesh_run.sh --remove <run_id>`, not by hand-editing.
- Set `SHARDS_DIR` to the shard directory and `DATASET_GB` to its real size.

## Installing the watcher

The watcher is a one-minute tick: `stern/watcher/watch_training.py`.
Only one host may tick, and the designation lives in `runs/state/watcher_host`.

```bash
cd $REPO
cluster/install_watcher.sh <login-node>   # designate and converge
cluster/install_watcher.sh                # later: converge this host
```

On the designated host this installs a systemd user timer (`stern-watcher.timer`,
every minute); elsewhere it stops the timer.

**Cron alternative.** Delete watcher lines from earlier layouts, then add:

```
* * * * *  cd $REPO && $HOME/.local/bin/uv run --no-sync python -m stern.watcher.watch_training
```

**Checking liveness.** Read the mtime of `runs/state/squeue.json`, which every
tick rewrites. Do not use `watcher.log`, whose ticks are mostly silent. A dead
watcher and a crash-looping one look identical in the log, but only the mtime
separates them.

## Class-hours gate

No job may run Monday to Friday, 08:00 to 20:00, Europe/Paris. The rule is
enforced in code. Do not work around it.

```bash
python3 stern/watcher/class_hours.py   # exit 0 allowed, 1 refused, reason on stderr
```

To gate hand-typed submissions, link the shim as `sbatch`, `srun` and `salloc` in
`$HOME/.local/bin`, which must precede `/usr/bin` on PATH. The shim passes `srun`
and `salloc` through inside an allocation. Job scripts call `class_hours.py` themselves.

## Monitoring with TensorBoard

```bash
cd $REPO && cluster/view_mesh_tensorboard.sh
```

The script picks one serving host, reachable over SSH, and reads
`$DATA_ROOT/stern_mesh_local`, one subdirectory per run. Open
`http://localhost:3003` through an SSH tunnel to the serving host. If the SSH
connection drops, rerun the script. The server keeps running.

- Tags include `loss/train/pool_mean`, `throughput/samples_per_sec/<node>`,
  `merge/n_contributors`, `merge/timeout_forced`, `slurm_job_id/pool`.
- Per-round scalars lag live training by up to two rounds. A finished run is
  flushed at DONE. `slurm_job_id/pool` names the first reporting node's job.

## Retrieving checkpoints

On DONE, the watcher moves a run's results to the results host:

```bash
rsync -a <results-host>:$DATA_ROOT/stern_mesh_results/<run_id>/master.pt ./<run_id>/
```

Never run `rsync --delete` against the repo: it wiped `runs/state` and
`slurm/logs` once. `.rsync-exclude` protects both.

## Exporting ViT-internals plots

Needs a node that holds one WebDataset shard locally.

```bash
# 1. Lay the checkpoint out where the export looks for it.
mkdir -p $DATA_ROOT/stern_mesh_results_local/checkpoints/<run_id>
cp <run_id>/master.pt $DATA_ROOT/stern_mesh_results_local/checkpoints/<run_id>/

# 2. Export activations and attention for one shard.
cd $REPO && $HOME/.local/bin/uv run --no-sync python -m \
  stern.evaluation.diagnostics.export_vit_internals \
  --shard-path $DATA_ROOT/voxceleb2_shards/shard-000000.tar \
  --mesh-shared $DATA_ROOT/stern_mesh_results_local \
  --run-ids <run_id_1>,<run_id_2> --out internals.npz

# 3. Plot.
$HOME/.local/bin/uv run --no-sync python -m \
  stern.evaluation.diagnostics.plot_vit_internals --npz internals.npz --out-dir plots/
```

This writes `plots/contact_sheet.png` and `plots/layer_diagnostics.png`. The
decoder shape comes from the run id (`per_frame`, `paper`, `e2b-*`). Add any new
naming scheme to `model_cfg_for_run` in `export_vit_internals.py`.

## Stopping a run

With the watcher paused, mark the run `DONE` in `runs/state/watcher_state.json`,
then cancel its jobs. `scancel` alone resurrects the run: `advance_all_runs`
recreates state for any config that lacks it.

## Known failure modes

- **Stale sync flag.** Flags were keyed by run id and round only, so a reused id
  let workers see an old flag and skip the merged checkpoint. Fixed:
  `_clean_stale_run_artifacts` (`watcher/mesh_tick.py`) clears them, with a
  digit-only glob so `pool-fish` cannot delete `pool-fish-5r`'s.
- **Merge crash loop.** A merge killed at 300 s, after its writes finished, left
  every retry seeing deltas from an already-applied round. Fixed in `5addb16`: a
  no-op for an applied round, and a 900 s timeout.
- **Shard-assignment race.** `shard_assignments_<run>.json` changed while queued
  jobs read it. A job could fail on a host missing from the new file. Fixed in
  `e4880b7`: each job reads an immutable per-node snapshot.
- **Oversubscription.** Nine live runs claimed about 126 roster slots for about
  62 schedulable slots. Runs below quorum relaunch from the same round until
  capacity returns. Check `squeue` before adding pools.
- **Silent watcher death.** The crontab was once found empty, and the watcher was
  dead for hours. Check the `squeue.json` mtime.
