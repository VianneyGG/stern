## Video arm (face, VA probing)

See `docs/superpowers/specs/2026-06-29-video-arm-design.md` for the design
and `video_encoding/` for the code. Shared contract code the skeleton
(landmark) arm will reuse lives in flat top-level packages: `data/`
(harmonisation, manifests, labels), `eval/` (CCC metric), and `cache/`
(embedding cache). The landmark arm is a placeholder in `landmark_encoding/`.

Pruned scaffolding (encoder/eval stubs, config-only datasets) is recorded
verbatim in `SCAFFOLD_ARCHIVE.md`, ready to restore when implemented.

Four decisions are still open — see `REPO_SETUP.md` §9 (D-MODEL-1, D9,
D-DATA-1, D-DATA-2; an external build-spec doc, not checked into this repo)
— and are wired as Hydra config defaults, not hardcoded.

## Monitoring mesh training runs

View live TensorBoard for all mesh (DiLoCo) runs at once:

    video_encoding/scripts/view_mesh_tensorboard.sh

Then open http://localhost:3003 and pick a run in the sidebar to compare
it against the others — no run_id argument needed; `--logdir` points at
the shared parent directory, so every run_id shows up as its own entry.
If your SSH connection drops, just rerun the same command — TensorBoard
keeps running remotely independently of your laptop; only the local
tunnel needs to be re-established.

The script prefers node089 and falls back to login-node (node089 is not yet
online), always landing on exactly one host — any stray instance left on
the other candidate host from a previous run is killed on failover.

Scalar tags are grouped by metric so TensorBoard's own tag-prefix grouping
keeps related lines together: `loss/train/<node>` and `loss/train/pool_mean`
(same for `val`), `throughput/samples_per_sec/<node>` (and `pool_sum`),
`data/n_shards/<node>`, `config/lr/<node>` (fixed per run — no scheduler on
this path, so it plots as a flat reference line), `slurm_job_id/<node>` (and
`/pool`), and `merge/n_contributors`, `merge/timeout_forced`,
`merge/alpha/<node>` (each contributor's merge weight).

Per-round scalars lag live training by up to two rounds while a run is in
progress: a worker only appends its round-R JSONL record after round R's
merge has already run, and the round replay then holds back the newest
round it sees for one more merge cycle so a straggler has a chance to
report before that round's pool mean is finalized. `merge/n_contributors`
and `merge/timeout_forced` show quorum size and whether a round was forced
by timeout, with no lag. Once a run reaches DONE, its held-back round(s)
are flushed immediately, so a finished run's curve always ends at its true
last round — the lag only ever affects the live tail of an in-progress run.

**Run-id rule:** the merge itself runs as a plain process on the login
node, which never gets a SLURM job id of its own — only the per-node
workers do. So `slurm_job_id/<node>` records each node's own current job
id, and `slurm_job_id/pool` is always the job id of that round's
alphabetically-first reporting node — a simple, stable pick, not a claim
that one node "owns" the merge. Use it to correlate a TB data point back
to a specific `sbatch` job/log; it changes whenever that node is requeued
into a new job.

## Generating the ViT-internals contact sheet

On `DONE`, the watcher moves a run's `master.pt`/`outer_momentum.pt` off the
NFS-quota-constrained `~/stern_mesh/checkpoints/<run_id>/` to
`node089:/Data/$USER/stern_mesh_results/<run_id>/` to free NFS space
— `export_vit_internals.py` only looks under
`<mesh-shared>/checkpoints/<run_id>/master.pt`, so pull a finished run's
checkpoint back into that shape on a node that also has a WebDataset shard
locally (e.g. `login-node`, which has `/Data/vox2_val_shards_8fps/`), then run
the two-stage export → plot pipeline:

    ssh login-node 'mkdir -p ~/stern_mesh_results_local/checkpoints/<run_id> && \
        rsync -a node089:/Data/$USER/stern_mesh_results/<run_id>/master.pt \
            ~/stern_mesh_results_local/checkpoints/<run_id>/master.pt'

    ssh login-node 'cd ~/stern && ~/.local/bin/uv run --no-sync python -m \
        video_encoding.scripts.export_vit_internals \
        --shard-path /Data/vox2_val_shards_8fps/shard-000000.tar \
        --mesh-shared ~/stern_mesh_results_local \
        --run-ids <run_id_1>,<run_id_2>,... \
        --out internals.npz'

    ssh login-node 'cd ~/stern && ~/.local/bin/uv run --no-sync python -m \
        video_encoding.scripts.plot_vit_internals \
        --npz internals.npz --out-dir plots/'

which writes `plots/contact_sheet.png` (per-run attention maps + PCA-RGB,
one row per `--run-ids` entry) and `plots/layer_diagnostics.png`
(per-layer consecutive-cosine-similarity and attention-norm density).
`model_cfg_for_run` in `export_vit_internals.py` picks the decoder shape
from substrings in `run_id` (`"per_frame"`, `"paper"`) — `e2b-*` run ids
don't carry the recipe name (they're named by pool: fish/birds/cars/
countries), so `run_id.startswith("e2b-")` is special-cased there too,
since all four `e2b-*` arms launched from the same paper-decoder recipe.
Add any future naming scheme to that same function rather than assuming
the default (legacy 256/8) decoder shape.
