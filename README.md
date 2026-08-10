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

View live TensorBoard for a mesh (DiLoCo) run:

    video_encoding/scripts/view_mesh_tensorboard.sh <run_id>

Then open http://localhost:3003. If your SSH connection drops, just rerun
the same command — TensorBoard keeps running on login-node independently of
your laptop; only the local tunnel needs to be re-established.

Per-round scalars (`mesh_round/<node>/*`, `mesh_round/pool/*`) lag live
training by up to one round — each round is held back one merge cycle so
a straggler node has a chance to report before that round's pool mean is
finalized. `mesh_merge/n_contributors` and `mesh_merge/timeout_forced`
show quorum size and whether a round was forced by timeout, with no lag.
An unrecognized `<run_id>` (or one whose first merge hasn't landed yet)
fails with an explicit error rather than opening an empty dashboard.

**Run-id rule:** the merge itself runs as a plain process on login-node (the
login node), which never gets a SLURM job id of its own — only the
per-node workers do. So `mesh_round/<node>/slurm_job_id` records each
node's own current job id, and `mesh_round/pool/slurm_job_id` is always
the job id of that round's alphabetically-first reporting node — a
simple, stable pick, not a claim that one node "owns" the merge. Use it
to correlate a TB data point back to a specific `sbatch` job/log; it
changes whenever that node is requeued into a new job.
