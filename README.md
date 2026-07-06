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
