# AFEW-VA probe staging notes (2026-09-10)

Staged on `login-node` at `/Data/$USER/afewva/`:
- `zips/` — the 12 source zips (01.zip .. 12.zip), 6.9G
- `raw/` — extracted clips + `manifest.csv` + `labels/`, 7.1G

Built with `video_encoding/scripts/build_afewva_manifest.py`, invoked as:
```
PYTHONPATH=. uv run python video_encoding/scripts/build_afewva_manifest.py \
    --afewva-root /Data/$USER/afewva/raw
```

## Numbers

- **Clips:** 600 / 600 extracted and included. **0 skipped** — every clip's frame
  count matched its annotation count.
- **Total frames:** 30,051 across all clips.
- **Valence/arousal range:** native AFEW-VA integers are `[-10, 10]`; label CSVs
  are written pre-divided by 10.0, and the observed range after scaling is
  `[-1.0, 0.9]` (within the expected `[-1, 1]`).
- **`identity_id`:** the clip's `actor` field (240 distinct actors across 600
  clips), falling back to the clip id when `actor` is missing/empty — none of
  the 600 clips needed the fallback.
- **`annotation_hz`:** 30.0, **approximate** — AFEW-VA ships no measured
  per-clip frame rate; `duration_s` in the manifest is `n_frames / 30.0` and
  should not be read as a measured value.

## Decisions (per user direction, superseding the original handoff draft)

- VA scaling: divide by 10.0 in the staging script (not by editing
  `PROBE_VA_RANGE`); `conf/dataset/afewva.yaml`'s `va_range` updated to
  `[-1.0, 1.0]` to match.
- Misaligned clips: skip and report, not warn-and-keep-and-align-to-shorter.
- `identity_id`: use `actor` (actor-disjoint split) in preference to clip id
  (clip-disjoint only) — the AFEW-VA JSON does carry an `actor` field, contrary
  to the original handoff's assumption that no actor labels exist.

## Verification (run on login-node, 2026-09-10)

```
clips: 600
misaligned clips: 0
va min/max: -1.0 0.8999999999999999
distinct identity_id values: 240
```

```
windows: 1617 pixel_values: (3, 16, 224, 224) label: tensor([0.0000, 0.5000])
```

## Outstanding

- `AFEWVA_ROOT=/Data/$USER/afewva/raw` and `AFEWVA_PRESENT=true`
  need to be exported wherever `conf/dataset/afewva.yaml` is resolved (not yet
  persisted in a shell profile or job script).
- `diag_probe.py` (which reads this manifest) exists only in the
  `STERN-ssl-diag-02` worktree, not in this checkout or login-node's `~/stern` —
  it needs to land there (or be pointed at this data) before `probe_ccc` can
  actually run against AFEW-VA.
