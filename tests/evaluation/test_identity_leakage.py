"""No scored identity reaches the probe's fit windows.

Both probe entry points (probe_ccc for one layer, probe_by_depth for the paired
depth sweep) split the manifest by identity, then build the fit set from the fit
records only. The seam below stops after that step: the dataset builder, the
token pass, and the fit are replaced by recorders, so no encoder or model is
needed. The windows themselves are built inside `_probe_dataset`, which is not
exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from stern.data.manifest import ManifestRecord, write_manifest
from stern.evaluation.diagnostics import probe

# 10 identities, 2 clips each. PROBE_SPLIT_FRACTION = 0.2 scores the last
# round(10 * 0.2) = 2 identities in sorted order: id08 and id09.
IDENTITIES = [f"id{i:02d}" for i in range(10)]
SCORED = {"id08", "id09"}


@pytest.fixture
def seam(tmp_path, monkeypatch):
    manifest = tmp_path / "probe_manifest.csv"
    write_manifest(
        manifest,
        [
            ManifestRecord(
                identity_id=ident,
                clip_path=f"{ident}_{k}.mp4",
                duration_s=2.0,
                role="probe",
                label_path=f"{ident}_{k}.csv",
            )
            for ident in IDENTITIES
            for k in (0, 1)
        ],
    )
    built = {}

    def fake_dataset(records, root, *, num_frames, image_size):
        built.setdefault("datasets", []).append(list(records))
        return list(records)

    def fake_tokens(encoder, loader, layer):
        n = len(loader.dataset)
        return torch.zeros(n, 1), torch.zeros(n, 2)

    monkeypatch.setattr(probe, "encoder_geometry", lambda _e: (12, 8, 4, 32))
    monkeypatch.setattr(probe, "_probe_dataset", fake_dataset)
    monkeypatch.setattr(probe, "_materialize_tokens", fake_tokens)
    monkeypatch.setattr(probe, "standardise_on_fit", lambda *a, **k: None)
    monkeypatch.setattr(probe, "_fit_and_score_probe", lambda *a, **k: 0.0)
    bundle = SimpleNamespace(encoder=object())
    return manifest, bundle, built


class _Stop(Exception):
    pass


def _probe_ccc_split(seam, monkeypatch):
    manifest, bundle, built = seam
    probe.probe_ccc(bundle, manifest, layer=12, seed=0)
    fit_records, scored_records = built["datasets"]
    return fit_records, scored_records


def _probe_by_depth_split(seam, monkeypatch):
    manifest, bundle, _ = seam
    seen = {}

    def record_and_stop(_bundle, fit_records, scored_records, _root, _layers):
        seen["split"] = (list(fit_records), list(scored_records))
        raise _Stop

    monkeypatch.setattr(probe, "_bundle_layer_tokens", record_and_stop)
    with pytest.raises(_Stop):
        probe.probe_by_depth(bundle, bundle, manifest, layers=[4])
    return seen["split"]


@pytest.mark.parametrize(
    "split",
    [
        pytest.param(_probe_ccc_split, id="probe_ccc"),
        pytest.param(_probe_by_depth_split, id="probe_by_depth"),
    ],
)
def test_fit_set_is_exactly_the_fit_identities_clips(seam, monkeypatch, split):
    fit_records, scored_records = split(seam, monkeypatch)
    assert sorted(r.clip_path for r in fit_records) == sorted(
        f"id0{i}_{k}.mp4" for i in range(8) for k in (0, 1)
    )
    assert {r.identity_id for r in scored_records} == SCORED
    assert [r.clip_path for r in fit_records if r.identity_id in SCORED] == []
