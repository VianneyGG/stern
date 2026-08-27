import sys

import numpy as np
from PIL import Image

from video_encoding.scripts.plot_vit_internals import (
    main,
    plot_contact_sheet,
    plot_layer_diagnostics,
)


def _fake_data(n_runs=2, n_windows=2, grid=4, img_size=8, depth=3, n_bins=5):
    rng = np.random.default_rng(0)
    return {
        "run_ids": np.array([f"run{i}" for i in range(n_runs)]),
        "clip_keys": np.array([f"clip{i}" for i in range(n_windows)]),
        "middle_frames": rng.integers(
            0, 255, size=(n_windows, img_size, img_size, 3), dtype=np.uint8
        ),
        "bin_edges": np.logspace(0, 2, n_bins + 1).astype(np.float32),
        "received_maps": rng.random((n_runs, n_windows, grid, grid)).astype(np.float32),
        "pca_maps": rng.integers(
            0, 255, size=(n_runs, n_windows, grid, grid, 3), dtype=np.uint8
        ),
        "cos_sim": rng.uniform(0.5, 1.0, size=(n_runs, depth - 1)).astype(np.float32),
        "norm_hist": rng.dirichlet(np.ones(n_bins), size=(n_runs, depth)).astype(
            np.float32
        ),
    }


def test_plot_contact_sheet_writes_a_readable_png(tmp_path):
    out = tmp_path / "contact_sheet.png"
    plot_contact_sheet(_fake_data(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    img = Image.open(out)
    assert img.width > 0 and img.height > 0


def test_plot_layer_diagnostics_writes_a_readable_png(tmp_path):
    out = tmp_path / "layer_diagnostics.png"
    plot_layer_diagnostics(_fake_data(), out)

    assert out.exists()
    assert out.stat().st_size > 0
    img = Image.open(out)
    assert img.width > 0 and img.height > 0


def test_main_reads_npz_and_writes_both_pngs(tmp_path, monkeypatch):
    npz_path = tmp_path / "internals.npz"
    np.savez_compressed(npz_path, **_fake_data())
    out_dir = tmp_path / "out"

    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_vit_internals", "--npz", str(npz_path), "--out-dir", str(out_dir)],
    )
    main()

    assert (out_dir / "contact_sheet.png").exists()
    assert (out_dir / "layer_diagnostics.png").exists()
