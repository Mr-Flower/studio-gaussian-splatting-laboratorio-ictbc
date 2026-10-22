import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from app import clean, runs, site  # noqa: E402

from .conftest import make_run  # noqa: E402


def test_nerfstudio_points_go_back_to_colmap(workspace):
    run_dir = make_run(workspace, "prova", "splatfacto", "2026-01-01_000000")
    angle = 0.4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    shift, scale = np.array([0.5, -1.0, 2.0]), 0.25
    (run_dir / "dataparser_transforms.json").write_text(json.dumps(
        {"transform": np.hstack([rotation, shift[:, None]]).tolist(), "scale": scale}))
    colmap = np.random.default_rng(0).normal(size=(20, 3))
    exported = scale * (colmap @ rotation.T + shift)  # come li salva nerfstudio
    run = runs.Run("prova", "splatfacto", "2026-01-01_000000")
    assert np.allclose(clean.to_colmap(exported, run, workspace / "data" / "prova"), colmap)


def test_splat_file_layout_and_order(tmp_path):
    g = {"xyz": np.array([[0, 0, 0], [1, 2, 3]], float), "scales": np.array([[0.1] * 3, [0.5] * 3]),
         "quat": np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float), "rgb": np.array([[1, 0, 0], [0, 0, 1]]),
         "opacity": np.array([0.9, 0.9])}
    target = tmp_path / "modello.splat"
    assert site.write_splat(g, target) == 2 and target.stat().st_size == 2 * 32
    raw = np.fromfile(target, dtype=[("xyz", "<f4", 3), ("scales", "<f4", 3), ("rgba", "u1", 4), ("quat", "u1", 4)])
    # La gaussiana piu' grande viene prima; colore e opacita' a 8 bit, rotazione centrata su 128.
    assert np.allclose(raw["xyz"][0], [1, 2, 3]) and list(raw["rgba"][0]) == [0, 0, 255, 230]
    assert list(raw["quat"][0]) == [128, 255, 128, 128] and list(raw["quat"][1]) == [255, 128, 128, 128]
