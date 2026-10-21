import csv
import json
import os

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from app import paper, runs  # noqa: E402

from .conftest import make_run  # noqa: E402

VIEWS = [f"jpg/v{i:02d}.JPG" for i in range(14)] + ["dng/v14.jpg", "dng/v15.jpg"]


def test_holm_correction():
    # Esempio classico: p ordinati 0.01, 0.02, 0.03, 0.04 su m = 4 confronti.
    assert paper.holm([0.04, 0.01, 0.03, 0.02]) == pytest.approx([0.06, 0.04, 0.06, 0.06])
    assert paper.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])  # mai sopra 1, e monotona


def test_bootstrap_is_reproducible_and_brackets_the_statistic():
    values = np.linspace(10, 30, 50)
    low, high = paper.bootstrap_ci(values)
    assert (low, high) == paper.bootstrap_ci(values)  # stesso seme, stesso intervallo
    assert low < values.mean() < high and high - low < 4
    assert paper.bootstrap_ci([5.0] * 10) == (5.0, 5.0)


def test_discarded_photos_are_counted_once(tmp_path):
    note = tmp_path / "LEGGIMI.txt"
    note.write_text("Scartate (3):\n  jpg/a.JPG: poco nitida\n  jpg/b.JPG: poco nitida; quasi doppione di jpg/c.JPG\n"
                    "  dng/d.jpg: quasi doppione di dng/e.jpg\n\nScartata a mano: lo scatto 0636.\n", encoding="utf-8")
    discarded = paper.read_discarded(note)
    assert len(discarded) == 4 and discarded["jpg/b.JPG"] == ["poco nitida", "quasi doppione"]


def fake_project(workspace):
    scene = workspace / "data" / "prova"
    scene.mkdir(parents=True)
    (scene / "transforms.json").write_text("allineamento")
    (scene / "split.json").write_text(json.dumps({"train": ["jpg/t1.JPG"] * 100, "test": VIEWS}))
    runs.update_json(scene / runs.ALIGNMENT_FILE, photos=116, registered_images=116, points=1000,
                     mean_reprojection_error_px=1.0, mean_track_length=4.0)

    def trained(method, family, timestamp, offset, age, seed=None, views=VIEWS):
        run_dir = make_run(workspace, "prova", method, timestamp)
        checkpoint = next((run_dir / "nerfstudio_models").iterdir())
        if method == "inria-3dgs":  # il codice Inria salva il modello come .ply, non come checkpoint di nerfstudio
            checkpoint = run_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"ply\nelement vertex 5\nend_header\n")
        os.utime(checkpoint, (10000 - age, 10000 - age))
        runs.update_json(run_dir / runs.RUN_FILE, family=family, downscale=1, iterations=30000, seed=seed,
                         alignment=runs.alignment_id(scene), train_seconds=3600, gpu_peak_mb=9000)
        per_image = [{"name": n, "psnr": 20 + offset + i * 0.1, "ssim": 0.6 + offset / 100, "lpips": 0.5 - offset / 100}
                     for i, n in enumerate(views)]
        results = {m: float(np.mean([v[m] for v in per_image])) for m in ("psnr", "ssim", "lpips")}
        (run_dir / runs.METRICS_FILE).write_text(json.dumps({"results": {**results, "fps": 10, "width": 100, "height": 75},
                                                             "per_image": per_image}))

    trained("splatfacto", "gaussian", "2026-01-01_000000", 1.0, age=4, seed=1)
    trained("splatfacto", "gaussian", "2026-01-02_000000", 2.0, age=3, seed=2)  # ripetizione con un altro seme
    trained("inria-3dgs", "gaussian", "2026-01-03_000000", 0.0, age=2, views=VIEWS[:-1])  # una vista in meno
    trained("nerfacto", "nerf", "2026-01-04_000000", -3.0, age=1)  # NeRF della 0.3: altre condizioni, escluso
    return scene


def test_tables_and_numbers(workspace, tmp_path):
    fake_project(workspace)
    out = tmp_path / "articolo"
    numbers = paper.build("prova", out, figures=False)

    for name in ("tab1_dati", "tab2_configurazioni", "tab3_risultati", "tab4_appaiati"):
        assert (out / f"{name}.csv").exists() and r"\toprule" in (out / f"{name}.tex").read_text(encoding="utf-8")
    assert not (out / "tab5_fotogrammetria.csv").exists()  # nessuna mesh nel progetto
    assert json.loads((out / "numeri.json").read_text(encoding="utf-8")) == numbers

    # Due semi per splatfacto: per ogni vista la media tra i run (offset 1 e 2 -> 1,5).
    assert numbers["splatfacto.psnr_mean"] == pytest.approx(21.5 + 0.1 * 7.5)
    assert numbers["splatfacto.run"] == 2 and numbers["splatfacto.psnr_sd_tra_run"] == pytest.approx(np.sqrt(0.5))
    assert "nerfacto.psnr_mean" not in numbers  # condizioni diverse: fuori dal confronto principale

    # Confronto appaiato solo sulle viste comuni, differenza costante di 1,5 dB.
    key = "appaiati.splatfacto_vs_inria-3dgs.psnr"
    assert numbers[f"{key}.viste"] == len(VIEWS) - 1 and numbers[f"{key}.mediana"] == pytest.approx(1.5)
    with open(out / "tab4_appaiati.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    assert len(rows) == 1 + 3  # una coppia di metodi, tre metriche
    assert numbers["dati.foto_test"] == len(VIEWS) and numbers["dati.camera_dng"] == 2
