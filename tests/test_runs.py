import csv
import json
import os

from app import config, runs
from app.config import Settings

from .conftest import make_run


def test_runs_are_listed_newest_first_and_need_a_checkpoint(workspace):
    old = make_run(workspace, "prova", "splatfacto", "2026-01-01_000000")
    new = make_run(workspace, "prova", "nerfacto", "2026-01-02_000000")
    os.utime(next((old / "nerfstudio_models").iterdir()), (1000, 1000))
    os.utime(next((new / "nerfstudio_models").iterdir()), (2000, 2000))
    (workspace / "outputs" / "prova" / "splatfacto" / "interrotto").mkdir()
    (workspace / "outputs" / "prova" / "splatfacto" / "interrotto" / "config.yml").write_text("config")

    found = runs.list_runs("prova")
    assert [(r.method, r.timestamp) for r in found] == [("nerfacto", "2026-01-02_000000"), ("splatfacto", "2026-01-01_000000")]
    assert runs.latest_run("prova", "splatfacto").timestamp == "2026-01-01_000000"
    assert runs.latest_run("prova", "splatfacto-big") is None


def test_row_with_and_without_recorded_data(workspace):
    run_dir = make_run(workspace, "prova", "splatfacto", "2026-01-01_000000", step=29999)
    run = runs.latest_run("prova", "splatfacto")

    bare = runs.row(run)  # run fatto prima che esistesse run.json
    assert bare["iterations"] == 30000 and bare["psnr"] == "" and bare["alignment"] == "non registrato"

    scene = workspace / "data" / "prova"
    scene.mkdir(parents=True)
    (scene / "transforms.json").write_text("allineamento")
    run.update(iterations=30000, downscale=0, train_seconds=1080, alignment=runs.alignment_id(scene))
    (run_dir / runs.METRICS_FILE).write_text(json.dumps({"results": {"psnr": 27.456, "ssim": 0.87654, "lpips": 0.12345, "fps": 151.26}}))
    run.export_dir.mkdir(parents=True)
    (run.export_dir / "splat.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement vertex 733441\nend_header\n")

    full = runs.row(run)
    assert full["train_minutes"] == 18.0 and full["resolution"] == "automatica"
    assert (full["psnr"], full["ssim"], full["lpips"]) == (27.46, 0.8765, 0.1235)
    assert full["elements"] == 733441 and full["alignment"] == "corrente"

    (scene / "transforms.json").write_text("allineamento rifatto")
    assert runs.row(run)["alignment"] == "precedente"


def test_csv_export(workspace, tmp_path):
    make_run(workspace, "prova", "splatfacto", "2026-01-01_000000")
    target = tmp_path / "confronto.csv"
    assert runs.write_csv("prova", target) == 1
    with open(target, encoding="utf-8-sig", newline="") as f:
        header, first = list(csv.reader(f, delimiter=";"))
    assert header[0] == "Metodo" and header[-1] == "Cartella del run"
    assert first[0] == config.METHODS["splatfacto"].label and len(first) == len(header)


def test_alignment_summary(workspace):
    scene = workspace / "data" / "prova"
    assert runs.alignment_summary(scene) == ""
    runs.update_json(scene / runs.ALIGNMENT_FILE, photos=906, registered_images=904, mean_reprojection_error_px=1.061005)
    assert runs.alignment_summary(scene) == "904 foto allineate su 906, errore medio di riproiezione 1.06 px"


def test_settings_round_trip_and_legacy_format(workspace):
    s = Settings(name="prova", photos="C:/foto", methods=["nerfacto", "splatfacto"], iterations=5000)
    s.save()
    assert Settings.load("prova") == s
    assert config.projects() == ["prova"]

    legacy = {"name": "vecchio", "photos": "C:/foto", "method": "splatfacto-big", "mesh_size": 2400}
    (workspace / "data" / "vecchio").mkdir()
    (workspace / "data" / "vecchio" / "project.json").write_text(json.dumps(legacy))
    loaded = Settings.load("vecchio")
    assert loaded.methods == ["splatfacto-big"] and loaded.mesh_size == 2400
    assert Settings.load("inesistente") is None


def test_quality_levels_set_iterations_and_resolution(workspace):
    s = Settings(name="prova")
    assert s.quality == "massima" and (s.iterations, s.downscale, s.max_side) == (30000, 0, 0)
    assert s.camera == "auto"
    s.set_quality("bozza")
    assert s.quality == "bozza" and (s.iterations, s.max_side) == (7000, 800)
    s.iterations = 1234  # impostazioni scelte a mano
    assert s.quality is None
    s.set_quality("alta")
    s.downscale = 2
    assert s.quality is None


def test_only_runs_made_in_the_same_conditions_are_compared(workspace):
    scene = workspace / "data" / "prova"
    scene.mkdir(parents=True)
    (scene / "transforms.json").write_text("allineamento")

    def trained(method, timestamp, downscale, iterations=30000, evaluated=True, age=0):
        run_dir = make_run(workspace, "prova", method, timestamp)
        os.utime(next((run_dir / "nerfstudio_models").iterdir()), (10000 - age, 10000 - age))
        runs.update_json(run_dir / runs.RUN_FILE, downscale=downscale, iterations=iterations,
                         alignment=runs.alignment_id(scene))
        if evaluated:
            (run_dir / runs.METRICS_FILE).write_text(json.dumps({"results": {"psnr": 25.0, "width": 1320, "height": 989}}))

    trained("splatfacto", "2026-01-05_000000", downscale=2, age=0)   # creato da solo, a risoluzione piu' alta
    trained("splatfacto", "2026-01-01_000000", downscale=4, age=4)   # i tre del test
    trained("nerfacto", "2026-01-02_000000", downscale=4, age=3)
    trained("splatfacto-big", "2026-01-03_000000", downscale=4, age=2)
    trained("nerfacto-big", "2026-01-04_000000", downscale=4, iterations=7000, age=1)
    trained("nerfacto", "2026-01-06_000000", downscale=4, evaluated=False, age=5)

    group = runs.comparable_group(runs.list_runs("prova"))
    assert sorted((r.method, r.timestamp) for r in group) == [
        ("nerfacto", "2026-01-02_000000"), ("splatfacto", "2026-01-01_000000"), ("splatfacto-big", "2026-01-03_000000")]
    assert runs.row(group[0])["resolution"] == "1320 × 989 px"
    assert runs.comparable_group([]) == []


def test_missing_components_are_reported(workspace):
    assert config.missing_components() == []
    (workspace / "tools" / "colmap-4.2.1" / "bin" / "colmap.exe").unlink()
    assert any("COLMAP" in item for item in config.missing_components())
