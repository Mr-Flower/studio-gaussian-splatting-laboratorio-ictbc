import os

import pytest

from app import pipeline, runs
from app.config import Settings

from .conftest import make_run, make_sparse_model


def settings(photos, **overrides):
    values = dict(name="prova", photos=str(photos), methods=["splatfacto", "nerfacto"], iterations=1000)
    values.update(overrides)
    return Settings(**values)


def labels(steps):
    return [step.label for step in steps]


def test_full_pipeline_order(workspace, photos):
    steps = pipeline.build_steps(settings(photos), ["sfm", "train", "eval", "export", "mesh"])
    assert labels(steps) == [
        "Estrazione delle feature", "Matching tra le foto", "Ricostruzione delle camere",
        "Verifica dell'allineamento", "Conversione per nerfstudio",
        "Training — Gaussian splatting (splatfacto)", "Valutazione — Gaussian splatting (splatfacto)",
        "Esportazione — Gaussian splatting (splatfacto)",
        "Training — NeRF (nerfacto)", "Valutazione — NeRF (nerfacto)", "Esportazione — NeRF (nerfacto)",
        "Mesh: correzione delle immagini", "Mesh: mappe di profondità", "Mesh: fusione in nuvola densa",
        "Mesh: superficie di Poisson",
    ]


def test_alignment_uses_colmap_4_options_and_camera_mode(workspace, photos):
    steps = pipeline.build_steps(settings(photos, camera="per_folder"), ["sfm"])
    features = steps[0].command()
    assert "--FeatureExtraction.use_gpu" in features and "--SiftExtraction.use_gpu" not in features
    assert "--ImageReader.single_camera_per_folder" in features
    assert steps[1].command()[1] == "exhaustive_matcher"


def test_existing_alignment_survives_until_the_new_one_is_ready(workspace, photos):
    s = settings(photos)
    old = make_sparse_model(s.scene / "colmap", "0", registered=5)
    (s.scene / "transforms.json").write_text("{}")
    steps = pipeline.build_steps(s, ["sfm"])

    steps[0].before()  # prepara la cartella di lavoro
    assert old.exists() and (s.scene / "transforms.json").exists()
    assert "colmap.tmp" in steps[0].command()[3]

    make_sparse_model(s.scene / "colmap.tmp", "0", registered=3)
    make_sparse_model(s.scene / "colmap.tmp", "1", registered=4)
    assert steps[3].command()[-1].endswith(os.path.join("colmap.tmp", "sparse", "1"))

    steps[4].before()  # scambio: solo ora il vecchio allineamento viene sostituito
    assert not (s.scene / "colmap.tmp").exists() and not (s.scene / "transforms.json").exists()
    command = steps[4].command()
    assert command[command.index("--colmap-model-path") + 1] == "colmap/sparse/1"


def test_no_reconstruction_is_reported(workspace, photos):
    s = settings(photos)
    (s.scene / "colmap" / "sparse").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="nessuna ricostruzione"):
        pipeline.best_sparse_model(s.scene / "colmap")


def test_training_command_and_run_record(workspace, photos):
    s = settings(photos, methods=["splatfacto"], downscale=2)
    (s.scene / "images").mkdir(parents=True)
    (s.scene / "transforms.json").write_text('{"frames": []}')
    train, evaluate, export = pipeline.build_steps(s, ["train", "eval", "export"])

    command = train.command()
    timestamp = command[command.index("--timestamp") + 1]
    assert command[1] == "splatfacto" and command[-3:] == ["nerfstudio-data", "--downscale-factor", "2"]
    assert command[command.index("--max-num-iterations") + 1] == "1000"

    make_run(workspace, "prova", "splatfacto", timestamp, step=999)  # simula il training completato
    train.after(123.4)
    run = runs.latest_run("prova", "splatfacto")
    info = run.info()
    assert info["train_seconds"] == 123.4 and info["iterations"] == 1000 and info["downscale"] == 2
    assert info["alignment"] == runs.alignment_id(s.scene) and run.comparable() is True

    assert evaluate.command()[-1] == str(run.path / "metrics.json")
    assert export.command()[1] == "gaussian-splat" and export.command()[-1] == str(run.export_dir)


def test_big_variants_get_their_own_run_folder(workspace, photos):
    # Per nerfstudio "splatfacto-big" ha method_name "splatfacto": va forzato, o i run finiscono insieme.
    s = settings(photos, methods=["splatfacto-big"])
    (s.scene).mkdir(parents=True)
    (s.scene / "transforms.json").write_text("{}")
    (train,) = pipeline.build_steps(s, ["train"])
    command = train.command()
    assert command[1] == "splatfacto-big"
    assert command[command.index("--method-name") + 1] == "splatfacto-big"


def test_nerf_is_exported_as_point_cloud(workspace, photos):
    s = settings(photos, methods=["nerfacto"])
    make_run(workspace, "prova", "nerfacto", "2026-01-01_000000")
    (export,) = pipeline.build_steps(s, ["export"])
    command = export.command()
    assert command[1] == "pointcloud" and "--normal-method" in command


def test_runs_from_a_previous_alignment_are_not_evaluated(workspace, photos):
    s = settings(photos, methods=["splatfacto"])
    s.scene.mkdir(parents=True)
    (s.scene / "transforms.json").write_text("nuovo allineamento")
    run_dir = make_run(workspace, "prova", "splatfacto", "2026-01-01_000000")
    runs.update_json(run_dir / runs.RUN_FILE, alignment="impronta-vecchia")
    (evaluate,) = pipeline.build_steps(s, ["eval"])
    with pytest.raises(RuntimeError, match="allineamento precedente"):
        evaluate.command()


@pytest.mark.parametrize("overrides, groups, expected", [
    (dict(name="nome non valido"), ["sfm"], "nome del progetto"),
    (dict(), [], "almeno un passo"),
    (dict(photos=""), ["sfm"], "cartella che contenga le foto"),
    (dict(methods=[]), ["sfm", "train"], "almeno un metodo"),
    (dict(), ["train"], "prima l'allineamento"),
    (dict(), ["mesh"], "prima l'allineamento"),
    (dict(), ["eval"], "Nessun training esistente"),
    (dict(), ["sfm", "train", "eval", "export", "mesh"], None),
])
def test_validation(workspace, photos, overrides, groups, expected):
    s = settings(photos)
    for name, value in overrides.items():
        setattr(s, name, value)
    error = pipeline.validate(s, groups)
    assert (expected in error) if expected else error is None


def test_project_lock(workspace, photos, monkeypatch):
    scene = settings(photos).scene
    assert pipeline.acquire_lock(scene) is None
    assert pipeline.acquire_lock(scene) is None  # lo stesso processo puo' riprendere il proprio blocco

    (scene / ".lock").write_text("424242")
    monkeypatch.setattr(pipeline.psutil, "pid_exists", lambda pid: True)
    assert "già in elaborazione" in pipeline.acquire_lock(scene)

    monkeypatch.setattr(pipeline.psutil, "pid_exists", lambda pid: False)  # blocco lasciato da un processo terminato
    assert pipeline.acquire_lock(scene) is None
    pipeline.release_lock(scene)
    assert not (scene / ".lock").exists()
