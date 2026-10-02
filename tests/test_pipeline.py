import json
import os

import pytest

from app import colmap_model, config, pipeline, runs
from app import photos as photo_analysis
from app.config import METHODS, Settings

from .conftest import GB, make_run, make_sparse_model


def settings(photos, **overrides):
    values = dict(name="prova", photos=str(photos), methods=["splatfacto", "nerfacto"], iterations=1000)
    values.update(overrides)
    return Settings(**values)


def labels(steps):
    return [step.label for step in steps]


def test_full_pipeline_order(workspace, photos):
    steps = pipeline.build_steps(settings(photos), ["sfm", "train", "eval", "export", "mesh", "report"])
    splat, nerf = METHODS["splatfacto"].label, METHODS["nerfacto"].label
    assert labels(steps) == [
        "Estrazione delle feature", "Matching tra le foto", "Ricostruzione delle camere",
        "Verifica dell'allineamento", "Conversione per nerfstudio",
        f"Training — {splat}", f"Valutazione — {splat}", f"Esportazione — {splat}",
        f"Training — {nerf}", f"Valutazione — {nerf}", f"Esportazione — {nerf}",
        "Mesh: correzione delle immagini", "Mesh: mappe di profondità", "Mesh: fusione in nuvola densa",
        "Mesh: superficie di Poisson",
        "Report del confronto",
    ]


def test_excluded_photos_are_left_out_of_the_alignment(workspace, photos):
    s = settings(photos)
    photo_analysis.set_excluded(s.scene, {"img_1.jpg", "img_3.jpg"})
    steps = pipeline.build_steps(s, ["sfm"])
    steps[0].before()
    command = steps[0].command()
    image_list = command[command.index("--image_list_path") + 1]
    assert open(image_list).read().split() == ["img_0.jpg", "img_2.jpg", "img_4.jpg"]

    photo_analysis.set_excluded(s.scene, {"img_0.jpg", "img_1.jpg", "img_2.jpg"})
    assert "meno di tre foto" in pipeline.validate(s, ["sfm"])


def aligned(s, images=1, width=5280, height=3956):
    s.scene.mkdir(parents=True, exist_ok=True)
    (s.scene / "transforms.json").write_text(json.dumps({"frames": [{"w": width, "h": height}] * images}))


@pytest.mark.parametrize("longest, requested, max_side, expected", [
    (5280, 0, 1600, 4), (3072, 0, 1600, 2), (1600, 0, 1600, 1), (30000, 0, 1600, 8), (5280, 0, 800, 8),
    (5280, 0, 0, 1), (5280, 2, 1600, 2), (5280, 1, 800, 1),
])
def test_training_resolution_follows_the_requested_limit(workspace, photos, longest, requested, max_side, expected):
    s = settings(photos, downscale=requested, max_side=max_side)
    aligned(s, width=longest, height=longest * 3 // 4)
    assert pipeline.training_factor(s) == expected


@pytest.mark.parametrize("methods, expected, limited_by", [
    # 887 foto da 5280 x 3956 px, 64 GB di memoria e 48 GB sulla scheda grafica (il caso di studio).
    (["splatfacto"], 2, "splatfacto"),          # 3 byte per pixel: a risoluzione piena 55 GB, a meta' 14
    (["splatfacto-big"], 2, "splatfacto-big"),
    (["inria-3dgs"], 4, "inria-3dgs"),          # 16 byte per pixel: a meta' risoluzione 74 GB
    (["nerfacto"], 4, "nerfacto"),              # 12 byte per pixel: a meta' risoluzione 55 GB
    (["splatfacto", "inria-3dgs"], 4, "inria-3dgs"),  # insieme: la risoluzione che regge anche il piu' esigente
])
def test_best_resolution_is_the_highest_that_fits_in_memory(workspace, photos, methods, expected, limited_by):
    s = settings(photos, methods=methods)
    aligned(s, images=887)
    factor, culprit = pipeline.training_plan(s)
    assert factor == expected and culprit == METHODS[limited_by].label
    assert pipeline.training_factor(s, ["splatfacto"]) == 2  # i metodi si possono indicare a parte


def test_few_small_photos_train_at_full_resolution(workspace, photos):
    s = settings(photos, methods=list(METHODS))
    aligned(s, images=40, width=2000, height=1500)
    assert pipeline.training_plan(s) == (1, None)


def test_where_the_photos_are_kept_during_training(workspace, monkeypatch):
    inria, splat, nerf = METHODS["inria-3dgs"], METHODS["splatfacto"], METHODS["nerfacto"]
    pixels = 1320 * 989
    assert pipeline.photo_placement(inria, 887, 776, pixels) == "gpu"      # 18 GB su 48
    assert pipeline.photo_placement(splat, 887, 776, pixels) == "ram"      # oltre 500 foto nerfstudio usa la RAM
    assert pipeline.photo_placement(splat, 100, 88, pixels) == "gpu"
    assert pipeline.photo_placement(nerf, 100, 88, pixels) == "ram"
    monkeypatch.setattr(pipeline, "hardware", lambda: (64 * GB, 24 * GB))
    assert pipeline.photo_placement(inria, 887, 776, pixels) == "ram"      # non entra in 12 GB: memoria centrale
    monkeypatch.setattr(pipeline, "hardware", lambda: (16 * GB, 24 * GB))
    assert pipeline.photo_placement(inria, 887, 776, pixels) is None
    monkeypatch.setattr(pipeline, "hardware", lambda: (0, 0))               # memoria non nota: nessun limite
    assert pipeline.photo_placement(inria, 887, 776, pixels) == "gpu"


def test_resolution_is_known_before_the_alignment(workspace, tmp_path):
    from PIL import Image
    folder = tmp_path / "vere"
    folder.mkdir()
    for i in range(16):
        Image.new("RGB", (400, 300)).save(folder / f"img_{i:02d}.jpg")
    s = settings(folder)
    assert pipeline.photo_stats(s) == (16, 14, 400, 300)  # una foto su otto va al test
    assert pipeline.training_factor(s) == 1


def test_inria_method_steps(workspace, photos, monkeypatch):
    s = settings(photos, methods=["inria-3dgs"])
    make_sparse_model(s.scene / "colmap", "0", registered=5)
    (s.scene / "transforms.json").write_text(json.dumps({"frames": [{"w": 5280, "h": 3956}] * 887}))
    (s.scene / colmap_model.SPLIT_FILE).write_text(json.dumps({"train": ["a"], "test": ["b"]}))
    prepare, train, evaluate, export = pipeline.build_steps(s, ["train", "eval", "export"])

    undistort = prepare.command()
    dataset = s.scene / "inria_4"
    assert undistort[1] == "image_undistorter" and undistort[undistort.index("--max_image_size") + 1] == "1320"
    (dataset / "sparse").mkdir(parents=True)
    (dataset / "sparse" / "cameras.bin").touch()
    prepare.after(1.0)
    assert (dataset / "sparse" / "0" / "cameras.bin").exists()
    assert prepare.command() is None  # gia' pronta per questo allineamento: non si rifa'

    command = train.command()
    assert command[1] == "train.py" and train.cwd == config.inria_repo()
    assert command[command.index("-s") + 1] == str(dataset) and "--eval" in command
    # Senza "-r 1" il codice Inria riduce da solo a 1600 px le foto piu' grandi.
    assert command[command.index("-r") + 1] == "1" and command[command.index("--data_device") + 1] == "cuda"
    monkeypatch.setattr(pipeline, "hardware", lambda: (64 * GB, 24 * GB))
    assert train.command()[train.command().index("--data_device") + 1] == "cpu"
    monkeypatch.setattr(pipeline, "hardware", lambda: (64 * GB, 48 * GB))

    run_dir = command[command.index("-m") + 1]
    saved = os.path.join(run_dir, "point_cloud", "iteration_1000")
    os.makedirs(saved)
    open(os.path.join(saved, "point_cloud.ply"), "wb").write(b"ply\nelement vertex 7\nend_header\n")
    train.after(60.0)
    run = runs.latest_run("prova", "inria-3dgs")
    assert run.trained_iterations() == 1000 and run.info()["engine"] == "inria"

    assert evaluate.command()[3:5] == ["--engine", "inria"] and "--repo" in evaluate.command()
    assert export.command() is None  # copia diretta del .ply, senza processo esterno
    export.before()
    assert runs.ply_elements(run.export_dir / "splat.ply") == 7


def test_alignment_uses_colmap_4_options_and_camera_mode(workspace, photos):
    steps = pipeline.build_steps(settings(photos, camera="per_folder"), ["sfm"])
    features = steps[0].command()
    assert "--FeatureExtraction.use_gpu" in features and "--SiftExtraction.use_gpu" not in features
    assert "--ImageReader.single_camera_per_folder" in features
    assert steps[1].command()[1] == "exhaustive_matcher"


def test_automatic_camera_mode_follows_the_folder_layout(workspace, photos, tmp_path):
    flat = pipeline.build_steps(settings(photos), ["sfm"])[0].command()
    assert "--ImageReader.single_camera" in flat
    nested = tmp_path / "due_camere"
    for folder in ("jpg", "dng"):
        (nested / folder).mkdir(parents=True)
        for i in range(3):
            (nested / folder / f"img_{i}.jpg").touch()
    assert "--ImageReader.single_camera_per_folder" in pipeline.build_steps(settings(nested), ["sfm"])[0].command()
    forced = pipeline.build_steps(settings(nested, camera="single"), ["sfm"])[0].command()
    assert "--ImageReader.single_camera" in forced


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
    s.scene.mkdir(parents=True)
    (s.scene / "transforms.json").write_text('{"frames": []}')
    (s.scene / colmap_model.SPLIT_FILE).write_text(json.dumps({"train": ["a", "b"], "test": ["c"]}))
    train, evaluate, export = pipeline.build_steps(s, ["train", "eval", "export"])

    command = train.command()
    timestamp = command[command.index("--timestamp") + 1]
    assert command[1] == "splatfacto" and command[-3:] == ["nerfstudio-data", "--downscale-factor", "2"]
    assert "--pipeline.datamanager.cache-images" not in command  # poche foto: restano sulla scheda grafica
    assert command[command.index("--max-num-iterations") + 1] == "1000"

    make_run(workspace, "prova", "splatfacto", timestamp, step=999)  # simula il training completato
    train.after(123.4)
    run = runs.latest_run("prova", "splatfacto")
    info = run.info()
    assert info["train_seconds"] == 123.4 and info["iterations"] == 1000 and info["downscale"] == 2
    assert info["alignment"] == runs.alignment_id(s.scene) and run.comparable() is True
    assert (info["train_images"], info["test_images"]) == (2, 1)

    # Stessa valutazione per tutti i metodi: il nostro modulo, non quello del programma di training.
    assert evaluate.command()[1:5] == ["-m", "app.evaluate", "--engine", "nerfstudio"]
    assert evaluate.command()[evaluate.command().index("--run") + 1] == str(run.path)
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
    (s.scene / colmap_model.SPLIT_FILE).write_text("{}")
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
