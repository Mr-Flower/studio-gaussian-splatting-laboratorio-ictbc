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
    (5280, 0, 0, 1), (5280, 0, 3200, 2), (5280, 0, 1600, 4), (3072, 0, 1600, 2), (1600, 0, 1600, 1),
    (30000, 0, 1600, 8), (5280, 0, 800, 8), (5280, 2, 1600, 2), (5280, 1, 800, 1),
])
def test_training_resolution_follows_the_requested_limit(workspace, photos, longest, requested, max_side, expected):
    # La memoria non riduce mai la risoluzione: anche con 887 foto grandi vale solo il limite richiesto.
    s = settings(photos, downscale=requested, max_side=max_side)
    aligned(s, images=887, width=longest, height=longest * 3 // 4)
    assert pipeline.training_factor(s) == expected


@pytest.mark.parametrize("max_side, expected", [
    # 887 foto da 5280 x 3956 px, 64 GB di memoria e 48 GB sulla scheda grafica (il caso di studio).
    (0, {"splatfacto": "disk", "inria-3dgs": "disk", "nerfacto": "disk"}),      # piena: da 56 a 296 GB di foto
    (3200, {"splatfacto": "ram", "inria-3dgs": "disk", "nerfacto": "disk"}),    # 2640 px: 14, 74 e 55 GB
    (1600, {"splatfacto": "ram", "inria-3dgs": "gpu", "nerfacto": "ram"}),      # 1320 px: 3,5, 18,5 e 14 GB
])
def test_photos_that_do_not_fit_in_memory_are_read_from_disk(workspace, photos, max_side, expected):
    s = settings(photos, methods=list(expected), max_side=max_side)
    aligned(s, images=887)
    assert {key: pipeline.placement(s, METHODS[key]) for key in expected} == expected
    assert [m.key for m in pipeline.disk_methods(s)] == [k for k, where in expected.items() if where == "disk"]


def test_few_small_photos_stay_in_memory_at_full_resolution(workspace, photos):
    s = settings(photos, methods=list(METHODS))
    aligned(s, images=40, width=2000, height=1500)
    assert pipeline.training_factor(s) == 1 and pipeline.disk_methods(s) == []


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
    assert pipeline.photo_placement(inria, 887, 776, pixels) == "disk"
    monkeypatch.setattr(pipeline, "hardware", lambda: (0, 0))               # memoria non nota: nessun limite
    assert pipeline.photo_placement(inria, 887, 776, pixels) == "gpu"


def test_training_commands_when_photos_are_read_from_disk(workspace, photos):
    s = settings(photos, methods=["splatfacto-big", "nerfacto", "inria-3dgs"])
    make_sparse_model(s.scene / "colmap", "0", registered=5)
    aligned(s, images=887)
    (s.scene / colmap_model.SPLIT_FILE).write_text(json.dumps({"train": ["a"] * 776, "test": ["b"] * 111}))
    splat, nerf, _, inria = pipeline.build_steps(s, ["train"])

    command = splat.command()
    assert command[1] == "splatfacto-big-disco"  # la variante di app/ns_disk.py
    assert command[command.index("--method-name") + 1] == "splatfacto-big-disco"
    # Il run sta nella cartella della variante, ma resta un run del metodo di partenza.
    timestamp = command[command.index("--timestamp") + 1]
    folder = make_run(workspace, "prova", "splatfacto-big-disco", timestamp)
    (run,) = runs.list_runs("prova")
    assert (run.method, run.path) == ("splatfacto-big", folder) and run.export_dir.name == f"splatfacto-big_{timestamp}"
    splat.after(10.0)
    assert run.info()["photos_in"] == "disk" and (folder / runs.RUN_FILE).exists()
    assert "--pipeline.datamanager.cache-images" not in command

    command = nerf.command()
    assert command[1] == "nerfacto-disco"
    # 250 MB per foto in virgola mobile, tre copie al cambio di gruppo, meta' di 64 GB: 42 foto alla volta.
    assert command[command.index("--pipeline.datamanager.train-num-images-to-sample-from") + 1] == "45"
    assert command[command.index("--pipeline.datamanager.train-num-times-to-repeat-images") + 1] == "200"
    assert command.index("--pipeline.datamanager.train-num-images-to-sample-from") < command.index("nerfstudio-data")

    command = inria.command()
    assert command[1].endswith("inria_avvio.py") and command[2:4] == ["--disco", "train.py"]
    assert command[command.index("-r") + 1] == "1"

    # Le varianti sono note a nerfstudio tramite l'ambiente dei processi, e 'app' deve essere importabile.
    env = config.environment()
    assert "splatfacto-big-disco=app.ns_disk:splatfacto_big" in env["NERFSTUDIO_METHOD_CONFIGS"].split(",")
    assert "inria" not in env["NERFSTUDIO_METHOD_CONFIGS"] and str(config.ROOT) in env["PYTHONPATH"].split(os.pathsep)


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
    s = settings(photos, methods=["inria-3dgs"], max_side=1600)
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
    assert command[1] == "train.py" and train.cwd == config.inria_repo()  # in memoria: codice Inria com'e'
    assert command[command.index("-s") + 1] == str(dataset) and "--eval" in command
    # Senza "-r 1" il codice Inria riduce da solo a 1600 px le foto piu' grandi.
    assert command[command.index("-r") + 1] == "1" and command[command.index("--data_device") + 1] == "cuda"
    # Salvataggi intermedi, per guardare il modello durante il training: dieci, l'ultimo alla fine.
    first = command.index("--save_iterations") + 1
    assert command[first:command.index("--test_iterations")] == [str(i) for i in range(100, 1001, 100)]
    assert pipeline.inria_save_iterations(30000)[-2:] == [27000, 30000] and pipeline.inria_save_iterations(5) == [1, 2, 3, 4, 5]
    assert train.snapshot() is None  # nessun salvataggio ancora
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
    older = os.path.join(run_dir, "point_cloud", "iteration_900")
    os.makedirs(older)
    train.on_line("[ITER 1000] Saving Gaussians")  # i salvataggi precedenti vengono tolti
    assert not os.path.exists(older) and train.snapshot() == run.checkpoint()
    assert run.info()["photos_in"] == "gpu" and run.info()["downscale"] == 4

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


@pytest.mark.parametrize("pose, appearance", [(False, False), (True, True), (True, False)])
def test_protocol_options_are_always_explicit(workspace, photos, pose, appearance):
    s = settings(photos, methods=["splatfacto", "nerfacto-big", "inria-3dgs"], downscale=2,
                 pose_refinement=pose, appearance_embedding=appearance)
    aligned(s, images=40, width=800, height=600)
    splat, nerf, _, inria = pipeline.build_steps(s, ["train"])

    def option(command, name):
        return command[command.index(name) + 1] if name in command else None

    mode = "SO3xR3" if pose else "off"
    for command in (splat.command(), nerf.command()):
        assert option(command, "--pipeline.model.camera-optimizer.mode") == mode
        assert command.index("--pipeline.model.camera-optimizer.mode") < command.index("nerfstudio-data")
    assert option(splat.command(), "--pipeline.model.use-appearance-embedding") is None  # solo i NeRF
    assert option(nerf.command(), "--pipeline.model.use-appearance-embedding") == str(appearance)
    assert not any("camera-optimizer" in part or "appearance" in part for part in inria.command())

    # Le opzioni chieste finiscono nel run, anche per il metodo Inria che non puo' applicarle.
    timestamp = option(nerf.command(), "--timestamp")
    make_run(workspace, "prova", "nerfacto-big", timestamp)
    nerf.after(1.0)
    info = runs.latest_run("prova", "nerfacto-big").info()
    assert (info["pose_refinement"], info["appearance_embedding"]) == (pose, appearance)


def test_runs_with_different_protocol_options_are_not_compared(workspace):
    scene = workspace / "data" / "prova"
    scene.mkdir(parents=True)
    (scene / "transforms.json").write_text("allineamento")

    def trained(method, timestamp, family, age, **options):
        run_dir = make_run(workspace, "prova", method, timestamp)
        os.utime(next((run_dir / "nerfstudio_models").iterdir()), (10000 - age, 10000 - age))
        runs.update_json(run_dir / runs.RUN_FILE, family=family, downscale=1, iterations=30000,
                         alignment=runs.alignment_id(scene), **options)
        (run_dir / runs.METRICS_FILE).write_text(json.dumps({"results": {"psnr": 20.0}}))

    # Run della 0.3: senza le opzioni registrate. Per il 3DGS equivalgono a opzioni spente.
    trained("splatfacto", "2026-01-01_000000", "gaussian", age=5)
    # NeRF della 0.3: valevano i default di nerfstudio, pose corrette e aspetto per foto.
    trained("nerfacto", "2026-01-02_000000", "nerf", age=4)
    trained("nerfacto-big", "2026-01-03_000000", "nerf", age=3, pose_refinement=False, appearance_embedding=False)
    trained("splatfacto-big", "2026-01-04_000000", "gaussian", age=2, pose_refinement=False, appearance_embedding=False)

    group = runs.comparable_group(runs.list_runs("prova"))
    assert sorted(r.method for r in group) == ["nerfacto-big", "splatfacto", "splatfacto-big"]
    assert pipeline.run_conditions({"family": "nerf"})[2:] == (True, True)
    assert pipeline.run_conditions({"family": "gaussian"})[2:] == (False, False)


def test_gpu_peak_keeps_the_highest_reading():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QCoreApplication
    from app import runner

    app = QCoreApplication.instance() or QCoreApplication([])  # noqa: F841 - serve agli oggetti Qt
    monitor = runner.GpuMonitor()
    monitor.stop()  # la prima lettura vera, se c'e' una scheda, si aggiunge alle altre
    first = monitor.peak
    for reading in ("1200", "5300", "N/A", "4100"):
        monitor.feed(reading + "\n")
    assert monitor.peak == max(first, 5300)


@pytest.mark.parametrize("seed", [None, 7])
def test_seed_reaches_every_training_program(workspace, photos, seed):
    s = settings(photos, methods=["splatfacto", "inria-3dgs"], downscale=2, seed=seed)
    make_sparse_model(s.scene / "colmap", "0", registered=5)
    aligned(s, images=40, width=800, height=600)
    splat, _, inria = pipeline.build_steps(s, ["train"])
    command = splat.command()
    assert (command[command.index("--machine.seed") + 1] == "7") if seed else "--machine.seed" not in command
    command = inria.command()
    if seed:  # il codice Inria fissa il seme a 0: lo si cambia con il lanciatore, senza modificarlo
        assert command[1].endswith("inria_avvio.py") and command[2:5] == ["--seme", "7", "train.py"]
    else:
        assert command[1] == "train.py"

    timestamp = command[command.index("-m") + 1].rsplit(os.sep, 1)[-1]
    saved = os.path.join(command[command.index("-m") + 1], "point_cloud", "iteration_1000")
    os.makedirs(saved)
    open(os.path.join(saved, "point_cloud.ply"), "wb").write(b"ply\nelement vertex 7\nend_header\n")
    inria.after(1.0)
    assert runs.Run("prova", "inria-3dgs", timestamp).info()["seed"] == (7 if seed else 0)  # 0: quello di Inria


def test_inria_launcher_reseeds_after_safe_state(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from app import inria_avvio

    (tmp_path / "utils").mkdir()
    (tmp_path / "utils" / "__init__.py").write_text("")
    (tmp_path / "utils" / "general_utils.py").write_text(
        "import random\ndef safe_state(silent):\n    random.seed(0)\n")
    (tmp_path / "train.py").write_text(
        "import random, sys\nfrom utils.general_utils import safe_state\nsafe_state(False)\n"
        "print('argomenti', sys.argv[1:], 'valore', random.random())\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    for name in ("utils", "utils.general_utils"):
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    inria_avvio.main(["--seme", "7", "train.py", "-s", "dati", "--eval"])
    out = capsys.readouterr().out
    import random
    # Gli argomenti arrivano allo script e il primo numero casuale e' quello del seme 7, non dello 0 di safe_state.
    assert "['-s', 'dati', '--eval']" in out and f"valore {random.Random(7).random()}" in out


def test_reevaluation_of_existing_runs_with_common_sides(workspace, photos):
    s = settings(photos, eval_sides=[2640, 1320])
    s.scene.mkdir(parents=True)
    (s.scene / colmap_model.SPLIT_FILE).write_text("{}")
    make_run(workspace, "prova", "splatfacto", "2026-01-01_000000")
    make_run(workspace, "prova", "splatfacto", "2026-01-02_000000")
    targets = runs.list_runs("prova")
    steps = pipeline.reevaluation_steps(s, targets)
    assert len(steps) == 2 and all(step.group == "eval" for step in steps)
    command = steps[0].command()
    assert command[command.index("--run") + 1] == str(targets[0].path)
    assert [command[i + 1] for i, part in enumerate(command) if part == "--eval-side"] == ["2640", "1320"]
    steps[1].after(12.0)
    assert targets[1].info()["eval_sides"] == [2640, 1320]
