"""Costruzione dei passi della pipeline: allineamento, training, valutazione, esportazione, mesh.

Ogni passo e' un comando esterno (COLMAP o nerfstudio). Qui non si esegue nulla: l'esecuzione
e' in runner.py, usato sia dall'interfaccia grafica sia dalla riga di comando.
"""
from __future__ import annotations

import functools
import os
import shutil
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import psutil

from . import colmap_model, config, photos, progress, runs
from .config import METHODS, Method, Settings
from .runs import Run


@dataclass
class Step:
    group: str
    label: str
    command: Callable[[], Optional[List[str]]]  # None: il passo fa tutto in `before`, senza processo esterno
    progress: Optional[progress.Fraction] = None
    cwd: Optional[Path] = None  # cartella di lavoro del processo (predefinita: radice del repository)
    detail: Optional[Callable[[str], Optional[str]]] = None  # testo di stato ricavato da una riga di log
    on_line: Optional[Callable[[str], None]] = None
    quiet: Optional[Callable[[str], bool]] = None  # righe da non mostrare nel log a video
    before: Optional[Callable[[], None]] = None
    after: Optional[Callable[[float], None]] = None  # riceve la durata del passo in secondi


# --- stato su disco

def state(s: Settings) -> Dict[str, bool]:
    return {
        "sfm": (s.scene / "transforms.json").exists(),
        "mesh": (s.scene / "colmap" / "dense" / "mesh-poisson.ply").exists(),
    }


def best_sparse_model(colmap_dir: Path) -> Path:
    """Tra i modelli prodotti dal mapper sceglie quello con piu' immagini registrate."""

    def registered(model: Path) -> int:
        with open(model / "images.bin", "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]

    sparse = colmap_dir / "sparse"
    models = [d for d in sparse.iterdir() if (d / "images.bin").exists()] if sparse.is_dir() else []
    if not models:
        raise RuntimeError("COLMAP non ha prodotto nessuna ricostruzione: le foto non si sovrappongono abbastanza.")
    return max(models, key=registered)


@functools.lru_cache(maxsize=1)
def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW).stdout
        return out.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


# --- blocco del progetto: una sola elaborazione alla volta per progetto

def acquire_lock(scene: Path) -> Optional[str]:
    """Blocca il progetto per questo processo. Restituisce un messaggio se e' gia' in uso."""
    lock = scene / ".lock"
    try:
        pid = int(lock.read_text())
    except (OSError, ValueError):
        pid = None
    if pid and pid != os.getpid() and psutil.pid_exists(pid):
        return f"Il progetto è già in elaborazione in un'altra finestra (processo {pid})."
    scene.mkdir(parents=True, exist_ok=True)
    lock.write_text(str(os.getpid()))
    return None


def release_lock(scene: Path) -> None:
    (scene / ".lock").unlink(missing_ok=True)


# --- controlli prima dell'avvio

def validate(s: Settings, groups: List[str]) -> Optional[str]:
    """Messaggio di errore se la richiesta non e' eseguibile, altrimenti None."""
    if not config.PROJECT_NAME.fullmatch(s.name):
        return "Il nome del progetto può contenere solo lettere, numeri, '-' e '_'."
    if not groups:
        return "Seleziona almeno un passo."
    unknown = [g for g in groups if g not in config.GROUPS]
    if unknown:
        return f"Passi sconosciuti: {', '.join(unknown)}"
    if ("sfm" in groups or "mesh" in groups) and config.count_images(s.photos) < 3:
        return "Scegli una cartella che contenga le foto."
    if "sfm" in groups and len(photos.kept_images(s.scene, s.photos)) < 3:
        return "Dopo le esclusioni restano meno di tre foto."
    per_method = [g for g in groups if g in ("train", "eval", "export")]
    if per_method and not s.methods:
        return "Seleziona almeno un metodo."
    unknown = [m for m in s.methods if m not in METHODS]
    if unknown:
        return f"Metodi sconosciuti: {', '.join(unknown)}"
    if per_method:
        unavailable = [METHODS[m].label for m in s.methods if m not in config.available_methods()]
        if unavailable:
            return "Metodi non installati: " + ", ".join(unavailable)
    aligned = "sfm" in groups or state(s)["sfm"]
    if ("train" in groups or "mesh" in groups) and not aligned:
        return "Serve prima l'allineamento delle foto."
    if per_method and "train" not in groups:
        missing = [METHODS[m].label for m in s.methods if runs.latest_run(s.name, m) is None]
        if missing:
            return "Nessun training esistente per: " + ", ".join(missing)
    return None


# --- passi

def _sfm_steps(s: Settings) -> List[Step]:
    colmap = str(config.colmap_exe())
    scene = s.scene
    work = scene / "colmap.tmp"  # si lavora a parte: l'allineamento esistente resta valido fino al successo
    final = scene / "colmap"
    db = str(work / "database.db")
    stats = progress.AlignmentStats()
    names = photos.kept_images(scene, s.photos)
    excluded = config.count_images(s.photos) - len(names)
    image_list = work / "image_list.txt"
    camera_flag = {
        "single": "--ImageReader.single_camera",
        "per_folder": "--ImageReader.single_camera_per_folder",
        "per_image": "--ImageReader.single_camera_per_image",
    }[s.camera]

    def prepare() -> None:
        shutil.rmtree(work, ignore_errors=True)
        (work / "sparse").mkdir(parents=True)
        image_list.write_text("\n".join(names) + "\n", encoding="utf-8")

    def swap() -> None:
        (scene / "transforms.json").unlink(missing_ok=True)
        (scene / runs.ALIGNMENT_FILE).unlink(missing_ok=True)
        if final.exists():
            shutil.rmtree(final)
        work.rename(final)

    def record(_seconds: float) -> None:
        split = colmap_model.write_split(scene, best_sparse_model(final))
        runs.update_json(scene / runs.ALIGNMENT_FILE, photos=len(names), excluded=excluded,
                         models=len(list((final / "sparse").iterdir())),
                         train_images=len(split["train"]), test_images=len(split["test"]),
                         camera=s.camera, matcher=s.matcher, **stats.values)

    # Non si usa il COLMAP integrato in ns-process-data: nerfstudio 1.1.5 passa opzioni
    # (--SiftExtraction.use_gpu) che in COLMAP 4.x non esistono piu'.
    return [
        Step("sfm", "Estrazione delle feature",
             lambda: [colmap, "feature_extractor", "--database_path", db, "--image_path", s.photos,
                      "--image_list_path", str(image_list),
                      "--ImageReader.camera_model", "OPENCV", camera_flag, "1", "--FeatureExtraction.use_gpu", "1"],
             progress.ratio(r"Processed file \[(\d+)/(\d+)\]"), before=prepare),
        Step("sfm", "Matching tra le foto",
             lambda: [colmap, f"{s.matcher}_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "1"],
             progress.matching),
        Step("sfm", "Ricostruzione delle camere",
             lambda: [colmap, "mapper", "--database_path", db, "--image_path", s.photos,
                      "--output_path", str(work / "sparse")],
             progress.mapper(len(names))),
        Step("sfm", "Verifica dell'allineamento",
             lambda: [colmap, "model_analyzer", "--path", str(best_sparse_model(work))],
             on_line=stats.feed),
        Step("sfm", "Conversione per nerfstudio",
             lambda: [config.ns("ns-process-data"), "images", "--data", s.photos, "--output-dir", str(scene),
                      "--skip-colmap", "--colmap-model-path", f"colmap/sparse/{best_sparse_model(final).name}"],
             before=swap, after=record),
    ]


def image_size(scene: Path) -> int:
    """Lato maggiore, in pixel, delle foto allineate."""
    transforms = runs.read_json(scene / "transforms.json")
    first = (transforms.get("frames") or [{}])[0]
    return max(transforms.get("w") or first.get("w", 0), transforms.get("h") or first.get("h", 0))


def training_factor(scene: Path, requested: int) -> int:
    """Fattore di riduzione delle immagini, uguale per tutti i metodi.

    Con 0 (automatico) si applica la regola di nerfstudio: si dimezza finche' il lato maggiore
    non scende a 1600 pixel o meno. Il fattore e' sempre passato in modo esplicito ai programmi
    di training, cosi' metodi diversi lavorano alla stessa risoluzione.
    """
    if requested:
        return requested
    factor = 1
    while image_size(scene) / factor > 1600 and factor < 8:
        factor *= 2
    return factor


def inria_dataset(scene: Path, factor: int) -> Path:
    return scene / f"inria_{factor}"


def _inria_prepare_step(s: Settings) -> Step:
    """Il codice Inria legge solo camere senza distorsione: gli si prepara una copia corretta delle foto."""
    scene = s.scene

    def dataset() -> Path:
        return inria_dataset(scene, training_factor(scene, s.downscale))

    def command() -> Optional[List[str]]:
        target = dataset()
        if (target / "alignment.txt").exists() and (target / "alignment.txt").read_text() == runs.alignment_id(scene):
            return None  # gia' pronta per questo allineamento
        shutil.rmtree(target, ignore_errors=True)
        size = round(image_size(scene) / training_factor(scene, s.downscale))
        return [str(config.colmap_exe()), "image_undistorter", "--image_path", s.photos,
                "--input_path", str(best_sparse_model(scene / "colmap")), "--output_path", str(target),
                "--output_type", "COLMAP", "--max_image_size", str(size)]

    def finish(_seconds: float) -> None:
        target = dataset()
        if (target / "alignment.txt").exists():
            return
        model = target / "sparse" / "0"  # il codice Inria si aspetta il modello in sparse/0
        model.mkdir(exist_ok=True)
        for item in (target / "sparse").iterdir():
            if item.is_file():
                item.rename(model / item.name)
        (target / "alignment.txt").write_text(runs.alignment_id(scene))

    return Step("train", "Preparazione delle foto per il metodo Inria", command,
                progress.ratio(r"Undistorting image \[(\d+)/(\d+)\]"), after=finish)


def _method_steps(s: Settings, method: Method, groups: List[str]) -> List[Step]:
    scene = s.scene
    inria = method.engine == "inria"
    trained: Optional[Run] = runs.new_run(s.name, method.key) if "train" in groups else None
    steps: List[Step] = []

    def target() -> Run:
        run = trained or runs.latest_run(s.name, method.key)
        if run is None or run.checkpoint() is None:
            raise RuntimeError(f"Nessun training disponibile per {method.label}.")
        if run.comparable() is False:
            raise RuntimeError(f"L'ultimo training di {method.label} usa un allineamento precedente: va rifatto.")
        return run

    def ensure_split() -> None:
        # Progetti allineati prima che esistesse la suddivisione comune: la si crea ora. Cambia
        # l'impronta dell'allineamento, quindi i run precedenti risultano (giustamente) non confrontabili.
        if not (scene / colmap_model.SPLIT_FILE).exists():
            colmap_model.write_split(scene, best_sparse_model(scene / "colmap"))

    def evaluation_command() -> List[str]:
        if not (scene / colmap_model.SPLIT_FILE).exists():
            raise RuntimeError("Il progetto non ha la suddivisione training/test comune: va rifatto il training.")
        return [config.python(), "-m", "app.evaluate", "--engine", method.engine, "--run", str(target().path),
                "--split", str(scene / colmap_model.SPLIT_FILE)] + (["--repo", str(config.inria_repo())] if inria else [])

    if trained is not None:

        def record_training(seconds: float) -> None:
            split = colmap_model.read_split(scene)
            trained.update(method=method.key, family=method.family, engine=method.engine, iterations=s.iterations,
                           downscale=training_factor(scene, s.downscale), train_seconds=round(seconds, 1),
                           alignment=runs.alignment_id(scene), train_images=len(split["train"]),
                           test_images=len(split["test"]), gpu=gpu_name(), versions=config.versions(),
                           finished=time.strftime("%Y-%m-%dT%H:%M:%S"))

        if inria:
            prepare = _inria_prepare_step(s)
            prepare.before = ensure_split
            steps.append(prepare)
            steps.append(Step(
                "train", f"Training — {method.label}",
                lambda: [config.python(), "train.py", "-s", str(inria_dataset(scene, training_factor(scene, s.downscale))),
                         "-m", str(trained.path), "--eval", "--iterations", str(s.iterations),
                         "--save_iterations", str(s.iterations), "--test_iterations", str(s.iterations),
                         "--disable_viewer"],
                progress.percent(r"Training progress:\s*(\d+)%"),
                cwd=config.inria_repo(), quiet=lambda line: "Training progress" in line, after=record_training))
        else:
            # --method-name: senza, le varianti "-big" scrivono nella cartella del metodo base
            # (per nerfstudio splatfacto-big si chiama "splatfacto") e i run si confonderebbero.
            steps.append(Step(
                "train", f"Training — {method.label}",
                lambda: [config.ns("ns-train"), method.key, "--data", str(scene),
                         "--output-dir", str(config.ROOT / "outputs"), "--experiment-name", s.name,
                         "--method-name", method.key,
                         "--timestamp", trained.timestamp, "--max-num-iterations", str(s.iterations),
                         "--viewer.quit-on-train-completion", "True",
                         "nerfstudio-data", "--downscale-factor", str(training_factor(scene, s.downscale))],
                progress.train, detail=progress.train_detail(s.iterations), quiet=progress.is_train_table,
                before=ensure_split, after=record_training))

    if "eval" in groups:
        steps.append(Step(
            "eval", f"Valutazione — {method.label}",
            evaluation_command, progress.ratio(r"vista (\d+)/(\d+)"),
            after=lambda seconds: target().update(eval_seconds=round(seconds, 1))))

    if "export" in groups:

        def copy_inria_model() -> None:
            run = target()
            run.export_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run.checkpoint(), run.export_dir / method.export_file)

        def export_command() -> Optional[List[str]]:
            if inria:  # il modello Inria e' gia' un .ply nel formato standard: basta copiarlo
                return None
            run = target()
            command = [config.ns("ns-export"), method.export_kind, "--load-config", str(run.config_file),
                       "--output-dir", str(run.export_dir)]
            if method.family == "nerf":
                command += ["--num-points", "1000000", "--normal-method", "open3d"]
            return command

        steps.append(Step("export", f"Esportazione — {method.label}", export_command,
                          before=copy_inria_model if inria else None,
                          after=lambda seconds: target().update(export_seconds=round(seconds, 1))))
    return steps


def _mesh_steps(s: Settings) -> List[Step]:
    colmap = str(config.colmap_exe())
    scene = s.scene
    dense = scene / "colmap" / "dense"

    def reset() -> None:
        shutil.rmtree(dense, ignore_errors=True)

    def timed(phase: str) -> Callable[[float], None]:
        def record(seconds: float) -> None:
            data = runs.read_json(dense / runs.MESH_FILE)
            data.setdefault("seconds", {})[phase] = round(seconds, 1)
            runs.update_json(dense / runs.MESH_FILE, max_image_size=s.mesh_size, seconds=data["seconds"])

        return record

    return [
        Step("mesh", "Mesh: correzione delle immagini",
             lambda: [colmap, "image_undistorter", "--image_path", s.photos,
                      "--input_path", str(best_sparse_model(scene / "colmap")), "--output_path", str(dense),
                      "--output_type", "COLMAP", "--max_image_size", str(s.mesh_size)],
             progress.ratio(r"Undistorting image \[(\d+)/(\d+)\]"), before=reset, after=timed("undistortion")),
        Step("mesh", "Mesh: mappe di profondità",
             lambda: [colmap, "patch_match_stereo", "--workspace_path", str(dense),
                      "--PatchMatchStereo.geom_consistency", "1"],
             progress.patch_match(), after=timed("patch_match_stereo")),
        Step("mesh", "Mesh: fusione in nuvola densa",
             lambda: [colmap, "stereo_fusion", "--workspace_path", str(dense), "--output_path", str(dense / "fused.ply")],
             after=timed("stereo_fusion")),
        Step("mesh", "Mesh: superficie di Poisson",
             lambda: [colmap, "poisson_mesher", "--input_path", str(dense / "fused.ply"),
                      "--output_path", str(dense / "mesh-poisson.ply")],
             after=timed("poisson_mesher")),
    ]


def build_steps(s: Settings, groups: List[str]) -> List[Step]:
    steps: List[Step] = []
    if "sfm" in groups:
        steps += _sfm_steps(s)
    if any(g in groups for g in ("train", "eval", "export")):
        for key in s.methods:
            steps += _method_steps(s, METHODS[key], groups)
    if "mesh" in groups:
        steps += _mesh_steps(s)
    if "report" in groups:
        steps.append(Step("report", "Report del confronto",
                          lambda: [config.python(), "-m", "app.report", "--project", s.name]))
    return steps
