"""Passi della pipeline foto -> gaussian splat -> mesh, usati dall'interfaccia grafica.

Ogni passo e' un comando esterno (COLMAP o nerfstudio) con una funzione che ricava
l'avanzamento dalle righe di log. Layout su disco, relativo alla radice del progetto:
data/<nome> (allineamento e immagini), outputs/<nome> (training), exports/<nome> (splat.ply).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
VENV_SCRIPTS = ROOT / ".venv" / "Scripts"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

GROUPS = {
    "sfm": "Allineamento delle foto (COLMAP)",
    "train": "Training del gaussian splat",
    "export": "Esportazione del file .ply",
    "mesh": "Mesh (ricostruzione densa, lenta)",
}


def _tool(pattern: str) -> Path:
    found = sorted(TOOLS.glob(pattern))
    if not found:
        raise FileNotFoundError(f"Componente mancante in {TOOLS}: {pattern}")
    return found[-1]


def colmap_exe() -> Path:
    return _tool("colmap-*/bin/colmap.exe")


def environment() -> dict:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(colmap_exe().parent), str(_tool("ffmpeg-*/bin")), str(TOOLS / "git" / "cmd"), str(VENV_SCRIPTS), env.get("PATH", "")]
    )
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["COLUMNS"] = "160"
    return env


@dataclass
class Settings:
    name: str = ""
    photos: str = ""
    camera: str = "single"  # single | per_folder | per_image
    matcher: str = "exhaustive"  # exhaustive | sequential
    method: str = "splatfacto"
    iterations: int = 30000
    downscale: int = 0  # 0 = scelta automatica di nerfstudio (lato massimo 1600 px)
    mesh_size: int = 1600

    @property
    def scene(self) -> Path:
        return ROOT / "data" / self.name

    @property
    def export_dir(self) -> Path:
        return ROOT / "exports" / self.name

    def save(self) -> None:
        self.scene.mkdir(parents=True, exist_ok=True)
        (self.scene / "project.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, name: str) -> Optional["Settings"]:
        path = ROOT / "data" / name / "project.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Step:
    group: str
    label: str
    command: Callable[[], List[str]]
    progress: Optional[Callable[[str], Optional[float]]] = None
    before: Optional[Callable[[], None]] = None
    detail: Optional[Callable[[str], Optional[str]]] = None  # testo di stato ricavato da una riga di log


def count_images(folder: str) -> int:
    path = Path(folder)
    if not folder or not path.is_dir():
        return 0
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def best_sparse_model(scene: Path) -> Path:
    """Tra i modelli prodotti dal mapper sceglie quello con piu' immagini registrate."""

    def registered(model: Path) -> int:
        with open(model / "images.bin", "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]

    models = [d for d in (scene / "colmap" / "sparse").iterdir() if (d / "images.bin").exists()]
    if not models:
        raise RuntimeError("COLMAP non ha prodotto nessuna ricostruzione: le foto non si sovrappongono abbastanza.")
    return max(models, key=registered)


def latest_config(name: str) -> Optional[Path]:
    configs = sorted((ROOT / "outputs" / name).glob("**/config.yml"), key=lambda p: p.stat().st_mtime)
    return configs[-1] if configs else None


def done(s: Settings) -> dict:
    """Quali gruppi di passi hanno gia' un risultato su disco."""
    config = latest_config(s.name) if s.name else None
    return {
        "sfm": (s.scene / "transforms.json").exists(),
        "train": bool(config and any((config.parent / "nerfstudio_models").glob("*.ckpt"))),
        "export": (s.export_dir / "splat.ply").exists(),
        "mesh": (s.scene / "colmap" / "dense" / "mesh-poisson.ply").exists(),
    }


def _ratio(pattern: str) -> Callable[[str], Optional[float]]:
    rx = re.compile(pattern)

    def parse(line: str) -> Optional[float]:
        m = rx.search(line)
        return int(m.group(1)) / max(int(m.group(2)), 1) if m else None

    return parse


def _matching_progress(line: str) -> Optional[float]:
    m = re.search(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]", line)
    if m:
        a, rows, b, cols = map(int, m.groups())
        return ((a - 1) * cols + b) / (rows * cols)
    return _ratio(r"\[(\d+)/(\d+)\]")(line)


def _mapper_progress(total: int) -> Callable[[str], Optional[float]]:
    def parse(line: str) -> Optional[float]:
        m = re.search(r"num_reg_frames=(\d+)", line)
        return int(m.group(1)) / max(total, 1) if m else None

    return parse


def _train_progress(line: str) -> Optional[float]:
    m = re.search(r"\((\d+(?:\.\d+)?)%\)", line)
    return float(m.group(1)) / 100 if m else None


def _train_detail(iterations: int) -> Callable[[str], Optional[str]]:
    # Riga tipo: "2760 (9.20%)        14.956 ms            6 m, 47 s            122.44 M"
    rx = re.compile(r"^\s*(\d+) \(\d+(?:\.\d+)?%\)\s+\S+ ms\s+(.+?)\s{2,}")

    def parse(line: str) -> Optional[str]:
        m = rx.search(line)
        return f"iterazione {int(m.group(1)) + 1} di {iterations}, restano circa {m.group(2)}" if m else None

    return parse


def train_summary(name: str) -> str:
    """Descrizione dell'ultimo training salvato, es. '30000 iterazioni, 02/10 13:52'."""
    config = latest_config(name)
    checkpoints = sorted((config.parent / "nerfstudio_models").glob("step-*.ckpt")) if config else []
    if not checkpoints:
        return ""
    last = checkpoints[-1]
    when = time.strftime("%d/%m %H:%M", time.localtime(last.stat().st_mtime))
    return f"{int(last.stem.split('-')[1]) + 1} iterazioni, {when}"


def gpu_status() -> str:
    """Scheda NVIDIA usata per i calcoli, con uso e memoria correnti."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
        ).stdout.strip().splitlines()[0]
        name, load, used, total = [x.strip() for x in out.split(",")]
        return f"Calcolo su {name} — uso {load}%, memoria {int(used) / 1024:.1f} / {int(total) / 1024:.0f} GB"
    except Exception:  # nvidia-smi assente o scheda non rilevata
        return "Scheda NVIDIA non rilevata"


def _patch_match_progress() -> Callable[[str], Optional[float]]:
    # Con geom_consistency COLMAP elabora ogni vista due volte (fotometrica, poi geometrica).
    seen = [0]

    def parse(line: str) -> Optional[float]:
        m = re.search(r"Processing view \d+ / (\d+)", line)
        if not m:
            return None
        seen[0] += 1
        return seen[0] / (2 * int(m.group(1)))

    return parse


def build_steps(s: Settings, groups: List[str]) -> List[Step]:
    colmap = str(colmap_exe())
    scene = s.scene
    work = scene / "colmap"
    db = str(work / "database.db")
    dense = work / "dense"
    steps: List[Step] = []

    if "sfm" in groups:
        camera_flag = {
            "single": "--ImageReader.single_camera",
            "per_folder": "--ImageReader.single_camera_per_folder",
            "per_image": "--ImageReader.single_camera_per_image",
        }[s.camera]

        def reset_colmap() -> None:
            shutil.rmtree(work, ignore_errors=True)
            (work / "sparse").mkdir(parents=True)

        # Non si usa il COLMAP integrato in ns-process-data: nerfstudio 1.1.5 passa opzioni
        # (--SiftExtraction.use_gpu) che in COLMAP 4.x non esistono piu'.
        steps += [
            Step(
                "sfm",
                "Estrazione delle feature",
                lambda: [colmap, "feature_extractor", "--database_path", db, "--image_path", s.photos,
                         "--ImageReader.camera_model", "OPENCV", camera_flag, "1", "--FeatureExtraction.use_gpu", "1"],
                _ratio(r"Processed file \[(\d+)/(\d+)\]"),
                before=reset_colmap,
            ),
            Step(
                "sfm",
                "Matching tra le foto",
                lambda: [colmap, f"{s.matcher}_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "1"],
                _matching_progress,
            ),
            Step(
                "sfm",
                "Ricostruzione delle camere",
                lambda: [colmap, "mapper", "--database_path", db, "--image_path", s.photos,
                         "--output_path", str(work / "sparse")],
                _mapper_progress(count_images(s.photos)),
            ),
            Step(
                "sfm",
                "Conversione per nerfstudio",
                lambda: [str(VENV_SCRIPTS / "ns-process-data.exe"), "images", "--data", s.photos,
                         "--output-dir", str(scene), "--skip-colmap",
                         "--colmap-model-path", f"colmap/sparse/{best_sparse_model(scene).name}"],
            ),
        ]

    if "train" in groups:
        dataparser = ["nerfstudio-data", "--downscale-factor", str(s.downscale)] if s.downscale else []
        steps.append(
            Step(
                "train",
                "Training",
                lambda: [str(VENV_SCRIPTS / "ns-train.exe"), s.method, "--data", str(scene),
                         "--output-dir", str(ROOT / "outputs"), "--experiment-name", s.name,
                         "--max-num-iterations", str(s.iterations),
                         "--viewer.quit-on-train-completion", "True"] + dataparser,
                _train_progress,
                detail=_train_detail(s.iterations),
            )
        )

    if "export" in groups:

        def export_command() -> List[str]:
            config = latest_config(s.name)
            if config is None:
                raise RuntimeError("Nessun training da esportare.")
            return [str(VENV_SCRIPTS / "ns-export.exe"), "gaussian-splat", "--load-config", str(config),
                    "--output-dir", str(s.export_dir)]

        steps.append(Step("export", "Esportazione splat.ply", export_command))

    if "mesh" in groups:
        steps += [
            Step(
                "mesh",
                "Mesh: correzione delle immagini",
                lambda: [colmap, "image_undistorter", "--image_path", s.photos,
                         "--input_path", str(best_sparse_model(scene)), "--output_path", str(dense),
                         "--output_type", "COLMAP", "--max_image_size", str(s.mesh_size)],
                _ratio(r"Undistorting image \[(\d+)/(\d+)\]"),
                before=lambda: shutil.rmtree(dense, ignore_errors=True),
            ),
            Step(
                "mesh",
                "Mesh: mappe di profondita'",
                lambda: [colmap, "patch_match_stereo", "--workspace_path", str(dense),
                         "--PatchMatchStereo.geom_consistency", "1"],
                _patch_match_progress(),
            ),
            Step(
                "mesh",
                "Mesh: fusione in nuvola densa",
                lambda: [colmap, "stereo_fusion", "--workspace_path", str(dense),
                         "--output_path", str(dense / "fused.ply")],
            ),
            Step(
                "mesh",
                "Mesh: superficie di Poisson",
                lambda: [colmap, "poisson_mesher", "--input_path", str(dense / "fused.ply"),
                         "--output_path", str(dense / "mesh-poisson.ply")],
            ),
        ]
    return steps
