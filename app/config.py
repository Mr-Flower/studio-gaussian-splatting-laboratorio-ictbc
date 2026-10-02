"""Percorsi, componenti esterni, metodi disponibili e impostazioni di un progetto.

Layout su disco, relativo alla radice del repository:
  data/<progetto>/                      allineamento condiviso da tutti i metodi
  outputs/<progetto>/<metodo>/<data>/   un run di training (nerfstudio) con run.json e metrics.json
  exports/<progetto>/<metodo>_<data>/   modello esportato
"""
from __future__ import annotations

import importlib.metadata
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
VENV_SCRIPTS = ROOT / ".venv" / "Scripts"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PROJECT_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Method:
    key: str  # nome del metodo in nerfstudio
    label: str
    family: str  # "gaussian" | "nerf"
    note: str

    @property
    def export_kind(self) -> str:
        return "gaussian-splat" if self.family == "gaussian" else "pointcloud"

    @property
    def export_file(self) -> str:
        return "splat.ply" if self.family == "gaussian" else "point_cloud.ply"


METHODS: Dict[str, Method] = {
    m.key: m
    for m in (
        Method("splatfacto", "Gaussian splatting (splatfacto)", "gaussian",
               "3D Gaussian Splatting nell'implementazione gsplat. Veloce, modello esportabile in .ply."),
        Method("splatfacto-big", "Gaussian splatting, alta qualità (splatfacto-big)", "gaussian",
               "Come splatfacto ma con più gaussiane: più dettaglio, più tempo e file più grandi."),
        Method("nerfacto", "NeRF (nerfacto)", "nerf",
               "Campo di radianza neurale. Senza tiny-cuda-nn usa l'implementazione PyTorch, molto più lenta."),
        Method("nerfacto-big", "NeRF, alta qualità (nerfacto-big)", "nerf",
               "Rete più grande di nerfacto: più qualità, tempi molto più lunghi."),
    )
}

# Gruppi di passi, nell'ordine di esecuzione.
GROUPS: Dict[str, str] = {
    "sfm": "Allineamento delle foto (COLMAP)",
    "train": "Training dei metodi selezionati",
    "eval": "Valutazione sulle viste escluse dal training (PSNR, SSIM, LPIPS)",
    "export": "Esportazione dei modelli",
    "mesh": "Fotogrammetria classica: nuvola densa e mesh (COLMAP, lenta)",
}


@dataclass
class Settings:
    name: str = ""
    photos: str = ""
    camera: str = "single"  # single | per_folder | per_image
    matcher: str = "exhaustive"  # exhaustive | sequential
    methods: List[str] = field(default_factory=lambda: ["splatfacto"])
    iterations: int = 30000
    downscale: int = 0  # 0 = scelta automatica di nerfstudio (lato massimo 1600 px)
    mesh_size: int = 1600

    @property
    def scene(self) -> Path:
        return ROOT / "data" / self.name

    def save(self) -> None:
        self.scene.mkdir(parents=True, exist_ok=True)
        (self.scene / "project.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, name: str) -> Optional["Settings"]:
        path = ROOT / "data" / name / "project.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if "method" in data:  # formato precedente: un solo metodo
            data.setdefault("methods", [data["method"]])
        known = {f.name for f in fields(cls)}
        settings = cls(**{k: v for k, v in data.items() if k in known})
        settings.methods = [m for m in settings.methods if m in METHODS] or ["splatfacto"]
        return settings


def projects() -> List[str]:
    data = ROOT / "data"
    return sorted(d.name for d in data.iterdir() if d.is_dir()) if data.is_dir() else []


def count_images(folder: str) -> int:
    path = Path(folder)
    if not folder or not path.is_dir():
        return 0
    return sum(1 for f in path.rglob("*") if f.suffix.lower() in IMAGE_EXTS)


def _tool(pattern: str) -> Path:
    found = sorted(TOOLS.glob(pattern))
    if not found:
        raise FileNotFoundError(f"Componente mancante in {TOOLS}: {pattern}")
    return found[-1]


def colmap_exe() -> Path:
    return _tool("colmap-*/bin/colmap.exe")


def ns(command: str) -> str:
    """Percorso di un eseguibile di nerfstudio (ns-train, ns-eval, ...)."""
    return str(VENV_SCRIPTS / f"{command}.exe")


def environment() -> Dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join(
        [str(colmap_exe().parent), str(_tool("ffmpeg-*/bin")), str(VENV_SCRIPTS), env.get("PATH", "")]
    )
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["COLUMNS"] = "160"
    return env


def missing_components() -> List[str]:
    """Componenti esterni senza i quali la pipeline non puo' partire."""
    missing = []
    for label, pattern in (("COLMAP", "colmap-*/bin/colmap.exe"), ("FFmpeg", "ffmpeg-*/bin/ffmpeg.exe")):
        if not sorted(TOOLS.glob(pattern)):
            missing.append(f"{label} (atteso in tools\\{pattern})")
    for command in ("ns-train", "ns-eval", "ns-export", "ns-process-data", "ns-viewer"):
        if not Path(ns(command)).exists():
            missing.append(f"nerfstudio: {command} (atteso in .venv\\Scripts)")
            break
    return missing


def versions() -> Dict[str, str]:
    """Versioni dei componenti, salvate con ogni run per poterlo riprodurre."""
    out = {"app": __version__}
    for package in ("nerfstudio", "gsplat", "torch"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "non installato"
    try:
        out["colmap"] = colmap_exe().parent.parent.name.replace("colmap-", "")
    except FileNotFoundError:
        out["colmap"] = "non installato"
    return out
