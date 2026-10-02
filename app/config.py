"""Percorsi, componenti esterni, metodi disponibili e impostazioni di un progetto.

Layout su disco, relativo alla radice del repository:
  data/<progetto>/                      allineamento condiviso da tutti i metodi
  outputs/<progetto>/<metodo>/<data>/   un run di training (nerfstudio) con run.json e metrics.json
  exports/<progetto>/<metodo>_<data>/   modello esportato
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
VENV_SCRIPTS = Path(sys.executable).parent  # gli eseguibili di nerfstudio dell'ambiente che esegue l'applicazione
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
PROJECT_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Method:
    key: str  # nome del metodo (per nerfstudio, quello passato a ns-train)
    label: str
    family: str  # "gaussian" | "nerf"
    engine: str  # programma che lo allena: "nerfstudio" | "inria"
    short: str  # etichetta breve per i grafici
    note: str

    @property
    def export_kind(self) -> str:
        return "gaussian-splat" if self.family == "gaussian" else "pointcloud"

    @property
    def export_file(self) -> str:
        return "splat.ply" if self.family == "gaussian" else "point_cloud.ply"


INRIA = "inria-3dgs"

METHODS: Dict[str, Method] = {
    m.key: m
    for m in (
        Method("splatfacto", "Gaussian splatting (gsplat, splatfacto)", "gaussian", "nerfstudio", "3DGS gsplat",
               "3D Gaussian Splatting nell'implementazione gsplat. Veloce, modello esportabile in .ply."),
        Method("splatfacto-big", "Gaussian splatting, alta qualità (gsplat, splatfacto-big)", "gaussian",
               "nerfstudio", "3DGS gsplat big",
               "Come splatfacto ma con più gaussiane: più dettaglio, più tempo e file più grandi."),
        Method(INRIA, "Gaussian splatting originale (Inria)", "gaussian", "inria", "3DGS Inria",
               "Implementazione di riferimento di Kerbl et al. 2023. Solo per uso di ricerca, non commerciale."),
        Method("nerfacto", "NeRF (nerfacto)", "nerf", "nerfstudio", "NeRF nerfacto",
               "Campo di radianza neurale. Senza tiny-cuda-nn usa l'implementazione PyTorch, molto più lenta."),
        Method("nerfacto-big", "NeRF, alta qualità (nerfacto-big)", "nerf", "nerfstudio", "NeRF nerfacto big",
               "Rete più grande di nerfacto: più qualità, tempi molto più lunghi."),
    )
}

@dataclass(frozen=True)
class Quality:
    key: str
    label: str
    iterations: int
    max_side: int  # lato massimo delle foto in pixel; 0 = nessun limite oltre alla memoria del computer


# Livelli di qualita' del training, dal migliore. Quello predefinito e' il primo.
QUALITIES: Dict[str, Quality] = {
    q.key: q
    for q in (
        Quality("massima", "Massima (consigliata)", 30000, 0),
        Quality("alta", "Alta — più veloce", 30000, 1600),
        Quality("media", "Media", 15000, 1600),
        Quality("bozza", "Bozza veloce", 7000, 800),
    )
}

# Gruppi di passi, nell'ordine di esecuzione.
GROUPS: Dict[str, str] = {
    "sfm": "Allineamento delle foto (COLMAP)",
    "train": "Training dei metodi selezionati",
    "eval": "Valutazione sulle viste escluse dal training (PSNR, SSIM, LPIPS)",
    "export": "Esportazione dei modelli",
    "mesh": "Fotogrammetria classica: nuvola densa e mesh (COLMAP, lenta)",
    "report": "Report del confronto con grafici",
}


@dataclass
class Settings:
    name: str = ""
    photos: str = ""
    camera: str = "auto"  # auto | single | per_folder | per_image
    matcher: str = "exhaustive"  # exhaustive | sequential
    methods: List[str] = field(default_factory=lambda: ["splatfacto"])
    iterations: int = 30000
    downscale: int = 0  # fattore di riduzione delle foto; 0 = il piu' piccolo che rispetta max_side e la memoria
    max_side: int = 0  # con downscale 0: lato massimo in pixel (0 = nessun limite oltre alla memoria)
    mesh_size: int = 1600

    @property
    def quality(self) -> Optional[str]:
        """Livello di qualita' corrispondente a queste impostazioni, None se sono state scelte a mano."""
        if self.downscale:
            return None
        return next((q.key for q in QUALITIES.values()
                     if (q.iterations, q.max_side) == (self.iterations, self.max_side)), None)

    def set_quality(self, key: str) -> None:
        quality = QUALITIES[key]
        self.iterations, self.downscale, self.max_side = quality.iterations, 0, quality.max_side

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


def python() -> str:
    """Interprete dell'ambiente (python.exe anche quando l'interfaccia gira con pythonw)."""
    return str(VENV_SCRIPTS / "python.exe")


def inria_repo() -> Path:
    return TOOLS / "gaussian-splatting"


def available_methods() -> Dict[str, Method]:
    """Metodi utilizzabili su questa installazione: quello Inria richiede il suo codice e i moduli compilati."""
    def usable(method: Method) -> bool:
        if method.engine != "inria":
            return True
        return (inria_repo() / "train.py").exists() and all(
            importlib.util.find_spec(module) for module in ("diff_gaussian_rasterization", "simple_knn"))

    return {key: method for key, method in METHODS.items() if usable(method)}


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
            missing.append(f"nerfstudio: {command} (atteso in {VENV_SCRIPTS})")
            break
    return missing


def versions() -> Dict[str, str]:
    """Versioni dei componenti, salvate con ogni run per poterlo riprodurre."""
    out = {"app": __version__}
    # tinycudann incide molto sui tempi dei metodi NeRF: va registrato se c'era o no.
    for package in ("nerfstudio", "gsplat", "torch", "tinycudann"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = "non installato"
    try:
        out["colmap"] = colmap_exe().parent.parent.name.replace("colmap-", "")
    except FileNotFoundError:
        out["colmap"] = "non installato"
    out["gaussian-splatting-inria"] = _git_commit(inria_repo()) or "non installato"
    return out


def _git_commit(repo: Path) -> str:
    """Commit corrente di un repository clonato, letto senza eseguire git."""
    try:
        head = (repo / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            head = (repo / ".git" / head[5:].strip()).read_text().strip()
        return head[:10]
    except OSError:
        return ""
