"""Registro dei run di training e dati per la tabella di confronto tra metodi."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

RUN_FILE = "run.json"
METRICS_FILE = "metrics.json"
ALIGNMENT_FILE = "alignment.json"
MESH_FILE = "mesh.json"


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update_json(path: Path, **values: Any) -> None:
    data = read_json(path)
    data.update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def alignment_id(scene: Path) -> str:
    """Impronta dell'allineamento corrente: un run e' confrontabile solo con run della stessa impronta."""
    path = scene / "transforms.json"
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12] if path.exists() else ""


def ply_elements(path: Path) -> Optional[int]:
    """Numero di elementi (gaussiane o punti) dichiarato nell'intestazione di un file .ply."""
    try:
        with open(path, "rb") as f:
            for _ in range(60):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("element vertex"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except OSError:
        pass
    return None


@dataclass(frozen=True)
class Run:
    project: str
    method: str
    timestamp: str

    @property
    def path(self) -> Path:
        return config.ROOT / "outputs" / self.project / self.method / self.timestamp

    @property
    def config_file(self) -> Path:
        return self.path / "config.yml"

    @property
    def export_dir(self) -> Path:
        return config.ROOT / "exports" / self.project / f"{self.method}_{self.timestamp}"

    @property
    def export_file(self) -> Optional[Path]:
        method = config.METHODS.get(self.method)
        path = self.export_dir / method.export_file if method else None
        return path if path and path.exists() else None

    @property
    def engine(self) -> str:
        method = config.METHODS.get(self.method)
        return method.engine if method else "nerfstudio"

    def checkpoint(self) -> Optional[Path]:
        """Modello salvato a fine training; la sua presenza distingue un run completo da uno interrotto."""
        if self.engine == "inria":
            saved = sorted(self.path.glob("point_cloud/iteration_*/point_cloud.ply"),
                           key=lambda p: int(p.parent.name.split("_")[1]))
        else:
            saved = sorted((self.path / "nerfstudio_models").glob("step-*.ckpt"))
        return saved[-1] if saved else None

    def trained_iterations(self) -> int:
        checkpoint = self.checkpoint()
        if checkpoint is None:
            return 0
        if self.engine == "inria":
            return int(checkpoint.parent.name.split("_")[1])
        return int(checkpoint.stem.split("-")[1]) + 1

    def info(self) -> Dict[str, Any]:
        return read_json(self.path / RUN_FILE)

    def metrics(self) -> Dict[str, Any]:
        return read_json(self.path / METRICS_FILE).get("results", {})

    def update(self, **values: Any) -> None:
        update_json(self.path / RUN_FILE, **values)

    def comparable(self) -> Optional[bool]:
        """True se il run usa l'allineamento corrente, None se non e' noto (run senza run.json)."""
        recorded = self.info().get("alignment")
        if not recorded:
            return None
        return recorded == alignment_id(config.ROOT / "data" / self.project)


def new_run(project: str, method: str) -> Run:
    return Run(project, method, time.strftime("%Y-%m-%d_%H%M%S"))


def list_runs(project: str) -> List[Run]:
    """Run con un checkpoint salvato, dal piu' recente."""
    base = config.ROOT / "outputs" / project
    runs = [Run(project, folder.parent.name, folder.name) for folder in base.glob("*/*") if folder.is_dir()]
    runs = [r for r in runs if r.checkpoint() is not None]
    return sorted(runs, key=lambda r: r.checkpoint().stat().st_mtime, reverse=True)


def latest_run(project: str, method: str) -> Optional[Run]:
    return next((r for r in list_runs(project) if r.method == method), None)


COLUMNS = [
    ("method", "Metodo"),
    ("date", "Data"),
    ("iterations", "Iterazioni"),
    ("resolution", "Risoluzione"),
    ("train_minutes", "Training (min)"),
    ("psnr", "PSNR (dB)"),
    ("ssim", "SSIM"),
    ("lpips", "LPIPS"),
    ("fps", "FPS render"),
    ("elements", "Gaussiane / punti"),
    ("model_mb", "Modello (MB)"),
    ("alignment", "Allineamento"),
]


def row(run: Run) -> Dict[str, Any]:
    """Valori di un run per la tabella di confronto; i campi non disponibili restano vuoti."""
    info, metrics, checkpoint = run.info(), run.metrics(), run.checkpoint()
    method = config.METHODS.get(run.method)
    downscale = info.get("downscale")
    comparable = run.comparable()
    export = run.export_file
    return {
        "method": method.label if method else run.method,
        "date": time.strftime("%d/%m/%Y %H:%M", time.localtime(checkpoint.stat().st_mtime)),
        "iterations": info.get("iterations") or run.trained_iterations(),
        "resolution": "" if downscale is None else ("automatica" if downscale == 0 else f"1/{downscale}"),
        "train_minutes": round(info["train_seconds"] / 60, 1) if "train_seconds" in info else "",
        "psnr": round(metrics["psnr"], 2) if "psnr" in metrics else "",
        "ssim": round(metrics["ssim"], 4) if "ssim" in metrics else "",
        "lpips": round(metrics["lpips"], 4) if "lpips" in metrics else "",
        "fps": round(metrics["fps"], 2) if "fps" in metrics else "",
        "elements": (ply_elements(export) or "") if export else "",
        "model_mb": round(checkpoint.stat().st_size / 2**20, 1),
        "alignment": {True: "corrente", False: "precedente", None: "non registrato"}[comparable],
    }


def write_csv(project: str, path: Path) -> int:
    runs = list_runs(project)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([label for _, label in COLUMNS] + ["Cartella del run"])
        for run in runs:
            values = row(run)
            writer.writerow([values[key] for key, _ in COLUMNS] + [str(run.path)])
    return len(runs)


def alignment_summary(scene: Path) -> str:
    data = read_json(scene / ALIGNMENT_FILE)
    if "registered_images" not in data:
        return ""
    text = f"{data['registered_images']} foto allineate su {data.get('photos', '?')}"
    if "mean_reprojection_error_px" in data:
        text += f", errore medio di riproiezione {data['mean_reprojection_error_px']:.2f} px"
    return text


def mesh_summary(scene: Path) -> str:
    dense = scene / "colmap" / "dense"
    if not (dense / "mesh-poisson.ply").exists():
        return ""
    data = read_json(dense / MESH_FILE)
    parts = []
    points = ply_elements(dense / "fused.ply")
    if points:
        parts.append(f"nuvola densa di {points:,} punti".replace(",", "."))
    vertices = ply_elements(dense / "mesh-poisson.ply")
    if vertices:
        parts.append(f"mesh di {vertices:,} vertici".replace(",", "."))
    if "seconds" in data:
        parts.append(f"{sum(data['seconds'].values()) / 60:.0f} min a {data.get('max_image_size', '?')} px")
    return ", ".join(parts)
