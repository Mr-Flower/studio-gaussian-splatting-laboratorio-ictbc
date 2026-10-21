"""Scala metrica del progetto dalle posizioni GPS delle foto, senza toccare l'allineamento condiviso.

L'allineamento di COLMAP ha una scala arbitraria. Qui si allinea una COPIA del modello sparso
alle posizioni GPS lette dalle EXIF (gia' nel database di COLMAP), con model_aligner in coordinate
ENU (est, nord, alto, in metri, con origine nella prima foto). L'allineamento usato dai run non
cambia, quindi i run restano validi; la trasformazione trovata si salva in data/<progetto>/scala.json
e la usa solo app/paper.py per riportare misure in metri.

La precisione dipende dal GPS del drone. Sul caso di studio le posizioni orizzontali sono buone
(residui di circa 1 m) mentre la quota e' molto meno affidabile, e trascina la scala di
model_aligner. Per questo scala.json contiene due stime: quella di model_aligner (tre assi) e una
dalle sole posizioni orizzontali (similitudine 2D ai minimi quadrati, ripetuta scartando i residui
oltre tre volte la mediana, con intervallo bootstrap). Quale usare lo dice il campo "use".

Uso:  python -m app.scala --project <nome>
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import struct
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from . import config, pipeline, runs

SCALE_FILE = "scala.json"
MAX_ERROR_M = 3.0  # soglia di model_aligner per considerare una foto coerente con il proprio GPS


def scale_file(project: str) -> Path:
    return config.WORK / "data" / project / SCALE_FILE


def load(project: str) -> Optional[Dict]:
    data = runs.read_json(scale_file(project))
    return data or None


def parse_log(text: str) -> Dict:
    """Numeri utili dall'uscita di model_aligner."""
    values: Dict = {}
    if match := re.search(r"ENU origin: ([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", text):
        values["enu_origin_lat_lon_alt"] = [float(x) for x in match.groups()]
    if match := re.search(r"Using (\d+) reference images", text):
        values["reference_images"] = int(match.group(1))
    if match := re.search(r"Alignment error: ([\d.]+) \(mean\), ([\d.]+) \(median\)", text):
        values["error_mean_m"], values["error_median_m"] = float(match.group(1)), float(match.group(2))
    values["succeeded"] = "Alignment succeeded" in text
    return values


def parse_transform(text: str) -> Dict:
    """Trasformazione di similitudine scritta da model_aligner: scala, quaternione (w, x, y, z), traslazione."""
    numbers = [float(x) for x in text.split()]
    if len(numbers) != 8:
        raise ValueError(f"Trasformazione inattesa: {text!r}")
    return {"scale": numbers[0], "rotation_wxyz": numbers[1:5], "translation": numbers[5:8]}


EARTH_RADIUS_M = 6378137.0


def gps_positions(database: Path) -> Dict[str, Tuple[float, float, float]]:
    """Latitudine, longitudine e quota di ogni foto con GPS, dal database di COLMAP (tabella pose_priors)."""
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        rows = db.execute("select p.position, i.name from pose_priors p join images i on i.image_id = p.corr_data_id "
                          "where p.coordinate_system = 0").fetchall()  # 0: coordinate WGS84
    return {name: struct.unpack("<3d", position) for position, name in rows}


def similarity_2d(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Scala, rotazione e traslazione che portano source su target nel piano, ai minimi quadrati (Umeyama)."""
    mean_s, mean_t = source.mean(axis=0), target.mean(axis=0)
    s, d = source - mean_s, target - mean_t
    u, sigma, vt = np.linalg.svd(d.T @ s / len(source))
    flip = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rotation = u @ flip @ vt
    scale = float(np.trace(np.diag(sigma) @ flip) / s.var(axis=0).sum())
    return scale, rotation, mean_t - scale * rotation @ mean_s


def horizontal_scale(geo_model: Path, gps: Dict[str, Tuple[float, float, float]], aligner_scale: float,
                     resamples: int = 500) -> Dict:
    """Scala dalle sole posizioni orizzontali: corregge quella di model_aligner, gia' applicata a geo_model."""
    from .paper import camera_centers  # centri delle camere da images.bin

    centers, _ = camera_centers(geo_model)
    names = sorted(n for n in gps if n in centers)
    lat0, lon0, _ = gps[min(gps)]
    east_north = np.array([[np.radians(gps[n][1] - lon0) * EARTH_RADIUS_M * np.cos(np.radians(lat0)),
                            np.radians(gps[n][0] - lat0) * EARTH_RADIUS_M] for n in names])
    model = np.array([centers[n][:2] for n in names])
    keep = np.ones(len(names), dtype=bool)
    for _ in range(5):
        scale, rotation, shift = similarity_2d(model[keep], east_north[keep])
        residuals = np.linalg.norm((scale * (rotation @ model.T)).T + shift - east_north, axis=1)
        keep = residuals < 3 * np.median(residuals)
    rng = np.random.default_rng(0)
    kept = np.flatnonzero(keep)
    boot = [similarity_2d(model[b], east_north[b])[0] for b in (rng.choice(kept, len(kept)) for _ in range(resamples))]
    return {"scale_horizontal": aligner_scale * scale,
            "scale_horizontal_ci95": [aligner_scale * float(np.percentile(boot, 2.5)),
                                      aligner_scale * float(np.percentile(boot, 97.5))],
            "horizontal_images": int(keep.sum()), "horizontal_residual_median_m": float(np.median(residuals[keep]))}


def align(project: str) -> Dict:
    scene = config.WORK / "data" / project
    colmap_dir = scene / "colmap"
    model = pipeline.best_sparse_model(colmap_dir)
    target = colmap_dir / "sparse_geo"  # fuori da colmap/sparse: non e' un modello tra cui scegliere
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    transform = target / "trasformazione.txt"
    command = [str(config.colmap_exe()), "model_aligner", "--input_path", str(model), "--output_path", str(target),
               "--database_path", str(colmap_dir / "database.db"), "--ref_is_gps", "1", "--alignment_type", "enu",
               "--alignment_max_error", str(MAX_ERROR_M), "--transform_path", str(transform)]
    before = runs.alignment_id(scene)
    out = subprocess.run(command, capture_output=True, text=True, errors="replace",
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = out.stdout + out.stderr
    values = parse_log(log)
    if out.returncode != 0 or not values.get("succeeded") or not transform.exists():
        raise SystemExit("Allineamento al GPS non riuscito. Le foto hanno le coordinate GPS nelle EXIF?\n" + log[-2000:])
    if runs.alignment_id(scene) != before:  # non deve mai succedere: si lavora su una copia
        raise SystemExit("L'allineamento condiviso e' cambiato: controllare.")
    data = {**parse_transform(transform.read_text()), **values}
    data.update(horizontal_scale(target, gps_positions(colmap_dir / "database.db"), data["scale"]))
    previous = load(project) or {}
    data = {**data, "use": previous.get("use"), "model": str(model.relative_to(scene)),
            "alignment": before, "max_error_m": MAX_ERROR_M, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "metri per unita' dell'allineamento: 'scale' (model_aligner, tre assi) o 'scale_horizontal' "
                    "(solo posizioni orizzontali); 'use' indica quale usare (null: nessuna, si resta senza scala)"}
    scale_file(project).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    data = align(args.project)
    low, high = data["scale_horizontal_ci95"]
    print(f"model_aligner: {data['scale']:.3f} m per unita', da {data['reference_images']} foto con GPS, errore "
          f"residuo {data['error_mean_m']:.1f} m medio.")
    print(f"Solo posizioni orizzontali: {data['scale_horizontal']:.3f} m per unita' (IC 95% {low:.3f}-{high:.3f}), "
          f"{data['horizontal_images']} foto, residuo mediano {data['horizontal_residual_median_m']:.2f} m.")
    print(f"Salvato in {scale_file(args.project)}; stima in uso: {data['use']}")


if __name__ == "__main__":
    main()
