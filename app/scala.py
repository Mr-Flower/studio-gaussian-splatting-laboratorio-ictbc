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

La scala si verifica poi con una misura del monumento (fase 13 del piano): il basamento di un
pilone, misurato in sito con il distanziometro laser. Nella nuvola densa raddrizzata si isolano i
due piloni in una fascia sopra il suolo, si calcola il rettangolo di area minima di ciascuno e se ne
ricava la scala (misura_piloni). Il risultato va in scala.json sotto "verifica", con la scala
verificata in "scale_verified"; "use" = "scale_verified" la mette in uso.

Uso:  python -m app.scala --project <nome>              allineamento al GPS (due stime)
      python -m app.scala --project <nome> --verifica   verifica con il basamento dei piloni
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import struct
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config, pipeline, runs

SCALE_FILE = "scala.json"
MAX_ERROR_M = 3.0  # soglia di model_aligner per considerare una foto coerente con il proprio GPS
SCALE_KEYS = ("scale", "scale_horizontal", "scale_verified")  # valori ammessi per "use"

# Verifica: basamento di un pilone misurato in sito (lato corto e lato lungo, in metri)
PIER_SIDES_M = (4.33, 6.013)
PIER_SOURCE = "misura in sito, distanziometro laser"
PIER_BAND_M = (1.0, 2.2)  # fascia sopra il suolo: il dado del basamento, sotto le modanature e la zoccolatura
MAX_RATIO_GAP = 0.03  # scarto massimo del rapporto tra i lati, controllo indipendente dalla scala
GROUND_RADIUS_M = 10.0  # il suolo e' il 2o percentile delle quote entro questa distanza dall'arco
CELL_M = 0.025  # lato delle celle della pianta per separare i piloni
HEIGHT_M, HEIGHT_SOURCE = 15.60, "italia.it, portale del turismo (fonte divulgativa)"  # controllo secondario


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
    kept = {k: previous[k] for k in ("scale_verified", "verifica", "use_note") if k in previous}
    data = {**data, **kept, "use": previous.get("use"), "model": str(model.relative_to(scene)),
            "alignment": before, "max_error_m": MAX_ERROR_M, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "metri per unita' dell'allineamento: 'scale' (model_aligner, tre assi) o 'scale_horizontal' "
                    "(solo posizioni orizzontali), 'scale_verified' (verifica con il basamento dei piloni, --verifica); "
                    "'use' indica quale usare (null: nessuna, si resta senza scala)"}
    scale_file(project).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def min_area_rectangle(xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """Lati (corto, lungo), orientamento del lato lungo in gradi e centro del rettangolo di area minima.

    Il rettangolo di area minima ha un lato allineato a un lato dell'inviluppo convesso: si provano tutti.
    """
    from scipy.spatial import ConvexHull

    hull = xy[ConvexHull(xy).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    best = (math.inf, None, 0.0, None)
    for angle in np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2)):
        rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        turned = hull @ rotation.T
        low, high = turned.min(axis=0), turned.max(axis=0)
        if (high - low).prod() < best[0]:
            best = ((high - low).prod(), high - low, angle, rotation.T @ ((low + high) / 2))
    _, sides, angle, centre = best
    if sides[0] < sides[1]:  # l'orientamento e' quello del lato lungo
        angle += np.pi / 2
    return np.sort(sides), float(np.degrees(angle) % 180), centre


def largest_blobs(xy: np.ndarray, cell: float, count: int = 2, min_points: int = 3) -> List[np.ndarray]:
    """I `count` gruppi di punti piu' estesi della pianta: celle con almeno min_points punti, chiusura, componenti."""
    from scipy import ndimage

    ij = np.floor((xy - xy.min(axis=0)) / cell).astype(int)
    grid = np.zeros(ij.max(axis=0) + 1, dtype=int)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    occupied = ndimage.binary_closing(grid >= min_points, iterations=2)
    labels, found = ndimage.label(occupied, structure=np.ones((3, 3)))  # anche in diagonale: pareti oblique
    sizes = ndimage.sum(occupied, labels, range(1, found + 1))
    point_label = labels[ij[:, 0], ij[:, 1]]
    return [xy[point_label == k + 1] for k in np.argsort(sizes)[::-1][:count]]


def pier_scale(sides_units: np.ndarray, sides_m: Sequence[float] = PIER_SIDES_M) -> float:
    """Metri per unita' che portano i due lati misurati su quelli in sito, ai minimi quadrati."""
    measured, real = np.asarray(sides_units, dtype=float), np.asarray(sorted(sides_m), dtype=float)
    return float(measured @ real / (measured @ measured))


def misura_piloni(project: str, metres: float) -> Dict:
    """Rettangolo di area minima del basamento di ciascun pilone nella nuvola densa, e scala che ne segue.

    `metres` e' la scala di partenza per collocare la fascia sopra il suolo (PIER_BAND_M); la funzione
    ripete il calcolo una volta con la scala trovata.
    """
    from . import geometry

    scene = config.WORK / "data" / project
    rotation = geometry.upright_frame(scene)
    box = geometry.region(scene, rotation)
    dense = geometry.read_xyz(scene / "colmap" / "dense" / "fused.ply")[0] @ rotation.T
    dense = dense[np.all((dense >= box[0]) & (dense <= box[1]), axis=1)]
    heights = dense[:, 2]
    centre = np.median(dense[heights > np.percentile(heights, 80), :2], axis=0)  # il corpo alto dell'arco
    plan_distance = np.linalg.norm(dense[:, :2] - centre, axis=1)
    for _ in range(2):
        ground = float(np.percentile(heights[plan_distance < GROUND_RADIUS_M / metres], 2))
        band = (heights > ground + PIER_BAND_M[0] / metres) & (heights < ground + PIER_BAND_M[1] / metres)
        piers = largest_blobs(dense[band & (plan_distance < GROUND_RADIUS_M / metres), :2], CELL_M / metres)
        measured = [min_area_rectangle(xy) for xy in piers]
        scales = [pier_scale(sides) for sides, _, _ in measured]
        metres = float(np.mean(scales))
    expected = PIER_SIDES_M[1] / PIER_SIDES_M[0]
    side_scales = [m / s for sides, _, _ in measured for s, m in zip(sides, PIER_SIDES_M)]
    height_units = float(np.percentile(heights[plan_distance < GROUND_RADIUS_M / metres], 99.5)) - ground
    return {"piloni": [{"punti": int(len(xy)), "lati_unita": [float(v) for v in sides],
                        "lati_m": [float(v * scale) for v in sides], "orientamento_gradi": angle,
                        "centro_unita": [float(v) for v in centre],
                        "rapporto": float(sides[1] / sides[0]),
                        "scarto_rapporto": float(sides[1] / sides[0] / expected - 1), "scala": scale}
                       for xy, (sides, angle, centre), scale in zip(piers, measured, scales)],
            "scala": metres, "scala_min_max_lati": [min(side_scales), max(side_scales)],
            "suolo_unita": ground, "altezza_unita": height_units}


def verify(project: str, accept_ratio: Optional[str] = None) -> Dict:
    """Verifica le stime dal GPS con il basamento dei piloni e scrive il risultato in scala.json."""
    data = load(project)
    if not data:
        raise SystemExit("Manca scala.json: eseguire prima python -m app.scala --project " + project)
    start = data.get("scale_horizontal") or data["scale"]
    found = misura_piloni(project, start)
    expected = PIER_SIDES_M[1] / PIER_SIDES_M[0]
    worst = max(abs(p["scarto_rapporto"]) for p in found["piloni"])
    if worst > MAX_RATIO_GAP and not accept_ratio:
        ratios = ", ".join(f"{p['rapporto']:.3f}" for p in found["piloni"])
        raise SystemExit(f"Rapporto tra i lati dei piloni {ratios} contro {expected:.3f} atteso: scarto oltre il "
                         f"{MAX_RATIO_GAP:.0%}. Controllare la misura; per accettarlo --accetta-rapporto \"<motivo>\".")
    verified = found["scala"]
    estimates = {"scale": data.get("scale"), "scale_horizontal": data.get("scale_horizontal")}
    check = {"fonte": PIER_SOURCE, "oggetto": "basamento di un pilone (lato corto, lato lungo)",
             "lati_in_sito_m": list(PIER_SIDES_M), "fascia_sopra_il_suolo_m": list(PIER_BAND_M),
             "rapporto_atteso": expected, "scarto_rapporto_max": MAX_RATIO_GAP, **found,
             "scarto_stime_gps": {k: v / verified - 1 for k, v in estimates.items() if v},
             "lati_con_stime_gps_m": {k: [[s * v for s in p["lati_unita"]] for p in found["piloni"]]
                                      for k, v in estimates.items() if v},
             "altezza_m": found["altezza_unita"] * verified, "altezza_confronto_m": HEIGHT_M,
             "altezza_fonte": HEIGHT_SOURCE, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if accept_ratio:
        check["rapporto_accettato"] = accept_ratio
    data.update({"scale_verified": verified, "verifica": check, "use": "scale_verified",
                 "use_note": "scala verificata con il basamento dei piloni (" + PIER_SOURCE + ")"})
    scale_file(project).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--verifica", action="store_true", help="verifica la scala con il basamento dei piloni")
    parser.add_argument("--accetta-rapporto", metavar="MOTIVO",
                        help="accetta uno scarto del rapporto tra i lati oltre il 3%%, scrivendo il motivo")
    args = parser.parse_args()
    if args.verifica:
        data = verify(args.project, args.accetta_rapporto)
        check = data["verifica"]
        for i, pier in enumerate(check["piloni"], 1):
            print(f"Pilone {i}: {pier['lati_unita'][0]:.3f} x {pier['lati_unita'][1]:.3f} unita', rapporto "
                  f"{pier['rapporto']:.3f} ({pier['scarto_rapporto']:+.1%}), scala {pier['scala']:.3f} m per unita'")
        low, high = check["scala_min_max_lati"]
        print(f"Scala verificata: {data['scale_verified']:.3f} m per unita' (dai singoli lati {low:.3f}-{high:.3f}).")
        for key, gap in check["scarto_stime_gps"].items():
            print(f"  {key} = {data[key]:.3f}: {gap:+.1%}")
        print(f"Altezza: {check['altezza_m']:.2f} m (confronto {HEIGHT_M} m, {HEIGHT_SOURCE}).")
        return
    data = align(args.project)
    low, high = data["scale_horizontal_ci95"]
    print(f"model_aligner: {data['scale']:.3f} m per unita', da {data['reference_images']} foto con GPS, errore "
          f"residuo {data['error_mean_m']:.1f} m medio.")
    print(f"Solo posizioni orizzontali: {data['scale_horizontal']:.3f} m per unita' (IC 95% {low:.3f}-{high:.3f}), "
          f"{data['horizontal_images']} foto, residuo mediano {data['horizontal_residual_median_m']:.2f} m.")
    print(f"Salvato in {scale_file(args.project)}; stima in uso: {data['use']}")


if __name__ == "__main__":
    main()
