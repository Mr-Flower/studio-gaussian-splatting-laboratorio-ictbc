"""Confronto geometrico tra i modelli e la nuvola densa della fotogrammetria (fase 10 del piano).

La nuvola densa di COLMAP (colmap/dense/fused.ply) e' il RIFERIMENTO, non la verita': anche lei ha
errori. Si confrontano con essa:
  - i centri delle gaussiane con opacita' > MIN_OPACITY di ogni modello 3DGS;
  - la nuvola di punti esportata da ogni NeRF (point_cloud.ply).
Tutto e' riportato nel sistema di COLMAP (clean.to_colmap) e ritagliato a un riquadro attorno ai
punti della nuvola sparsa, ridotto ai percentili 2-98 su ogni asse nel sistema raddrizzato del
volo (verticale delle camere), cosi' si escludono cielo e intorno lontano.

Per ogni metodo, nei due versi:
  - accuratezza: distanza da ogni punto del metodo al punto piu' vicino della nuvola densa;
  - completezza: distanza da ogni punto della nuvola densa al punto piu' vicino del metodo;
con mediana, 90o percentile e frazione entro 5, 10 e 20 cm. Le distanze sono in metri con la
scala scelta in scala.json (app/scala.py); senza scala restano in unita' dell'allineamento.
Ogni nuvola e' sottocampionata a SAMPLE punti (seme fisso).
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import clean, colmap_model, config, pipeline, runs

MIN_OPACITY = 0.5
SAMPLE = 2_000_000
THRESHOLDS_M = (0.05, 0.10, 0.20)
SEED = 0


def read_points3d(model: Path) -> np.ndarray:
    """Coordinate dei punti della nuvola sparsa di COLMAP (file points3D.bin)."""
    points = []
    with open(model / "points3D.bin", "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            _id, x, y, z = struct.unpack("<Q3d", f.read(32))
            f.seek(3 + 8, 1)  # colore e errore di riproiezione
            (track,) = struct.unpack("<Q", f.read(8))
            f.seek(8 * track, 1)
            points.append((x, y, z))
    return np.asarray(points, dtype=np.float64)


def read_xyz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Posizioni di un .ply e, se e' un modello di gaussiane, le opacita' (0..1)."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    names = {p.name for p in vertex.properties}
    opacity = 1 / (1 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float64))) if "opacity" in names else None
    return xyz, opacity


def subsample(points: np.ndarray, size: int = SAMPLE, seed: int = SEED) -> np.ndarray:
    if len(points) <= size:
        return points
    return points[np.random.default_rng(seed).choice(len(points), size=size, replace=False)]


def upright_frame(scene: Path) -> np.ndarray:
    """Rotazione (righe = assi) dal sistema di COLMAP a uno con la verticale del volo sul terzo asse."""
    _, up, _, _ = clean.flight_frame(scene)
    first = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(first) < 1e-6:
        first = np.cross(up, [0.0, 1.0, 0.0])
    first /= np.linalg.norm(first)
    return np.stack([first, np.cross(up, first), up])


def region(scene: Path, rotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Riquadro di confronto nel sistema raddrizzato: percentili 2-98 della nuvola sparsa."""
    sparse = read_points3d(pipeline.best_sparse_model(scene / "colmap")) @ rotation.T
    return np.percentile(sparse, 2, axis=0), np.percentile(sparse, 98, axis=0)


def inside(points: np.ndarray, rotation: np.ndarray, box: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    turned = points @ rotation.T
    return points[np.all((turned >= box[0]) & (turned <= box[1]), axis=1)]


def distance_stats(distances: np.ndarray, metres: float) -> Dict[str, float]:
    d = distances * metres
    stats = {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "mean": float(d.mean())}
    for threshold in THRESHOLDS_M:
        stats[f"within_{int(threshold * 100)}cm"] = float((d <= threshold).mean())
    return stats


def method_points(run: runs.Run, scene: Path) -> Optional[np.ndarray]:
    """Nuvola di un run nel sistema di COLMAP: centri delle gaussiane opache o punti esportati dal NeRF."""
    export = run.export_file
    if export is None:
        return None
    xyz, opacity = read_xyz(export)
    if opacity is not None:
        xyz = xyz[opacity > MIN_OPACITY]
    return clean.to_colmap(xyz, run, scene)


def compare(project: str, metres: Optional[float]) -> Dict:
    """Accuratezza e completezza di ogni metodo del confronto principale rispetto alla nuvola densa."""
    from scipy.spatial import cKDTree

    scene = config.WORK / "data" / project
    dense_file = scene / "colmap" / "dense" / "fused.ply"
    if not dense_file.exists():
        raise FileNotFoundError("Manca la nuvola densa: eseguire la fotogrammetria (passo mesh).")
    rotation = upright_frame(scene)
    box = region(scene, rotation)
    dense = subsample(inside(read_xyz(dense_file)[0], rotation, box))
    dense_tree = cKDTree(dense)
    scale = metres or 1.0
    results: Dict[str, Dict] = {"_riferimento": {"punti": int(len(dense)), "metri": bool(metres)}}
    groups = runs.condition_groups(runs.list_runs(project))
    for key, method_runs in (groups[0].items() if groups else []):
        points = method_points(method_runs[0], scene)
        if points is None:
            continue
        points = subsample(inside(points, rotation, box))
        if len(points) == 0:
            continue
        accuracy = dense_tree.query(points, k=1, workers=-1)[0]
        completeness = cKDTree(points).query(dense, k=1, workers=-1)[0]
        results[key] = {"punti": int(len(points)), "accuratezza": distance_stats(accuracy, scale),
                        "completezza": distance_stats(completeness, scale),
                        "_accuracy": accuracy * scale, "_points": points @ rotation.T}
    return results


def spearman_quality_geometry(results: Dict, quality: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """RQ4: correlazione di rango tra qualita' d'immagine (PSNR corretto) e accuratezza mediana."""
    from scipy.stats import spearmanr

    keys = [k for k in results if not k.startswith("_") and k in quality]
    if len(keys) < 3:
        return None
    rho, p_value = spearmanr([quality[k] for k in keys], [results[k]["accuratezza"]["median"] for k in keys])
    return float(rho), float(p_value)


def figures(results: Dict, out: Path, unit: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import FAMILY_COLOR, GRID, _style
    from .paper import _save

    _style()
    keys = [k for k in results if not k.startswith("_")]
    if not keys:
        return
    limit = max(np.percentile(results[k]["_accuracy"], 95) for k in keys)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    bins = np.linspace(0, limit, 60)
    for key in keys:
        method = config.METHODS[key]
        ax.hist(results[key]["_accuracy"], bins=bins, histtype="step", linewidth=1.8, density=True,
                color=FAMILY_COLOR[method.family], label=method.short,
                linestyle="-" if "big" not in key and key != config.INRIA else ("--" if "big" in key else ":"))
    ax.set_xlabel(f"distanza dalla nuvola densa ({unit})")
    ax.set_ylabel("densità")
    ax.set_title("Accuratezza: distanza di ogni punto del modello dalla nuvola densa")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out, "fig6_distanze")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.8), constrained_layout=True, squeeze=False)
    for ax, key in zip(axes[0], keys):
        points, distance = results[key]["_points"], results[key]["_accuracy"]
        rng = np.random.default_rng(SEED)
        chosen = rng.choice(len(points), size=min(len(points), 150_000), replace=False)
        image = ax.scatter(points[chosen, 0], points[chosen, 1], c=np.minimum(distance[chosen], limit), s=0.3,
                           cmap="viridis", vmin=0, vmax=limit, linewidth=0)
        ax.set_aspect("equal")
        ax.set_title(config.METHODS[key].short, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label=f"distanza ({unit})")
    fig.suptitle("Distanza dalla nuvola densa, vista dall'alto", x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, out, "fig6b_distanze_pianta")
    plt.close(fig)


def table(results: Dict, out: Path, numbers: dict, unit: str) -> None:
    from .paper import _f, write_table

    rows = []
    for key, r in results.items():
        if key.startswith("_"):
            continue
        a, c = r["accuratezza"], r["completezza"]
        rows.append([config.METHODS[key].label, r["punti"], _f(a["median"], 3), _f(a["p90"], 3),
                     *[f"{a[f'within_{int(t * 100)}cm']:.1%}" for t in THRESHOLDS_M],
                     _f(c["median"], 3), _f(c["p90"], 3), f"{c['within_10cm']:.1%}"])
        for side, values in (("accuratezza", a), ("completezza", c)):
            for name, value in values.items():
                numbers[f"geometria.{key}.{side}.{name}"] = value
    numbers["geometria.punti_riferimento"] = results["_riferimento"]["punti"]
    numbers["geometria.unita"] = unit
    write_table(out, "tab8_geometria",
                ["Metodo", "Punti", f"Accuratezza mediana ({unit})", f"Accuratezza 90° perc. ({unit})",
                 "Entro 5 cm", "Entro 10 cm", "Entro 20 cm", f"Completezza mediana ({unit})",
                 f"Completezza 90° perc. ({unit})", "Completezza entro 10 cm"], rows,
                "Distanze rispetto alla nuvola densa della fotogrammetria (riferimento, non verità)")


def build(project: str, out: Path, numbers: dict, metres: Optional[float], quality: Dict[str, float]) -> bool:
    """Tabella 8, figure 6 e numeri della fase 10; False se manca la nuvola densa."""
    try:
        results = compare(project, metres)
    except FileNotFoundError:
        return False
    unit = "m" if metres else "unità"
    table(results, out, numbers, unit)
    figures(results, out, unit)
    rq4 = spearman_quality_geometry(results, quality)
    if rq4:
        numbers["geometria.rq4_spearman_rho"], numbers["geometria.rq4_spearman_p"] = rq4
    return True
