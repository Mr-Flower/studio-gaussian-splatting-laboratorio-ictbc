"""Pulizia automatica di un modello di gaussiane: toglie cio' che non fa parte della scena rilevata.

Il modello originale non viene toccato: il risultato va in un file nuovo accanto all'esportato
(splat_pulito.ply) con le regole applicate e i conteggi (pulizia.json). La pulizia serve a
guardare e pubblicare il modello; le misure del confronto restano quelle del modello originale.

Regole, tutte riferite alla geometria del volo, quindi indipendenti dalla scala del modello:
  1. gaussiane quasi trasparenti (opacita' sotto MIN_OPACITY);
  2. gaussiane enormi (asse maggiore oltre MAX_SIZE volte il raggio della zona del volo), di
     solito cielo o sfondo lontano;
  3. fuori dalla zona del volo: piu' lontane in orizzontale del raggio entro cui stanno le camere,
     sopra la camera piu' alta o sotto il terreno;
  4. sopra la parte solida piu' alta della scena (gaussiane fluttuanti nel cielo);
  5. isolate: distanza media dalle vicine molto superiore a quella tipica (fluttuanti), con una
     soglia piu' severa per quelle semitrasparenti, che sono quasi sempre artefatti.
Il riferimento e' il sistema di COLMAP; i modelli di nerfstudio vi vengono riportati con le
trasformazioni salvate da nerfstudio, e il file pulito resta nel sistema originale del modello.

Uso:  python -m app.clean --project <nome> [--run <metodo>_<data>] ...
      (senza --run: i modelli di gaussiane del confronto principale)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import config, pipeline, runs

CLEAN_FILE = "splat_pulito.ply"
REPORT_FILE = "pulizia.json"
MIN_OPACITY = 0.1
SOLID_OPACITY = 0.7  # gaussiane considerate superficie solida (per trovare la cima della scena)
TOP_PERCENTILE = 99.8  # la cima: le gaussiane solide piu' alte dentro la zona del volo
MAX_SIZE = 0.25  # asse maggiore massimo, in frazioni del raggio della zona del volo
RING_PERCENTILE = 90  # le camere entro cui sta la scena: il 90% delle distanze orizzontali dal centro
GROUND_PERCENTILE = 2  # il terreno: le gaussiane piu' basse dentro la zona del volo
MARGIN = 0.05  # tolleranza sopra e sotto, in frazioni del raggio
NEIGHBOURS = 16
ISOLATION_PERCENTILE = 99.0
FAINT_OPACITY = 0.5  # sotto questa opacita' una gaussiana isolata e' trattata con la soglia severa
FAINT_ISOLATION_PERCENTILE = 90.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def to_colmap(points: np.ndarray, run: runs.Run, scene: Path) -> np.ndarray:
    """Porta i centri delle gaussiane di un run nel sistema di COLMAP."""
    if run.engine == "inria":
        return points  # il codice Inria lavora direttamente nel sistema di COLMAP
    data = runs.read_json(run.path / "dataparser_transforms.json")
    transform, scale = np.asarray(data["transform"]), float(data["scale"])
    # nerfstudio salva in dataparser_transforms.json la trasformazione dalle coordinate originali
    # (quelle di COLMAP, applied_transform compreso): p = scala * (R q + t).
    return (points / scale - transform[:, 3]) @ transform[:, :3]


def flight_frame(scene: Path):
    """Centro, direzione dell'alto, raggio e quote delle camere, nel sistema di COLMAP."""
    from .paper import camera_centers

    centers, up = camera_centers(pipeline.best_sparse_model(scene / "colmap"))
    positions = np.array(list(centers.values()))
    heights = positions @ up
    flat = positions - np.outer(heights, up)
    middle = np.median(flat, axis=0)
    radius = float(np.percentile(np.linalg.norm(flat - middle, axis=1), RING_PERCENTILE))
    return middle, up, radius, float(heights.max())


def clean(run: runs.Run, scene: Path) -> Dict:
    from plyfile import PlyData, PlyElement
    from scipy.spatial import cKDTree

    source = run.export_file
    if source is None or config.METHODS[run.method].family != "gaussian":
        raise SystemExit(f"{run.method} {run.timestamp}: nessun modello di gaussiane esportato.")
    ply = PlyData.read(str(source))
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    points = to_colmap(xyz, run, scene)
    middle, up, radius, top = flight_frame(scene)

    heights = points @ up
    horizontal = np.linalg.norm(points - np.outer(heights, up) - middle, axis=1)
    # Le dimensioni delle gaussiane sono nel sistema del modello: si riportano a quello di COLMAP.
    to_colmap_scale = 1.0 if run.engine == "inria" else 1.0 / float(
        runs.read_json(run.path / "dataparser_transforms.json")["scale"])
    size = np.exp(np.stack([vertex[f"scale_{i}"] for i in range(3)], axis=1)).max(axis=1) * to_colmap_scale
    opacity = _sigmoid(np.asarray(vertex["opacity"], dtype=np.float64))

    inside_ring = horizontal <= radius
    ground = float(np.percentile(heights[inside_ring], GROUND_PERCENTILE)) if inside_ring.any() else heights.min()
    rules = {
        "trasparenti": opacity < MIN_OPACITY,
        "enormi": size > MAX_SIZE * radius,
        "fuori dalla zona del volo": (~inside_ring) | (heights > top + MARGIN * radius)
                                     | (heights < ground - MARGIN * radius),
    }
    solid = inside_ring & (opacity >= SOLID_OPACITY)
    if solid.any():
        summit = float(np.percentile(heights[solid], TOP_PERCENTILE))
        rules["sopra la scena"] = heights > summit + MARGIN * radius
    keep = ~np.logical_or.reduce(list(rules.values()))
    # Isolate: si valutano solo tra le rimaste, che sono la scena vera.
    tree = cKDTree(points[keep])
    distances, _ = tree.query(points[keep], k=NEIGHBOURS + 1)
    spread = distances[:, 1:].mean(axis=1)
    isolated = np.zeros(len(points), dtype=bool)
    faint = opacity[keep] < FAINT_OPACITY
    isolated[np.flatnonzero(keep)] = (spread > np.percentile(spread, ISOLATION_PERCENTILE)) | (
        faint & (spread > np.percentile(spread, FAINT_ISOLATION_PERCENTILE)))
    rules["isolate"] = isolated & keep
    keep &= ~isolated

    target = run.export_dir / CLEAN_FILE
    PlyData([PlyElement.describe(vertex.data[keep], "vertex")], text=False,
            byte_order=ply.byte_order).write(str(target))
    report = {
        "source": str(source), "target": str(target), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gaussians": int(len(points)), "kept": int(keep.sum()),
        "removed": {name: int(mask.sum()) for name, mask in rules.items()},
        "parameters": {"min_opacity": MIN_OPACITY, "max_size_of_radius": MAX_SIZE, "ring_percentile": RING_PERCENTILE,
                       "ground_percentile": GROUND_PERCENTILE, "margin_of_radius": MARGIN, "neighbours": NEIGHBOURS,
                       "isolation_percentile": ISOLATION_PERCENTILE, "solid_opacity": SOLID_OPACITY,
                       "top_percentile": TOP_PERCENTILE, "faint_opacity": FAINT_OPACITY,
                       "faint_isolation_percentile": FAINT_ISOLATION_PERCENTILE},
        "flight_radius_colmap_units": radius,
        "note": "una gaussiana puo' rientrare in piu' regole: i conteggi per regola non si sommano",
    }
    (run.export_dir / REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    dc = np.stack([vertex[f"f_dc_{i}"] for i in range(3)], axis=1)
    colors = np.clip(0.5 + 0.28209479177387814 * dc, 0, 1)  # colore di base di ogni gaussiana
    preview(points, keep, colors, up, middle, radius, run.export_dir / "pulizia.png", config.METHODS[run.method].short)
    return report


def preview(points: np.ndarray, keep: np.ndarray, colors: np.ndarray, up: np.ndarray, middle: np.ndarray,
            radius: float, path: Path, title: str) -> None:
    """Pianta e prospetto dei centri vicino alla scena: tenuti con il loro colore, tolti in grigio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import _style

    _style()
    heights = points @ up
    flat = points - np.outer(heights, up) - middle
    axis_a = np.cross(up, [1.0, 0.0, 0.0])
    axis_a = axis_a / np.linalg.norm(axis_a) if np.linalg.norm(axis_a) > 1e-6 else np.array([0.0, 1.0, 0.0])
    axis_b = np.cross(up, axis_a)
    a, b = flat @ axis_a, flat @ axis_b
    h = heights - np.median(heights[keep])
    limit = 2.0 * radius
    near = (np.abs(a) < limit) & (np.abs(b) < limit) & (np.abs(h) < limit)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), constrained_layout=True)
    views = ((a, b, "pianta"), (a, h, "prospetto"), (b, h, "fianco"))
    for ax, (x, y, label) in zip(axes, views):
        removed = np.flatnonzero(near & ~keep)
        kept = np.flatnonzero(keep)
        removed = rng.choice(removed, size=min(len(removed), 60000), replace=False) if len(removed) else removed
        kept = rng.choice(kept, size=min(len(kept), 200000), replace=False)
        # Nel prospetto e nel fianco si disegnano prima i punti piu' lontani, cosi' davanti resta la facciata.
        order = np.argsort(b[kept] if label == "prospetto" else -a[kept] if label == "fianco" else h[kept])
        ax.scatter(x[removed], y[removed], s=0.4, color="#c3c2b7", linewidth=0,
                   label=f"tolte ({int((~keep).sum()):,})".replace(",", "."))
        ax.scatter(x[kept][order], y[kept][order], s=0.5, c=colors[kept][order], linewidth=0,
                   label=f"tenute ({int(keep.sum()):,})".replace(",", "."))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal")
        ax.set_title(f"{title}: {label}")
        ax.tick_params(length=0)
    axes[0].legend(frameon=False, markerscale=15, loc="upper right")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main_runs(project: str) -> List[runs.Run]:
    groups = runs.condition_groups(runs.list_runs(project))
    if not groups:
        return []
    return [found[0] for key, found in groups[0].items() if config.METHODS[key].family == "gaussian"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", action="append", help="cartella esportata <metodo>_<data>; si puo' ripetere")
    args = parser.parse_args()
    scene = config.WORK / "data" / args.project
    chosen = main_runs(args.project) if not args.run else [
        r for r in runs.list_runs(args.project) if f"{r.method}_{r.timestamp}" in args.run]
    if not chosen:
        raise SystemExit("Nessun modello di gaussiane da pulire.")
    for run in chosen:
        report = clean(run, scene)
        def n(value: int) -> str:
            return f"{value:,}".replace(",", ".")

        removed = ", ".join(f"{k} {n(v)}" for k, v in report["removed"].items())
        print(f"{run.method} {run.timestamp}: tenute {n(report['kept'])} su {n(report['gaussians'])} ({removed})")
        print(f"  -> {report['target']}")


if __name__ == "__main__":
    main()
