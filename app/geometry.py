"""Confronto geometrico tra i modelli e la nuvola densa della fotogrammetria (fasi 10 e 12 del piano).

La nuvola densa di COLMAP (colmap/dense/fused.ply) e' il RIFERIMENTO, non la verita': anche lei ha
errori. Si confrontano con essa, per ogni metodo:
  - i punti di superficie (app/surface.py), ricavati allo stesso modo per tutti: e' il confronto
    principale;
  - gli stessi punti dopo il filtro statistico uguale per tutti ("filtrata");
  - per il 3DGS, i centri delle gaussiane con opacita' > MIN_OPACITY ("centri"), analisi secondaria.
Le stesse misure si ripetono ristrette all'arco ("arch only"): riquadro che contiene i rettangoli dei
due piloni della verifica della scala (app/scala.py) e va dal suolo ad ARCH_HEIGHT_M, con ARCH_MARGIN_M
di margine su ogni lato; accuratezza dei punti del metodo nel riquadro, completezza dei punti della
nuvola densa nel riquadro.
Tutto e' riportato nel sistema di COLMAP (clean.to_colmap) e ritagliato a un riquadro attorno ai
punti della nuvola sparsa, ridotto ai percentili 2-98 su ogni asse nel sistema raddrizzato del
volo (verticale delle camere), cosi' si escludono cielo e intorno lontano.

Per ogni metodo, nei due versi:
  - accuratezza: distanza da ogni punto del metodo al punto piu' vicino della nuvola densa;
  - completezza: distanza da ogni punto della nuvola densa al punto piu' vicino del metodo;
con mediana, 90o percentile e frazione entro 5, 10 e 20 cm. Le distanze sono in metri con la
scala in uso in scala.json (app/scala.py, letta da paper.metres_per_unit): la scala verificata con
il basamento dei piloni misurato in sito, se c'e'. Senza scala restano in unita' dell'allineamento.
La nuvola densa e' sottocampionata a SAMPLE punti, quelle dei metodi a METHOD_SAMPLE (seme fisso).
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
METHOD_SAMPLE = 1_000_000  # come i punti di superficie (surface.SAMPLE)
VARIANTS = {"superficie": "superficie", "filtrata": "superficie filtrata", "centri": "centri delle gaussiane"}
SECTION_M = 0.05  # spessore della sezione verticale della figura fig6c
ARCH_HEIGHT_M = 15.7  # riquadro dell'arco ("arch only"): dal suolo a questa quota sopra i piloni
ARCH_MARGIN_M = 0.5  # margine del riquadro dell'arco su ogni lato
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
    """Centri delle gaussiane opache di un run 3DGS nel sistema di COLMAP; None per i NeRF."""
    export = run.export_file
    if export is None or config.METHODS[run.method].family != "gaussian":
        return None
    xyz, opacity = read_xyz(export)
    return clean.to_colmap(xyz[opacity > MIN_OPACITY], run, scene)


def variant_points(run: runs.Run, scene: Path, variant: str) -> Optional[np.ndarray]:
    """Nuvola di un run per una variante del confronto, nel sistema di COLMAP."""
    from . import surface

    if variant == "centri":
        return method_points(run, scene)
    path = surface.surface_file(run, filtered=variant == "filtrata")
    return read_xyz(path)[0] if path else None


def compare(project: str, metres: Optional[float]) -> Dict:
    """Accuratezza e completezza di ogni metodo e variante rispetto alla nuvola densa.

    I valori principali di ogni metodo sono quelli dei punti di superficie; le altre varianti stanno in
    results[metodo]["varianti"].
    """
    from scipy.spatial import cKDTree

    scene = config.WORK / "data" / project
    dense_file = scene / "colmap" / "dense" / "fused.ply"
    if not dense_file.exists():
        raise FileNotFoundError("Manca la nuvola densa: eseguire la fotogrammetria (passo mesh).")
    rotation = upright_frame(scene)
    box = region(scene, rotation)
    dense_all = inside(read_xyz(dense_file)[0], rotation, box)
    dense = subsample(dense_all)
    dense_tree = cKDTree(dense)
    scale = metres or 1.0
    arch = arch_box(project, metres) if metres else None
    dense_in_arch = in_arch(dense @ rotation.T, arch) if arch else None
    results: Dict[str, Dict] = {"_riferimento": {"punti": int(len(dense)), "metri": bool(metres),
                                                 "_dense": dense_all @ rotation.T, "_rotation": rotation}}
    groups = runs.condition_groups(runs.list_runs(project))
    for key, method_runs in (groups[0].items() if groups else []):
        variants = {}
        for variant in VARIANTS:
            points = variant_points(method_runs[0], scene, variant)
            if points is None:
                continue
            points = subsample(inside(points, rotation, box), METHOD_SAMPLE)
            if len(points) == 0:
                continue
            accuracy = dense_tree.query(points, k=1, workers=-1)[0]
            completeness = cKDTree(points).query(dense, k=1, workers=-1)[0]
            turned = points @ rotation.T
            variants[variant] = {"punti": int(len(points)), "accuratezza": distance_stats(accuracy, scale),
                                 "completezza": distance_stats(completeness, scale),
                                 "_accuracy": accuracy * scale, "_points": turned}
            if arch and dense_in_arch.any():
                mine = in_arch(turned, arch)
                if mine.any():
                    variants[variant]["arco"] = {"punti": int(mine.sum()),
                                                 "accuratezza": distance_stats(accuracy[mine], scale),
                                                 "completezza": distance_stats(completeness[dense_in_arch], scale)}
        if "superficie" in variants:
            results[key] = {**variants["superficie"], "varianti": variants}
    if arch:
        results["_riferimento"]["arco"] = {"punti": int(dense_in_arch.sum()),
                                           "lati_m": [float(v * scale) for v in arch["high"] - arch["low"]]}
    return results


def arch_box(project: str, metres: float) -> Optional[Dict]:
    """Riquadro dell'arco nel sistema raddrizzato, dai rettangoli dei due piloni in scala.json.

    Assi in pianta: lungo la facciata e in profondita' (lato lungo dei piloni); in quota dal suolo ad
    ARCH_HEIGHT_M. Tutto allargato di ARCH_MARGIN_M. None se manca la verifica della scala.
    """
    from . import scala

    check = (scala.load(project) or {}).get("verifica")
    if not check or len(check.get("piloni", [])) != 2 or "centro_unita" not in check["piloni"][0]:
        return None
    angle = np.radians(np.mean([p["orientamento_gradi"] for p in check["piloni"]]))
    depth = np.array([np.cos(angle), np.sin(angle)])
    axes = np.stack([np.array([-depth[1], depth[0]]), depth])  # righe: facciata, profondita'
    corners = []
    for pier in check["piloni"]:
        a = np.radians(pier["orientamento_gradi"])
        along, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        short, long = pier["lati_unita"]
        for i in (-0.5, 0.5):
            for j in (-0.5, 0.5):
                corners.append(np.asarray(pier["centro_unita"]) + i * long * along + j * short * across)
    flat = np.array(corners) @ axes.T
    margin = ARCH_MARGIN_M / metres
    ground = check["suolo_unita"]
    low = np.array([*flat.min(axis=0), ground]) - margin
    high = np.array([*flat.max(axis=0), ground + ARCH_HEIGHT_M / metres]) + margin
    return {"axes": axes, "low": low, "high": high}


def in_arch(turned: np.ndarray, arch: Dict) -> np.ndarray:
    """Maschera dei punti (gia' nel sistema raddrizzato) dentro il riquadro dell'arco."""
    local = np.column_stack([turned[:, :2] @ arch["axes"].T, turned[:, 2]])
    return np.all((local >= arch["low"]) & (local <= arch["high"]), axis=1)


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
    ax.set_title("Accuratezza: distanza di ogni punto di superficie dalla nuvola densa")
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
    fig.suptitle("Distanza dei punti di superficie dalla nuvola densa, vista dall'alto", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    _save(fig, out, "fig6b_distanze_pianta")
    plt.close(fig)


def section(results: Dict, project: str, metres: float) -> Optional[Dict]:
    """Sezione verticale sottile attraverso il fornice, in metri: nuvola densa e punti di superficie per metodo.

    Il piano passa per il centro tra i due piloni ed e' parallelo alla facciata (perpendicolare al lato
    lungo dei piloni, da scala.json); x corre lungo la facciata, z e' la quota sul suolo.
    """
    from . import scala

    check = (scala.load(project) or {}).get("verifica")
    if not check or len(check.get("piloni", [])) != 2 or "centro_unita" not in check["piloni"][0]:
        return None
    centre = np.mean([p["centro_unita"] for p in check["piloni"]], axis=0)
    depth_angle = np.radians(np.mean([p["orientamento_gradi"] for p in check["piloni"]]))
    depth_axis = np.array([np.cos(depth_angle), np.sin(depth_angle)])
    facade_axis = np.array([-depth_axis[1], depth_axis[0]])
    half = SECTION_M / 2 / metres
    ground = check["suolo_unita"]

    def cut(points: np.ndarray) -> np.ndarray:
        flat = points[:, :2] - centre
        keep = np.abs(flat @ depth_axis) <= half
        return np.column_stack([flat[keep] @ facade_axis, points[keep, 2] - ground]) * metres

    return {"_dense": cut(results["_riferimento"]["_dense"]),
            **{key: cut(r["_points"]) for key, r in results.items() if not key.startswith("_")}}


def figure_section(cuts: Dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .report import FAMILY_COLOR, _style
    from .paper import _save

    _style()
    keys = [k for k in cuts if not k.startswith("_")]
    dense = cuts["_dense"]
    span = np.percentile(np.abs(dense[:, 0]), 99) * 1.05
    shown = (np.abs(dense[:, 0]) <= span) & (dense[:, 1] > -1)
    top = np.percentile(dense[shown, 1], 99.9) + 0.5
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 4.4), sharey=True, constrained_layout=True,
                             squeeze=False)
    for ax, key in zip(axes[0], keys):
        points = cuts[key]
        visible = (np.abs(points[:, 0]) <= span) & (points[:, 1] > -1) & (points[:, 1] < top)
        ax.scatter(dense[shown, 0], dense[shown, 1], s=0.4, color="#b9b8b3", linewidth=0, label="nuvola densa")
        ax.scatter(points[visible, 0], points[visible, 1], s=1.2, linewidth=0,
                   color=FAMILY_COLOR[config.METHODS[key].family], label="superficie del metodo")
        ax.set_title(config.METHODS[key].short, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-span, span)
        ax.set_ylim(-1, top)
        ax.set_xlabel("lungo la facciata (m)")
        ax.tick_params(length=0)
    axes[0][0].set_ylabel("quota sul suolo (m)")
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left", markerscale=6)
    fig.suptitle(f"Sezione verticale attraverso il fornice, spessore {SECTION_M * 100:.0f} cm", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    _save(fig, out, "fig6c_profilo")
    plt.close(fig)


def table(results: Dict, out: Path, numbers: dict, unit: str) -> None:
    from .paper import _f, write_table

    rows = []
    for key, r in results.items():
        if key.startswith("_"):
            continue
        for arch_only in (False, True):
            for variant, label in VARIANTS.items():
                v = r["varianti"].get(variant)
                if arch_only:
                    if v is None and not any("arco" in x for x in r["varianti"].values()):
                        continue
                    v = v.get("arco") if v else None
                    label = f"{label}, arch only"
                if v is None:
                    rows.append([config.METHODS[key].label, label] + ["–"] * 9)
                    continue
                a, c = v["accuratezza"], v["completezza"]
                rows.append([config.METHODS[key].label, label, v["punti"], _f(a["median"], 3), _f(a["p90"], 3),
                             *[f"{a[f'within_{int(t * 100)}cm']:.1%}" for t in THRESHOLDS_M],
                             _f(c["median"], 3), _f(c["p90"], 3), f"{c['within_10cm']:.1%}"])
                prefix = f"geometria.{key}" if variant == "superficie" else f"geometria.{key}.{variant}"
                if arch_only:
                    prefix += ".arco"
                numbers[f"{prefix}.punti"] = v["punti"]
                for side, values in (("accuratezza", a), ("completezza", c)):
                    for name, value in values.items():
                        numbers[f"{prefix}.{side}.{name}"] = value
    numbers["geometria.punti_riferimento"] = results["_riferimento"]["punti"]
    if "arco" in results["_riferimento"]:
        numbers["geometria.arco.punti_riferimento"] = results["_riferimento"]["arco"]["punti"]
        numbers["geometria.arco.riquadro_m"] = results["_riferimento"]["arco"]["lati_m"]
        numbers["geometria.arco.altezza_m"], numbers["geometria.arco.margine_m"] = ARCH_HEIGHT_M, ARCH_MARGIN_M
    numbers["geometria.unita"] = unit
    write_table(out, "tab8_geometria",
                ["Metodo", "Nuvola", "Punti", f"Accuratezza mediana ({unit})", f"Accuratezza 90° perc. ({unit})",
                 "Entro 5 cm", "Entro 10 cm", "Entro 20 cm", f"Completezza mediana ({unit})",
                 f"Completezza 90° perc. ({unit})", "Completezza entro 10 cm"], rows,
                "Distanze rispetto alla nuvola densa della fotogrammetria (riferimento, non verità); "
                "«superficie»: punti di superficie con la stessa procedura per tutti i metodi (confronto principale); "
                f"«arch only»: riquadro dei due piloni dal suolo a {ARCH_HEIGHT_M} m, margine {ARCH_MARGIN_M} m")


def surface_numbers(project: str, numbers: dict) -> None:
    """Procedura dei punti di superficie (superficie.json) e controllo con ns-export per i NeRF."""
    from . import surface

    groups = runs.condition_groups(runs.list_runs(project))
    for key, method_runs in (groups[0].items() if groups else []):
        run = method_runs[0]
        info = runs.read_json(run.export_dir / surface.INFO_FILE)
        if info:
            numbers.update({f"geometria.{key}.superficie.viste": len(info["viste"]),
                            f"geometria.{key}.superficie.pixel_validi": info["pixel_validi"] / info["pixel"],
                            f"geometria.{key}.superficie.punti_nel_riquadro": info["punti_nel_riquadro"],
                            f"geometria.{key}.superficie.convenzione": info["convenzione_profondita"],
                            f"geometria.{key}.superficie.filtro_tenuti": info["filtro"]["punti_tenuti"]})
            numbers.update({"geometria.superficie.lato_px": info["lato_px"],
                            "geometria.superficie.vista_ogni": surface.VIEW_EVERY,
                            "geometria.superficie.opacita_minima": info["opacita_minima"],
                            "geometria.superficie.filtro_vicini": info["filtro"]["vicini"],
                            "geometria.superficie.filtro_std_ratio": info["filtro"]["std_ratio"]})
        check = runs.read_json(run.export_dir / surface.CHECK_FILE)
        for name, value in check.items():
            if name.startswith(("mediana", "p90", "punti")):
                numbers[f"geometria.{key}.controllo_ns_export.{name}"] = value


def build(project: str, out: Path, numbers: dict, metres: Optional[float], quality: Dict[str, float]) -> bool:
    """Tabella 8, figure 6 e numeri delle fasi 10 e 12; False se manca la nuvola densa."""
    try:
        results = compare(project, metres)
    except FileNotFoundError:
        return False
    unit = "m" if metres else "unità"
    table(results, out, numbers, unit)
    surface_numbers(project, numbers)
    figures(results, out, unit)
    cuts = section(results, project, metres) if metres else None
    if cuts:
        figure_section(cuts, out)
    rq4 = spearman_quality_geometry(results, quality)
    if rq4:
        numbers["geometria.rq4_spearman_rho"], numbers["geometria.rq4_spearman_p"] = rq4
        numbers["geometria.rq4_nota"] = (f"punti di superficie senza filtro; con {len(quality)} metodi il risultato "
                                         "è solo descrittivo")
    return True
