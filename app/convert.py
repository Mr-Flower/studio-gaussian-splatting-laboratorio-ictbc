"""Conversione dei modelli esportati in altri formati.

- Un modello di gaussiane (.ply di 3D Gaussian Splatting) diventa una nuvola di punti colorata:
  un punto al centro di ogni gaussiana, con il suo colore di base. Si perdono forma, trasparenza e
  riflessi delle gaussiane: e' una nuvola per misurare o per programmi che non leggono le gaussiane.
- Una nuvola di punti o una mesh .ply diventa un file .glb (glTF binario).

Il formato glTF non ha ancora un modo diffuso di contenere le gaussiane: un .glb ottenuto da un
modello di gaussiane contiene la nuvola di punti, non le gaussiane.

Uso:  python -m app.convert <modello.ply> <uscita.ply | uscita.glb> [--min-opacity 0.1]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

SH_C0 = 0.28209479177387814  # coefficiente dell'armonica sferica di grado zero: da f_dc al colore
MIN_OPACITY = 0.1  # sotto questa opacita' una gaussiana e' quasi invisibile: non diventa un punto


def read_points(source: Path, min_opacity: float = MIN_OPACITY) -> Tuple[np.ndarray, np.ndarray]:
    """Posizioni [N, 3] e colori [N, 3] (0-255) dei punti di un .ply di gaussiane o di una nuvola di punti."""
    from plyfile import PlyData

    vertex = PlyData.read(str(source))["vertex"]
    names = {p.name for p in vertex.properties}
    positions = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float32)
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:  # gaussiane
        dc = np.stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]], axis=1).astype(np.float32)
        colors = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0) * 255.0
        if "opacity" in names and min_opacity > 0:
            opacity = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))
            keep = opacity >= min_opacity
            positions, colors = positions[keep], colors[keep]
    elif {"red", "green", "blue"} <= names:
        colors = np.stack([vertex["red"], vertex["green"], vertex["blue"]], axis=1).astype(np.float32)
    else:
        colors = np.full_like(positions, 200.0)
    finite = np.isfinite(positions).all(axis=1)
    return positions[finite], np.round(colors[finite]).astype(np.uint8)


def write_point_cloud(positions: np.ndarray, colors: np.ndarray, target: Path) -> None:
    from plyfile import PlyData, PlyElement

    points = np.empty(len(positions), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                              ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    points["x"], points["y"], points["z"] = positions.T
    points["red"], points["green"], points["blue"] = colors.T
    target.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(points, "vertex")]).write(str(target))


def has_faces(source: Path) -> bool:
    """True se il .ply e' una mesh (ha facce), False se contiene solo punti."""
    with open(source, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element face"):
                return int(line.split()[-1]) > 0
            if line == "end_header" or not line:
                break
    return False


def write_glb(source: Path, target: Path, min_opacity: float = MIN_OPACITY) -> int:
    """Scrive un .glb con la mesh o con la nuvola di punti contenuta in `source`. Restituisce il numero di vertici."""
    import trimesh

    target.parent.mkdir(parents=True, exist_ok=True)
    if has_faces(source):
        mesh = trimesh.load(str(source), process=False)
        mesh.export(str(target))
        return len(mesh.vertices)
    positions, colors = read_points(source, min_opacity)
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    trimesh.PointCloud(positions, colors=np.hstack([colors, alpha])).export(str(target))
    return len(positions)


def convert(source: Path, target: Path, min_opacity: float = MIN_OPACITY) -> int:
    """Converte secondo l'estensione di `target` (.ply: nuvola di punti; .glb). Restituisce punti o vertici scritti."""
    suffix = target.suffix.lower()
    if suffix == ".glb":
        return write_glb(source, target, min_opacity)
    if suffix == ".ply":
        positions, colors = read_points(source, min_opacity)
        write_point_cloud(positions, colors, target)
        return len(positions)
    raise ValueError(f"Formato non previsto: {suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--min-opacity", type=float, default=MIN_OPACITY,
                        help="gaussiane meno opache di cosi' non diventano punti (0 = tutte)")
    args = parser.parse_args()
    count = convert(args.source, args.target, args.min_opacity)
    print(f"{args.target}: {count:,} punti o vertici".replace(",", "."))


if __name__ == "__main__":
    main()
