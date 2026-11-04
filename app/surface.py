"""Punti di superficie ricavati allo stesso modo per tutti i metodi (fase 12 del piano).

Il confronto geometrico della fase 10 usava nuvole di natura diversa: per i NeRF i punti esportati da
nerfstudio (profondita' lungo i raggi delle foto, con un filtro degli outlier), per il 3DGS i centri
delle gaussiane opache. Qui ogni modello e' trattato con la stessa procedura:

  - viste: una foto di training ogni VIEW_EVERY, in ordine di nome (le stesse per tutti), rese con
    lato maggiore SIDE px; ogni metodo usa le proprie camere, come in valutazione (nerfstudio con la
    distorsione OPENCV per i NeRF e le camere corrette per splatfacto, il codice Inria con le camere
    gia' raddrizzate di cameras.json);
  - per pixel: profondita' attesa e opacita' accumulata; si tengono i pixel con opacita' > MIN_ALPHA,
    si ritroiettano in 3D e si portano nel sistema di COLMAP (clean.to_colmap);
  - campionamento: si uniscono i punti di tutte le viste, si ritagliano al riquadro di
    geometry.region e se ne estraggono SAMPLE con seme SEED;
  - filtro: nessuno. A parte si salva la variante con lo stesso filtro statistico per tutti
    (FILTER_NEIGHBOURS vicini, FILTER_STD_RATIO, come ns-export di nerfstudio), fatto con cKDTree.

Convenzioni di profondita': in splatfacto (gsplat, modo "RGB+ED") e' la z nel sistema della camera,
in nerfacto e' la distanza lungo il raggio (direzioni unitarie). Per il codice Inria si usa un metodo
indipendente dalla versione: si rende con override_color = z in camera di ogni gaussiana e con
override_color = 1 (opacita' accumulata); la profondita' e' il rapporto. Anche questa e' una z.

Uscita: exports/<progetto>/<metodo>_<data>/superficie.ply, superficie_filtrata.ply e superficie.json.

Uso (dall'ambiente con PyTorch):
  python -m app.surface --project <nome>                 tutti i run del confronto principale
  python -m app.surface --project <nome> --only <metodo>
  python -m app.surface --project <nome> --controllo     confronto con ns-export per i NeRF
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import clean, colmap_model, config, geometry, runs

VIEW_EVERY = 8
SIDE = 1320
MIN_ALPHA = 0.5
SAMPLE = 1_000_000
SEED = 0
FILTER_NEIGHBOURS = 20
FILTER_STD_RATIO = 10.0
SURFACE_FILE = "superficie.ply"
FILTERED_FILE = "superficie_filtrata.ply"
INFO_FILE = "superficie.json"
CHECK_FILE = "superficie_controllo.json"
CONVENTION = {"splatfacto": "z", "splatfacto-big": "z", "inria-3dgs": "z", "nerfacto": "ray", "nerfacto-big": "ray"}


# --- geometria, senza scheda grafica

def surface_views(split: dict) -> List[str]:
    """Le viste di training usate: una ogni VIEW_EVERY in ordine di nome, uguali per tutti i metodi."""
    return sorted(split["train"])[::VIEW_EVERY]


def backproject(origins: np.ndarray, directions: np.ndarray, depth: np.ndarray, forward: np.ndarray,
                convention: str) -> np.ndarray:
    """Punti 3D dai raggi (direzioni unitarie) e dalla profondita' di ogni pixel.

    convention "ray": la profondita' e' la distanza lungo il raggio; "z": e' la coordinata lungo l'asse
    ottico della camera (`forward`, unitario), quindi la distanza lungo il raggio e' depth / cos.
    """
    if convention == "ray":
        t = depth
    elif convention == "z":
        t = depth / (directions @ forward)
    else:
        raise ValueError(convention)
    return origins + directions * t[..., None]


def pinhole_rays(width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 camera_to_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origini, direzioni unitarie (centri dei pixel) e asse ottico di una camera stenopeica.

    camera_to_world e' 3x4 nella convenzione di COLMAP/OpenCV: x a destra, y in basso, z in avanti.
    """
    u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    local = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    rotation, centre = camera_to_world[:, :3], camera_to_world[:, 3]
    directions = local @ rotation.T
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.broadcast_to(centre, directions.shape), directions, rotation[:, 2]


def project(points: np.ndarray, fx: float, fy: float, cx: float, cy: float, camera_to_world: np.ndarray) -> np.ndarray:
    """Coordinate in pixel (u, v) dei punti nella camera stenopeica (inversa di pinhole_rays)."""
    rotation, centre = camera_to_world[:, :3], camera_to_world[:, 3]
    local = (points - centre) @ rotation
    return np.stack([fx * local[..., 0] / local[..., 2] + cx, fy * local[..., 1] / local[..., 2] + cy], axis=-1)


def statistical_filter(points: np.ndarray, neighbours: int = FILTER_NEIGHBOURS,
                       std_ratio: float = FILTER_STD_RATIO) -> np.ndarray:
    """Maschera dei punti tenuti dal filtro statistico (come remove_statistical_outlier di Open3D).

    Per ogni punto si calcola la distanza media dai `neighbours` vicini; si scartano i punti con
    distanza media oltre media + std_ratio * deviazione standard, calcolate su tutti i punti.
    """
    from scipy.spatial import cKDTree

    distances = cKDTree(points).query(points, k=neighbours + 1, workers=-1)[0][:, 1:].mean(axis=1)
    return distances <= distances.mean() + std_ratio * distances.std()


def write_ply(path: Path, points: np.ndarray) -> None:
    """Nuvola di punti in un .ply binario con le sole posizioni (float32)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(points, dtype="<f4")
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(data)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def read_ply(path: Path) -> np.ndarray:
    return geometry.read_xyz(path)[0]


# --- rendering (PyTorch)

# Una vista: nome della foto, punti nel sistema del run (pixel con opacita' > MIN_ALPHA), pixel totali.
ViewPoints = Tuple[str, np.ndarray, int]


def nerfstudio_points(run: runs.Run, names: Sequence[str], frames: Dict[str, str]) -> Iterator[ViewPoints]:
    import torch

    config.use_nerfstudio_variants()
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(run.config_file, test_mode="inference")
    pipeline.eval()
    dataset = pipeline.datamanager.train_dataset
    index = {frames.get(Path(f).name, Path(f).name).replace("\\", "/"): i for i, f in enumerate(dataset.image_filenames)}
    convention = CONVENTION[run.method]
    for name in names:
        camera = deepcopy(dataset.cameras[index[name]:index[name] + 1])
        if convention == "z":  # splatfacto rende una camera stenopeica: i raggi devono esserlo anche loro
            camera.distortion_params = torch.zeros_like(camera.distortion_params)
        camera.rescale_output_resolution(SIDE / max(int(camera.width), int(camera.height)))
        camera = camera.to(pipeline.device)
        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)
            rays = camera.generate_rays(camera_indices=0, keep_shape=True)
        depth = outputs["depth" if convention == "z" else "expected_depth"][..., 0].double().cpu().numpy()
        alpha = outputs["accumulation"][..., 0].cpu().numpy()
        origins = rays.origins.double().cpu().numpy()
        directions = rays.directions.double().cpu().numpy()
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        forward = -camera.camera_to_worlds[0, :3, 2].double().cpu().numpy()  # nerfstudio: la camera guarda verso -z
        keep = alpha > MIN_ALPHA
        points = backproject(origins[keep], directions[keep], depth[keep], forward, convention)
        yield name, points, int(alpha.size)


def inria_points(run: runs.Run, names: Sequence[str]) -> Iterator[ViewPoints]:
    import torch

    sys.path.insert(0, str(config.inria_repo()))
    from gaussian_renderer import GaussianModel, render
    from scene.cameras import MiniCam
    from utils.graphics_utils import getProjectionMatrix

    gaussians = GaussianModel(3)
    gaussians.load_ply(str(run.checkpoint()))
    pipe = argparse.Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, device="cuda")
    cameras = {c["img_name"].replace("\\", "/"): c for c in json.loads((run.path / "cameras.json").read_text())}
    by_stem = {name.rsplit(".", 1)[0]: c for name, c in cameras.items()}
    means = gaussians.get_xyz
    ones = torch.ones((len(means), 3), device="cuda")
    for name in names:
        c = cameras.get(name) or by_stem[name.rsplit(".", 1)[0]]
        camera_to_world = np.asarray(c["rotation"], dtype=np.float64)  # cameras.json: rotazione camera -> mondo
        centre = np.asarray(c["position"], dtype=np.float64)
        scale = SIDE / max(c["width"], c["height"])
        width, height = round(c["width"] * scale), round(c["height"] * scale)
        fovx, fovy = 2 * math.atan(c["width"] / (2 * c["fx"])), 2 * math.atan(c["height"] / (2 * c["fy"]))
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = camera_to_world.T
        world_to_camera[:3, 3] = -camera_to_world.T @ centre
        view = torch.tensor(world_to_camera, dtype=torch.float32, device="cuda").transpose(0, 1)
        projection = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        mini = MiniCam(width, height, fovy, fovx, 0.01, 100.0, view, view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0))
        with torch.no_grad():
            z = (means @ view[:3, :3] + view[3, :3])[:, 2]  # z in camera (matrici trasposte, come nel codice Inria)
            # render() limita l'immagine a 0..1: la z si passa divisa per la massima e si rimoltiplica dopo.
            top = float(z.max().clamp_min(1e-6))
            weighted = render(mini, gaussians, pipe, background,
                              override_color=(z / top).clamp(0, 1)[:, None].repeat(1, 3))["render"][0]
            alpha = render(mini, gaussians, pipe, background, override_color=ones)["render"][0]
        alpha_np = alpha.double().cpu().numpy()
        depth = (weighted.double() * top / alpha.double().clamp_min(1e-12)).cpu().numpy()
        # Il rasterizzatore Inria ha il punto principale al centro e i pixel centrati (come pinhole_rays).
        fx, fy = width / (2 * math.tan(fovx / 2)), height / (2 * math.tan(fovy / 2))
        origins, directions, forward = pinhole_rays(width, height, fx, fy, width / 2, height / 2,
                                                    np.column_stack([camera_to_world, centre]))
        keep = alpha_np > MIN_ALPHA
        yield name, backproject(origins[keep], directions[keep], depth[keep], forward, "z"), int(alpha_np.size)


# --- un run

def main_runs(project: str) -> Dict[str, runs.Run]:
    groups = runs.condition_groups(runs.list_runs(project))
    return {key: found[0] for key, found in (groups[0].items() if groups else [])}


def all_points(run: runs.Run, scene: Path) -> Tuple[List[str], np.ndarray, int, int]:
    """Viste usate, punti di tutte le viste nel riquadro (sistema di COLMAP), pixel totali e validi."""
    split = colmap_model.read_split(scene)
    names = surface_views(split)
    rotation = geometry.upright_frame(scene)
    box = geometry.region(scene, rotation)
    views = (inria_points(run, names) if run.engine == "inria"
             else nerfstudio_points(run, names, split.get("frames", {})))
    kept, pixels, valid = [], 0, 0
    for i, (name, points, total) in enumerate(views, 1):
        pixels += total
        valid += len(points)
        kept.append(geometry.inside(clean.to_colmap(points, run, scene), rotation, box).astype(np.float32))
        print(f"{run.method}: vista {i}/{len(names)}", flush=True)
    return names, np.concatenate(kept), pixels, valid


def build_run(run: runs.Run, scene: Path) -> Dict:
    start = time.time()
    names, joined, pixels, valid = all_points(run, scene)
    sampled = geometry.subsample(joined, SAMPLE, SEED)
    write_ply(run.export_dir / SURFACE_FILE, sampled)
    keep = statistical_filter(sampled.astype(np.float64))
    write_ply(run.export_dir / FILTERED_FILE, sampled[keep])
    info = {"metodo": run.method, "run": run.timestamp, "viste": names, "lato_px": SIDE, "opacita_minima": MIN_ALPHA,
            "convenzione_profondita": CONVENTION[run.method], "pixel": pixels, "pixel_validi": valid,
            "punti_nel_riquadro": int(len(joined)), "punti_campionati": int(len(sampled)), "seme": SEED,
            "filtro": {"vicini": FILTER_NEIGHBOURS, "std_ratio": FILTER_STD_RATIO, "punti_tenuti": int(keep.sum())},
            "secondi": round(time.time() - start, 1), "creato": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (run.export_dir / INFO_FILE).write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def surface_file(run: runs.Run, filtered: bool = False) -> Optional[Path]:
    path = run.export_dir / (FILTERED_FILE if filtered else SURFACE_FILE)
    return path if path.exists() else None


# --- controllo di coerenza per i NeRF (ns-export senza filtro)

def check_nerf(run: runs.Run, scene: Path, metres: Optional[float]) -> Dict:
    """Confronta superficie.ply con ns-export pointcloud dello stesso run, senza filtro degli outlier.

    ns-export campiona 1 milione di raggi a caso dalle foto di training, con la stessa profondita'
    attesa: si misura la distanza mediana di quei punti dalla superficie di tutte le viste PRIMA del
    campionamento (circa 100 milioni di punti, ricalcolati qui), cosi' la distanza non dipende dalla
    densita' della nuvola. Il verso opposto, dai punti di superficie campionati a quelli di ns-export,
    dipende dalla densita' di ns-export ed e' solo indicativo.
    """
    from scipy.spatial import cKDTree

    folder = run.export_dir / "controllo_ns_export"
    command = [config.ns("ns-export"), "pointcloud", "--load-config", str(run.config_file), "--output-dir", str(folder),
               "--num-points", str(SAMPLE), "--remove-outliers", "False", "--depth-output-name", "expected_depth",
               "--normal-method", "open3d", "--save-world-frame", "False"]
    if not (folder / "point_cloud.ply").exists():
        subprocess.run(command, check=True, env=config.environment())
    rotation = geometry.upright_frame(scene)
    box = geometry.region(scene, rotation)
    exported = geometry.inside(clean.to_colmap(read_ply(folder / "point_cloud.ply"), run, scene), rotation, box)
    full = all_points(run, scene)[1]
    scale = metres or 1.0
    to_ours = cKDTree(full).query(exported, k=1, workers=-1)[0] * scale
    to_export = cKDTree(exported).query(read_ply(run.export_dir / SURFACE_FILE), k=1, workers=-1)[0] * scale
    result = {"metodo": run.method, "punti_ns_export": int(len(exported)), "punti_superficie_completa": int(len(full)),
              "metri": bool(metres),
              "mediana_ns_export_verso_superficie": float(np.median(to_ours)),
              "mediana_superficie_verso_ns_export": float(np.median(to_export)),
              "p90_ns_export_verso_superficie": float(np.percentile(to_ours, 90))}
    (run.export_dir / CHECK_FILE).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--only", help="solo questo metodo (chiave, es. splatfacto)")
    parser.add_argument("--controllo", action="store_true", help="confronto con ns-export per i NeRF")
    args = parser.parse_args()
    scene = config.WORK / "data" / args.project
    found = main_runs(args.project)
    if args.controllo:
        from .paper import metres_per_unit
        for key, run in found.items():
            if config.METHODS[key].family == "nerf":
                print(json.dumps(check_nerf(run, scene, metres_per_unit(args.project)), indent=1))
        return
    if args.only:
        info = build_run(found[args.only], scene)
        print(f"{args.only}: {info['punti_nel_riquadro']} punti nel riquadro, {info['punti_campionati']} campionati, "
              f"{info['filtro']['punti_tenuti']} dopo il filtro, {info['secondi']:.0f} s")
        return
    # Un processo per run: il codice Inria e nerfstudio non convivono bene nello stesso interprete.
    for key in found:
        subprocess.run([sys.executable, "-m", "app.surface", "--project", args.project, "--only", key], check=True, env=config.environment(),
                       cwd=config.ROOT)


if __name__ == "__main__":
    main()
