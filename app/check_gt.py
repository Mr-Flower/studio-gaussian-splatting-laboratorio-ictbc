"""Verifica delle immagini di riferimento: quanto differiscono quelle di nerfstudio e del codice Inria.

Ogni metodo e' valutato contro l'immagine vera preparata dal programma che lo ha allenato.
nerfstudio (gsplat) corregge la distorsione con OpenCV e ritaglia la zona valida; il codice Inria
usa le foto corrette da COLMAP, con un ritaglio e un centro ottico diversi. Le due immagini vere
di una vista sono quindi due camere ideali con la stessa posa e parametri interni diversi.

Per confrontarle si riproietta l'immagine Inria nella camera di nerfstudio: a ogni pixel di
nerfstudio corrisponde una direzione, e quella direzione cade in un punto preciso dell'immagine
Inria, che si legge con interpolazione bilineare. Si misurano PSNR e SSIM sulla sola zona comune.
L'interpolazione aggiunge a sua volta un piccolo errore: il risultato e' un limite superiore della
differenza tra le due procedure.

Le immagini vere dei NeRF sono le foto non corrette (nerfacto tiene conto della distorsione nel
calcolo dei raggi): non si confrontano pixel per pixel con le altre e qui non sono considerate.

Uso (dall'ambiente con PyTorch e la scheda grafica):
  python -m app.check_gt --project <nome>   ->  reports/<nome>/check_gt.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from . import colmap_model, config, pipeline, runs
from .evaluate import inria_scene, original_names

NEGLIGIBLE_PSNR = 40.0  # sopra questa soglia media la differenza si cita come limite trascurabile
BORDER = 2  # pixel esclusi lungo il bordo della zona comune, dove l'interpolazione legge fuori immagine


def choose_runs(project: str, downscale: Optional[int] = None) -> Tuple[runs.Run, runs.Run]:
    """Un run di gsplat (splatfacto o splatfacto-big) e uno del codice Inria fatti nelle stesse condizioni.

    Senza `downscale` si usa il gruppo del confronto principale (quello con piu' metodi, come in
    report e app/paper.py); con `downscale` il primo gruppo a quel fattore di riduzione.
    """
    for group in runs.condition_groups(runs.list_runs(project)):
        gsplat = next((group[m][0] for m in ("splatfacto", "splatfacto-big") if m in group), None)
        inria = group.get(config.INRIA, [None])[0]
        if gsplat is None or inria is None:
            continue
        if downscale is None or gsplat.info().get("downscale") == downscale:
            return gsplat, inria
    raise SystemExit("Servono un run valutato di gsplat (splatfacto o splatfacto-big) e uno del metodo Inria nelle "
                     "stesse condizioni, sull'allineamento corrente" + (f" e con riduzione {downscale}." if downscale else "."))


def warp(image: torch.Tensor, source: Tuple[float, float, float, float], target: Tuple[float, float, float, float],
         size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Riproietta `image` [H, W, 3] dalla camera `source` alla camera `target` con la stessa posa.

    Le camere sono (fx, fy, cx, cy) con l'origine nell'angolo del primo pixel, la convenzione di
    COLMAP e di nerfstudio. Restituisce l'immagine [h, w, 3] della camera `target` (size = (w, h))
    e la maschera dei pixel che cadono dentro l'immagine di partenza, lontano dal bordo.
    """
    fx_s, fy_s, cx_s, cy_s = source
    fx_t, fy_t, cx_t, cy_t = target
    width, height = size
    device = image.device
    u = (torch.arange(width, device=device, dtype=torch.float64) + 0.5 - cx_t) / fx_t * fx_s + cx_s
    v = (torch.arange(height, device=device, dtype=torch.float64) + 0.5 - cy_t) / fy_t * fy_s + cy_s
    src_h, src_w = image.shape[:2]
    # grid_sample con align_corners=False: -1 e +1 sono i bordi esterni dei pixel, come l'origine qui.
    grid = torch.stack(torch.meshgrid(2 * v / src_h - 1, 2 * u / src_w - 1, indexing="ij")[::-1], dim=-1)
    warped = torch.nn.functional.grid_sample(image.permute(2, 0, 1)[None].float(), grid[None].float(),
                                             mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    inside_u = (u >= 0.5 + BORDER) & (u <= src_w - 0.5 - BORDER)
    inside_v = (v >= 0.5 + BORDER) & (v <= src_h - 0.5 - BORDER)
    return warped, inside_v[:, None] & inside_u[None, :]


def compare(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """PSNR e SSIM tra due immagini [H, W, 3] in 0..1 sul rettangolo coperto dalla maschera."""
    from torchmetrics.functional import structural_similarity_index_measure

    rows, cols = torch.where(mask.any(dim=1))[0], torch.where(mask.any(dim=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Le due immagini non hanno una zona comune.")
    crop = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1))
    x, y = a[crop].clamp(0, 1), b[crop].clamp(0, 1)
    mse = torch.mean((x - y) ** 2).item()
    psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(1.0 / mse)).item()
    ssim = structural_similarity_index_measure(x.permute(2, 0, 1)[None], y.permute(2, 0, 1)[None], data_range=1.0).item()
    return {"psnr": psnr, "ssim": ssim, "common_fraction": mask.float().mean().item()}


def nerfstudio_references(run: runs.Run, split: dict):
    """Per ogni vista di test: nome originale, immagine vera di nerfstudio e camera (fx, fy, cx, cy)."""
    config.use_nerfstudio_variants()
    from nerfstudio.utils.eval_utils import eval_setup

    _, ns_pipeline, _, _ = eval_setup(run.config_file, test_mode="test")
    manager = ns_pipeline.datamanager
    images = manager.cached_eval  # caricarle aggiorna i parametri delle camere dopo la correzione
    cameras = manager.eval_dataset.cameras
    named = original_names([(Path(name).name, index) for index, name in enumerate(manager.eval_dataset.image_filenames)],
                           split)
    for name, index in named:
        camera = cameras[index]
        intrinsics = tuple(float(getattr(camera, k).item()) for k in ("fx", "fy", "cx", "cy"))
        yield name, (lambda index=index: ns_pipeline.model.get_gt_img(images[index]["image"])), intrinsics


def inria_references(run: runs.Run, scene_dir: Path, split: dict) -> Dict[str, tuple]:
    """Per ogni vista di test: (camera Inria, parametri fx, fy, cx, cy dal modello COLMAP corretto)."""
    scene, *_ = inria_scene(run.path, config.inria_repo())
    model = pipeline.inria_dataset(scene_dir, run.info().get("downscale") or 1) / "sparse" / "0"
    cameras, poses = colmap_model.read_cameras(model), colmap_model.read_poses(model)
    by_stem = {name.rsplit(".", 1)[0]: name for name in poses}
    references = {}
    for name, view in original_names([(view.image_name, view) for view in scene.getTestCameras()], split):
        camera = cameras[poses[by_stem.get(name.rsplit(".", 1)[0], name)].camera_id]
        if camera.model != "PINHOLE":
            raise SystemExit(f"Camera {camera.model} inattesa nel modello corretto da COLMAP: serve PINHOLE.")
        references[name] = (view, tuple(camera.params))
    return references


def check(project: str, downscale: Optional[int] = None) -> Path:
    scene_dir = config.WORK / "data" / project
    split = json.loads((scene_dir / colmap_model.SPLIT_FILE).read_text(encoding="utf-8"))
    gsplat, inria = choose_runs(project, downscale)
    factor = gsplat.info().get("downscale")
    inria_refs = inria_references(inria, scene_dir, split)
    per_view: List[dict] = []
    with torch.no_grad():
        for index, (name, load, ns_camera) in enumerate(nerfstudio_references(gsplat, split)):
            ns_image = load()  # [H, W, 3] in 0..1, sulla scheda grafica
            view, inria_camera = inria_refs[name]
            inria_image = view.original_image.cuda().permute(1, 2, 0)
            size = (ns_image.shape[1], ns_image.shape[0])
            warped, mask = warp(inria_image, inria_camera, ns_camera, size)
            per_view.append({
                "name": name, "camera": name.split("/")[0] if "/" in name else "",
                "size_nerfstudio": list(size), "size_inria": [inria_image.shape[1], inria_image.shape[0]],
                "camera_nerfstudio": ns_camera, "camera_inria": inria_camera,
                **compare(ns_image.float(), warped, mask),
            })
            print(f"vista {index + 1}/{len(split['test'])}: PSNR {per_view[-1]['psnr']:.2f} dB", flush=True)
    finite = [v["psnr"] for v in per_view if v["psnr"] != float("inf")]
    summary = {
        "views": len(per_view),
        "psnr_mean": statistics.mean(finite) if finite else None,
        "psnr_min": min(finite) if finite else None,
        "ssim_mean": statistics.mean(v["ssim"] for v in per_view),
        "common_fraction_mean": statistics.mean(v["common_fraction"] for v in per_view),
        "same_size_views": sum(v["size_nerfstudio"] == v["size_inria"] for v in per_view),
        "negligible": bool(finite) and statistics.mean(finite) > NEGLIGIBLE_PSNR,
        "threshold_psnr": NEGLIGIBLE_PSNR,
    }
    # Il confronto principale va in check_gt.json; le altre risoluzioni in check_gt_<fattore>.json.
    name = "check_gt.json" if downscale is None else f"check_gt_{factor}.json"
    out = config.WORK / "reports" / project / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "runs": {"gsplat": str(gsplat.path), "inria": str(inria.path)},
        "downscale": factor, "size_nerfstudio": per_view[0]["size_nerfstudio"] if per_view else None,
        "method": "riproiezione dell'immagine Inria nella camera di nerfstudio, interpolazione bilineare, "
                  f"zona comune meno {BORDER} pixel di bordo",
        "summary": summary, "per_view": per_view,
    }, indent=1), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--downscale", type=int,
                        help="fattore di riduzione dei run da usare (predefinito: quelli del confronto principale)")
    args = parser.parse_args()
    out = check(args.project, args.downscale)
    s = json.loads(out.read_text(encoding="utf-8"))["summary"]
    verdict = "trascurabile" if s["negligible"] else f"NON trascurabile (soglia {s['threshold_psnr']:.0f} dB)"
    print(f"{s['views']} viste: PSNR medio {s['psnr_mean']:.2f} dB (minimo {s['psnr_min']:.2f}), SSIM medio "
          f"{s['ssim_mean']:.4f}, zona comune {s['common_fraction_mean'] * 100:.1f}% -> differenza {verdict}.")
    print(f"Salvato in {out}")


if __name__ == "__main__":
    main()
