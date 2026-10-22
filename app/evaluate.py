"""Valutazione uniforme di un run sulle viste di test.

Per ogni vista esclusa dal training si genera l'immagine con il modello allenato, si misura il solo
tempo di rendering e si calcolano PSNR, SSIM e LPIPS con lo stesso codice per tutti i metodi,
qualunque sia il programma che li ha allenati. Alcune viste sono salvate per il confronto visivo.

Oltre alle metriche grezze si calcolano quelle dopo una correzione affine del colore per canale
(psnr_cc, ssim_cc, lpips_cc): per ogni canale si trovano a e b che minimizzano
||a * generata + b - vera||^2 sui pixel della vista, si applicano e si limita a 0..1. Le foto del
rilievo hanno esposizione e bilanciamento del bianco automatici, mentre un modello senza
compensazione rende un aspetto medio: la correzione assorbe le differenze globali di esposizione
e di colore, non gli errori locali. Poiche' usa l'immagine vera, il valore corretto e' un limite
superiore della fedelta' del modello: vanno riportati entrambi.

Con --eval-side N (ripetibile) le metriche si ricalcolano anche dopo aver ridotto entrambe le
immagini a lato maggiore N (media sulle aree), cosi' modelli allenati a risoluzioni diverse si
confrontano alla stessa scala; le scale pari o superiori a quella della vista si saltano.

Uso (dall'ambiente con PyTorch):
  python -m app.evaluate --engine nerfstudio --run <cartella del run> --split <split.json>
  python -m app.evaluate --engine inria --run <cartella del run> --split <split.json> --repo <gaussian-splatting>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Sequence, Tuple

import torch

VIEWS_DIR = "test_views"
SAVED_VIEWS = 6

# Una vista: nome della foto, funzione che genera (immagine prodotta, immagine vera) come [H, W, 3] in 0..1.
View = Tuple[str, Callable[[], Tuple[torch.Tensor, torch.Tensor]]]


def nerfstudio_views(run: Path) -> Iterator[View]:
    from . import config
    config.use_nerfstudio_variants()  # anche quando il modulo e' eseguito a mano, fuori dal programma
    from nerfstudio.utils.eval_utils import eval_setup

    _, pipeline, _, _ = eval_setup(run / "config.yml", test_mode="test")
    pipeline.eval()
    names = pipeline.datamanager.eval_dataset.image_filenames
    for index, (camera, batch) in enumerate(pipeline.datamanager.fixed_indices_eval_dataloader):

        def render(camera=camera, batch=batch):
            outputs = pipeline.model.get_outputs_for_camera(camera=camera)
            torch.cuda.synchronize()
            return outputs, batch

        def finish(outputs, batch):
            # Ogni modello prepara a modo suo l'immagine vera (sfondo, ritaglio): si riusa la sua
            # funzione, che restituisce vera e prodotta affiancate.
            _, images = pipeline.model.get_image_metrics_and_images(outputs, batch)
            both = images["img"]
            half = both.shape[1] // 2
            return both[:, half:, :3], both[:, :half, :3]

        yield Path(names[int(batch.get("image_idx", index))]).name, (render, finish)


def inria_scene(run: Path, repo: Path):
    """Scena del codice Inria per un run: telecamere (foto lette una alla volta), gaussiane e opzioni."""
    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], "-m", str(run)]
    # Le foto si leggono una alla volta: caricarle tutte, comprese quelle di training, non serve
    # e ad alta risoluzione non entrerebbero in memoria.
    from . import inria_lazy
    inria_lazy.install()
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel
    from scene import Scene

    parser = argparse.ArgumentParser()
    model, pipe = ModelParams(parser, sentinel=True), PipelineParams(parser)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    return scene, gaussians, dataset, pipe.extract(args)


def inria_views(run: Path, repo: Path) -> Iterator[View]:
    scene, gaussians, dataset, pipeline = inria_scene(run, repo)
    from gaussian_renderer import render as render_view

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
        separate_sh = True
    except ImportError:
        separate_sh = False

    background = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")
    for view in scene.getTestCameras():

        def render(view=view):
            image = render_view(view, gaussians, pipeline, background, use_trained_exp=dataset.train_test_exp,
                                separate_sh=separate_sh)["render"]
            torch.cuda.synchronize()
            return image, view

        def finish(image, view):
            return image.clamp(0, 1).permute(1, 2, 0), view.original_image[0:3].clamp(0, 1).permute(1, 2, 0)

        yield view.image_name, (render, finish)


def original_names(views: List[View], split: dict) -> List[View]:
    """Riporta ogni vista al nome originale della foto e verifica che siano proprio quelle di test.

    nerfstudio lavora su copie rinominate (frame_00012.jpg), il codice Inria puo' togliere
    l'estensione: senza questo passaggio le viste dei diversi metodi non si potrebbero abbinare.
    """
    test = split["test"]
    by_stem = {name.rsplit(".", 1)[0]: name for name in test}
    renamed = []
    for name, functions in views:
        name = split.get("frames", {}).get(name, name).replace("\\", "/")
        name = name if name in test else by_stem.get(name, by_stem.get(name.rsplit(".", 1)[0]))
        if name is None:
            raise SystemExit("Una vista valutata non appartiene alle viste di test del progetto.")
        renamed.append((name, functions))
    if sorted(name for name, _ in renamed) != sorted(test):
        raise SystemExit(f"Le viste valutate ({len(renamed)}) non coincidono con le viste di test del progetto ({len(test)}).")
    return sorted(renamed, key=lambda view: view[0])


METRICS = ("psnr", "ssim", "lpips")
CORRECTED = tuple(m + "_cc" for m in METRICS)


def color_correct(predicted: torch.Tensor, truth: torch.Tensor) -> Tuple[torch.Tensor, List[float], List[float]]:
    """Correzione affine per canale di `predicted` [H, W, 3] verso `truth`, ai minimi quadrati.

    Restituisce l'immagine corretta (limitata a 0..1, stesso tipo e dispositivo) e i coefficienti
    a e b dei tre canali. Calcolo in float64; un canale costante nella generata resta con a = 1.
    """
    p = predicted.reshape(-1, predicted.shape[-1]).double()
    t = truth.reshape(-1, truth.shape[-1]).double()
    mean_p, mean_t = p.mean(dim=0), t.mean(dim=0)
    variance = ((p - mean_p) ** 2).mean(dim=0)
    covariance = ((p - mean_p) * (t - mean_t)).mean(dim=0)
    a = torch.where(variance > 0, covariance / variance.clamp_min(1e-30), torch.ones_like(variance))
    b = mean_t - a * mean_p
    corrected = (predicted.double() * a + b).clamp(0, 1).to(predicted.dtype)
    return corrected, a.tolist(), b.tolist()


def reduce_to_side(image: torch.Tensor, side: int) -> torch.Tensor:
    """Riduce [1, 3, H, W] a lato maggiore `side`, mediando le aree dei pixel."""
    height, width = image.shape[-2:]
    scale = side / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    return torch.nn.functional.interpolate(image, size=size, mode="area")


def evaluate(views: List[View], out_dir: Path, sides: Sequence[int] = ()) -> dict:
    from torchmetrics.functional import structural_similarity_index_measure
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from PIL import Image

    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()

    def measure(p: torch.Tensor, t: torch.Tensor) -> Dict[str, float]:
        """Metriche grezze e corrette per due immagini [1, 3, H, W] sulla scheda grafica."""
        corrected, a, b = color_correct(p[0].permute(1, 2, 0), t[0].permute(1, 2, 0))
        c = corrected.permute(2, 0, 1)[None]
        values = {}
        for suffix, image in (("", p), ("_cc", c)):
            values["psnr" + suffix] = psnr(image, t).item()
            values["ssim" + suffix] = structural_similarity_index_measure(image, t, data_range=1.0).item()
            values["lpips" + suffix] = lpips(image, t).item()
        values["cc_a"], values["cc_b"] = a, b
        return values

    per_image: List[dict] = []
    saved = []
    with torch.no_grad():
        keep = set(range(0, len(views), max(len(views) // SAVED_VIEWS, 1))[:SAVED_VIEWS])
        for index, (name, (render, finish)) in enumerate(views):
            torch.cuda.synchronize()
            start = time.perf_counter()
            rendered = render()
            seconds = time.perf_counter() - start
            predicted, truth = finish(*rendered)
            p = predicted.permute(2, 0, 1)[None].float().cuda()
            t = truth.permute(2, 0, 1)[None].float().cuda()
            entry = {"name": name, **measure(p, t), "render_seconds": seconds}
            native = max(p.shape[-2:])
            at_side = {str(side): {k: v for k, v in measure(reduce_to_side(p, side), reduce_to_side(t, side)).items()
                                   if k in METRICS + CORRECTED}
                       for side in sides if side < native}
            if at_side:
                entry["at_side"] = at_side
            per_image.append(entry)
            if index in keep:
                views_dir = out_dir / VIEWS_DIR
                views_dir.mkdir(parents=True, exist_ok=True)
                safe = name.replace("/", "__").replace("\\", "__")
                for image, suffix in ((p, "pred"), (t, "gt")):
                    pixels = (image[0].permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()
                    Image.fromarray(pixels).save(views_dir / f"{safe}_{suffix}.jpg", quality=92)
                saved.append(name)
            print(f"vista {index + 1}/{len(views)}", flush=True)
    if not per_image:
        raise SystemExit("Nessuna vista di test: il progetto non ha una suddivisione training/test.")
    # La prima vista include l'inizializzazione (compilazione dei kernel, cache): si esclude dal tempo medio.
    timed = per_image[1:] or per_image
    render_seconds = statistics.mean(v["render_seconds"] for v in timed)
    results = summarize(per_image)
    sides_done = sorted({side for v in per_image for side in v.get("at_side", {})}, key=int)
    if sides_done:
        results["at_side"] = {side: summarize([v["at_side"][side] for v in per_image]) for side in sides_done}
    results.update(render_seconds=render_seconds, fps=1.0 / render_seconds, test_images=len(per_image),
                   height=int(predicted.shape[0]), width=int(predicted.shape[1]))
    return {"results": results, "saved_views": saved, "per_image": per_image}


def summarize(per_image: List[dict]) -> Dict[str, float]:
    """Media e deviazione standard di ogni metrica, grezza e corretta."""
    results = {}
    for key in METRICS + CORRECTED:
        values = [v[key] for v in per_image if key in v]
        if values:
            results[key] = statistics.mean(values)
            results[key + "_std"] = statistics.pstdev(values)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["nerfstudio", "inria"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split.json del progetto")
    parser.add_argument("--repo", type=Path, help="cartella del codice Inria (solo per --engine inria)")
    parser.add_argument("--eval-side", type=int, action="append", default=[],
                        help="calcola le metriche anche a questo lato maggiore, in pixel (ripetibile)")
    args = parser.parse_args()
    run = args.run.resolve()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    views = nerfstudio_views(run) if args.engine == "nerfstudio" else inria_views(run, args.repo.resolve())
    data = evaluate(original_names(list(views), split), run, args.eval_side)
    (run / "metrics.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    r = data["results"]
    print(f"PSNR {r['psnr']:.2f} dB, SSIM {r['ssim']:.4f}, LPIPS {r['lpips']:.4f}, "
          f"{r['fps']:.1f} fotogrammi al secondo su {r['test_images']} viste")
    print(f"con correzione del colore: PSNR {r['psnr_cc']:.2f} dB, SSIM {r['ssim_cc']:.4f}, LPIPS {r['lpips_cc']:.4f}")
    for side, s in r.get("at_side", {}).items():
        print(f"a {side} px: PSNR {s['psnr']:.2f} dB ({s['psnr_cc']:.2f} corretto), LPIPS {s['lpips']:.4f} "
              f"({s['lpips_cc']:.4f} corretto)")


if __name__ == "__main__":
    main()
