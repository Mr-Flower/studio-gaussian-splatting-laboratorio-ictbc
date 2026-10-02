"""Valutazione uniforme di un run sulle viste di test.

Per ogni vista esclusa dal training si genera l'immagine con il modello allenato, si misura il solo
tempo di rendering e si calcolano PSNR, SSIM e LPIPS con lo stesso codice per tutti i metodi,
qualunque sia il programma che li ha allenati. Alcune viste sono salvate per il confronto visivo.

Uso (dall'ambiente con PyTorch):
  python -m app.evaluate --engine nerfstudio --run <cartella del run>
  python -m app.evaluate --engine inria --run <cartella del run> --repo <gaussian-splatting>
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterator, List, Tuple

import torch

VIEWS_DIR = "test_views"
SAVED_VIEWS = 6

# Una vista: nome della foto, funzione che genera (immagine prodotta, immagine vera) come [H, W, 3] in 0..1.
View = Tuple[str, Callable[[], Tuple[torch.Tensor, torch.Tensor]]]


def nerfstudio_views(run: Path) -> Iterator[View]:
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


def inria_views(run: Path, repo: Path) -> Iterator[View]:
    sys.path.insert(0, str(repo))
    sys.argv = [sys.argv[0], "-m", str(run)]
    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import GaussianModel, render as render_view
    from scene import Scene

    try:
        from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
        separate_sh = True
    except ImportError:
        separate_sh = False

    parser = argparse.ArgumentParser()
    model, pipe = ModelParams(parser, sentinel=True), PipelineParams(parser)
    args = get_combined_args(parser)
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    background = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0], device="cuda")
    pipeline = pipe.extract(args)
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


def evaluate(views: List[View], out_dir: Path) -> dict:
    from torchmetrics.functional import structural_similarity_index_measure
    from torchmetrics.image import PeakSignalNoiseRatio
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    from PIL import Image

    psnr =PeakSignalNoiseRatio(data_range=1.0).cuda()
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).cuda()
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
            per_image.append({
                "name": name,
                "psnr": psnr(p, t).item(),
                "ssim": structural_similarity_index_measure(p, t, data_range=1.0).item(),
                "lpips": lpips(p, t).item(),
                "render_seconds": seconds,
            })
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
    results = {key: statistics.mean(v[key] for v in per_image) for key in ("psnr", "ssim", "lpips")}
    for key in ("psnr", "ssim", "lpips"):
        results[key + "_std"] = statistics.pstdev(v[key] for v in per_image)
    results.update(render_seconds=render_seconds, fps=1.0 / render_seconds, test_images=len(per_image),
                   height=int(predicted.shape[0]), width=int(predicted.shape[1]))
    return {"results": results, "saved_views": saved, "per_image": per_image}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["nerfstudio", "inria"], required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True, help="split.json del progetto")
    parser.add_argument("--repo", type=Path, help="cartella del codice Inria (solo per --engine inria)")
    args = parser.parse_args()
    run = args.run.resolve()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    views = nerfstudio_views(run) if args.engine == "nerfstudio" else inria_views(run, args.repo.resolve())
    data = evaluate(original_names(list(views), split), run)
    (run / "metrics.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    r = data["results"]
    print(f"PSNR {r['psnr']:.2f} dB, SSIM {r['ssim']:.4f}, LPIPS {r['lpips']:.4f}, "
          f"{r['fps']:.1f} fotogrammi al secondo su {r['test_images']} viste")


if __name__ == "__main__":
    main()
