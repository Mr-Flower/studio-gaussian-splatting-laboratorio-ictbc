"""Sito pubblicabile (per esempio su GitHub Pages) con i risultati di un progetto.

Contiene una pagina di presentazione, il report completo con grafici e viste a confronto, e le
schede dei modelli di gaussiane. I file dei modelli NON vanno nel sito: un sito statico li
renderebbe scaricabili. Si preparano a parte, puliti (app/clean.py) e nel formato compatto .splat,
raddrizzati con la verticale del volo sull'asse +y e l'arco al centro (tutti con lo stesso
orientamento), per pubblicarli su un visualizzatore esterno come SuperSplat; l'indirizzo di ogni
visualizzazione si passa con --viewer e diventa il pulsante della scheda.

Nessun visualizzatore nel browser impedisce davvero di salvare un modello: per mostrarlo, il
browser deve riceverne i dati. Si puo' solo evitare di offrirlo come file.

Uso:  python -m app.site --project <nome> --out <cartella> [--viewer <metodo>=<indirizzo> ...]
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import __version__, clean, config, runs

REPOSITORY = "https://github.com/Mr-Flower/studio-gaussian-splatting-laboratorio-ictbc"
SH_C0 = 0.28209479177387814


def gaussians_in_colmap(path: Path, run: runs.Run) -> Dict[str, np.ndarray]:
    """Posizioni, dimensioni, rotazioni (w, x, y, z), colori e opacita' nel sistema di COLMAP."""
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    v = PlyData.read(str(path))["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)
    scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))
    quat = np.stack([v[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)  # w, x, y, z
    quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    if run.engine != "inria":  # nerfstudio: p = scala * (R q + t), vedi clean.to_colmap
        data = runs.read_json(run.path / "dataparser_transforms.json")
        rotation, scale = np.asarray(data["transform"])[:, :3], float(data["scale"])
        xyz = clean.to_colmap(xyz, run, config.WORK / "data" / run.project)
        scales = scales / scale
        turned = Rotation.from_matrix(rotation.T) * Rotation.from_quat(quat[:, [1, 2, 3, 0]])
        quat = turned.as_quat()[:, [3, 0, 1, 2]]
    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1)
    return {"xyz": xyz, "scales": scales, "quat": quat, "rgb": np.clip(0.5 + SH_C0 * dc, 0, 1),
            "opacity": 1 / (1 + np.exp(-np.asarray(v["opacity"], dtype=np.float64)))}


def upright(g: Dict[str, np.ndarray], scene: Path) -> Dict[str, np.ndarray]:
    """Raddrizza il modello: verticale del volo sull'asse +y (l'alto per SuperSplat), arco al centro.

    La verticale e' la direzione media dell'alto delle camere del drone; il lato lungo dell'arco
    (asse principale in pianta delle gaussiane) va sull'asse x. Si ruotano posizioni e gaussiane.
    """
    from scipy.spatial.transform import Rotation

    middle, up, radius, _ = clean.flight_frame(scene)
    near = np.linalg.norm(g["xyz"] - np.outer(g["xyz"] @ up, up) - middle, axis=1) < radius
    flat = g["xyz"][near] - np.outer(g["xyz"][near] @ up, up)
    flat -= flat.mean(axis=0)
    long_axis = np.linalg.svd(flat, full_matrices=False)[2][0]
    long_axis -= (long_axis @ up) * up
    long_axis /= np.linalg.norm(long_axis)
    third = np.cross(long_axis, up)
    rotation = np.stack([long_axis, up, third])  # righe: nuovi assi x, y, z nel sistema di COLMAP
    origin = middle + up * np.percentile(g["xyz"][near] @ up, 2)  # al piede dell'arco
    turned = Rotation.from_matrix(rotation) * Rotation.from_quat(g["quat"][:, [1, 2, 3, 0]])
    return {**g, "xyz": (g["xyz"] - origin) @ rotation.T, "quat": turned.as_quat()[:, [3, 0, 1, 2]]}


def write_splat(g: Dict[str, np.ndarray], target: Path) -> int:
    """Formato .splat: 32 byte per gaussiana (posizione, dimensioni, colore RGBA, rotazione), dalla piu' visibile."""
    order = np.argsort(-(g["scales"].prod(axis=1) * g["opacity"]))
    record = np.zeros(len(order), dtype=[("xyz", "<f4", 3), ("scales", "<f4", 3), ("rgba", "u1", 4), ("quat", "u1", 4)])
    record["xyz"] = g["xyz"][order]
    record["scales"] = g["scales"][order]
    record["rgba"][:, :3] = np.round(g["rgb"][order] * 255)
    record["rgba"][:, 3] = np.round(g["opacity"][order] * 255)
    record["quat"] = np.clip(np.round(g["quat"][order] * 128 + 128), 0, 255)
    target.parent.mkdir(parents=True, exist_ok=True)
    record.tofile(target)
    return len(order)


def _number(value, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def build(project: str, out: Path, viewers: Optional[Dict[str, str]] = None,
          models_out: Optional[Path] = None) -> Path:
    """Scrive il sito in `out` e i modelli .splat da pubblicare altrove in `models_out` (fuori dal sito)."""
    viewers = viewers or {}
    report_dir = config.WORK / "reports" / project
    if not (report_dir / "index.html").exists():
        raise SystemExit("Manca il report: generarlo prima con  python -m app.report --project " + project)
    models_out = models_out or config.WORK / "exports" / project / "da_pubblicare"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(report_dir, out / "report", ignore=shutil.ignore_patterns("check_gt*.json"))
    models: List[dict] = []
    for run in clean.main_runs(project):
        cleaned = run.export_dir / clean.CLEAN_FILE
        if not cleaned.exists():
            clean.clean(run, config.WORK / "data" / project)
        method = config.METHODS[run.method]
        model = upright(gaussians_in_colmap(cleaned, run), config.WORK / "data" / project)
        count = write_splat(model, models_out / f"{run.method}.splat")
        metrics = run.metrics()
        models.append({"key": run.method, "label": method.label, "gaussians": count,
                       "psnr": metrics.get("psnr"), "ssim": metrics.get("ssim"), "lpips": metrics.get("lpips"),
                       "hours": run.info().get("train_seconds", 0) / 3600, "viewer": viewers.get(run.method)})
    (out / "index.html").write_text(page(project, models), encoding="utf-8")
    (out / ".nojekyll").write_text("")  # GitHub Pages: pubblica i file cosi' come sono
    return out / "index.html"


def page(project: str, models: List[dict]) -> str:
    cards = "".join(f"""
<article class="card">
  <h3>{html.escape(m['label'])}</h3>
  <p class="nums">PSNR {_number(m['psnr'], 2)} dB · SSIM {_number(m['ssim'], 3)} · LPIPS {_number(m['lpips'], 3)}<br>
  training {_number(m['hours'], 1)} h · {f"{m['gaussians']:,}".replace(",", ".")} gaussiane</p>
  <p>{f'<a class="button" href="{html.escape(m["viewer"])}" target="_blank" rel="noopener">Guarda in 3D</a>'
      if m["viewer"] else '<span class="note">Visualizzazione 3D in preparazione.</span>'}</p>
</article>""" for m in models)
    return f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Arco di Traiano: 3DGS e NeRF a confronto</title>
<meta name="description" content="Confronto tra 3D Gaussian Splatting e NeRF a parità di condizioni sull'Arco di Traiano a Benevento.">
<style>
 :root {{ --bg: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781; --line: #e1e0d9; --accent: #2a78d6; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg: #141414; --surface: #1c1c1b; --ink: #f1f0ec; --ink2: #c3c2b7;
   --muted: #9a9890; --line: #33332f; --accent: #6aa8ef; }} }}
 body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: system-ui, "Segoe UI", sans-serif; line-height: 1.5; }}
 main {{ max-width: 1080px; margin: 0 auto; padding: 32px 16px 56px; }}
 h1 {{ font-size: 28px; margin: 0 0 6px; }} h2 {{ font-size: 20px; margin: 36px 0 12px; }} h3 {{ font-size: 16px; margin: 0 0 6px; }}
 .lead {{ color: var(--ink2); font-size: 17px; max-width: 760px; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
 .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
 .nums {{ color: var(--ink2); font-size: 14px; font-variant-numeric: tabular-nums; }}
 .button {{ display: inline-block; background: var(--accent); color: #fff; text-decoration: none; padding: 8px 12px;
   border-radius: 6px; margin-right: 10px; font-weight: 600; }}
 a {{ color: var(--accent); }} img {{ max-width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fcfcfb; }}
 .note {{ color: var(--muted); font-size: 14px; }}
 footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
</style></head><body><main>
<h1>Arco di Traiano, Benevento: 3D Gaussian Splatting e NeRF a confronto</h1>
<p class="lead">Cinque metodi di ricostruzione 3D allenati sulle stesse 887 fotografie da drone, con lo stesso
allineamento delle camere, le stesse 111 viste di test, la risoluzione piena (5280 px) e lo stesso codice di misura.
Laboratorio ICTBC, ricerca non commerciale.</p>

<h2>Modelli 3D</h2>
<p class="note">Modelli di gaussiane puliti automaticamente (sfondo, gaussiane trasparenti e fluttuanti tolti), da
guardare nel browser: tasto sinistro per ruotare, destro per spostarsi, rotella per lo zoom. I modelli non sono
distribuiti come file.</p>
<div class="grid">{cards}</div>

<h2>Qualità e costo</h2>
<img src="report/figure/qualita.png" alt="PSNR, SSIM e LPIPS per metodo">
<img src="report/figure/costo.png" alt="Tempo di calcolo, velocità di rendering e dimensione dei modelli per metodo">

<h2>Viste di test a confronto</h2>
<p class="note">Foto vera e immagini generate dai cinque metodi da punti di vista esclusi dal training.</p>
<img src="report/figure/vista_1.jpg" alt="Vista di test 1: foto vera e immagini dei cinque metodi">
<img src="report/figure/vista_2.jpg" alt="Vista di test 2">
<img src="report/figure/vista_3.jpg" alt="Vista di test 3">

<h2>Approfondimenti</h2>
<ul>
 <li><a href="report/index.html">Report completo</a>: tabelle delle misure, differenze di processo, condizioni dell'esperimento.</li>
 <li><a href="report/confronto.csv">Tabella di tutti i run (CSV)</a>.</li>
 <li><a href="{REPOSITORY}">Codice, protocollo e risultati</a> (README con abstract e risultati dettagliati).</li>
</ul>

<footer>Generato il {time.strftime('%d/%m/%Y')} con Studio Gaussian Splatting {__version__} (progetto «{html.escape(project)}»).
Codice con licenza Apache 2.0; il codice 3D Gaussian Splatting di Inria ha una licenza propria, solo per ricerca non commerciale.</footer>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--viewer", action="append", default=[], metavar="METODO=INDIRIZZO",
                        help="pagina dove si guarda il modello del metodo (per esempio pubblicata con SuperSplat)")
    parser.add_argument("--modelli", type=Path, help="dove scrivere i .splat da pubblicare (fuori dal sito)")
    args = parser.parse_args()
    viewers = dict(item.split("=", 1) for item in args.viewer)
    print(f"Sito in {build(args.project, args.out, viewers, args.modelli)}")


if __name__ == "__main__":
    main()
