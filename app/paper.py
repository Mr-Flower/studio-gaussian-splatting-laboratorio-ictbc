"""Tabelle, figure e numeri per l'articolo, dai file gia' prodotti dal programma.

Non esegue training ne' valutazioni: legge run.json, metrics.json, photos.json, alignment.json,
split.json, mesh.json e il modello COLMAP, e scrive in una cartella:

  tab1_dati, tab2_configurazioni, tab3_risultati, tab4_appaiati, tab5_fotogrammetria (.csv e .tex)
  fig2_camere, fig5_ora, fig5b_jpg_dng (.pdf e .png a 300 dpi)
  numeri.json  tutti i numeri citati nel testo, con chiavi stabili (es. "inria-3dgs.psnr_mean")

Uso:  python -m app.paper --project <nome> --out paper\\<nome> [--scartate foto\\LEGGIMI.txt]

--scartate indica il file con le foto tolte prima del progetto, una per riga nella forma
"  <nome>: <motivo>[; <motivo>]" (come il LEGGIMI.txt della cartella delle foto), piu' le righe
che iniziano con "Scartata a mano".
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import colmap_model, config, photos, pipeline, runs
from .runs import Run

METRICS = ("psnr", "ssim", "lpips")
CORRECTED = tuple(m + "_cc" for m in METRICS)  # dopo la correzione affine del colore (app/evaluate.py)
ALL_METRICS = METRICS + CORRECTED
MEASURE = {"": "grezza", "_cc": "corretta"}
BOOTSTRAP = 10000  # ricampionamenti delle viste per gli intervalli di confidenza
SEED = 0  # seme del bootstrap: gli stessi file danno sempre gli stessi intervalli
CONFIDENCE = 0.95
IMPLEMENTATION = {"inria": "codice Inria", "nerfstudio": "nerfstudio"}
DJI_NAME = re.compile(r"DJI_(\d{14})_")  # i file DJI riportano data e ora di scatto nel nome


# --- statistica

def bootstrap_ci(values: Sequence[float], statistic=np.mean, resamples: int = BOOTSTRAP,
                 seed: int = SEED) -> Tuple[float, float]:
    """Intervallo di confidenza percentile al 95% della statistica, ricampionando i valori."""
    data = np.asarray(values, dtype=float)
    if len(data) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, len(data), size=(resamples, len(data)))]
    stats = statistic(samples, axis=1)
    tail = (1 - CONFIDENCE) / 2 * 100
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100 - tail))


def holm(p_values: Sequence[float]) -> List[float]:
    """Correzione di Holm-Bonferroni: p corretti, nello stesso ordine di quelli dati."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def wilcoxon_p(differences: Sequence[float]) -> float:
    """p bilaterale del test di Wilcoxon per dati appaiati; 1 se tutte le differenze sono nulle."""
    from scipy.stats import wilcoxon

    data = np.asarray(differences, dtype=float)
    if len(data) == 0 or np.all(data == 0):
        return 1.0
    return float(wilcoxon(data).pvalue)


# --- dati

@dataclass
class MethodResult:
    key: str
    runs: List[Run]
    per_view: Dict[str, Dict[str, float]] = field(default_factory=dict)  # vista -> metrica -> media tra i run

    @property
    def method(self) -> config.Method:
        return config.METHODS[self.key]


def collect(project: str) -> List[MethodResult]:
    """Metodi del confronto principale: i run valutati sull'allineamento corrente nelle condizioni piu' comuni.

    Con piu' run per metodo (semi diversi) i valori di ogni vista sono la media tra i run.
    """
    groups = runs.condition_groups(runs.list_runs(project))
    if not groups:
        raise SystemExit("Nessun run valutato sull'allineamento corrente.")
    group = groups[0]
    results = []
    for key in config.METHODS:
        method_runs = group.get(key)
        if not method_runs:
            continue
        result = MethodResult(key, method_runs)
        per_run = [{v["name"]: v for v in runs.read_json(r.path / runs.METRICS_FILE).get("per_image", [])}
                   for r in method_runs]
        result.per_view = per_view(method_runs)
        results.append(result)
    return results


def per_view(method_runs: List[Run], side: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Metriche per vista, media tra i run; con `side` quelle alla scala comune di quel lato (at_side)."""
    per_run = []
    for run in method_runs:
        views = {}
        for v in runs.read_json(run.path / runs.METRICS_FILE).get("per_image", []):
            values = v if side is None else v.get("at_side", {}).get(side)
            if values is not None:
                views[v["name"]] = values
        per_run.append(views)
    if not per_run:
        return {}
    common = set.intersection(*(set(p) for p in per_run))
    out = {}
    for name in sorted(common):
        out[name] = {m: statistics.mean(p[name][m] for p in per_run)
                     for m in ALL_METRICS if all(m in p[name] for p in per_run)}
    return out


def available(results: List["MethodResult"], metrics=ALL_METRICS) -> List[str]:
    """Metriche presenti per tutti i metodi e tutte le viste (i run vecchi non hanno quelle corrette)."""
    return [m for m in metrics if all(m in v for r in results for v in r.per_view.values())]


def camera_of(name: str) -> str:
    """Camera di una foto dalla sottocartella del nome (jpg/... o dng/...)."""
    return name.split("/")[0].lower() if "/" in name else ""


def taken_times(scene: Path) -> Dict[str, datetime]:
    """Ora di scatto di ogni foto: dall'EXIF (photos.json) o, se manca, dal nome dei file DJI."""
    times = {}
    for photo in photos.load(scene):
        stamp = None
        if photo.taken:
            try:
                stamp = datetime.strptime(photo.taken.strip()[:19], "%Y:%m:%d %H:%M:%S")
            except ValueError:
                stamp = None
        match = DJI_NAME.search(photo.name)
        if stamp is None and match:
            stamp = datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        if stamp is not None:
            times[photo.name] = stamp
    return times


def read_discarded(path: Optional[Path]) -> Dict[str, List[str]]:
    """Foto tolte prima del progetto: nome (o "a mano N") -> motivi (vedi l'aiuto del modulo)."""
    discarded: Dict[str, List[str]] = {}
    if path is None:
        return discarded
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().lower().startswith("scartata a mano"):
            discarded[f"a mano {len(discarded) + 1}"] = ["a mano"]
            continue
        match = re.match(r"^\s+(\S+): (.+)$", line)
        if match:  # "quasi doppione di X" -> "quasi doppione"
            discarded[match.group(1)] = [re.sub(r" di .*$", "", r.strip()) for r in match.group(2).split(";")]
    return discarded


def ply_faces(path: Path) -> Optional[int]:
    try:
        with open(path, "rb") as f:
            for _ in range(60):
                line = f.readline().decode("ascii", "replace").strip()
                if line.startswith("element face"):
                    return int(line.split()[-1])
                if line == "end_header":
                    break
    except OSError:
        pass
    return None


# --- scrittura delle tabelle

def _latex_escape(text: str) -> str:
    for char, escaped in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                          ("±", r"$\pm$"), ("–", "--")):
        text = text.replace(char, escaped)
    return text


def write_table(folder: Path, name: str, headers: List[str], rows: List[List], caption: str = "") -> None:
    """Scrive la tabella in CSV (separatore ';', UTF-8) e in LaTeX con booktabs."""
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(headers)
        writer.writerows(rows)
    align = "l" + "r" * (len(headers) - 1)
    lines = [f"% {caption}" if caption else "% generata da app/paper.py", r"\begin{tabular}{" + align + "}",
             r"\toprule", " & ".join(_latex_escape(str(h)) for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(_latex_escape(str(c)) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (folder / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def _f(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value:.{digits}f}"


# --- tabelle

def table_data(scene: Path, discarded: Dict[str, List[str]], out: Path, numbers: dict) -> None:
    alignment = runs.read_json(scene / runs.ALIGNMENT_FILE)
    split = colmap_model.read_split(scene)
    used = split["train"] + split["test"]
    per_camera = Counter(camera_of(n) for n in used)
    analysed = photos.load(scene)
    excluded = photos.load_excluded(scene)
    removed_reasons = dict(discarded)
    for photo in analysed:
        if photo.name in excluded:
            removed_reasons[photo.name] = photo.reasons or ["a mano"]
    # Una foto puo' avere piu' motivi: si contano le foto per motivo, e a parte il totale delle foto.
    reasons = Counter(reason for motives in removed_reasons.values() for reason in set(motives))
    removed = len(removed_reasons)
    total = alignment.get("photos", len(used)) + removed
    rows = [["Foto del rilievo", total], ["Foto scartate", removed]]
    rows += [[f"  per motivo: {reason}", count] for reason, count in sorted(reasons.items())]
    rows += [["Foto usate", alignment.get("photos")]]
    rows += [[f"  camera {cam.upper()}", count] for cam, count in sorted(per_camera.items())]
    rows += [["Foto registrate da COLMAP", alignment.get("registered_images")],
             ["Foto di training / di test", f"{len(split['train'])} / {len(split['test'])}"],
             ["Errore medio di riproiezione (px)", _f(alignment.get("mean_reprojection_error_px"), 3)],
             ["Punti 3D", alignment.get("points")],
             ["Lunghezza media delle tracce", _f(alignment.get("mean_track_length"), 2)]]
    write_table(out, "tab1_dati", ["Voce", "Valore"], rows, "Dati del rilievo e dell'allineamento")
    numbers.update({"dati.foto_rilievo": total, "dati.foto_scartate": removed, "dati.foto_usate": alignment.get("photos"),
                    "dati.foto_registrate": alignment.get("registered_images"), "dati.foto_training": len(split["train"]),
                    "dati.foto_test": len(split["test"]), "dati.errore_riproiezione_px": alignment.get("mean_reprojection_error_px"),
                    "dati.punti_3d": alignment.get("points"), "dati.lunghezza_tracce": alignment.get("mean_track_length")})
    numbers.update({f"dati.camera_{cam}": count for cam, count in per_camera.items()})
    numbers.update({f"dati.scartate.{reason}": count for reason, count in reasons.items()})


def _side(run: Run) -> Optional[int]:
    metrics = run.metrics()
    return max(metrics["width"], metrics["height"]) if "width" in metrics else None


def table_configurations(results: List[MethodResult], out: Path, numbers: dict) -> None:
    rows = []
    for r in results:
        info = r.runs[0].info()
        nerfstudio = r.method.engine == "nerfstudio"
        pose, appearance = pipeline.run_conditions(info)[2:]
        rows.append([r.method.label, r.method.family, IMPLEMENTATION[r.method.engine], info.get("iterations"),
                     info.get("downscale"), _side(r.runs[0]) or "–",
                     ("sì" if pose else "no") if nerfstudio else "non prevista",
                     ("sì" if appearance else "no") if r.method.family == "nerf" else "non prevista",
                     info.get("photos_in", "–"), len(r.runs)])
        numbers[f"{r.key}.lato_px"] = _side(r.runs[0])
        numbers[f"{r.key}.run"] = len(r.runs)
    write_table(out, "tab2_configurazioni",
                ["Metodo", "Famiglia", "Implementazione", "Iterazioni", "Riduzione", "Lato (px)",
                 "Correzione pose", "Aspetto per foto", "Foto in", "Run"], rows, "Configurazioni confrontate")


def table_results(results: List[MethodResult], out: Path, numbers: dict) -> None:
    rows = []
    shown = available(results)
    for r in results:
        row = [r.method.label, len(r.per_view)]
        for metric in shown:
            values = [v[metric] for v in r.per_view.values()]
            mean, sd = statistics.mean(values), statistics.pstdev(values)
            low, high = bootstrap_ci(values)
            row += [f"{mean:.3f} ± {sd:.3f}", f"[{low:.3f}, {high:.3f}]"]
            numbers.update({f"{r.key}.{metric}_mean": mean, f"{r.key}.{metric}_sd": sd,
                            f"{r.key}.{metric}_ci_low": low, f"{r.key}.{metric}_ci_high": high})
            if len(r.runs) > 1:  # variabilita' tra esecuzioni: media del metodo per ogni run
                per_run = [runs.read_json(run.path / runs.METRICS_FILE)["results"][metric] for run in r.runs]
                numbers[f"{r.key}.{metric}_sd_tra_run"] = statistics.stdev(per_run)
        infos = [run.info() for run in r.runs]
        metrics = [run.metrics() for run in r.runs]
        hours = statistics.mean(i.get("train_seconds", math.nan) for i in infos) / 3600
        peaks = [i["gpu_peak_mb"] for i in infos if "gpu_peak_mb" in i]
        fps = statistics.mean(m.get("fps", math.nan) for m in metrics)
        export = r.runs[0].export_file
        size_mb = export.stat().st_size / 2**20 if export else None
        elements = runs.ply_elements(export) if export else None
        row += [_f(hours, 2), max(peaks) if peaks else "–", _f(fps, 1), _f(size_mb, 1), elements or "–"]
        rows.append(row)
        numbers.update({f"{r.key}.training_ore": hours, f"{r.key}.gpu_peak_mb": max(peaks) if peaks else None,
                        f"{r.key}.fps": fps, f"{r.key}.ply_mb": size_mb, f"{r.key}.elementi": elements,
                        f"{r.key}.viste": len(r.per_view)})
    headers = ["Metodo", "Viste"]
    for metric in shown:
        name = metric.replace("_cc", "").upper() + (" corretto" if metric.endswith("_cc") else "")
        headers += [f"{name} media ± d.s.", f"{name} IC 95%"]
    headers += ["Training (h)", "Picco GPU (MB)", "Rendering (fps)", ".ply (MB)", "Gaussiane / punti"]
    write_table(out, "tab3_risultati", headers, rows,
                f"Risultati sulle viste di test; IC bootstrap percentile, {BOOTSTRAP} ricampionamenti, seme {SEED}; "
                "«corretto»: dopo la correzione affine del colore per canale (limite superiore)")


def paired(a: MethodResult, b: MethodResult, metric: str) -> np.ndarray:
    """Differenze a - b vista per vista, sulle viste valutate da entrambi."""
    common = sorted(set(a.per_view) & set(b.per_view))
    return np.array([a.per_view[n][metric] - b.per_view[n][metric] for n in common])


def table_paired(results: List[MethodResult], out: Path, numbers: dict) -> None:
    rows = []
    shown = set(available(results))
    for suffix, measure in MEASURE.items():
        metrics = [m + suffix for m in METRICS if m + suffix in shown]
        tests = []
        for a, b in combinations(results, 2):
            for metric in metrics:
                diff = paired(a, b, metric)
                low, high = bootstrap_ci(diff, statistic=np.median)
                tests.append((a, b, metric, diff, float(np.median(diff)) if len(diff) else math.nan, low, high,
                              wilcoxon_p(diff)))
        # Holm su tutti i confronti di una famiglia (coppie per metriche, grezze o corrette): prudente, e le
        # metriche grezze restano corrette come nella versione 0.4.0.
        adjusted = holm([test[-1] for test in tests])
        for (a, b, metric, diff, median, low, high, p_value), p_holm in zip(tests, adjusted):
            rows.append([a.method.label, b.method.label, metric.replace("_cc", "").upper(), measure, len(diff),
                         f"{median:+.4f}", f"[{low:+.4f}, {high:+.4f}]", f"{p_value:.2e}", f"{p_holm:.2e}"])
            key = f"appaiati.{a.key}_vs_{b.key}.{metric}"
            numbers.update({f"{key}.mediana": median, f"{key}.ci_low": low, f"{key}.ci_high": high,
                            f"{key}.p": p_value, f"{key}.p_holm": p_holm, f"{key}.viste": len(diff)})
    write_table(out, "tab4_appaiati",
                ["Metodo A", "Metodo B", "Metrica", "Misura", "Viste", "Mediana A−B", "IC 95%", "p (Wilcoxon)",
                 "p (Holm)"], rows,
                "Confronti appaiati vista per vista; Holm separato per le metriche grezze e per quelle corrette")


# --- esposizione delle foto

EXIF_IFD, EXPOSURE_TIME, F_NUMBER, ISO = 0x8769, 0x829A, 0x829D, 0x8827


def exposure_values(folder: Path, names: Sequence[str]) -> Dict[str, Optional[float]]:
    """EV100 di ogni foto dalle EXIF: log2(N^2 / t) - log2(ISO / 100); None se mancano i dati."""
    from PIL import Image

    values: Dict[str, Optional[float]] = {}
    for name in names:
        try:
            with Image.open(folder / name) as image:
                exif = image.getexif().get_ifd(EXIF_IFD)
            t, n, iso = float(exif[EXPOSURE_TIME]), float(exif[F_NUMBER]), exif[ISO]
            iso = float(iso[0] if isinstance(iso, (tuple, list)) else iso)
            values[name] = math.log2(n * n / t) - math.log2(iso / 100) if t > 0 and n > 0 and iso > 0 else None
        except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError):
            values[name] = None
    return values


def exposure_distance(scene: Path, project: str) -> Dict[str, float]:
    """Per ogni vista di test, distanza di esposizione dalla mediana delle foto di training (in EV)."""
    settings = config.Settings.load(project)
    if settings is None or not settings.photos:
        return {}
    split = colmap_model.read_split(scene)
    ev = exposure_values(Path(settings.photos), split["train"] + split["test"])
    train = [ev[n] for n in split["train"] if ev.get(n) is not None]
    if not train:
        return {}
    middle = statistics.median(train)
    return {n: ev[n] - middle for n in split["test"] if ev.get(n) is not None}


def exposure_numbers(results: List[MethodResult], distance: Dict[str, float], numbers: dict) -> None:
    from scipy.stats import spearmanr

    numbers["esposizione.viste_con_dati"] = len(distance)
    numbers["esposizione.delta_ev_mediana_assoluta"] = statistics.median(abs(d) for d in distance.values()) if distance else None
    for r in results:
        for metric in ("psnr", "psnr_cc"):
            names = [n for n in r.per_view if n in distance and metric in r.per_view[n]]
            if len(names) > 2:
                rho, p_value = spearmanr([abs(distance[n]) for n in names], [r.per_view[n][metric] for n in names])
                numbers[f"{r.key}.esposizione.rho_{metric}"] = float(rho)
                numbers[f"{r.key}.esposizione.p_{metric}"] = float(p_value)
        if all("psnr_cc" in v for v in r.per_view.values()):
            numbers[f"{r.key}.psnr_grezzo_meno_corretto"] = statistics.mean(
                v["psnr"] - v["psnr_cc"] for v in r.per_view.values())
    by_key = {r.key: r for r in results}
    if "inria-3dgs" in by_key and "splatfacto" in by_key:
        a, b = by_key["inria-3dgs"].per_view, by_key["splatfacto"].per_view
        names = [n for n in a if n in b and "psnr_cc" in a[n] and "psnr_cc" in b[n]]
        flips = sum(1 for n in names if np.sign(a[n]["psnr"] - b[n]["psnr"]) != np.sign(a[n]["psnr_cc"] - b[n]["psnr_cc"]))
        numbers["esposizione.inversioni_inria_splatfacto_psnr"] = flips
        numbers["esposizione.viste_confrontate_inria_splatfacto"] = len(names)


def figure_exposure(results: List[MethodResult], distance: Dict[str, float], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    from .report import FAMILY_COLOR, GRID, MUTED, _style

    _style()
    shown = [r for r in results if any(n in distance for n in r.per_view)]
    fig, axes = plt.subplots(1, len(shown), figsize=(3.2 * len(shown), 3.8), sharey=True, constrained_layout=True,
                             squeeze=False)
    for ax, r in zip(axes[0], shown):
        names = [n for n in r.per_view if n in distance]
        x = np.array([abs(distance[n]) for n in names])
        for metric, color, label in (("psnr", MUTED, "grezzo"), ("psnr_cc", FAMILY_COLOR[r.method.family], "corretto")):
            if not all(metric in r.per_view[n] for n in names):
                continue
            y = np.array([r.per_view[n][metric] for n in names])
            rho = spearmanr(x, y)[0] if len(x) > 2 else math.nan
            ax.scatter(x, y, s=12, color=color, linewidth=0, alpha=0.75, label=f"{label} (ρ = {rho:.2f})")
            if len(x) > 2 and np.ptp(x) > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.linspace(x.min(), x.max(), 10)
                ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.6)
        ax.set_title(r.method.short, fontsize=10)
        ax.set_xlabel("|ΔEV| dalla mediana del training")
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("PSNR per vista (dB)")
    fig.suptitle("PSNR rispetto alla distanza di esposizione della vista: grezzo e dopo la correzione del colore",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, out, "fig5c_esposizione")
    plt.close(fig)


# --- risoluzione di training e scala di valutazione

def condition_index(project: str) -> Dict[tuple, Dict[str, List[Run]]]:
    """Run valutati sull'allineamento corrente, per condizioni e metodo (dal piu' recente)."""
    index: Dict[tuple, Dict[str, List[Run]]] = {}
    for run in runs.list_runs(project):
        if run.method in config.METHODS and run.comparable() is True and "psnr" in run.metrics():
            index.setdefault(pipeline.run_conditions(run.info()), {}).setdefault(run.method, []).append(run)
    return index


SCALES = ((1, None), (2, "2640"), (4, "1320"))  # colonna: fattore di riduzione e lato della scala comune


def views_at(method_runs: List[Run], train_factor: int, eval_factor: int) -> Dict[str, Dict[str, float]]:
    """Metriche per vista di modelli allenati a 1/train_factor e valutati alla scala 1/eval_factor."""
    if eval_factor < train_factor:
        return {}  # non si valuta a risoluzione piu' alta di quella di training
    if eval_factor == train_factor:
        return per_view(method_runs)
    return per_view(method_runs, dict(SCALES)[eval_factor])


def table_resolution(project: str, main: List[MethodResult], out: Path, numbers: dict) -> bool:
    index = condition_index(project)
    if not main:
        return False
    iterations = main[0].runs[0].info().get("iterations")
    rows = []
    for key in ("inria-3dgs", "splatfacto"):
        trained = {f: index.get((f, iterations, False, False), {}).get(key) for f, _ in SCALES}
        if not trained[1]:
            continue
        for factor, method_runs in trained.items():
            if not method_runs:
                continue
            cells = []
            for eval_factor, _ in SCALES:
                views = views_at(method_runs, factor, eval_factor)
                metric = "psnr_cc" if views and all("psnr_cc" in v for v in views.values()) else "psnr"
                if not views:
                    cells.append("–")
                    continue
                means = {m: statistics.mean(v[m + metric[4:]] for v in views.values()) for m in METRICS}
                cells.append(f"{means['psnr']:.2f} / {means['ssim']:.3f} / {means['lpips']:.3f}")
                numbers[f"risoluzione.{key}.train{factor}.eval{eval_factor}.{metric}"] = means["psnr"]
                numbers[f"risoluzione.{key}.train{factor}.eval{eval_factor}.lpips{metric[4:]}"] = means["lpips"]
            infos = [r.info() for r in method_runs]
            export = method_runs[0].export_file
            gaussians = runs.ply_elements(export) if export else None
            hours = statistics.mean(i.get("train_seconds", math.nan) for i in infos) / 3600
            peaks = [i["gpu_peak_mb"] for i in infos if "gpu_peak_mb" in i]
            side = _side(method_runs[0])
            rows.append([config.METHODS[key].label, side, *cells, gaussians or "–", _f(hours, 2),
                         max(peaks) if peaks else "–", len(method_runs)])
            numbers[f"risoluzione.{key}.train{factor}.gaussiane"] = gaussians
            numbers[f"risoluzione.{key}.train{factor}.training_ore"] = hours
        # La domanda: a parita' di scala (1320 px), il modello allenato a piena risoluzione e' migliore?
        if trained[1] and trained[4]:
            full, small = views_at(trained[1], 1, 4), views_at(trained[4], 4, 4)
            names = [n for n in full if n in small]
            for metric in ("psnr_cc", "lpips_cc", "psnr", "lpips"):
                if names and all(metric in full[n] and metric in small[n] for n in names):
                    diff = np.array([full[n][metric] - small[n][metric] for n in names])
                    base = f"risoluzione.{key}.piena_meno_1320_a_1320.{metric}"
                    numbers[f"{base}.mediana"] = float(np.median(diff))
                    numbers[f"{base}.media"] = float(diff.mean())
                    numbers[f"{base}.p"] = wilcoxon_p(diff)
                    numbers[f"{base}.viste"] = len(diff)
    if not rows:
        return False
    write_table(out, "tab6_risoluzione",
                ["Metodo", "Lato di training (px)", "Valutato a piena risoluzione", "Valutato a 2640 px",
                 "Valutato a 1320 px", "Gaussiane", "Training (h)", "Picco GPU (MB)", "Run"], rows,
                "PSNR / SSIM / LPIPS dopo la correzione del colore (grezzi se mancano), per scala di valutazione")
    return True


# --- disegno 2x2 delle opzioni di nerfstudio

def table_nerf_options(project: str, main: List[MethodResult], out: Path, numbers: dict) -> bool:
    index = condition_index(project)
    if not main:
        return False
    iterations = main[0].runs[0].info().get("iterations")
    cells = {}
    for pose in (False, True):
        for appearance in (False, True):
            method_runs = index.get((1, iterations, pose, appearance), {}).get("nerfacto")
            if method_runs:
                cells[(pose, appearance)] = per_view(method_runs)
    if len(cells) < 2:
        return False
    rows = []
    means = {}
    for (pose, appearance), views in sorted(cells.items()):
        shown = [m for m in ALL_METRICS if views and all(m in v for v in views.values())]
        means[(pose, appearance)] = {m: statistics.mean(v[m] for v in views.values()) for m in shown}
        rows.append(["sì" if pose else "no", "sì" if appearance else "no", len(views),
                     *[_f(means[(pose, appearance)].get(m), 3) for m in ALL_METRICS]])
        for m, value in means[(pose, appearance)].items():
            numbers[f"nerfacto_2x2.pose_{int(pose)}.aspetto_{int(appearance)}.{m}"] = value
    if len(cells) == 4:
        for m in ALL_METRICS:
            if all(m in means[c] for c in cells):
                pose_effect = ((means[(True, False)][m] - means[(False, False)][m])
                               + (means[(True, True)][m] - means[(False, True)][m])) / 2
                appearance_effect = ((means[(False, True)][m] - means[(False, False)][m])
                                     + (means[(True, True)][m] - means[(True, False)][m])) / 2
                numbers[f"nerfacto_2x2.effetto_pose.{m}"] = pose_effect
                numbers[f"nerfacto_2x2.effetto_aspetto.{m}"] = appearance_effect
        rows.append(["effetto principale", "", "", *[
            _f(numbers.get(f"nerfacto_2x2.effetto_pose.{m}"), 3) for m in ALL_METRICS]])
        rows[-1][0] = "effetto: correzione delle pose"
        rows.append(["effetto: vettore di aspetto", "", "", *[
            _f(numbers.get(f"nerfacto_2x2.effetto_aspetto.{m}"), 3) for m in ALL_METRICS]])
    write_table(out, "tab7_opzioni_nerfstudio",
                ["Correzione delle pose", "Vettore di aspetto", "Viste", "PSNR", "SSIM", "LPIPS", "PSNR corretto",
                 "SSIM corretto", "LPIPS corretto"], rows,
                "nerfacto a piena risoluzione: le due opzioni di nerfstudio una alla volta e insieme; "
                "con il vettore di aspetto le viste di test usano il vettore medio")
    return True


def table_photogrammetry(scene: Path, out: Path, numbers: dict) -> bool:
    dense = scene / "colmap" / "dense"
    data = runs.read_json(dense / runs.MESH_FILE)
    if not (dense / "mesh-poisson.ply").exists() or "seconds" not in data:
        return False
    rows = [[f"Tempo: {phase}", _f(seconds / 60, 1) + " min"] for phase, seconds in data["seconds"].items()]
    rows += [["Tempo totale", _f(sum(data["seconds"].values()) / 60, 1) + " min"],
             ["Lato delle immagini (px)", data.get("max_image_size", "–")],
             ["Punti della nuvola densa", runs.ply_elements(dense / "fused.ply") or "–"],
             ["Vertici della mesh", runs.ply_elements(dense / "mesh-poisson.ply") or "–"],
             ["Triangoli della mesh", ply_faces(dense / "mesh-poisson.ply") or "–"]]
    write_table(out, "tab5_fotogrammetria", ["Voce", "Valore"], rows, "Fotogrammetria classica (COLMAP MVS + Poisson)")
    numbers.update({"mvs.minuti": sum(data["seconds"].values()) / 60, "mvs.punti": runs.ply_elements(dense / "fused.ply"),
                    "mvs.vertici": runs.ply_elements(dense / "mesh-poisson.ply"),
                    "mvs.triangoli": ply_faces(dense / "mesh-poisson.ply")})
    return True


# --- figure

def camera_centers(model: Path) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Centri delle camere per nome e direzione media dell'alto, nel sistema di COLMAP."""
    centers, ups = {}, []
    for name, pose in colmap_model.read_poses(model).items():
        w, x, y, z = pose.qvec
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                             [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                             [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        centers[name] = -rotation.T @ np.asarray(pose.tvec)
        ups.append(-rotation.T @ np.array([0.0, 1.0, 0.0]))  # l'asse y della camera punta in basso nell'immagine
    up = np.mean(ups, axis=0)
    return centers, up / np.linalg.norm(up)


def top_view(centers: Dict[str, np.ndarray], up: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Coordinate in pianta: proiezione sul piano perpendicolare all'alto, assi principali dei centri."""
    names = list(centers)
    points = np.array([centers[n] for n in names])
    flat = points - np.outer(points @ up, up)
    flat -= flat.mean(axis=0)
    _, _, axes = np.linalg.svd(flat, full_matrices=False)
    plane = flat @ axes[:2].T
    return {n: (float(p[0]), float(p[1])) for n, p in zip(names, plane)}


def metres_per_unit(project: str) -> Optional[float]:
    """Scala scelta in data/<progetto>/scala.json (app/scala.py), None se non ce n'e' una in uso."""
    from . import scala

    data = scala.load(project) or {}
    key = data.get("use")
    return float(data[key]) if key in ("scale", "scale_horizontal") else None


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)


CAMERA_COLOR = {"jpg": "#2a78d6", "dng": "#eb6834"}  # palette categorica di riferimento, ordine fisso


def figure_cameras(scene: Path, out: Path, metres: Optional[float] = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import INK_SECONDARY, MUTED, SURFACE, _style

    _style()
    centers, up = camera_centers(pipeline.best_sparse_model(scene / "colmap"))
    plan = top_view(centers, up)
    if metres:
        plan = {n: (x * metres, y * metres) for n, (x, y) in plan.items()}
    test = set(colmap_model.read_split(scene)["test"])
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    for cam in sorted({camera_of(n) for n in plan}, reverse=True):  # jpg sotto, dng (poche) sopra
        for is_test in (False, True):
            names = [n for n in plan if camera_of(n) == cam and (n in test) == is_test]
            if not names:
                continue
            xs, ys = zip(*(plan[n] for n in names))
            color = CAMERA_COLOR.get(cam, MUTED)
            label = f"{cam.upper()} {'test' if is_test else 'training'} ({len(names)})"
            if is_test:
                ax.scatter(xs, ys, s=46, facecolor="none", edgecolor=color, linewidth=1.6, label=label, zorder=3)
            else:
                ax.scatter(xs, ys, s=10, color=color, alpha=0.55, linewidth=0, label=label, zorder=2)
    ax.set_aspect("equal")
    ax.set_title("Centri delle camere, visti dall'alto")
    unit = "metri (scala dal GPS del drone)" if metres else "unità dell'allineamento (scala arbitraria)"
    ax.set_xlabel(unit, color=INK_SECONDARY)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel("metri" if metres else "unità dell'allineamento", color=INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), facecolor=SURFACE)
    _save(fig, out, "fig2_camere")
    plt.close(fig)


def figure_time(results: List[MethodResult], times: Dict[str, datetime], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import FAMILY_COLOR, GRID, _style

    _style()
    fig, axes = plt.subplots(2, len(results), figsize=(3.0 * len(results), 5.2), sharex=True, sharey="row",
                             constrained_layout=True, squeeze=False)
    for column, r in enumerate(results):
        names = [n for n in r.per_view if n in times]
        hours = [times[n].hour + times[n].minute / 60 for n in names]
        for row, metric in enumerate(("psnr", "lpips")):
            ax = axes[row][column]
            ax.scatter(hours, [r.per_view[n][metric] for n in names], s=12, color=FAMILY_COLOR[r.method.family],
                       linewidth=0, alpha=0.8)
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
            if row == 0:
                ax.set_title(r.method.short, fontsize=10)
            if column == 0:
                ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "LPIPS (più basso è meglio)")
            if row == 1:
                ax.set_xlabel("ora di scatto")
    fig.suptitle("Qualità per vista rispetto all'ora di scatto", x=0.01, ha="left", fontsize=12, fontweight="bold")
    _save(fig, out, "fig5_ora")
    plt.close(fig)


def figure_jpg_dng(results: List[MethodResult], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .report import GRID, _style

    _style()
    cams = sorted({camera_of(n) for r in results for n in r.per_view})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 1.6 + 0.6 * len(results)),
                             constrained_layout=True)
    for ax, metric in zip(axes, METRICS):
        for i, r in enumerate(results):
            for j, cam in enumerate(cams):
                values = [v[metric] for n, v in r.per_view.items() if camera_of(n) == cam]
                if not values:
                    continue
                y = i + (j - (len(cams) - 1) / 2) * 0.28
                ax.scatter(values, [y] * len(values), s=9, color=CAMERA_COLOR.get(cam, "#898781"), alpha=0.55,
                           linewidth=0, label=f"{cam.upper()}" if i == 0 else None)
                ax.plot([statistics.median(values)] * 2, [y - 0.11, y + 0.11], color="#0b0b0b", linewidth=1.6)
        ax.set_yticks(range(len(results)), [r.method.short for r in results])
        ax.invert_yaxis()
        ax.set_title(metric.upper() + (" (più basso è meglio)" if metric == "lpips" else ""))
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    counts = ", ".join(f"{cam.upper()} {sum(1 for n in results[0].per_view if camera_of(n) == cam)} viste" for cam in cams)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{label}" for label in labels], frameon=False, fontsize=8.5, loc="outside lower center",
               ncol=len(labels) + 1, title=counts, title_fontsize=8)
    fig.suptitle("Metriche per vista: foto JPG della camera e fotogrammi sviluppati da DNG (riga nera: mediana)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    _save(fig, out, "fig5b_jpg_dng")
    plt.close(fig)


# --- insieme

def build(project: str, out: Path, discarded_file: Optional[Path] = None, figures: bool = True) -> dict:
    scene = config.WORK / "data" / project
    out.mkdir(parents=True, exist_ok=True)
    numbers: dict = {"progetto": project}
    results = collect(project)
    table_data(scene, read_discarded(discarded_file), out, numbers)
    table_configurations(results, out, numbers)
    table_results(results, out, numbers)
    if len(results) > 1:
        table_paired(results, out, numbers)
    numbers["tab6.disponibile"] = table_resolution(project, results, out, numbers)
    numbers["tab7.disponibile"] = table_nerf_options(project, results, out, numbers)
    distance = exposure_distance(scene, project)
    exposure_numbers(results, distance, numbers)
    numbers["mvs.disponibile"] = table_photogrammetry(scene, out, numbers)
    metres = metres_per_unit(project)
    if metres:
        from . import scala
        data = scala.load(project)
        numbers.update({"scala.metri_per_unita": metres, "scala.stima": data["use"],
                        "scala.ic95": data.get("scale_horizontal_ci95") if data["use"] == "scale_horizontal" else None,
                        "scala.residuo_orizzontale_m": data.get("horizontal_residual_median_m"),
                        "scala.errore_medio_3d_m": data.get("error_mean_m")})
    if figures:
        figure_cameras(scene, out, metres)
        times = taken_times(scene)
        numbers["ora.prima"] = min(times.values()).strftime("%H:%M") if times else None
        numbers["ora.ultima"] = max(times.values()).strftime("%H:%M") if times else None
        figure_time(results, times, out)
        figure_jpg_dng(results, out)
        if distance:
            figure_exposure(results, distance, out)
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in numbers.items()}
    (out / "numeri.json").write_text(json.dumps(clean, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scartate", type=Path, help="file con le foto tolte prima del progetto (es. LEGGIMI.txt)")
    args = parser.parse_args()
    numbers = build(args.project, args.out, args.scartate)
    methods = sorted({k.split(".")[0] for k in numbers if k.endswith(".psnr_mean")})
    print(f"Metodi: {', '.join(methods)}. Tabelle, figure e numeri in {args.out.resolve()}")


if __name__ == "__main__":
    main()
