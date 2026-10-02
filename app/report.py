"""Report del confronto tra metodi: grafici, tabella delle misure, differenze di processo, viste a confronto.

Considera, per ogni metodo, il run piu' recente fatto sull'allineamento corrente e gia' valutato.
Uso:  python -m app.report --project <nome>   ->  reports/<nome>/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from . import __version__, config, runs  # noqa: E402
from .runs import Run  # noqa: E402

# Palette categorica di riferimento (primi tre colori: distinguibili tra loro anche con daltonismo).
# Il colore indica la famiglia del metodo, non la sua posizione in classifica.
FAMILY_COLOR = {"gaussian": "#2a78d6", "nerf": "#eb6834", "mvs": "#1baf7a"}
FAMILY_LABEL = {"gaussian": "3D Gaussian Splatting", "nerf": "NeRF", "mvs": "Fotogrammetria (MVS)"}
SURFACE, INK, INK_SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# Differenze di processo: descrizione qualitativa dei metodi, indipendente dalle misure.
PROCESS = {
    "gaussian": {
        "Rappresentazione": "Insieme esplicito di gaussiane 3D con posizione, forma, opacità e colore dipendente dalla direzione",
        "Punto di partenza": "Nuvola sparsa dell'allineamento",
        "Ottimizzazione": "Discesa del gradiente sui parametri delle gaussiane, con aggiunta e rimozione durante il training",
        "Rendering": "Rasterizzazione, in tempo reale",
        "Risultato": "File .ply di gaussiane, modificabile (pulizia, ritaglio) e pubblicabile sul web",
        "Geometria": "Implicita nelle gaussiane; nessuna superficie",
    },
    "nerf": {
        "Rappresentazione": "Campo di radianza continuo codificato in una rete neurale",
        "Punto di partenza": "Solo le pose delle camere",
        "Ottimizzazione": "Discesa del gradiente sui pesi della rete, campionando raggi dalle immagini",
        "Rendering": "Volumetrico lungo ogni raggio, non in tempo reale",
        "Risultato": "Pesi della rete; esportabile come nuvola di punti campionata",
        "Geometria": "Densità volumetrica; superfici solo per estrazione",
    },
    "mvs": {
        "Rappresentazione": "Nuvola di punti densa e mesh triangolare",
        "Punto di partenza": "Pose delle camere e nuvola sparsa",
        "Ottimizzazione": "Nessun training: mappe di profondità per corrispondenza tra viste, poi fusione",
        "Rendering": "Mesh con colori per vertice, in tempo reale; nessuna dipendenza dalla direzione di vista",
        "Risultato": "File .ply di punti e di mesh, misurabile e usabile in CAD e stampa 3D",
        "Geometria": "Esplicita: superficie ricostruita con il metodo di Poisson",
    },
}


def collect(project: str) -> List[Dict]:
    """Una voce per metodo: l'ultimo run valutato sull'allineamento corrente, a parita' di condizioni."""
    entries = []
    known = [run for run in runs.list_runs(project) if run.method in config.METHODS]
    for run in runs.comparable_group(known):
        method = config.METHODS[run.method]
        metrics = run.metrics()
        info = run.info()
        export = run.export_file
        entries.append({
            "key": method.key, "label": method.label, "short": method.short, "family": method.family, "run": run,
            "psnr": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics["lpips"],
            "fps": metrics.get("fps"), "test_images": metrics.get("test_images"),
            "width": metrics.get("width"), "height": metrics.get("height"),
            "train_minutes": info.get("train_seconds", 0) / 60, "iterations": info.get("iterations"),
            "elements": runs.ply_elements(export) if export else None,
            "export_mb": export.stat().st_size / 2**20 if export else None,
            "model_mb": run.checkpoint().stat().st_size / 2**20,
            "info": info,
        })
    order = list(config.METHODS)
    return sorted(entries, key=lambda e: order.index(e["key"]))


def mesh_entry(scene: Path) -> Optional[Dict]:
    dense = scene / "colmap" / "dense"
    data = runs.read_json(dense / runs.MESH_FILE)
    if not (dense / "mesh-poisson.ply").exists() or "seconds" not in data:
        return None
    return {
        "short": "Fotogrammetria MVS", "label": "Fotogrammetria classica (COLMAP MVS + Poisson)", "family": "mvs",
        "train_minutes": sum(data["seconds"].values()) / 60, "points": runs.ply_elements(dense / "fused.ply"),
        "vertices": runs.ply_elements(dense / "mesh-poisson.ply"), "max_image_size": data.get("max_image_size"),
        "export_mb": (dense / "mesh-poisson.ply").stat().st_size / 2**20,
    }


# --- grafici

def _style() -> None:
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK_SECONDARY,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": INK_SECONDARY, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.facecolor": SURFACE,
    })


def _number(value: float, decimals: int) -> str:
    return f"{value:,.{decimals}f}".replace(",", " ").replace(".", ",")


def _bars(ax, entries: List[Dict], key: str, title: str, note: str, decimals: int) -> None:
    """Barre orizzontali di una misura: una barra per metodo, colore per famiglia, valore scritto a fianco."""
    shown = [e for e in entries if e.get(key) is not None]
    labels = [e["short"] for e in shown]
    values = [e[key] for e in shown]
    positions = range(len(shown))
    ax.barh(positions, values, height=0.5, color=[FAMILY_COLOR[e["family"]] for e in shown], linewidth=0)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    limit = max(values) if values else 1
    ax.set_xlim(0, limit * 1.18)
    for position, value in zip(positions, values):
        ax.text(value + limit * 0.015, position, _number(value, decimals), va="center", ha="left", color=INK, fontsize=9)
    ax.set_title(title)
    ax.set_xlabel(note, color=MUTED, fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def _legend(fig, families: List[str]) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in families]
    fig.legend(handles, [FAMILY_LABEL[f] for f in families], loc="upper right", frameon=False, ncol=len(families),
               fontsize=9, handlelength=1.2, handleheight=0.9)


def quality_figure(entries: List[Dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.1 + 0.55 * len(entries)), constrained_layout=True)
    _bars(axes[0], entries, "psnr", "PSNR (dB)", "più alto è meglio", 2)
    _bars(axes[1], entries, "ssim", "SSIM", "più alto è meglio", 3)
    _bars(axes[2], entries, "lpips", "LPIPS", "più basso è meglio", 3)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.suptitle("Qualità delle viste di test", x=0.01, ha="left", fontsize=13, fontweight="bold")
    _legend(fig, sorted({e["family"] for e in entries}, key=list(FAMILY_COLOR).index))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cost_figure(entries: List[Dict], mesh: Optional[Dict], path: Path) -> None:
    timed = entries + ([mesh] if mesh else [])
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.1 + 0.55 * len(timed)), constrained_layout=True)
    _bars(axes[0], timed, "train_minutes", "Tempo di calcolo (minuti)", "più basso è meglio", 1)
    _bars(axes[1], entries, "fps", "Velocità di rendering (fotogrammi al secondo)", "più alto è meglio", 1)
    _bars(axes[2], timed, "export_mb", "Dimensione del modello esportato (MB)", "", 1)
    fig.suptitle("Costo di calcolo e dimensioni", x=0.01, ha="left", fontsize=13, fontweight="bold")
    _legend(fig, sorted({e["family"] for e in timed}, key=list(FAMILY_COLOR).index))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def tradeoff_figure(entries: List[Dict], path: Path) -> None:
    """Qualita' rispetto al tempo: un punto per metodo, con il nome scritto accanto."""
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)
    for e in entries:
        ax.scatter(e["train_minutes"], e["psnr"], s=90, color=FAMILY_COLOR[e["family"]], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        ax.annotate(e["short"], (e["train_minutes"], e["psnr"]), xytext=(8, 6), textcoords="offset points",
                    color=INK, fontsize=9)
    times = [e["train_minutes"] for e in entries]
    if min(times) > 0 and max(times) / min(times) > 20:
        ax.set_xscale("log")
    ax.margins(0.18)
    ax.set_xlabel("Tempo di training (minuti)")
    ax.set_ylabel("PSNR sulle viste di test (dB)")
    ax.set_title("Qualità rispetto al tempo di training")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    families = sorted({e["family"] for e in entries}, key=list(FAMILY_COLOR).index)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=FAMILY_COLOR[f]) for f in families]
    ax.legend(handles, [FAMILY_LABEL[f] for f in families], frameon=False, loc="lower right", fontsize=9)
    fig.savefig(path, dpi=160)
    plt.close(fig)


# --- viste a confronto

def view_strips(entries: List[Dict], figures: Path, limit: int = 3) -> List[Path]:
    """Per alcune viste di test: foto vera e immagine prodotta da ogni metodo, affiancate."""
    def saved(e: Dict) -> Dict[str, Path]:
        folder = e["run"].path / "test_views"
        return {p.name[: -len("_pred.jpg")]: p for p in folder.glob("*_pred.jpg")}

    per_method = [(e, saved(e)) for e in entries]
    common = set.intersection(*(set(views) for _, views in per_method)) if per_method else set()
    strips = []
    width = 520
    try:
        font = ImageFont.truetype("segoeui.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    for index, name in enumerate(sorted(common)[:limit]):
        truth = entries[0]["run"].path / "test_views" / f"{name}_gt.jpg"
        tiles = [("Foto originale", truth)] + [(e["short"], views[name]) for e, views in per_method]
        images = []
        for label, path in tiles:
            image = Image.open(path).convert("RGB")
            image = image.resize((width, round(image.height * width / image.width)))
            images.append((label, image))
        height = max(image.height for _, image in images)
        strip = Image.new("RGB", (width * len(images) + 8 * (len(images) - 1), height + 34), SURFACE)
        draw = ImageDraw.Draw(strip)
        for column, (label, image) in enumerate(images):
            x = column * (width + 8)
            draw.text((x + 2, 4), label, fill=INK, font=font)
            strip.paste(image, (x, 34))
        target = figures / f"vista_{index + 1}.jpg"
        strip.save(target, quality=90)
        strips.append(target)
    return strips


# --- pagina

def _table(headers: List[str], rows: List[List[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _fmt(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return f"{value:,}".replace(",", ".") if isinstance(value, int) else str(value)


def build(project: str) -> Path:
    scene = config.WORK / "data" / project
    entries = collect(project)
    if not entries:
        raise SystemExit("Nessun run valutato sull'allineamento corrente: eseguire training e valutazione.")
    mesh = mesh_entry(scene)
    out = config.WORK / "reports" / project
    figures = out / "figure"
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    quality_figure(entries, figures / "qualita.png")
    cost_figure(entries, mesh, figures / "costo.png")
    if len(entries) > 1:
        tradeoff_figure(entries, figures / "qualita_tempo.png")
    strips = view_strips(entries, figures)
    runs.write_csv(project, out / "confronto.csv")

    first = entries[0]
    alignment = runs.read_json(scene / runs.ALIGNMENT_FILE)
    measures = _table(
        ["Metodo", "Iterazioni", "Training (min)", "PSNR (dB)", "SSIM", "LPIPS", "Rendering (fps)",
         "Gaussiane / punti", "Modello esportato (MB)"],
        [[e["label"], _fmt(e["iterations"]), _fmt(e["train_minutes"], 1), _fmt(e["psnr"]), _fmt(e["ssim"], 4),
          _fmt(e["lpips"], 4), _fmt(e["fps"], 1), _fmt(e["elements"]), _fmt(e["export_mb"], 1)] for e in entries])
    families = sorted({e["family"] for e in entries} | ({"mvs"} if mesh else set()), key=list(FAMILY_COLOR).index)
    process = _table(["", *[FAMILY_LABEL[f] for f in families]],
                     [[aspect, *[PROCESS[f][aspect] for f in families]] for aspect in PROCESS["gaussian"]])
    mesh_html = ""
    if mesh:
        mesh_html = "<h2>Fotogrammetria classica</h2>" + _table(
            ["Tempo totale (min)", "Punti della nuvola densa", "Vertici della mesh", "Risoluzione delle immagini (px)"],
            [[_fmt(mesh["train_minutes"], 1), _fmt(mesh["points"]), _fmt(mesh["vertices"]), _fmt(mesh["max_image_size"])]])
        mesh_html += ("<p class='note'>La fotogrammetria non produce immagini da nuovi punti di vista, quindi non ha "
                      "PSNR, SSIM e LPIPS: il confronto con gli altri metodi su quel piano non è possibile.</p>")
    included = {e["key"] for e in entries}
    left_out = sorted({config.METHODS[run.method].label for run in runs.list_runs(project)
                       if run.method in config.METHODS and run.method not in included
                       and run.comparable() is True and "psnr" in run.metrics()})
    left_out_html = ("<p class='note'>Non inclusi, perché allenati con risoluzione o numero di iterazioni diversi: "
                     + html.escape("; ".join(left_out)) + ".</p>") if left_out else ""
    from_disk = [e["label"] for e in entries if e["info"].get("photos_in") == "disk"]
    if from_disk:
        left_out_html += ("<p class='note'>Foto lette dal disco durante il training (non entravano in memoria): "
                          + html.escape("; ".join(from_disk)) + ". Il loro tempo di training include la lettura.</p>")
    views_html = "".join(f"<img src='figure/{p.name}' alt='Vista di test a confronto'>" for p in strips)
    versions = first["info"].get("versions", {})
    conditions = _table(["Condizione", "Valore"], [
        ["Foto allineate", f"{_fmt(alignment.get('registered_images'))} su {_fmt(alignment.get('photos'))}"],
        ["Errore medio di riproiezione (px)", _fmt(alignment.get("mean_reprojection_error_px"))],
        ["Foto di training / di test", f"{_fmt(first['info'].get('train_images'))} / {_fmt(first['test_images'])}"],
        ["Risoluzione delle immagini valutate", f"{first['width']} × {first['height']} px"],
        ["Scheda grafica", first["info"].get("gpu", "—")],
        *[[f"Versione di {name}", version] for name, version in versions.items()],
    ])
    page = f"""<!doctype html>
<html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Confronto dei metodi — {html.escape(project)}</title>
<style>
 body {{ font-family: system-ui, "Segoe UI", sans-serif; background: #f9f9f7; color: {INK}; margin: 0; }}
 main {{ max-width: 1280px; margin: 0 auto; padding: 24px 16px 48px; }}
 h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 32px 0 10px; }}
 .lead, .note {{ color: {INK_SECONDARY}; }} .note {{ font-size: 14px; }}
 img {{ max-width: 100%; height: auto; display: block; margin: 8px 0 16px; border: 1px solid {GRID}; }}
 .scroll {{ overflow-x: auto; }}
 table {{ border-collapse: collapse; background: {SURFACE}; font-size: 14px; }}
 th, td {{ border: 1px solid {GRID}; padding: 6px 10px; text-align: left; vertical-align: top; }}
 th {{ background: #f0efec; }} td {{ font-variant-numeric: tabular-nums; }}
</style></head><body><main>
<h1>Confronto dei metodi di ricostruzione 3D — progetto «{html.escape(project)}»</h1>
<p class="lead">Generato il {time.strftime('%d/%m/%Y alle %H:%M')} con la versione {__version__}. Tutti i metodi usano lo stesso
allineamento delle camere, le stesse foto di training e di test e la stessa risoluzione; le metriche sono calcolate
con lo stesso codice sulle viste escluse dal training.</p>
<h2>Qualità</h2><img src="figure/qualita.png" alt="Grafici a barre di PSNR, SSIM e LPIPS per metodo">
<h2>Costo</h2><img src="figure/costo.png" alt="Grafici a barre di tempo di calcolo, velocità di rendering e dimensione per metodo">
{"<h2>Qualità rispetto al tempo</h2><img src='figure/qualita_tempo.png' alt='PSNR rispetto al tempo di training' style='max-width:760px'>" if len(entries) > 1 else ""}
<h2>Misure</h2><div class="scroll">{measures}</div>
<p class="note">Il tempo di training è la durata dell'intero comando, caricamento dei dati incluso. La velocità di rendering
è misurata sulle viste di test, alla risoluzione indicata sotto. La tabella completa di tutti i run è in
<a href="confronto.csv">confronto.csv</a>.</p>
{left_out_html}
{mesh_html}
{"<h2>Viste di test a confronto</h2>" + views_html if strips else ""}
<h2>Differenze di processo</h2><div class="scroll">{process}</div>
<h2>Condizioni dell'esperimento</h2><div class="scroll">{conditions}</div>
<p class="note">Ogni configurazione è stata eseguita una volta: le differenze piccole tra metodi possono rientrare nella
variabilità tra esecuzioni ripetute, che qui non è stimata.</p>
</main></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "dati.json").write_text(json.dumps(
        [{k: v for k, v in e.items() if k not in ("run", "info")} for e in entries] + ([mesh] if mesh else []),
        indent=1), encoding="utf-8")
    return out / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    print(f"Report salvato in {build(args.project)}")


if __name__ == "__main__":
    main()
