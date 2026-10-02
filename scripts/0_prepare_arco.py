"""Passo 0 (specifico per il rilievo dell'arco) - prepara data/arco/input.

Base: i JPG della camera (gia' corretti per distorsione e vignettatura), collegati con
hardlink in input/jpg. I fotogrammi presenti solo come DNG vengono sviluppati con rawpy in
input/dng, applicando una curva tonale stimata sulle coppie JPG/DNG degli scatti vicini, cosi'
da avere una resa simile ai JPG. Le due cartelle vanno trattate come camere distinte in COLMAP
(i DNG sviluppati non hanno la correzione di distorsione).

Uso:  python scripts/0_prepare_arco.py
"""
import os
import re
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
JPG_DIR = ROOT / "foto jpg"
DNG_DIR = ROOT / "foto dng"
OUT = ROOT / "data" / "arco" / "input"
# Scatti da escludere: 636 e' una foto di gruppo, non inquadra l'arco.
EXCLUDE = {636}
# Coppie JPG/DNG usate per stimare la curva tonale (attorno al blocco solo-DNG 184-236).
FIT_FRAMES = [176, 178, 180, 182, 183, 237, 238, 240, 242, 244]
CHECK_FRAMES = [177, 181, 239, 243]


def number(name: str) -> int:
    return int(re.search(r"_(\d{4})_D", name).group(1))


def develop(path: Path) -> np.ndarray:
    """DNG -> RGB 16 bit con bilanciamento del bianco della camera.

    Si lascia l'auto-luminosita' di rawpy: con luminosita' fissa la curva stimata sbaglia di
    piu' (il JPG della camera non e' una funzione fissa del raw). Anche cosi' la luminosita'
    media si discosta dai JPG fino a ~15-20% sugli scatti di controllo.
    """
    with rawpy.imread(str(path)) as raw:
        return raw.postprocess(use_camera_wb=True, output_bps=16)


def fit_lut(pairs) -> np.ndarray:
    """Curva per canale (65536 -> 0..255) per histogram matching sui pixel di tutte le coppie."""
    src_hist = np.zeros((3, 65536))
    ref_hist = np.zeros((3, 256))
    for dng, jpg in pairs:
        src = develop(dng)[::4, ::4]
        ref = np.asarray(Image.open(jpg).convert("RGB"))[::4, ::4]
        for c in range(3):
            src_hist[c] += np.bincount(src[..., c].ravel(), minlength=65536)
            ref_hist[c] += np.bincount(ref[..., c].ravel(), minlength=256)
    lut = np.zeros((3, 65536), dtype=np.uint8)
    for c in range(3):
        src_cdf = np.cumsum(src_hist[c]) / src_hist[c].sum()
        ref_cdf = np.cumsum(ref_hist[c]) / ref_hist[c].sum()
        lut[c] = np.clip(np.interp(src_cdf, ref_cdf, np.arange(256)), 0, 255).round()
    return lut


def apply_lut(img16: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return np.stack([lut[c][img16[..., c]] for c in range(3)], axis=-1)


def main() -> None:
    jpgs = {number(f.name): f for f in JPG_DIR.iterdir()}
    dngs = {number(f.name): f for f in DNG_DIR.iterdir()}
    (OUT / "jpg").mkdir(parents=True, exist_ok=True)
    (OUT / "dng").mkdir(parents=True, exist_ok=True)

    linked = 0
    for n, src in sorted(jpgs.items()):
        dst = OUT / "jpg" / src.name
        if n not in EXCLUDE and not dst.exists():
            os.link(src, dst)
            linked += 1
    print(f"JPG collegati: {linked}")

    lut = fit_lut([(dngs[n], jpgs[n]) for n in FIT_FRAMES])
    for n in CHECK_FRAMES:
        dev = apply_lut(develop(dngs[n]), lut).astype(np.float32)
        ref = np.asarray(Image.open(jpgs[n]).convert("RGB")).astype(np.float32)
        print(f"controllo {n}: media sviluppato {dev.mean((0, 1)).round(1)}  media JPG {ref.mean((0, 1)).round(1)}")

    only_dng = sorted(n for n in dngs if n not in jpgs and n not in EXCLUDE)
    for n in only_dng:
        dst = OUT / "dng" / (dngs[n].stem + ".jpg")
        if not dst.exists():
            Image.fromarray(apply_lut(develop(dngs[n]), lut)).save(dst, quality=95, subsampling=0)
    print(f"DNG sviluppati: {len(only_dng)} ({only_dng[0]}-{only_dng[-1]})")


if __name__ == "__main__":
    main()
