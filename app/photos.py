"""Analisi preliminare delle foto: nitidezza, esposizione, quasi-doppioni, foto diverse dal resto.

L'analisi propone le foto da escludere, la decisione resta all'utente. Nessun file viene toccato:
le foto escluse sono solo saltate dall'allineamento (elenco in data/<progetto>/photos.json).
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from . import config

PHOTOS_FILE = "photos.json"
THUMBS_DIR = "thumbs"
THUMB_SIZE = (192, 144)

# Soglie dei suggerimenti. Sono relative al set, perche' i valori assoluti dipendono dal soggetto.
BLUR_RATIO = 0.4  # nitidezza sotto il 40% della mediana della propria cartella
CLIPPED_FRACTION = 0.5  # piu' di meta' dei pixel bruciati o neri (un cielo bianco da solo non basta)
DUPLICATE_DISTANCE = 10  # bit diversi su 256 dell'impronta percettiva

BLURRY = "poco nitida"
OVEREXPOSED = "sovraesposta"
UNDEREXPOSED = "sottoesposta"
DUPLICATE = "quasi doppione di"
DIFFERENT = "formato diverso dalle altre"


@dataclass
class Photo:
    name: str  # percorso relativo alla cartella delle foto, con "/"
    width: int = 0
    height: int = 0
    sharpness: float = 0.0  # dettaglio fine rispetto al grossolano, vedi sharpness_of
    relative_sharpness: float = 1.0  # rispetto alla mediana delle foto della stessa cartella
    brightness: float = 0.0  # luminosita' media, 0..255
    over: float = 0.0  # frazione di pixel bruciati
    under: float = 0.0  # frazione di pixel neri
    camera: str = ""
    taken: str = ""
    fingerprint: str = ""  # impronta percettiva (dHash 16x16, esadecimale)
    error: str = ""
    reasons: List[str] = field(default_factory=list)  # motivi per cui se ne suggerisce l'esclusione
    notes: List[str] = field(default_factory=list)  # osservazioni che non comportano l'esclusione

    @property
    def suggested(self) -> bool:
        return bool(self.reasons)


def list_photos(folder: Path) -> List[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in config.IMAGE_EXTS)


def thumb_path(scene: Path, name: str) -> Path:
    return scene / THUMBS_DIR / (name.replace("/", "__") + ".jpg")


def _detail_energy(gray: Image.Image, scale: int, grid: int = 12) -> np.ndarray:
    """Energia del laplaciano dell'immagine ridotta di `scale`, mediata su una griglia di riquadri."""
    g = np.asarray(gray.resize((max(gray.width // scale, 3), max(gray.height // scale, 3)), Image.BOX), dtype=np.float32)
    energy = (g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]) ** 2
    rows = np.array_split(np.arange(energy.shape[0]), grid)
    cols = np.array_split(np.arange(energy.shape[1]), grid)
    return np.array([[energy[np.ix_(r, c)].mean() if len(r) and len(c) else 0.0 for c in cols] for r in rows])


def sharpness_of(gray: Image.Image) -> float:
    """Nitidezza indipendente dal soggetto: dettaglio fine rispetto al dettaglio grossolano.

    Una foto mossa o sfocata perde il dettaglio fine (scala 1/2) ma conserva quello grossolano
    (scala 1/8). Il rapporto e' calcolato solo sul quarto di riquadri piu' ricchi di dettaglio:
    cosi' cielo e superfici lisce, che non dicono nulla sulla messa a fuoco, non contano.
    """
    fine, coarse = _detail_energy(gray, 2), _detail_energy(gray, 8)
    textured = coarse >= np.quantile(coarse, 0.75)
    total = float(coarse[textured].sum())
    return float(fine[textured].sum()) / total if total > 0 else 0.0


def measure(folder: Path, path: Path, thumbs: Optional[Path] = None) -> Photo:
    """Misure di una foto: dimensioni, dati EXIF, nitidezza, esposizione, impronta percettiva."""
    photo = Photo(name=path.relative_to(folder).as_posix())
    try:
        with Image.open(path) as im:
            photo.width, photo.height = im.size
            exif = im.getexif()
            photo.camera = " ".join(str(exif.get(tag, "")).strip() for tag in (0x010F, 0x0110)).strip()
            photo.taken = str(exif.get_ifd(0x8769).get(0x9003, "") or exif.get(0x0132, ""))
            if thumbs is not None:
                im.draft("RGB", (THUMB_SIZE[0] * 2, THUMB_SIZE[1] * 2))
                thumb = im.convert("RGB")
                thumb.thumbnail(THUMB_SIZE)
                target = thumbs / (photo.name.replace("/", "__") + ".jpg")
                thumb.save(target, quality=80)
        with Image.open(path) as im:
            gray = im.convert("L")
        half = gray.resize((max(gray.width // 2, 1), max(gray.height // 2, 1)), Image.BOX)
        pixels = np.asarray(half)
        photo.sharpness = sharpness_of(gray)
        photo.brightness = float(pixels.mean())
        photo.over = float((pixels >= 250).mean())
        photo.under = float((pixels <= 5).mean())
        small = np.asarray(gray.resize((17, 16), Image.BOX), dtype=np.int16)
        bits = (small[:, 1:] > small[:, :-1]).flatten()
        photo.fingerprint = np.packbits(bits).tobytes().hex()
    except Exception as exc:  # file illeggibile: va segnalato, non deve fermare l'analisi
        photo.error = str(exc)
    return photo


def _distance(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def suggest(photos: List[Photo]) -> None:
    """Compila motivi e note di ogni foto confrontandola con il resto del set."""
    valid = [p for p in photos if not p.error]
    for p in photos:
        p.reasons, p.notes = (["file illeggibile"], []) if p.error else ([], [])
    if not valid:
        return
    common_size = Counter((p.width, p.height) for p in valid).most_common(1)[0][0]
    common_camera = Counter(p.camera for p in valid).most_common(1)[0][0]
    # La nitidezza si confronta tra foto della stessa sottocartella: sviluppi diversi (ad esempio
    # JPG della camera e raw convertiti) hanno livelli di dettaglio fine non paragonabili.
    groups: Dict[str, List[Photo]] = {}
    for p in valid:
        groups.setdefault(p.name.rpartition("/")[0], []).append(p)
    overall = statistics.median(p.sharpness for p in valid)
    medians = {key: statistics.median(p.sharpness for p in group) if len(group) >= 10 else overall
               for key, group in groups.items()}
    for p in valid:
        median = medians[p.name.rpartition("/")[0]]
        p.relative_sharpness = p.sharpness / median if median else 1.0
        if p.relative_sharpness < BLUR_RATIO:
            p.reasons.append(BLURRY)
        if p.over > CLIPPED_FRACTION:
            p.reasons.append(OVEREXPOSED)
        if p.under > CLIPPED_FRACTION:
            p.reasons.append(UNDEREXPOSED)
        if (p.width, p.height) != common_size or p.camera != common_camera:
            p.notes.append(DIFFERENT)
    # Quasi-doppioni: tra due foto quasi identiche si tiene la piu' nitida.
    kept: List[Photo] = []
    for p in sorted(valid, key=lambda x: -x.relative_sharpness):
        twin = next((k for k in kept if _distance(p.fingerprint, k.fingerprint) <= DUPLICATE_DISTANCE), None)
        if twin is None:
            kept.append(p)
        else:
            p.reasons.append(f"{DUPLICATE} {twin.name}")


def analyze(folder: Path, scene: Path, on_progress: Optional[Callable[[int, int], None]] = None) -> List[Photo]:
    """Analizza tutte le foto della cartella e salva misure e miniature nel progetto."""
    paths = list_photos(folder)
    thumbs = scene / THUMBS_DIR
    thumbs.mkdir(parents=True, exist_ok=True)
    photos: List[Photo] = []
    with ThreadPoolExecutor(max_workers=8) as pool:  # la decodifica delle immagini rilascia il GIL
        for done, photo in enumerate(pool.map(lambda p: measure(folder, p, thumbs), paths), start=1):
            photos.append(photo)
            if on_progress:
                on_progress(done, len(paths))
    suggest(photos)
    save(scene, photos, load_excluded(scene) & {p.name for p in photos}, folder)
    return photos


# --- salvataggio

def save(scene: Path, photos: List[Photo], excluded, folder: Optional[Path] = None) -> None:
    previous = _read(scene)
    data = {
        "folder": str(folder) if folder else previous.get("folder", ""),
        "excluded": sorted(excluded),
        "photos": [asdict(p) for p in photos],
    }
    scene.mkdir(parents=True, exist_ok=True)
    (scene / PHOTOS_FILE).write_text(json.dumps(data, indent=1), encoding="utf-8")


def _read(scene: Path) -> Dict:
    try:
        return json.loads((scene / PHOTOS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load(scene: Path, folder: Optional[str] = None) -> List[Photo]:
    """Analisi salvata; vuota se non c'e' o se riguarda un'altra cartella di foto."""
    data = _read(scene)
    if folder is not None and data.get("folder") and Path(data["folder"]) != Path(folder):
        return []
    return [Photo(**p) for p in data.get("photos", [])]


def load_excluded(scene: Path) -> set:
    return set(_read(scene).get("excluded", []))


def set_excluded(scene: Path, excluded) -> None:
    save(scene, load(scene), excluded)


def kept_images(scene: Path, folder: str) -> List[str]:
    """Foto da usare per l'allineamento: tutte quelle della cartella meno quelle escluse."""
    excluded = load_excluded(scene)
    root = Path(folder)
    return [name for name in (p.relative_to(root).as_posix() for p in list_photos(root)) if name not in excluded]


def summary(photos: List[Photo], excluded) -> str:
    if not photos:
        return "Foto non ancora analizzate."
    suggested = sum(1 for p in photos if p.suggested)
    return f"{len(photos)} foto analizzate, {suggested} da rivedere, {len(excluded)} escluse dall'allineamento."
