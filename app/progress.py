"""Lettura dell'avanzamento dalle righe di log di COLMAP e nerfstudio (funzioni pure)."""
from __future__ import annotations

import re
from typing import Callable, Dict, Optional

Fraction = Callable[[str], Optional[float]]

# Riga della tabella di training di nerfstudio, es.
# "2760 (9.20%)        14.956 ms            6 m, 47 s            122.44 M"
# Le colonne sono separate da piu' spazi: passo, tempo per iterazione, tempo rimanente, raggi al secondo.
_TRAIN_ROW = re.compile(r"^\s*(\d+) \((\d+(?:\.\d+)?)%\)\s{2,}\S.*?\s{2,}(\S.*?)(?:\s{2,}|$)")
_TRAIN_HEADER = re.compile(r"^\s*(Step \(% Done\)|-{20,})")
# ns-train scrive "Viewer running locally at: http://localhost:7007"; ns-viewer solo il riquadro
# di viser, "│   HTTP      │ http://0.0.0.0:7007   │".
_VIEWER = re.compile(r"(?:Viewer running locally at:|HTTP\s*\S?)\s*http://[^:\s]+:(\d+)")


def ratio(pattern: str) -> Fraction:
    """Avanzamento da righe del tipo '[12/40]': il pattern cattura (fatti, totale)."""
    rx = re.compile(pattern)

    def parse(line: str) -> Optional[float]:
        m = rx.search(line)
        return int(m.group(1)) / max(int(m.group(2)), 1) if m else None

    return parse


def percent(pattern: str) -> Fraction:
    """Avanzamento da righe che riportano una percentuale: il pattern la cattura."""
    rx = re.compile(pattern)

    def parse(line: str) -> Optional[float]:
        m = rx.search(line)
        return int(m.group(1)) / 100 if m else None

    return parse


def matching(line: str) -> Optional[float]:
    m = re.search(r"Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]", line)
    if m:
        a, rows, b, cols = map(int, m.groups())
        return ((a - 1) * cols + b) / (rows * cols)
    return ratio(r"\[(\d+)/(\d+)\]")(line)


def mapper(total_images: int) -> Fraction:
    def parse(line: str) -> Optional[float]:
        m = re.search(r"num_reg_frames=(\d+)", line)
        return int(m.group(1)) / max(total_images, 1) if m else None

    return parse


def train(line: str) -> Optional[float]:
    m = _TRAIN_ROW.search(line)
    return float(m.group(2)) / 100 if m else None


def train_detail(iterations: int) -> Callable[[str], Optional[str]]:
    def parse(line: str) -> Optional[str]:
        m = _TRAIN_ROW.search(line)
        return f"iterazione {int(m.group(1)) + 1} di {iterations}, restano circa {m.group(3)}" if m else None

    return parse


def is_train_table(line: str) -> bool:
    """Righe ripetute della tabella di training: utili per lo stato, rumore nel log a video."""
    return bool(_TRAIN_ROW.search(line) or _TRAIN_HEADER.search(line))


def patch_match() -> Fraction:
    # Con geom_consistency COLMAP elabora ogni vista due volte (fotometrica, poi geometrica).
    seen = [0]

    def parse(line: str) -> Optional[float]:
        m = re.search(r"Processing view \d+ / (\d+)", line)
        if not m:
            return None
        seen[0] += 1
        return seen[0] / (2 * int(m.group(1)))

    return parse


def viewer_url(line: str) -> Optional[str]:
    m = _VIEWER.search(line)
    return f"http://localhost:{m.group(1)}" if m else None


class AlignmentStats:
    """Raccoglie i numeri stampati da 'colmap model_analyzer'."""

    _FIELDS = {
        "registered_images": re.compile(r"Registered images: (\d+)"),
        "points": re.compile(r"Points: (\d+)"),
        "mean_track_length": re.compile(r"Mean track length: ([\d.]+)"),
        "mean_reprojection_error_px": re.compile(r"Mean reprojection error: ([\d.]+)"),
    }

    def __init__(self) -> None:
        self.values: Dict[str, float] = {}

    def feed(self, line: str) -> None:
        for key, rx in self._FIELDS.items():
            m = rx.search(line)
            if m:
                self.values[key] = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
