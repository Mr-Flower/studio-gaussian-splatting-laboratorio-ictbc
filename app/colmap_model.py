"""Lettura delle immagini di una ricostruzione COLMAP e suddivisione training/test comune a tutti i metodi."""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

SPLIT_FILE = "split.json"
TEST_EVERY = 8  # una vista di test ogni otto, la convenzione di Mip-NeRF 360 e 3D Gaussian Splatting


@dataclass(frozen=True)
class ModelImage:
    image_id: int
    name: str  # percorso relativo alla cartella delle foto, con "/"
    points2d: int  # punti chiave rilevati
    triangulated: int  # punti chiave associati a un punto 3D


def read_images(model: Path) -> List[ModelImage]:
    """Immagini registrate in un modello sparso (file images.bin)."""
    images = []
    with open(model / "images.bin", "rb") as f:
        (count,) = struct.unpack("<Q", f.read(8))
        for _ in range(count):
            image_id, *_pose, _camera_id = struct.unpack("<i7di", f.read(64))
            name = bytearray()
            while (char := f.read(1)) != b"\x00":
                name += char
            (points2d,) = struct.unpack("<Q", f.read(8))
            data = f.read(24 * points2d)  # per punto: x, y (double) e id del punto 3D (int64, -1 se assente)
            triangulated = sum(1 for i in range(points2d) if data[24 * i + 16: 24 * i + 24] != b"\xff" * 8)
            images.append(ModelImage(image_id, name.decode("utf-8"), points2d, triangulated))
    return images


def choose_split(names: List[str]) -> Dict[str, List[str]]:
    """Suddivisione deterministica: in ordine di nome, una foto ogni TEST_EVERY va al test."""
    ordered = sorted(names)
    test = ordered[::TEST_EVERY] if len(ordered) >= TEST_EVERY else ordered[:1]
    held = set(test)
    return {"train": [n for n in ordered if n not in held], "test": test}


def write_split(scene: Path, model: Path) -> Dict[str, List[str]]:
    """Sceglie le viste di test e le scrive dove le leggono i diversi metodi.

    - split.json: elenco con i nomi originali delle foto;
    - transforms.json: elenchi train/val/test che nerfstudio usa al posto della sua suddivisione.
    """
    images = read_images(model)
    split = choose_split([image.name for image in images])

    transforms_path = scene / "transforms.json"
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    name_of = {image.image_id: image.name for image in images}
    held = set(split["test"])
    train, test = [], []
    frames = {}  # nome dato da nerfstudio alla copia -> nome originale della foto
    for frame in transforms["frames"]:
        original = name_of[frame["colmap_im_id"]]
        frames[Path(frame["file_path"]).name] = original
        (test if original in held else train).append(frame["file_path"])
    (scene / SPLIT_FILE).write_text(json.dumps({**split, "frames": frames}, indent=1), encoding="utf-8")
    transforms["train_filenames"] = sorted(train)
    transforms["val_filenames"] = sorted(test)
    transforms["test_filenames"] = sorted(test)
    transforms_path.write_text(json.dumps(transforms, indent=4), encoding="utf-8")
    return split


def read_split(scene: Path) -> Dict[str, List[str]]:
    try:
        return json.loads((scene / SPLIT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"train": [], "test": []}
