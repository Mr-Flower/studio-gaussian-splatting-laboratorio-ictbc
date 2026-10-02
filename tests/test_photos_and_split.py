import json
import random
import struct

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app import colmap_model, photos
from app.photos import Photo


def write_images_bin(model, images):
    """images.bin nel formato di COLMAP: per immagine id, posa, camera, nome, punti 2D con id del punto 3D."""
    model.mkdir(parents=True)
    with open(model / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id, name, point_ids in images:
            f.write(struct.pack("<i7di", image_id, 1, 0, 0, 0, 0, 0, 0, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", len(point_ids)))
            for point_id in point_ids:
                f.write(struct.pack("<ddq", 1.5, 2.5, point_id))


def test_read_images(tmp_path):
    write_images_bin(tmp_path / "0", [(7, "jpg/b.JPG", [3, -1, 9]), (2, "dng/a.jpg", [-1, -1])])
    first, second = colmap_model.read_images(tmp_path / "0")
    assert (first.image_id, first.name, first.points2d, first.triangulated) == (7, "jpg/b.JPG", 3, 2)
    assert (second.name, second.triangulated) == ("dng/a.jpg", 0)


def test_one_view_in_eight_goes_to_the_test_set():
    names = [f"img_{i:03d}.jpg" for i in reversed(range(20))]
    split = colmap_model.choose_split(names)
    assert split["test"] == ["img_000.jpg", "img_008.jpg", "img_016.jpg"]
    assert len(split["train"]) == 17 and not set(split["train"]) & set(split["test"])
    assert colmap_model.choose_split(["b", "a"]) == {"train": ["b"], "test": ["a"]}


def test_split_is_written_for_every_method(tmp_path):
    scene = tmp_path
    images = [(i + 1, f"foto_{i:02d}.jpg", [1]) for i in range(10)]
    write_images_bin(scene / "colmap" / "sparse" / "0", images)
    frames = [{"file_path": f"images/frame_{i + 1:05d}.jpg", "colmap_im_id": i + 1} for i in range(10)]
    (scene / "transforms.json").write_text(json.dumps({"frames": frames}))

    split = colmap_model.write_split(scene, scene / "colmap" / "sparse" / "0")
    assert split["test"] == ["foto_00.jpg", "foto_08.jpg"]
    transforms = json.loads((scene / "transforms.json").read_text())
    # nerfstudio riceve gli stessi nomi, tradotti nelle sue copie rinominate
    assert transforms["test_filenames"] == ["images/frame_00001.jpg", "images/frame_00009.jpg"]
    assert transforms["val_filenames"] == transforms["test_filenames"] and len(transforms["train_filenames"]) == 8
    saved = colmap_model.read_split(scene)
    assert saved["frames"]["frame_00009.jpg"] == "foto_08.jpg"


def textured(seed: int = 0) -> Image.Image:
    """Immagine con dettaglio a tutte le scale: rumore fine sopra una trama grossolana."""
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 255, (30, 40), dtype=np.uint8)).resize((640, 480), Image.BICUBIC)
    fine = rng.integers(-40, 40, (480, 640))
    return Image.fromarray(np.clip(np.asarray(coarse, dtype=np.int16) + fine, 0, 255).astype(np.uint8))


def fingerprints(count: int, seed: int = 0) -> list:
    """Impronte casuali di 256 bit: tra loro distano circa 128 bit, quindi non sono doppioni."""
    rng = random.Random(seed)
    return [f"{rng.getrandbits(256):064x}" for _ in range(count)]


def test_sharpness_drops_with_blur_but_not_with_flat_areas():
    sharp = textured()
    blurred = sharp.filter(ImageFilter.GaussianBlur(6))
    with_sky = sharp.copy()
    with_sky.paste(180, (0, 0, 640, 300))  # meta' superiore uniforme, come un cielo
    assert photos.sharpness_of(blurred) < 0.4 * photos.sharpness_of(sharp)
    assert photos.sharpness_of(with_sky) == pytest.approx(photos.sharpness_of(sharp), rel=0.35)


def photo(name, sharpness=1.0, fingerprint="0" * 64, **values) -> Photo:
    return Photo(name=name, width=100, height=80, sharpness=sharpness, camera="X", fingerprint=fingerprint, **values)


def test_suggestions():
    prints = fingerprints(12)
    prints[1] = f"{int(prints[0], 16) ^ 0b11:064x}"  # 1.jpg e 2.jpg: a pochi bit da 0.jpg
    prints[2] = f"{int(prints[0], 16) ^ 0b1100:064x}"
    rows = [photo(f"jpg/{i}.jpg", fingerprint=prints[i]) for i in range(12)]
    rows[0].sharpness = 0.2  # poco nitida
    rows[5].over = 0.6  # sovraesposta
    rows[6].under = 0.7  # sottoesposta
    rows[8].over = 0.3  # un cielo bianco: non basta per suggerire l'esclusione
    rows[7].width = 50  # formato diverso: osservazione, non motivo di esclusione
    rows.append(Photo(name="jpg/rotta.jpg", error="file danneggiato"))
    photos.suggest(rows)
    by_name = {p.name: p for p in rows}
    assert photos.BLURRY in by_name["jpg/0.jpg"].reasons
    assert by_name["jpg/5.jpg"].reasons == [photos.OVEREXPOSED] and by_name["jpg/6.jpg"].reasons == [photos.UNDEREXPOSED]
    assert by_name["jpg/7.jpg"].notes == [photos.DIFFERENT] and not by_name["jpg/7.jpg"].suggested
    assert not by_name["jpg/8.jpg"].suggested
    assert by_name["jpg/rotta.jpg"].suggested
    # 1.jpg e 2.jpg hanno impronte quasi identiche a quella di 0.jpg: ne resta una sola senza segnalazione
    duplicates = [p.name for p in rows if any(r.startswith(photos.DUPLICATE) for r in p.reasons)]
    assert len(duplicates) == 2 and set(duplicates) < {"jpg/0.jpg", "jpg/1.jpg", "jpg/2.jpg"}


def test_sharpness_is_compared_within_the_same_folder():
    # I raw sviluppati (cartella dng) hanno meno dettaglio fine dei JPG della camera: non sono per questo mossi.
    prints = fingerprints(20, seed=1)
    rows = [photo(f"jpg/{i}.jpg", sharpness=1.0, fingerprint=prints[i]) for i in range(10)]
    rows += [photo(f"dng/{i}.jpg", sharpness=0.3, fingerprint=prints[10 + i]) for i in range(10)]
    photos.suggest(rows)
    assert not any(p.suggested for p in rows)


def test_exclusions_are_saved_and_applied(tmp_path):
    folder, scene = tmp_path / "foto", tmp_path / "progetto"
    (folder / "sub").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg", "sub/c.jpg", "note.txt"):
        (folder / name).touch()
    assert photos.kept_images(scene, str(folder)) == ["a.jpg", "b.jpg", "sub/c.jpg"]
    photos.set_excluded(scene, {"b.jpg"})
    assert photos.kept_images(scene, str(folder)) == ["a.jpg", "sub/c.jpg"]
    assert photos.load_excluded(scene) == {"b.jpg"}


def test_analysis_of_real_files(tmp_path):
    folder, scene = tmp_path / "foto", tmp_path / "progetto"
    folder.mkdir()
    for i in range(3):
        textured(i).convert("RGB").save(folder / f"{i}.jpg", quality=95)
    textured(0).convert("RGB").save(folder / "copia.jpg", quality=95)
    (folder / "rotta.jpg").write_bytes(b"non e' un'immagine")
    seen = []
    found = {p.name: p for p in photos.analyze(folder, scene, lambda done, total: seen.append((done, total)))}
    assert seen[-1] == (5, 5) and found["rotta.jpg"].reasons == ["file illeggibile"]
    assert sum(1 for name in ("0.jpg", "copia.jpg") if found[name].suggested) == 1  # uno dei due e' il doppione
    assert photos.thumb_path(scene, "0.jpg").exists()
    assert [p.name for p in photos.load(scene, str(folder))] == sorted(found)
    assert photos.load(scene, str(tmp_path / "altra_cartella")) == []
