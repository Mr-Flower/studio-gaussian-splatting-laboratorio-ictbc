"""Varianti dei metodi di nerfstudio che leggono le foto dal disco durante il training.

nerfstudio carica tutte le foto in memoria prima di partire: con molte foto ad alta risoluzione non
bastano ne' la memoria centrale ne' quella della scheda grafica. Queste varianti scambiano memoria
con tempo, senza toccare il modello ne' l'ottimizzazione:

- splatfacto: ogni foto e' letta dal disco, e corretta dalla distorsione, quando serve
  (DiskImageDatamanager); le prossime sono preparate in anticipo da altri thread;
- nerfacto: i raggi sono campionati da un gruppo di foto tenuto in memoria e sostituito
  periodicamente, la modalita' prevista da nerfstudio per i set di foto grandi.

nerfstudio le scopre tramite la variabile d'ambiente NERFSTUDIO_METHOD_CONFIGS (vedi
config.environment): il nome di ognuna e' quello del metodo di partenza piu' config.DISK_SUFFIX,
la funzione che la costruisce ha il nome del metodo con "_" al posto di "-".
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Type

import numpy as np
import torch
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager, FullImageDatamanagerConfig, _undistort_image,
)

AHEAD = 8  # foto preparate in anticipo, ognuna da un thread
NEVER = 10**9  # passi tra due valutazioni durante il training: mai (la valutazione si fa alla fine)


def _distorted(camera) -> bool:
    return camera.distortion_params is not None and not torch.all(camera.distortion_params == 0)


class DiskImages(Sequence):
    """Si comporta come l'elenco di foto in memoria di nerfstudio, ma carica ogni foto quando viene chiesta."""

    def __init__(self, dataset, cameras, image_type: str, upcoming: Optional[Callable[[], List[int]]] = None) -> None:
        self.dataset = dataset
        self.cameras = cameras  # parametri delle camere prima della correzione della distorsione
        self.image_type = image_type
        self.upcoming = upcoming  # indici delle prossime foto richieste, se sono noti
        self.pool = ThreadPoolExecutor(max_workers=AHEAD)
        self.pending: Dict[int, Future] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def load(self, idx: int) -> Dict[str, torch.Tensor]:
        """Stesse operazioni di FullImageDatamanager._load_images, per una foto sola."""
        data = self.dataset.get_data(idx, image_type=self.image_type)
        camera = self.cameras[idx].reshape(())
        if _distorted(camera):
            K = camera.get_intrinsics_matrices().numpy()
            _, image, mask = _undistort_image(camera, camera.distortion_params.numpy(), data, data["image"].numpy(), K)
            data["image"] = torch.from_numpy(np.ascontiguousarray(image))
            if mask is not None:
                data["mask"] = mask
        return data

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if not 0 <= idx < len(self):
            raise IndexError(idx)
        future = self.pending.pop(idx, None) or self.pool.submit(self.load, idx)
        if self.upcoming is not None:
            wanted = list(self.upcoming()[:AHEAD])
            for other in [i for i in self.pending if i not in wanted]:
                self.pending.pop(other).cancel()
            for other in wanted:
                if other not in self.pending:
                    self.pending[other] = self.pool.submit(self.load, other)
        return future.result()


@dataclass
class DiskImageDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: DiskImageDatamanager)


class DiskImageDatamanager(FullImageDatamanager):
    config: DiskImageDatamanagerConfig

    def _load_images(self, split, cache_images_device):
        dataset = self.train_dataset if split == "train" else self.eval_dataset
        original = deepcopy(dataset.cameras)
        # La correzione della distorsione ritaglia la foto e sposta il centro ottico: i parametri
        # corretti dipendono solo dalla camera, quindi si calcolano una volta per camera su una foto vuota.
        corrected: Dict[bytes, tuple] = {}
        for idx in range(len(dataset)):
            camera = original[idx].reshape(())
            if not _distorted(camera):
                continue
            K = camera.get_intrinsics_matrices().numpy()
            distortion = camera.distortion_params.numpy()
            width, height = int(camera.width.item()), int(camera.height.item())
            key = K.tobytes() + distortion.tobytes() + bytes(f"{width}x{height}", "ascii")
            if key not in corrected:
                blank = np.zeros((height, width, 3), dtype=np.uint8)
                new_K, image, _ = _undistort_image(camera, distortion, {}, blank, K.copy())
                corrected[key] = (new_K, image.shape[1], image.shape[0])
            new_K, new_width, new_height = corrected[key]
            dataset.cameras.fx[idx] = float(new_K[0, 0])
            dataset.cameras.fy[idx] = float(new_K[1, 1])
            dataset.cameras.cx[idx] = float(new_K[0, 2])
            dataset.cameras.cy[idx] = float(new_K[1, 2])
            dataset.cameras.width[idx] = new_width
            dataset.cameras.height[idx] = new_height
        self.train_cameras = self.train_dataset.cameras
        upcoming = (lambda: self.train_unseen_cameras) if split == "train" else None
        return DiskImages(dataset, original, self.config.cache_images_type, upcoming)

    def get_train_rays_per_batch(self) -> int:
        # nerfstudio lo chiede a ogni iterazione guardando la prima foto: qui vorrebbe dire rileggerla dal disco.
        camera = self.train_dataset.cameras[0]
        return int(camera.width.item()) * int(camera.height.item())


def _base(name: str):
    # Importato qui: nerfstudio carica questo modulo mentre sta ancora costruendo l'elenco dei metodi.
    from nerfstudio.configs.method_configs import method_configs

    config = deepcopy(method_configs[name])
    # Durante il training nerfstudio valuterebbe periodicamente le viste di test, caricandole tutte.
    config.steps_per_eval_batch = config.steps_per_eval_image = config.steps_per_eval_all_images = NEVER
    return config


def _splat(name: str):
    from nerfstudio.plugins.types import MethodSpecification

    config = _base(name)
    old = config.pipeline.datamanager
    config.pipeline.datamanager = DiskImageDatamanagerConfig(
        dataparser=old.dataparser, cache_images="cpu", cache_images_type=old.cache_images_type)
    return MethodSpecification(config=config, description=f"{name} con le foto lette dal disco")


def _nerf(name: str):
    from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
    from nerfstudio.plugins.types import MethodSpecification

    config = _base(name)
    old = config.pipeline.datamanager
    # Quante foto tenere in memoria e ogni quante iterazioni cambiarle lo decide la pipeline
    # (opzioni --pipeline.datamanager.train-num-images-to-sample-from e ...-times-to-repeat-images).
    config.pipeline.datamanager = VanillaDataManagerConfig(
        dataparser=old.dataparser, train_num_rays_per_batch=old.train_num_rays_per_batch,
        eval_num_rays_per_batch=old.eval_num_rays_per_batch,
        eval_num_images_to_sample_from=1, eval_num_times_to_repeat_images=-1)
    return MethodSpecification(config=config, description=f"{name} con le foto caricate a gruppi")


def splatfacto():
    return _splat("splatfacto")


def splatfacto_big():
    return _splat("splatfacto-big")


def nerfacto():
    return _nerf("nerfacto")


def nerfacto_big():
    return _nerf("nerfacto-big")
