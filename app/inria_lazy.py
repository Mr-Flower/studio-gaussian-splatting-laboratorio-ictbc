"""Fa leggere al codice Inria le foto dal disco quando servono, invece di tenerle tutte in memoria.

Il codice Inria carica ogni foto, in virgola mobile, sulla scheda grafica (o in memoria centrale)
prima di iniziare: 887 foto da 5280 x 3956 pixel occuperebbero quasi 300 GB. Qui si sostituisce
solo il modo in cui la foto viene conservata: i valori dei pixel sono prodotti dalla stessa funzione
del codice Inria, e modello, ottimizzazione e rendering non sono toccati. Il costo e' il tempo di
lettura di una foto a ogni iterazione.

Per il training (dalla cartella del codice Inria):
  python <questo file> train.py -s <dati> -m <run> ...
Per la valutazione: install(), dopo aver aggiunto la cartella del codice Inria a sys.path.
"""
from __future__ import annotations

import inspect
import os
import runpy
import sys


def install() -> None:
    import numpy as np
    import torch
    from PIL import Image

    import scene.cameras as cameras
    from utils.general_utils import PILtoTorch

    Camera = cameras.Camera
    original_init = Camera.__init__
    signature = inspect.signature(original_init)

    def read(self):
        photo = self.__dict__.get("_photo")
        if photo is None:
            return self.__dict__["_image"]
        path, resolution = photo
        with Image.open(path) as image:
            if image.size == resolution and image.mode == "RGB":
                # Stesse operazioni della funzione originale (PILtoTorch), ma con la conversione in
                # virgola mobile fatta sulla scheda grafica: stessi valori, meta' del tempo.
                pixels = (torch.from_numpy(np.array(image)).cuda() / 255.0).permute(2, 0, 1)
            else:
                pixels = PILtoTorch(image, resolution)
        return pixels[:3, ...].clamp(0.0, 1.0)

    def store(self, value) -> None:
        self.__dict__["_image"] = value

    def lazy_init(self, *args, **kwargs) -> None:
        arguments = signature.bind(self, *args, **kwargs).arguments
        image, resolution = arguments["image"], tuple(arguments["resolution"])
        if arguments.get("train_test_exp"):
            raise SystemExit("La lettura delle foto dal disco non supporta l'opzione train_test_exp.")
        if len(image.getbands()) > 3:
            raise SystemExit("La lettura delle foto dal disco non supporta foto con canale alfa.")
        original_init(self, *args, **kwargs)  # con la funzione di caricamento sostituita qui sotto
        self.__dict__["_photo"] = (image.filename, resolution)
        self.__dict__.pop("_image", None)
        self.alpha_mask = None  # era una maschera tutta a uno: il training la salta se manca
        self.image_width, self.image_height = resolution
        image.close()

    # Durante la costruzione della camera la foto non viene letta: basta un segnaposto.
    cameras.PILtoTorch = lambda image, resolution: torch.zeros((3, 1, 1))
    Camera.__init__ = lazy_init
    Camera.original_image = property(read, store)


def main() -> None:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.getcwd())  # la cartella del codice Inria, come se si eseguisse direttamente lo script
    install()
    print("Foto lette dal disco a ogni iterazione (non entrano in memoria tutte insieme).", flush=True)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
