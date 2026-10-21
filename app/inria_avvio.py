"""Avvio di uno script del codice Inria con opzioni che il codice non prevede, senza modificarlo.

  python <questo file> [--disco] [--seme N] train.py <argomenti di train.py>

--disco  le foto si leggono dal disco quando servono (vedi inria_lazy.py);
--seme   seme dei numeri casuali. Il codice Inria lo fissa a 0 in utils.general_utils.safe_state,
         chiamata all'inizio di train.py: qui safe_state viene avvolta in modo da reimpostare
         subito dopo random, numpy e torch con il seme dato. Restano le differenze dovute al
         non-determinismo della scheda grafica (somme in ordine variabile nei kernel CUDA).

Va eseguito dalla cartella del codice Inria, come train.py.
"""
from __future__ import annotations

import os
import runpy
import sys


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def install_seed(seed: int) -> None:
    import utils.general_utils as general_utils

    original = general_utils.safe_state

    def safe_state(*args, **kwargs):
        result = original(*args, **kwargs)
        set_seed(seed)
        print(f"Seme dei numeri casuali: {seed}", flush=True)
        return result

    general_utils.safe_state = safe_state  # train.py la importa da qui quando viene eseguito


def main(argv=None) -> None:
    # Le opzioni di questo file vengono prima del nome dello script; tutto il resto va allo script.
    args = list(sys.argv[1:] if argv is None else argv)
    disco, seme = False, None
    while args and args[0] in ("--disco", "--seme"):
        flag = args.pop(0)
        if flag == "--disco":
            disco = True
        else:
            seme = int(args.pop(0))
    script, *arguments = args
    sys.argv = [script, *arguments]
    sys.path.insert(0, os.getcwd())  # la cartella del codice Inria, come se si eseguisse direttamente lo script
    if disco:
        from inria_lazy import install  # stessa cartella di questo file, che e' in sys.path come script
        install()
        print("Foto lette dal disco a ogni iterazione (non entrano in memoria tutte insieme).", flush=True)
    if seme is not None:
        install_seed(seme)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
