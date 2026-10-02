"""Scrive le note di una release: come si installa, piu' la sezione del registro delle modifiche.

Uso (lo esegue il workflow di release):  python scripts/note_release.py 0.3.0 note.md
"""
import re
import sys
from pathlib import Path

INSTALLAZIONE = """## Installazione

Scaricare **GaussianSplatting-Setup-{version}.exe** qui sotto e avviarlo: la procedura guidata copia il
programma, scarica e configura i componenti necessari (circa 6 GB, da 10 a 40 minuti) e crea le icone.
Non servono diritti di amministratore.

Requisiti: Windows 10/11 a 64 bit, scheda NVIDIA RTX con driver 551.61 o successivi, 25 GB liberi.

Il file non è firmato: Windows può mostrare l'avviso «PC protetto da Windows». In quel caso:
«Ulteriori informazioni», poi «Esegui comunque».

Il Setup contiene due moduli compilati del codice 3D Gaussian Splatting di Inria, distribuiti con la
licenza Inria (solo ricerca e valutazione non commerciali).
"""


def changes(version: str) -> str:
    text = (Path(__file__).resolve().parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(version)}\b.*?\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    if not match:
        raise SystemExit(f"CHANGELOG.md non ha una sezione per la versione {version}.")
    return match.group(1).strip()


if __name__ == "__main__":
    version, target = sys.argv[1], Path(sys.argv[2])
    target.write_text(INSTALLAZIONE.format(version=version) + "\n## Modifiche\n\n"
                      + changes(version) + "\n", encoding="utf-8")
    print(target)
