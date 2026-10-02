"""Avvio dell'interfaccia grafica:  python -m app"""
from __future__ import annotations

import sys
import time
import traceback

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from . import config
from .window import MainWindow

ERROR_LOG = config.WORK / "logs" / "app-errors.log"


def report_exception(kind, value, trace) -> None:
    """Con pythonw non c'e' console: gli errori imprevisti vanno su file e in una finestra."""
    text = "".join(traceback.format_exception(kind, value, trace))
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n{text}")
    except OSError:
        pass
    QMessageBox.critical(None, "Errore imprevisto", f"{value}\n\nDettagli salvati in {ERROR_LOG}")


def main() -> int:
    if sys.platform == "win32":  # nella barra delle applicazioni: icona e gruppo propri, non quelli di Python
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LaboratorioICTBC.StudioGaussianSplatting")
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(str(config.ROOT / "assets" / "icona.ico")))
    sys.excepthook = report_exception
    missing = config.missing_components()
    if missing:
        QMessageBox.critical(None, "Componenti mancanti",
                             "Non è possibile eseguire la pipeline perché mancano:\n\n• " + "\n• ".join(missing)
                             + "\n\nPer completare l'installazione: «Completa o ripara l'installazione» nel menu "
                             "Start, oppure Installa.bat nella cartella del programma.")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
