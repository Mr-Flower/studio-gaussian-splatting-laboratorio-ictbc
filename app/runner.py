"""Esecuzione dei passi, uno dopo l'altro, come processi esterni (solo QtCore: serve anche alla CLI)."""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from . import config
from .pipeline import Step

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STOPPED = "Interrotto."


def kill_tree(pid: int) -> None:
    """Termina il processo e i suoi figli (ns-train avvia un interprete Python figlio)."""
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def process_environment() -> QProcessEnvironment:
    env = QProcessEnvironment()
    for key, value in config.environment().items():
        env.insert(key, value)
    return env


def clean(raw: bytes) -> str:
    return ANSI.sub("", raw.decode("utf-8", "replace")).rstrip()


class Runner(QObject):
    line = Signal(str)  # riga da mostrare nel log
    step_started = Signal(int, int, str)  # indice, totale, etichetta
    progress = Signal(float)  # 0..1 del passo corrente
    detail = Signal(str)
    finished = Signal(bool, str)  # esito, messaggio di errore

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self.steps: List[Step] = []
        self.index = -1
        self.buffer = b""
        self.stopped = False
        self.active = False
        self.step_start = 0.0
        self.log = None

    def running(self) -> bool:
        return self.active

    def start(self, steps: List[Step], log_path: Path) -> None:
        self.steps, self.index, self.stopped, self.active = steps, -1, False, True
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open(log_path, "a", encoding="utf-8")
        self._write(f"\n##### {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(steps)} passi #####", show=False)
        self._next()

    def stop(self) -> None:
        if self.active and self.proc is not None:
            self.stopped = True
            kill_tree(self.proc.processId())

    def _finish(self, ok: bool, message: str) -> None:
        self.active = False
        if self.proc is not None:
            self.proc.deleteLater()
            self.proc = None
        if self.log:
            self._write(f"##### {'completato' if ok else message} #####", show=False)
            self.log.close()
            self.log = None
        self.finished.emit(ok, message)

    def _next(self) -> None:
        if self.proc is not None:
            self.proc.deleteLater()
            self.proc = None
        self.index += 1
        if self.index >= len(self.steps):
            self._finish(True, "")
            return
        step = self.steps[self.index]
        try:
            if step.before:
                step.before()
            command = step.command()
        except Exception as exc:  # errori di preparazione: vanno riportati all'utente, non sollevati
            self._finish(False, f"{step.label}: {exc}")
            return
        self.step_started.emit(self.index, len(self.steps), step.label)
        self.buffer = b""
        self.step_start = time.monotonic()
        if command is None:  # passo senza processo esterno: il lavoro e' gia' stato fatto in `before`
            self._write(f"\n=== {step.label} ===")
            QTimer.singleShot(0, self._completed)
            return
        self._write(f"\n=== {step.label} ===\n> {subprocess.list2cmdline(command)}")
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(step.cwd or config.ROOT))
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.setProcessEnvironment(process_environment())
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(command[0], command[1:])

    def _write(self, text: str, show: bool = True) -> None:
        if self.log:
            self.log.write(text + "\n")
            self.log.flush()
        if show:
            self.line.emit(text)

    def _read(self) -> None:
        if self.proc is None or not self.active:
            return
        self.buffer += bytes(self.proc.readAllStandardOutput())
        parts = re.split(rb"[\r\n]+", self.buffer)
        self.buffer = parts.pop()
        for raw in parts:
            self._handle(raw)

    def _handle(self, raw: bytes) -> None:
        text = clean(raw)
        if not text:
            return
        step = self.steps[self.index]
        self._write(text, show=not (step.quiet and step.quiet(text)))
        if step.on_line:
            step.on_line(text)
        value = step.progress(text) if step.progress else None
        if value is not None:
            self.progress.emit(min(value, 1.0))
        detail = step.detail(text) if step.detail else None
        if detail:
            self.detail.emit(detail)

    def _error(self, error: QProcess.ProcessError) -> None:
        # Solo il mancato avvio non genera 'finished'; gli altri errori arrivano a _done.
        if error == QProcess.ProcessError.FailedToStart and self.active:
            self._finish(False, f"{self.steps[self.index].label}: impossibile avviare il programma.")

    def _done(self, code: int, status: QProcess.ExitStatus) -> None:
        if not self.active:
            return
        self._read()
        self._handle(self.buffer)
        self.buffer = b""
        step = self.steps[self.index]
        if self.stopped:
            self._finish(False, STOPPED)
            return
        if code != 0 or status != QProcess.ExitStatus.NormalExit:
            self._finish(False, f"{step.label} non riuscito (codice {code}). I dettagli sono nel log.")
            return
        self._completed()

    def _completed(self) -> None:
        if not self.active:
            return
        step = self.steps[self.index]
        seconds = time.monotonic() - self.step_start
        self._write(f"--- {step.label}: {seconds:.0f} s", show=False)
        try:
            if step.after:
                step.after(seconds)
        except Exception as exc:
            self._finish(False, f"{step.label}: {exc}")
            return
        self._next()
