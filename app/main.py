"""Interfaccia grafica: si sceglie la cartella delle foto e la pipeline gira da sola."""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from . import pipeline
from .pipeline import GROUPS, ROOT, Settings, Step

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
VIEWER_URL = "http://localhost:7007"
SUPERSPLAT_URL = "https://superspl.at/editor"


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def process_environment() -> QProcessEnvironment:
    env = QProcessEnvironment()
    for key, value in pipeline.environment().items():
        env.insert(key, value)
    return env


class Runner(QObject):
    """Esegue i passi uno dopo l'altro e ne inoltra log e avanzamento."""

    line = Signal(str)
    step_started = Signal(int, int, str)  # indice, totale, etichetta
    progress = Signal(float)
    detail = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.proc: Optional[QProcess] = None
        self.steps: List[Step] = []
        self.index = -1
        self.buffer = b""
        self.stopped = False
        self.log = None

    def running(self) -> bool:
        return self.proc is not None

    def start(self, steps: List[Step], log_path: Path) -> None:
        self.steps, self.index, self.stopped = steps, -1, False
        self.log = open(log_path, "a", encoding="utf-8")
        self._next()

    def stop(self) -> None:
        if self.proc is not None:
            self.stopped = True
            kill_tree(self.proc.processId())

    def _finish(self, ok: bool, message: str) -> None:
        self.proc = None
        if self.log:
            self.log.close()
            self.log = None
        self.finished.emit(ok, message)

    def _next(self) -> None:
        self.index += 1
        if self.index >= len(self.steps):
            self._finish(True, "")
            return
        step = self.steps[self.index]
        try:
            if step.before:
                step.before()
            command = step.command()
        except Exception as exc:  # errori di preparazione: vanno mostrati all'utente, non sollevati
            self._finish(False, f"{step.label}: {exc}")
            return
        self.step_started.emit(self.index, len(self.steps), step.label)
        self._write(f"\n=== {step.label} ===\n> {' '.join(command)}")
        self.buffer = b""
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.setProcessEnvironment(process_environment())
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._done)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(command[0], command[1:])

    def _write(self, text: str) -> None:
        if self.log:
            self.log.write(text + "\n")
            self.log.flush()
        self.line.emit(text)

    def _read(self) -> None:
        self.buffer += bytes(self.proc.readAllStandardOutput())
        parts = re.split(rb"[\r\n]+", self.buffer)
        self.buffer = parts.pop()
        for raw in parts:
            self._handle(raw)

    def _handle(self, raw: bytes) -> None:
        text = ANSI.sub("", raw.decode("utf-8", "replace")).rstrip()
        if not text:
            return
        self._write(text)
        step = self.steps[self.index]
        value = step.progress(text) if step.progress else None
        if value is not None:
            self.progress.emit(min(value, 1.0))
        detail = step.detail(text) if step.detail else None
        if detail:
            self.detail.emit(detail)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._finish(False, f"{self.steps[self.index].label}: impossibile avviare il programma.")

    def _done(self, code: int, status: QProcess.ExitStatus) -> None:
        self._read()
        self._handle(self.buffer)
        label = self.steps[self.index].label
        if self.stopped:
            self._finish(False, "Interrotto.")
        elif code != 0 or status != QProcess.ExitStatus.NormalExit:
            self._finish(False, f"{label} non riuscito (codice {code}). I dettagli sono nel log.")
        else:
            self._next()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Foto → Gaussian Splat")
        self.resize(980, 820)
        self.runner = Runner(self)
        self.viewer: Optional[QProcess] = None
        self.started_at = 0.0
        self.clock = QTimer(self)
        self.clock.setInterval(1000)
        self.clock.timeout.connect(self._tick)

        # --- progetto e foto
        self.name = QComboBox()
        self.name.setEditable(True)
        data = ROOT / "data"
        if data.is_dir():
            self.name.addItems(sorted(d.name for d in data.iterdir() if d.is_dir()))
        self.name.setCurrentText("")
        self.photos = QLineEdit()
        browse = QPushButton("Sfoglia…")
        browse.clicked.connect(self._browse)
        photos_row = QHBoxLayout()
        photos_row.addWidget(self.photos)
        photos_row.addWidget(browse)
        self.photo_count = QLabel("")

        # --- opzioni
        self.camera = self._combo([("Una sola camera per tutte le foto", "single"),
                                   ("Una camera per sottocartella", "per_folder"),
                                   ("Una camera per ogni foto (zoom o camere diverse)", "per_image")])
        self.matcher = self._combo([("Foto sparse: confronta tutte le coppie", "exhaustive"),
                                    ("Sequenza ordinata (fotogrammi di un video)", "sequential")])
        self.method = self._combo([("Standard (splatfacto)", "splatfacto"),
                                   ("Alta qualità, più lento (splatfacto-big)", "splatfacto-big")])
        self.iterations = QSpinBox()
        self.iterations.setRange(500, 200000)
        self.iterations.setSingleStep(1000)
        self.downscale = self._combo([("Automatica (lato massimo 1600 px)", 0), ("Metà risoluzione", 2),
                                      ("Risoluzione piena", 1)])
        self.mesh_size = self._combo([("1000 px (veloce)", 1000), ("1600 px", 1600), ("2400 px (molto lenta)", 2400)])

        project_box = QGroupBox("Progetto")
        form = QFormLayout(project_box)
        form.addRow("Nome del progetto", self.name)
        form.addRow("Cartella delle foto", photos_row)
        form.addRow("", self.photo_count)

        options_box = QGroupBox("Opzioni")
        form = QFormLayout(options_box)
        form.addRow("Camere", self.camera)
        form.addRow("Matching", self.matcher)
        form.addRow("Metodo di training", self.method)
        form.addRow("Iterazioni", self.iterations)
        form.addRow("Risoluzione del training", self.downscale)
        form.addRow("Risoluzione della mesh", self.mesh_size)

        # --- passi
        steps_box = QGroupBox("Passi da eseguire")
        steps_layout = QVBoxLayout(steps_box)
        self.checks = {}
        for key, label in GROUPS.items():
            self.checks[key] = QCheckBox(label)
            steps_layout.addWidget(self.checks[key])

        # --- esecuzione
        self.start_button = QPushButton("Avvia")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Interrompi")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        run_row = QHBoxLayout()
        run_row.addWidget(self.start_button)
        run_row.addWidget(self.stop_button)
        run_row.addStretch(1)
        self.gpu = QLabel(pipeline.gpu_status())
        run_row.addWidget(self.gpu)
        self.gpu_clock = QTimer(self)
        self.gpu_clock.setInterval(3000)
        self.gpu_clock.timeout.connect(lambda: self.gpu.setText(pipeline.gpu_status()))
        self.gpu_clock.start()
        self.step_text = ""
        self.step_percent = ""
        self.step_detail = ""
        self.status = QLabel("Scegli un progetto e la cartella delle foto.")
        self.elapsed = QLabel("")
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.elapsed)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(4000)
        self.log.setFont(QFont("Consolas", 9))

        # --- risultati
        self.train_viewer_button = QPushButton("Guarda il training in corso")
        self.train_viewer_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(VIEWER_URL)))
        self.train_viewer_button.setEnabled(False)
        self.model_button = QPushButton("Visualizza il modello")
        self.model_button.clicked.connect(self._view_model)
        self.supersplat_button = QPushButton("Apri in SuperSplat")
        self.supersplat_button.clicked.connect(self._open_supersplat)
        self.mesh_button = QPushButton("Apri la mesh")
        self.mesh_button.clicked.connect(self._open_mesh)
        folder_button = QPushButton("Apri la cartella del progetto")
        folder_button.clicked.connect(self._open_folder)
        results_row = QHBoxLayout()
        for button in (self.train_viewer_button, self.model_button, self.supersplat_button, self.mesh_button, folder_button):
            results_row.addWidget(button)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(project_box)
        left.addWidget(steps_box)
        top.addLayout(left, 1)
        top.addWidget(options_box, 1)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(run_row)
        layout.addLayout(status_row)
        layout.addWidget(self.bar)
        layout.addWidget(self.log, 1)
        layout.addLayout(results_row)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.runner.line.connect(self.log.appendPlainText)
        self.runner.step_started.connect(self._step_started)
        self.runner.progress.connect(self._progress)
        self.runner.detail.connect(self._detail)
        self.runner.finished.connect(self._finished)
        self.name.currentTextChanged.connect(self._project_changed)
        self.photos.textChanged.connect(self._photos_changed)
        self._apply(Settings())
        self._refresh()

    # --- campi <-> impostazioni
    @staticmethod
    def _combo(items) -> QComboBox:
        combo = QComboBox()
        for text, value in items:
            combo.addItem(text, value)
        return combo

    def _settings(self) -> Settings:
        return Settings(
            name=self.name.currentText().strip(), photos=self.photos.text().strip(),
            camera=self.camera.currentData(), matcher=self.matcher.currentData(),
            method=self.method.currentData(), iterations=self.iterations.value(),
            downscale=self.downscale.currentData(), mesh_size=self.mesh_size.currentData(),
        )

    def _apply(self, s: Settings) -> None:
        self.photos.setText(s.photos)
        for combo, value in ((self.camera, s.camera), (self.matcher, s.matcher), (self.method, s.method),
                             (self.downscale, s.downscale), (self.mesh_size, s.mesh_size)):
            combo.setCurrentIndex(max(combo.findData(value), 0))
        self.iterations.setValue(s.iterations)

    def _project_changed(self, name: str) -> None:
        saved = Settings.load(name.strip()) if name.strip() else None
        if saved:
            self._apply(saved)
        self._refresh()

    def _photos_changed(self, text: str) -> None:
        folder = Path(text.strip())
        if text.strip() and folder.is_dir() and not self.name.currentText().strip():
            self.name.setCurrentText(re.sub(r"[^A-Za-z0-9_-]+", "_", folder.name))
        count = pipeline.count_images(text.strip())
        self.photo_count.setText(f"{count} immagini trovate" if text.strip() else "")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Cartella delle foto", self.photos.text() or str(ROOT))
        if folder:
            self.photos.setText(folder)

    def _refresh(self) -> None:
        """Aggiorna spunte e pulsanti in base a cio' che esiste gia' su disco."""
        s = self._settings()
        state = pipeline.done(s) if s.name else dict.fromkeys(GROUPS, False)
        for key, check in self.checks.items():
            note = "  — già fatto" if state[key] else ""
            if key == "train" and state[key]:
                note += f" ({pipeline.train_summary(s.name)})"
            check.setText(GROUPS[key] + note)
            check.setChecked(not state[key] and key != "mesh")
        idle = not self.runner.running()
        if idle and s.name and self.viewer is None:
            self.status.setText("Nessuna elaborazione in corso.")
        self.model_button.setEnabled(idle and state["train"])
        self.supersplat_button.setEnabled(state["export"])
        self.mesh_button.setEnabled(state["mesh"])

    # --- esecuzione
    def _start(self) -> None:
        s = self._settings()
        groups = [key for key, check in self.checks.items() if check.isChecked()]
        state = pipeline.done(s) if s.name else {}
        error = None
        if not re.fullmatch(r"[A-Za-z0-9_-]+", s.name):
            error = "Il nome del progetto può contenere solo lettere, numeri, '-' e '_'."
        elif not groups:
            error = "Seleziona almeno un passo."
        elif ("sfm" in groups or "mesh" in groups) and pipeline.count_images(s.photos) < 3:
            error = "Scegli una cartella che contenga le foto."
        elif ("train" in groups or "mesh" in groups) and "sfm" not in groups and not state["sfm"]:
            error = "Serve prima l'allineamento delle foto."
        elif "export" in groups and "train" not in groups and not state["train"]:
            error = "Serve prima il training."
        if error:
            QMessageBox.warning(self, "Impossibile avviare", error)
            return
        if "sfm" in groups and (s.scene / "colmap").exists():
            answer = QMessageBox.question(
                self, "Allineamento esistente",
                f"Il progetto '{s.name}' ha già un allineamento: verrà cancellato e rifatto, insieme alla mesh. Continuare?")
            if answer != QMessageBox.StandardButton.Yes:
                return
        if self.viewer is not None:
            self._close_viewer()
        s.save()
        self.log.clear()
        self._set_running(True)
        self.train_viewer_button.setEnabled(False)
        self.started_at = time.time()
        self.clock.start()
        self.runner.start(pipeline.build_steps(s, groups), s.scene / "pipeline.log")

    def _stop(self) -> None:
        self.status.setText("Interruzione in corso…")
        self.runner.stop()

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        for widget in (self.name, self.photos, self.camera, self.matcher, self.method, self.iterations,
                       self.downscale, self.mesh_size, *self.checks.values()):
            widget.setEnabled(not running)

    def _step_started(self, index: int, total: int, label: str) -> None:
        self.step_text, self.step_percent, self.step_detail = f"Passo {index + 1} di {total}: {label}", "", ""
        self._show_step()
        self.bar.setRange(0, 0)  # indeterminato finche' il passo non comunica un avanzamento
        self.train_viewer_button.setEnabled(label == "Training")

    def _show_step(self) -> None:
        self.status.setText(" — ".join(t for t in (self.step_text, self.step_percent, self.step_detail) if t))

    def _progress(self, value: float) -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(int(value * 1000))
        self.step_percent = f"{value * 100:.0f}%"
        self._show_step()

    def _detail(self, text: str) -> None:
        self.step_detail = text
        self._show_step()

    def _tick(self) -> None:
        seconds = int(time.time() - self.started_at)
        self.elapsed.setText(f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}")

    def _finished(self, ok: bool, message: str) -> None:
        self.clock.stop()
        self._set_running(False)
        self.train_viewer_button.setEnabled(False)
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if ok else 0)
        self._refresh()
        self.status.setText("Completato." if ok else message)
        if not ok and message != "Interrotto.":
            QMessageBox.critical(self, "Errore", message)

    # --- risultati
    def _view_model(self) -> None:
        config = pipeline.latest_config(self._settings().name)
        if config is None:
            return
        self._close_viewer()
        self.viewer = QProcess(self)
        self.viewer.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.viewer.setProcessEnvironment(process_environment())
        opened = [False]

        def read() -> None:
            text = bytes(self.viewer.readAllStandardOutput()).decode("utf-8", "replace")
            if not opened[0] and "http" in text:
                opened[0] = True
                QDesktopServices.openUrl(QUrl(VIEWER_URL))
                self.status.setText(f"Nessuna elaborazione in corso. Viewer del modello già allenato attivo su {VIEWER_URL}")

        self.viewer.readyReadStandardOutput.connect(read)
        self.viewer.start(str(pipeline.VENV_SCRIPTS / "ns-viewer.exe"), ["--load-config", str(config)])
        self.status.setText("Avvio del viewer…")

    def _close_viewer(self) -> None:
        if self.viewer is not None and self.viewer.state() != QProcess.ProcessState.NotRunning:
            kill_tree(self.viewer.processId())
            self.viewer.waitForFinished(3000)
        self.viewer = None

    def _open_supersplat(self) -> None:
        # SuperSplat gira nel browser: il file va trascinato nella pagina, quindi si apre anche la cartella.
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._settings().export_dir)))
        QDesktopServices.openUrl(QUrl(SUPERSPLAT_URL))

    def _open_mesh(self) -> None:
        mesh = self._settings().scene / "colmap" / "dense" / "mesh-poisson.ply"
        meshlab = pipeline.TOOLS / "meshlab" / "meshlab.exe"
        if meshlab.exists():
            QProcess.startDetached(str(meshlab), [str(mesh)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(mesh)))

    def _open_folder(self) -> None:
        scene = self._settings().scene
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(scene if scene.exists() else ROOT)))

    def closeEvent(self, event) -> None:
        if self.runner.running():
            answer = QMessageBox.question(self, "Elaborazione in corso", "Interrompere l'elaborazione e chiudere?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        self._close_viewer()
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
