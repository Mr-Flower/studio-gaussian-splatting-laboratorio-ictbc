"""Finestra principale: scelta del progetto, esecuzione della pipeline e tabella di confronto dei run."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QProcess, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from . import __version__, config, pipeline, progress, runs
from .config import GROUPS, METHODS, ROOT, Settings
from .runner import STOPPED, Runner, clean, kill_tree, process_environment
from .runs import Run

SUPERSPLAT_URL = "https://superspl.at/editor"
GPU_QUERY = ["--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]


def open_path(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Studio Gaussian Splatting — Laboratorio ICTBC  (v{__version__})")
        self.resize(1100, 860)
        self.runner = Runner(self)
        self.viewer: Optional[QProcess] = None
        self.train_viewer_url = ""
        self.locked_scene: Optional[Path] = None
        self.table_runs: List[Run] = []
        self.step_text = self.step_percent = self.step_detail = ""
        self.started_at = 0.0

        self._build_project_box()
        self._build_run_tab()
        self._build_compare_tab()
        tabs = QTabWidget()
        tabs.addTab(self.run_tab, "Elaborazione")
        tabs.addTab(self.compare_tab, "Confronto dei metodi")
        layout = QVBoxLayout()
        layout.addWidget(self.project_box)
        layout.addWidget(tabs, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.clock = QTimer(self, interval=1000)
        self.clock.timeout.connect(self._tick)
        self.count_timer = QTimer(self, interval=400, singleShot=True)  # il conteggio scorre tutta la cartella
        self.count_timer.timeout.connect(self._count_photos)
        self.gpu_proc: Optional[QProcess] = None
        self.gpu_timer = QTimer(self, interval=3000)
        self.gpu_timer.timeout.connect(self._poll_gpu)
        self.gpu_timer.start()
        self._poll_gpu()

        self.runner.line.connect(self._line)
        self.runner.step_started.connect(self._step_started)
        self.runner.progress.connect(self._progress)
        self.runner.detail.connect(self._detail)
        self.runner.finished.connect(self._finished)
        self.name_timer = QTimer(self, interval=400, singleShot=True)  # non a ogni carattere digitato
        self.name_timer.timeout.connect(lambda: self._project_changed(self.name.currentText()))
        self.name.currentTextChanged.connect(lambda _: self.name_timer.start())
        self.photos.textChanged.connect(lambda _: self.count_timer.start())
        self._apply(Settings())
        self._refresh(reset_checks=True)

    # --- costruzione dell'interfaccia
    def _build_project_box(self) -> None:
        self.name = QComboBox()
        self.name.setEditable(True)
        self.name.addItems(config.projects())
        self.name.setCurrentText("")
        self.name.setToolTip("Un progetto raccoglie un set di foto, il suo allineamento e tutti i training fatti su di esso.")
        self.photos = QLineEdit()
        browse = QPushButton("Sfoglia…")
        browse.clicked.connect(self._browse)
        photos_row = QHBoxLayout()
        photos_row.addWidget(self.photos)
        photos_row.addWidget(browse)
        self.photo_count = QLabel("")
        self.alignment = QLabel("")
        self.project_box = QGroupBox("Progetto")
        form = QFormLayout(self.project_box)
        form.addRow("Nome del progetto", self.name)
        form.addRow("Cartella delle foto", photos_row)
        form.addRow("", self.photo_count)
        form.addRow("Allineamento", self.alignment)

    def _build_run_tab(self) -> None:
        methods_box = QGroupBox("Metodi da allenare e confrontare")
        methods_layout = QVBoxLayout(methods_box)
        self.method_checks: Dict[str, QCheckBox] = {}
        for key, method in METHODS.items():
            check = QCheckBox(method.label)
            check.setToolTip(method.note)
            self.method_checks[key] = check
            methods_layout.addWidget(check)

        steps_box = QGroupBox("Passi da eseguire")
        steps_layout = QVBoxLayout(steps_box)
        self.checks: Dict[str, QCheckBox] = {}
        for key, label in GROUPS.items():
            self.checks[key] = QCheckBox(label)
            steps_layout.addWidget(self.checks[key])
        self.checks["sfm"].setToolTip("Calcola posizione e parametri delle camere. È comune a tutti i metodi: "
                                      "rifarlo rende non più confrontabili i training già fatti.")
        self.checks["train"].setToolTip("Ogni avvio crea un nuovo run: i precedenti non vengono sovrascritti.")
        self.checks["eval"].setToolTip("Misura la qualità sulle foto tenute fuori dal training (il 10% del set).")
        self.checks["mesh"].setToolTip("Multi-view stereo e superficie di Poisson. Con centinaia di foto richiede molte ore.")

        self.camera = self._combo([("Una sola camera per tutte le foto", "single"),
                                   ("Una camera per sottocartella", "per_folder"),
                                   ("Una camera per ogni foto (zoom o camere diverse)", "per_image")])
        self.matcher = self._combo([("Foto sparse: confronta tutte le coppie", "exhaustive"),
                                    ("Sequenza ordinata (fotogrammi di un video)", "sequential")])
        self.iterations = QSpinBox()
        self.iterations.setRange(100, 500000)
        self.iterations.setSingleStep(1000)
        self.iterations.setToolTip("Uguale per tutti i metodi selezionati.")
        self.downscale = self._combo([("Automatica (lato massimo 1600 px)", 0), ("Metà risoluzione", 2),
                                      ("Risoluzione piena", 1)])
        self.mesh_size = self._combo([("1000 px (veloce)", 1000), ("1600 px", 1600), ("2400 px (molto lenta)", 2400)])
        options_box = QGroupBox("Opzioni")
        form = QFormLayout(options_box)
        form.addRow("Camere", self.camera)
        form.addRow("Matching", self.matcher)
        form.addRow("Iterazioni di training", self.iterations)
        form.addRow("Risoluzione del training", self.downscale)
        form.addRow("Risoluzione della mesh", self.mesh_size)

        self.start_button = QPushButton("Avvia")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Interrompi")
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        self.gpu = QLabel("")
        run_row = QHBoxLayout()
        run_row.addWidget(self.start_button)
        run_row.addWidget(self.stop_button)
        run_row.addStretch(1)
        run_row.addWidget(self.gpu)
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
        self.train_viewer_button = QPushButton("Guarda il training in corso")
        self.train_viewer_button.setToolTip("Disponibile solo mentre un training è in esecuzione.")
        self.train_viewer_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.train_viewer_url)))
        self.train_viewer_button.setEnabled(False)
        folder_button = QPushButton("Apri la cartella del progetto")
        folder_button.clicked.connect(self._open_folder)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.train_viewer_button)
        bottom_row.addWidget(folder_button)
        bottom_row.addStretch(1)

        left = QVBoxLayout()
        left.addWidget(methods_box)
        left.addWidget(steps_box)
        top = QHBoxLayout()
        top.addLayout(left, 1)
        top.addWidget(options_box, 1)
        layout = QVBoxLayout()
        layout.addLayout(top)
        layout.addLayout(run_row)
        layout.addLayout(status_row)
        layout.addWidget(self.bar)
        layout.addWidget(self.log, 1)
        layout.addLayout(bottom_row)
        self.run_tab = QWidget()
        self.run_tab.setLayout(layout)
        self.inputs = [self.project_box, methods_box, steps_box, options_box]  # bloccati durante l'elaborazione

    def _build_compare_tab(self) -> None:
        self.table = QTableWidget(0, len(runs.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label in runs.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._update_result_buttons)
        self.compare_note = QLabel(
            "Le metriche sono calcolate sulle foto escluse dal training. Sono confrontabili solo i run con "
            "allineamento «corrente» e stessa risoluzione.")
        self.compare_note.setWordWrap(True)
        self.mesh_label = QLabel("")
        self.model_button = QPushButton("Visualizza nel viewer")
        self.model_button.clicked.connect(self._view_model)
        self.supersplat_button = QPushButton("Apri in SuperSplat")
        self.supersplat_button.clicked.connect(self._open_supersplat)
        self.export_button = QPushButton("Apri la cartella del modello esportato")
        self.export_button.clicked.connect(lambda: self._selected() and open_path(self._selected().export_dir))
        self.mesh_button = QPushButton("Apri la mesh")
        self.mesh_button.clicked.connect(self._open_mesh)
        self.csv_button = QPushButton("Esporta la tabella in CSV…")
        self.csv_button.clicked.connect(self._export_csv)
        buttons = QHBoxLayout()
        for button in (self.model_button, self.supersplat_button, self.export_button, self.mesh_button, self.csv_button):
            buttons.addWidget(button)
        layout = QVBoxLayout()
        layout.addWidget(self.compare_note)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.mesh_label)
        layout.addLayout(buttons)
        self.compare_tab = QWidget()
        self.compare_tab.setLayout(layout)

    @staticmethod
    def _combo(items) -> QComboBox:
        combo = QComboBox()
        for text, value in items:
            combo.addItem(text, value)
        return combo

    # --- campi <-> impostazioni
    def _settings(self) -> Settings:
        return Settings(
            name=self.name.currentText().strip(), photos=self.photos.text().strip(),
            camera=self.camera.currentData(), matcher=self.matcher.currentData(),
            methods=[key for key, check in self.method_checks.items() if check.isChecked()],
            iterations=self.iterations.value(), downscale=self.downscale.currentData(),
            mesh_size=self.mesh_size.currentData(),
        )

    def _apply(self, s: Settings) -> None:
        self.photos.setText(s.photos)
        for combo, value in ((self.camera, s.camera), (self.matcher, s.matcher),
                             (self.downscale, s.downscale), (self.mesh_size, s.mesh_size)):
            combo.setCurrentIndex(max(combo.findData(value), 0))
        self.iterations.setValue(s.iterations)
        for key, check in self.method_checks.items():
            check.setChecked(key in s.methods)

    def _project_changed(self, name: str) -> None:
        saved = Settings.load(name.strip()) if config.PROJECT_NAME.fullmatch(name.strip()) else None
        if saved:
            self._apply(saved)
        self._refresh(reset_checks=True)

    def _count_photos(self) -> None:
        text = self.photos.text().strip()
        folder = Path(text)
        if text and folder.is_dir() and not self.name.currentText().strip():
            self.name.setCurrentText(re.sub(r"[^A-Za-z0-9_-]+", "_", folder.name))
        self.photo_count.setText(f"{config.count_images(text)} immagini trovate" if text else "")

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Cartella delle foto", self.photos.text() or str(ROOT))
        if folder:
            self.photos.setText(folder)

    def _refresh(self, reset_checks: bool = False) -> None:
        """Aggiorna etichette, tabella e pulsanti in base a cio' che esiste su disco."""
        s = self._settings()
        valid = bool(config.PROJECT_NAME.fullmatch(s.name))
        state = pipeline.state(s) if valid else {"sfm": False, "mesh": False}
        self.checks["sfm"].setText(GROUPS["sfm"] + ("  — già fatto" if state["sfm"] else ""))
        self.checks["mesh"].setText(GROUPS["mesh"] + ("  — già fatta" if state["mesh"] else ""))
        if reset_checks:
            for key, check in self.checks.items():
                check.setChecked(key in ("train", "eval", "export") or (key == "sfm" and not state["sfm"]))
        self.alignment.setText((runs.alignment_summary(s.scene) or ("presente" if state["sfm"] else "non ancora fatto"))
                               if valid else "")
        self._fill_table(s.name if valid else "")
        self.mesh_label.setText(("Fotogrammetria classica: " + runs.mesh_summary(s.scene)) if state["mesh"] else "")
        self.mesh_button.setEnabled(state["mesh"])
        if valid and not self.runner.running() and self.viewer is None:
            self.status.setText("Nessuna elaborazione in corso.")

    def _fill_table(self, project: str) -> None:
        selected = self._selected()
        self.table_runs = runs.list_runs(project) if project else []
        self.table.setRowCount(len(self.table_runs))
        for r, run in enumerate(self.table_runs):
            values = runs.row(run)
            for c, (key, _) in enumerate(runs.COLUMNS):
                item = QTableWidgetItem(str(values[key]))
                if isinstance(values[key], (int, float)):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)
        if selected in self.table_runs:
            self.table.selectRow(self.table_runs.index(selected))
        elif self.table_runs:
            self.table.selectRow(0)
        self.csv_button.setEnabled(bool(self.table_runs))
        self._update_result_buttons()

    def _selected(self) -> Optional[Run]:
        row = self.table.currentRow()
        return self.table_runs[row] if 0 <= row < len(self.table_runs) else None

    def _update_result_buttons(self) -> None:
        run = self._selected()
        exported = bool(run and run.export_file)
        # Il viewer carica i dati del progetto: non va aperto durante un'elaborazione ne' su allineamenti superati.
        self.model_button.setEnabled(bool(run) and not self.runner.running() and run.comparable() is not False)
        self.supersplat_button.setEnabled(exported and METHODS[run.method].family == "gaussian")
        self.export_button.setEnabled(exported)

    # --- esecuzione
    def _start(self) -> None:
        s = self._settings()
        groups = [key for key, check in self.checks.items() if check.isChecked()]
        error = pipeline.validate(s, groups)
        if error:
            QMessageBox.warning(self, "Impossibile avviare", error)
            return
        if "sfm" in groups and pipeline.state(s)["sfm"]:
            answer = QMessageBox.question(
                self, "Rifare l'allineamento?",
                f"Il progetto «{s.name}» ha già un allineamento.\n\nRifarlo richiede tempo e rende non più "
                "confrontabili (né visualizzabili) i training già fatti; anche la mesh esistente viene cancellata.\n\n"
                "Per allenare altri metodi sullo stesso allineamento basta togliere la spunta da «Allineamento».\n\n"
                "Rifare comunque l'allineamento?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        if "mesh" in groups and config.count_images(s.photos) > 150 and s.mesh_size > 1000:
            answer = QMessageBox.question(
                self, "Mesh molto lenta",
                f"Con {config.count_images(s.photos)} foto a {s.mesh_size} px la mesh può richiedere molte ore "
                "(anche più di un giorno). Continuare?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
        error = pipeline.acquire_lock(s.scene)
        if error:
            QMessageBox.warning(self, "Progetto in uso", error)
            return
        self.locked_scene = s.scene
        self._close_viewer()
        s.save()
        self.log.clear()
        self._set_running(True)
        self.started_at = time.time()
        self.clock.start()
        self.runner.start(pipeline.build_steps(s, groups), s.scene / "pipeline.log")
        self._update_result_buttons()

    def _stop(self) -> None:
        self.status.setText("Interruzione in corso…")
        self.runner.stop()

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        for widget in self.inputs:
            widget.setEnabled(not running)

    def _line(self, text: str) -> None:
        self.log.appendPlainText(text)
        url = progress.viewer_url(text)
        if url and self.runner.running():
            self.train_viewer_url = url
            self.train_viewer_button.setEnabled(True)

    def _step_started(self, index: int, total: int, label: str) -> None:
        self.step_text, self.step_percent, self.step_detail = f"Passo {index + 1} di {total}: {label}", "", ""
        self._show_step()
        self.bar.setRange(0, 0)  # indeterminato finche' il passo non comunica un avanzamento
        self.train_viewer_button.setEnabled(False)

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
        if self.locked_scene is not None:
            pipeline.release_lock(self.locked_scene)
            self.locked_scene = None
        self._set_running(False)
        self.train_viewer_button.setEnabled(False)
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000 if ok else 0)
        self._refresh()
        self.status.setText("Completato." if ok else message)
        if not ok and message != STOPPED:
            QMessageBox.critical(self, "Errore", message)

    # --- scheda grafica
    def _poll_gpu(self) -> None:
        if self.gpu_proc is not None:  # la lettura precedente non e' ancora terminata
            return
        self.gpu_proc = QProcess(self)
        self.gpu_proc.finished.connect(self._gpu_read)
        self.gpu_proc.errorOccurred.connect(self._gpu_failed)
        self.gpu_proc.start("nvidia-smi", GPU_QUERY)

    def _gpu_failed(self, _error) -> None:
        self.gpu.setText("Scheda NVIDIA non rilevata")
        self.gpu_timer.stop()
        if self.gpu_proc is not None:
            self.gpu_proc.deleteLater()
            self.gpu_proc = None

    def _gpu_read(self) -> None:
        proc, self.gpu_proc = self.gpu_proc, None
        if proc is None:
            return
        try:
            fields = [x.strip() for x in bytes(proc.readAllStandardOutput()).decode().splitlines()[0].split(",")]
            name, load, used, total = fields
            self.gpu.setText(f"Calcolo su {name} — uso {load}%, memoria {int(used) / 1024:.1f} / {int(total) / 1024:.0f} GB")
        except (IndexError, ValueError):
            self.gpu.setText("Scheda NVIDIA non rilevata")
        proc.deleteLater()

    # --- risultati
    def _view_model(self) -> None:
        run = self._selected()
        if run is None:
            return
        self._close_viewer()
        self.viewer = QProcess(self)
        self.viewer.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.viewer.setProcessEnvironment(process_environment())
        output = [b"", False]  # testo ricevuto finora, browser gia' aperto

        def read() -> None:
            if self.viewer is None or output[1]:
                return
            output[0] += bytes(self.viewer.readAllStandardOutput())
            url = progress.viewer_url(clean(output[0]))
            if url:
                output[1] = True
                QDesktopServices.openUrl(QUrl(url))
                self.status.setText(f"Nessuna elaborazione in corso. Viewer del modello attivo su {url}")

        self.viewer.readyReadStandardOutput.connect(read)
        self.viewer.start(config.ns("ns-viewer"), ["--load-config", str(run.config_file)])
        self.status.setText("Avvio del viewer… (il browser si apre da solo tra qualche secondo)")

    def _close_viewer(self) -> None:
        if self.viewer is not None:
            viewer, self.viewer = self.viewer, None
            if viewer.state() != QProcess.ProcessState.NotRunning:
                kill_tree(viewer.processId())
                viewer.waitForFinished(3000)
            viewer.deleteLater()

    def _open_supersplat(self) -> None:
        # SuperSplat gira nel browser: il file va trascinato nella pagina, quindi si apre anche la cartella.
        run = self._selected()
        if run is not None:
            open_path(run.export_dir)
            QDesktopServices.openUrl(QUrl(SUPERSPLAT_URL))

    def _open_mesh(self) -> None:
        mesh = self._settings().scene / "colmap" / "dense" / "mesh-poisson.ply"
        meshlab = config.TOOLS / "meshlab" / "meshlab.exe"
        if meshlab.exists():
            QProcess.startDetached(str(meshlab), [str(mesh)])
        else:
            open_path(mesh)

    def _export_csv(self) -> None:
        project = self._settings().name
        path, _ = QFileDialog.getSaveFileName(self, "Esporta la tabella", str(ROOT / f"confronto_{project}.csv"), "CSV (*.csv)")
        if path:
            runs.write_csv(project, Path(path))

    def _open_folder(self) -> None:
        scene = self._settings().scene
        open_path(scene if scene.exists() else ROOT)

    def closeEvent(self, event) -> None:
        if self.runner.running():
            answer = QMessageBox.question(self, "Elaborazione in corso", "Interrompere l'elaborazione e chiudere?",
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                          QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        if self.locked_scene is not None:
            pipeline.release_lock(self.locked_scene)
        self._close_viewer()
        event.accept()
