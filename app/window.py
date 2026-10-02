"""Finestra principale: analisi delle foto, esecuzione della pipeline, confronto dei run."""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import QProcess, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from . import __version__, config, photos, pipeline, progress, runs
from .config import GROUPS, METHODS, QUALITIES, Settings
from .runner import STOPPED, Runner, clean, kill_tree, process_environment
from .runs import Run

SUPERSPLAT_URL = "https://superspl.at/editor"
GPU_QUERY = ["--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
PHOTO_COLUMNS = ["Escludi", "Anteprima", "Foto", "Nitidezza", "Esposizione", "Osservazioni"]


def open_path(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def confirm(parent, title: str, text: str) -> bool:
    """Domanda con «No» come risposta predefinita."""
    answer = QMessageBox.question(parent, title, text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                  QMessageBox.StandardButton.No)
    return answer == QMessageBox.StandardButton.Yes


class AnalysisWorker(QThread):
    """Analizza le foto fuori dal thread dell'interfaccia."""

    progressed = Signal(int, int)
    done = Signal(list)
    failed = Signal(str)

    def __init__(self, folder: Path, scene: Path, parent=None) -> None:
        super().__init__(parent)
        self.folder, self.scene = folder, scene

    def run(self) -> None:
        try:
            self.done.emit(photos.analyze(self.folder, self.scene, self.progressed.emit))
        except Exception as exc:  # qualunque errore va mostrato, non deve chiudere il thread in silenzio
            self.failed.emit(str(exc))


class ConvertWorker(QThread):
    """Converte un modello in un altro formato fuori dal thread dell'interfaccia (i file grandi richiedono tempo)."""

    done = Signal(str, int)
    failed = Signal(str)

    def __init__(self, source: Path, target: Path, parent=None) -> None:
        super().__init__(parent)
        self.source, self.target = source, target

    def run(self) -> None:
        try:
            from . import convert
            self.done.emit(str(self.target), convert.convert(self.source, self.target))
        except Exception as exc:  # va mostrato all'utente, non deve chiudere il thread in silenzio
            self.failed.emit(str(exc))


def quality_text(s: Settings, methods: List[str]) -> str:
    """Cosa comporta la qualita' scelta: iterazioni, risoluzione delle foto e come vengono caricate."""
    text = f"{s.iterations:,} iterazioni".replace(",", ".")
    side = pipeline.image_size(s)
    if not side:
        return text + "; la risoluzione delle foto si vede quando si indica la cartella."
    factor = pipeline.training_factor(s)
    if factor == 1:
        text += f", foto a risoluzione piena ({side} px di lato)."
    else:
        text += f", foto a {side // factor} px di lato (originali: {side} px)."
    slow = pipeline.disk_methods(s, methods)
    if slow:
        who = "" if len(methods) == 1 else " per " + ", ".join(f"«{m.short}»" for m in slow)
        text += (f" A questa risoluzione le foto non entrano in memoria tutte insieme{who}: vengono lette dal "
                 "disco durante il training, che dura parecchio di più.")
    return text


class TestDialog(QDialog):
    """Scelta dei metodi da mettere a confronto e della qualità comune a tutti."""

    def __init__(self, parent, settings: Settings, quality: Optional[str]) -> None:
        super().__init__(parent)
        self.setWindowTitle("Test: confronto tra metodi")
        self.settings = settings
        intro = QLabel(
            "I metodi spuntati vengono allenati uno dopo l'altro sulle stesse foto, alla stessa risoluzione, e "
            "valutati con lo stesso calcolo. Alla fine si apre il report con grafici e viste a confronto.")
        intro.setWordWrap(True)
        self.checks: Dict[str, QCheckBox] = {}
        methods_box = QGroupBox("Metodi da confrontare")
        methods_layout = QVBoxLayout(methods_box)
        for key, method in config.available_methods().items():
            check = QCheckBox(method.label)
            check.setToolTip(method.note)
            check.setChecked(True)
            check.toggled.connect(self._update)
            self.checks[key] = check
            methods_layout.addWidget(check)
        self.quality = QComboBox()
        for q in QUALITIES.values():
            self.quality.addItem(q.label, q.key)
        self.quality.setCurrentIndex(max(self.quality.findData(quality), 0))
        self.quality.currentIndexChanged.connect(self._update)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.mesh = QCheckBox("Aggiungi la mesh della fotogrammetria classica (molto lenta)")
        form = QFormLayout()
        form.addRow("Qualità", self.quality)
        form.addRow("", self.note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Avvia il test")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annulla")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(methods_box)
        layout.addLayout(form)
        layout.addWidget(self.mesh)
        layout.addWidget(self.buttons)
        self.setMinimumWidth(640)
        self._update()

    def methods(self) -> List[str]:
        return [key for key, check in self.checks.items() if check.isChecked()]

    def chosen(self) -> Settings:
        """Le impostazioni del progetto con i metodi e la qualità scelti qui."""
        s = Settings(**{**vars(self.settings), "methods": self.methods()})
        s.set_quality(self.quality.currentData())
        return s

    def _update(self) -> None:
        methods = self.methods()
        enough = len(methods) >= 2
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(enough)
        self.note.setText(quality_text(self.chosen(), methods) if enough else "Spunta almeno due metodi.")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Studio Gaussian Splatting — Laboratorio ICTBC  (v{__version__})")
        self.resize(1180, 900)
        self.runner = Runner(self)
        self.viewer: Optional[QProcess] = None
        self.worker: Optional[AnalysisWorker] = None
        self.converter: Optional[ConvertWorker] = None
        self.train_viewer_url = ""
        self.locked_scene: Optional[Path] = None
        self.table_runs: List[Run] = []
        self.photo_rows: List[photos.Photo] = []
        self.filling_photos = False
        self.open_report_when_done = False
        self.show_result_when_done = False
        self.custom_quality = (30000, 0, 0)  # iterazioni, fattore, lato massimo di un progetto impostato a mano
        self.step_text = self.step_percent = self.step_detail = ""
        self.started_at = 0.0

        self._build_project_box()
        self._build_photos_tab()
        self._build_run_tab()
        self._build_compare_tab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.photos_tab, "1. Foto")
        self.tabs.addTab(self.run_tab, "2. Crea il modello")
        self.tabs.addTab(self.compare_tab, "3. Risultati e confronto")
        self.tabs.setCurrentWidget(self.run_tab)
        layout = QVBoxLayout()
        layout.addWidget(self.project_box)
        layout.addWidget(self.tabs, 1)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.clock = QTimer(self, interval=1000)
        self.clock.timeout.connect(self._tick)
        self.count_timer = QTimer(self, interval=400, singleShot=True)  # il conteggio scorre tutta la cartella
        self.count_timer.timeout.connect(self._photos_folder_changed)
        self.name_timer = QTimer(self, interval=400, singleShot=True)  # non a ogni carattere digitato
        self.name_timer.timeout.connect(lambda: self._project_changed(self.name.currentText()))
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
        form.addRow("Cartella delle foto", photos_row)
        form.addRow("", self.photo_count)
        form.addRow("Nome del progetto", self.name)
        form.addRow("Allineamento", self.alignment)

    def _build_photos_tab(self) -> None:
        intro = QLabel(
            "L'analisi misura nitidezza ed esposizione di ogni foto e cerca i quasi-doppioni. Le foto spuntate "
            "vengono saltate dal prossimo allineamento; nessun file viene modificato o cancellato.")
        intro.setWordWrap(True)
        self.analyze_button = QPushButton("Analizza le foto")
        self.analyze_button.clicked.connect(self._analyze)
        self.follow_button = QPushButton("Escludi quelle suggerite")
        self.follow_button.clicked.connect(lambda: self._set_exclusions(suggested=True))
        self.keep_button = QPushButton("Non escludere nulla")
        self.keep_button.clicked.connect(lambda: self._set_exclusions(suggested=False))
        self.photo_bar = QProgressBar()
        self.photo_bar.setVisible(False)
        self.photo_summary = QLabel("")
        buttons = QHBoxLayout()
        for widget in (self.analyze_button, self.follow_button, self.keep_button):
            buttons.addWidget(widget)
        buttons.addWidget(self.photo_bar, 1)
        buttons.addStretch(1)
        self.photo_table = QTableWidget(0, len(PHOTO_COLUMNS))
        self.photo_table.setHorizontalHeaderLabels(PHOTO_COLUMNS)
        self.photo_table.setIconSize(QSize(96, 72))
        self.photo_table.verticalHeader().setVisible(False)
        self.photo_table.verticalHeader().setDefaultSectionSize(76)
        self.photo_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.photo_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.photo_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(PHOTO_COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        self.photo_table.itemChanged.connect(self._photo_toggled)
        self.photo_table.cellDoubleClicked.connect(self._open_photo)
        self.photo_table.setToolTip("Doppio clic su una riga per aprire la foto.")
        layout = QVBoxLayout()
        layout.addWidget(intro)
        layout.addLayout(buttons)
        layout.addWidget(self.photo_summary)
        layout.addWidget(self.photo_table, 1)
        self.photos_tab = QWidget()
        self.photos_tab.setLayout(layout)

    def _build_run_tab(self) -> None:
        available = config.available_methods()
        self.method = self._combo([(method.label, key) for key, method in available.items()])
        self.method_note = QLabel("")
        self.method_note.setWordWrap(True)
        self.quality = self._combo([(q.label, q.key) for q in QUALITIES.values()])
        self.quality.setToolTip("Si parte dalla resa migliore, con le foto a risoluzione piena: "
                                "abbassarla serve solo a fare prima.")
        self.quality_note = QLabel("")
        self.quality_note.setWordWrap(True)
        model_box = QGroupBox("Modello da creare")
        model_form = QFormLayout(model_box)
        model_form.addRow("Metodo", self.method)
        model_form.addRow("", self.method_note)
        model_form.addRow("Qualità", self.quality)
        model_form.addRow("", self.quality_note)
        missing = [method.label for key, method in METHODS.items() if key not in available]
        if missing:
            model_form.addRow("", QLabel("Non installati: " + "; ".join(missing)))

        self.camera = self._combo([("Automatico", "auto"),
                                   ("Una sola camera per tutte le foto", "single"),
                                   ("Una camera per sottocartella", "per_folder"),
                                   ("Una camera per ogni foto (zoom o camere diverse)", "per_image")])
        self.camera.setToolTip("Automatico: una camera per sottocartella se le foto sono divise in sottocartelle, "
                               "altrimenti una sola.")
        self.matcher = self._combo([("Foto sparse: confronta tutte le coppie", "exhaustive"),
                                    ("Sequenza ordinata (fotogrammi di un video)", "sequential")])
        self.redo_alignment = QCheckBox("Rifai l'allineamento delle foto")
        self.redo_alignment.setToolTip("L'allineamento si fa da solo la prima volta ed è comune a tutti i metodi. "
                                       "Rifarlo rende non più confrontabili i modelli già creati.")
        self.mesh_check = QCheckBox("Crea anche la mesh con la fotogrammetria classica (lenta)")
        self.mesh_check.setToolTip("Multi-view stereo e superficie di Poisson. Con centinaia di foto richiede molte ore.")
        self.mesh_size = self._combo([("1000 px (veloce)", 1000), ("1600 px", 1600), ("2400 px (molto lenta)", 2400)])
        self.advanced_box = QGroupBox("Opzioni avanzate")
        form = QFormLayout(self.advanced_box)
        form.addRow("Camere", self.camera)
        form.addRow("Matching", self.matcher)
        form.addRow("", self.redo_alignment)
        form.addRow("", self.mesh_check)
        form.addRow("Risoluzione della mesh", self.mesh_size)
        self.advanced_box.setVisible(False)
        self.advanced_button = QPushButton("Opzioni avanzate ▸")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setFlat(True)
        self.advanced_button.toggled.connect(self._toggle_advanced)

        self.start_button = QPushButton("Crea il modello 3D")
        self.start_button.setDefault(True)
        self.start_button.setMinimumHeight(36)
        self.start_button.setToolTip("Allinea le foto se non è già stato fatto, allena il metodo scelto, lo valuta "
                                     "e lo esporta.")
        self.start_button.clicked.connect(self._start)
        self.test_button = QPushButton("Test: confronta più metodi…")
        self.test_button.setMinimumHeight(36)
        self.test_button.setToolTip("Fa scegliere i metodi, li allena tutti nelle stesse condizioni e genera il "
                                    "report con i grafici del confronto.")
        self.test_button.clicked.connect(self._start_test)
        self.stop_button = QPushButton("Interrompi")
        self.stop_button.setMinimumHeight(36)
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setEnabled(False)
        self.gpu = QLabel("")
        run_row = QHBoxLayout()
        for button in (self.start_button, self.test_button, self.stop_button):
            run_row.addWidget(button)
        run_row.addStretch(1)
        run_row.addWidget(self.gpu)
        self.status = QLabel("Scegli la cartella delle foto.")
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
        self.train_viewer_button.setToolTip("Disponibile mentre è in corso il training di un metodo di nerfstudio.")
        self.train_viewer_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.train_viewer_url)))
        self.train_viewer_button.setEnabled(False)
        folder_button = QPushButton("Apri la cartella del progetto")
        folder_button.clicked.connect(self._open_folder)
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.train_viewer_button)
        bottom_row.addWidget(folder_button)
        bottom_row.addStretch(1)
        bottom_row.addWidget(self.advanced_button)

        layout = QVBoxLayout()
        layout.addWidget(model_box)
        layout.addLayout(run_row)
        layout.addLayout(status_row)
        layout.addWidget(self.bar)
        layout.addWidget(self.log, 1)
        layout.addLayout(bottom_row)
        layout.addWidget(self.advanced_box)
        self.run_tab = QWidget()
        self.run_tab.setLayout(layout)
        self.method.currentIndexChanged.connect(self._update_notes)
        self.quality.currentIndexChanged.connect(self._update_notes)
        # Bloccati durante l'elaborazione.
        self.inputs = [self.project_box, model_box, self.advanced_box, self.photos_tab]

    def _toggle_advanced(self, shown: bool) -> None:
        self.advanced_box.setVisible(shown)
        self.advanced_button.setText("Opzioni avanzate ▾" if shown else "Opzioni avanzate ▸")

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
        self.model_button.setToolTip("Viewer di nerfstudio nel browser. Per il metodo Inria usare SuperSplat.")
        self.model_button.clicked.connect(self._view_model)
        self.supersplat_button = QPushButton("Pulisci e pubblica con SuperSplat")
        self.supersplat_button.setToolTip("Apre SuperSplat nel browser e la cartella del modello: si trascina il file "
                                          "nella pagina, lo si pulisce e da lì lo si può pubblicare.")
        self.supersplat_button.clicked.connect(self._open_supersplat)
        self.export_button = QPushButton("Apri la cartella del modello")
        self.export_button.clicked.connect(lambda: self._selected() and open_path(self._selected().export_dir))
        self.formats_button = QPushButton("Altri formati…")
        self.formats_button.setToolTip("Converte il modello selezionato in nuvola di punti (.ply) o in .glb, "
                                       "e la mesh della fotogrammetria in .glb.")
        self.formats_menu = QMenu(self)
        self.points_action = self.formats_menu.addAction("Nuvola di punti (.ply) dal modello selezionato")
        self.points_action.triggered.connect(lambda: self._convert_model("punti.ply"))
        self.glb_action = self.formats_menu.addAction("GLB con la nuvola di punti del modello selezionato")
        self.glb_action.triggered.connect(lambda: self._convert_model("punti.glb"))
        self.mesh_glb_action = self.formats_menu.addAction("GLB con la mesh della fotogrammetria")
        self.mesh_glb_action.triggered.connect(self._convert_mesh)
        self.formats_button.setMenu(self.formats_menu)
        self.mesh_button = QPushButton("Apri la mesh")
        self.mesh_button.clicked.connect(self._open_mesh)
        self.report_button = QPushButton("Genera e apri il report")
        self.report_button.setToolTip("Grafici, tabelle e viste a confronto dei metodi valutati sull'allineamento corrente.")
        self.report_button.clicked.connect(self._make_report)
        self.csv_button = QPushButton("Esporta la tabella in CSV…")
        self.csv_button.clicked.connect(self._export_csv)
        buttons = QHBoxLayout()
        for button in (self.model_button, self.supersplat_button, self.export_button, self.formats_button,
                       self.mesh_button,
                       self.report_button, self.csv_button):
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
        s = Settings(
            name=self.name.currentText().strip(), photos=self.photos.text().strip(),
            camera=self.camera.currentData(), matcher=self.matcher.currentData(),
            methods=[self.method.currentData()] if self.method.count() else [],
            mesh_size=self.mesh_size.currentData(),
        )
        if self.quality.currentData() in QUALITIES:
            s.set_quality(self.quality.currentData())
        else:
            s.iterations, s.downscale, s.max_side = self.custom_quality
        return s

    def _valid_project(self) -> bool:
        return bool(config.PROJECT_NAME.fullmatch(self.name.currentText().strip()))

    def _apply(self, s: Settings) -> None:
        self.photos.setText(s.photos)
        for combo, value in ((self.camera, s.camera), (self.matcher, s.matcher), (self.mesh_size, s.mesh_size),
                             (self.method, next((m for m in s.methods if self.method.findData(m) >= 0), None))):
            combo.setCurrentIndex(max(combo.findData(value), 0))
        # Impostazioni scelte a mano (da riga di comando): restano disponibili come voce in più.
        custom = self.quality.findData("personalizzata")
        if custom >= 0:
            self.quality.removeItem(custom)
        if s.quality is None:
            self.custom_quality = (s.iterations, s.downscale, s.max_side)
            self.quality.addItem("Personalizzata (impostata nel progetto)", "personalizzata")
        self.quality.setCurrentIndex(max(self.quality.findData(s.quality or "personalizzata"), 0))
        self.redo_alignment.setChecked(False)
        self.mesh_check.setChecked(False)

    def _update_notes(self) -> None:
        method = METHODS.get(self.method.currentData())
        self.method_note.setText(method.note if method else "")
        s = self._settings()
        self.quality_note.setText(quality_text(s, s.methods))

    def _project_changed(self, name: str) -> None:
        saved = Settings.load(name.strip()) if config.PROJECT_NAME.fullmatch(name.strip()) else None
        if saved:
            self._apply(saved)
        self._refresh(reset_checks=True)
        self._load_photos()

    def _photos_folder_changed(self) -> None:
        text = self.photos.text().strip()
        folder = Path(text)
        if text and folder.is_dir() and not self.name.currentText().strip():
            self.name.setCurrentText(re.sub(r"[^A-Za-z0-9_-]+", "_", folder.name))
        self.photo_count.setText(f"{config.count_images(text)} immagini trovate" if text else "")
        self._load_photos()
        self._update_notes()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Cartella delle foto", self.photos.text() or str(config.WORK))
        if folder:
            self.photos.setText(folder)

    def _refresh(self, reset_checks: bool = False) -> None:
        """Aggiorna etichette, tabella e pulsanti in base a cio' che esiste su disco."""
        s = self._settings()
        valid = self._valid_project()
        state = pipeline.state(s) if valid else {"sfm": False, "mesh": False}
        self.redo_alignment.setEnabled(state["sfm"])
        self.mesh_check.setText("Crea anche la mesh con la fotogrammetria classica (lenta)"
                                + ("  — già fatta: verrà rifatta" if state["mesh"] else ""))
        if reset_checks:
            self.redo_alignment.setChecked(False)
            self.mesh_check.setChecked(False)
        self._update_notes()
        self.alignment.setText((runs.alignment_summary(s.scene) or ("presente" if state["sfm"] else "non ancora fatto"))
                               if valid else "")
        self._fill_table(s.name if valid else "")
        self.mesh_label.setText(("Fotogrammetria classica: " + runs.mesh_summary(s.scene)) if state["mesh"] else "")
        self.mesh_button.setEnabled(state["mesh"])
        self.mesh_glb_action.setEnabled(state["mesh"])
        if valid and not self.runner.running() and self.viewer is None:
            self.status.setText("Nessuna elaborazione in corso.")

    # --- scheda delle foto
    def _load_photos(self) -> None:
        s = self._settings()
        rows = photos.load(s.scene, s.photos) if self._valid_project() and s.photos else []
        self._fill_photos(rows)

    def _fill_photos(self, rows: List[photos.Photo]) -> None:
        s = self._settings()
        excluded = photos.load_excluded(s.scene) if rows else set()
        self.photo_rows = sorted(rows, key=lambda p: (not p.suggested, not p.notes, p.name))
        self.filling_photos = True
        self.photo_table.setRowCount(len(self.photo_rows))
        for r, p in enumerate(self.photo_rows):
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Checked if p.name in excluded else Qt.CheckState.Unchecked)
            self.photo_table.setItem(r, 0, check)
            thumb = QTableWidgetItem()
            thumb.setIcon(QIcon(str(photos.thumb_path(s.scene, p.name))))
            self.photo_table.setItem(r, 1, thumb)
            exposure = f"luminosità {p.brightness:.0f}, bruciati {p.over * 100:.0f}%, neri {p.under * 100:.0f}%"
            texts = [p.name, f"{p.relative_sharpness * 100:.0f}% della mediana", exposure,
                     "; ".join(p.reasons + p.notes)]
            for c, text in enumerate(texts, start=2):
                self.photo_table.setItem(r, c, QTableWidgetItem(text))
        self.filling_photos = False
        self.photo_summary.setText(photos.summary(self.photo_rows, excluded))
        has_rows = bool(self.photo_rows)
        self.follow_button.setEnabled(has_rows)
        self.keep_button.setEnabled(has_rows)

    def _analyze(self) -> None:
        s = self._settings()
        if not self._valid_project() or config.count_images(s.photos) == 0:
            QMessageBox.warning(self, "Analisi delle foto", "Scegli prima la cartella delle foto e il nome del progetto.")
            return
        self.analyze_button.setEnabled(False)
        self.photo_bar.setVisible(True)
        self.photo_bar.setRange(0, 0)
        self.worker = AnalysisWorker(Path(s.photos), s.scene, self)
        self.worker.progressed.connect(lambda done, total: (self.photo_bar.setRange(0, total), self.photo_bar.setValue(done)))
        self.worker.done.connect(self._analysis_done)
        self.worker.failed.connect(lambda message: (self._analysis_done([]), QMessageBox.critical(self, "Analisi delle foto", message)))
        self.worker.start()

    def _analysis_done(self, rows: list) -> None:
        self.analyze_button.setEnabled(True)
        self.photo_bar.setVisible(False)
        if rows:
            self._fill_photos(rows)

    def _photo_toggled(self, item: QTableWidgetItem) -> None:
        if self.filling_photos or item.column() != 0:
            return
        self._save_exclusions()

    def _save_exclusions(self) -> None:
        excluded = {p.name for r, p in enumerate(self.photo_rows)
                    if self.photo_table.item(r, 0).checkState() == Qt.CheckState.Checked}
        photos.set_excluded(self._settings().scene, excluded)
        self.photo_summary.setText(photos.summary(self.photo_rows, excluded))

    def _set_exclusions(self, suggested: bool) -> None:
        self.filling_photos = True
        for r, p in enumerate(self.photo_rows):
            wanted = suggested and p.suggested
            self.photo_table.item(r, 0).setCheckState(Qt.CheckState.Checked if wanted else Qt.CheckState.Unchecked)
        self.filling_photos = False
        self._save_exclusions()

    def _open_photo(self, row: int, _column: int) -> None:
        if 0 <= row < len(self.photo_rows):
            open_path(Path(self._settings().photos) / self.photo_rows[row].name)

    # --- tabella dei run
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
        idle = not self.runner.running()
        exported = bool(run and run.export_file)
        method = METHODS.get(run.method) if run else None
        # Il viewer carica i dati del progetto: non va aperto durante un'elaborazione ne' su allineamenti superati.
        self.model_button.setEnabled(bool(run) and idle and run.engine == "nerfstudio" and run.comparable() is not False)
        self.supersplat_button.setEnabled(exported and bool(method) and method.family == "gaussian")
        self.export_button.setEnabled(exported)
        self.points_action.setEnabled(exported and bool(method) and method.family == "gaussian")
        self.glb_action.setEnabled(exported)
        self.report_button.setEnabled(idle and any(r.comparable() is True and r.metrics() for r in self.table_runs))

    # --- esecuzione
    def _groups(self, s: Settings, report: bool, mesh: bool) -> List[str]:
        """Passi da eseguire: l'allineamento solo se manca o se e' stato chiesto di rifarlo."""
        wanted = {"train", "eval", "export"}
        if self.redo_alignment.isChecked() or not (self._valid_project() and pipeline.state(s)["sfm"]):
            wanted.add("sfm")
        if mesh:
            wanted.add("mesh")
        if report:
            wanted.add("report")
        return [group for group in GROUPS if group in wanted]

    def _start_test(self) -> None:
        if len(config.available_methods()) < 2:
            QMessageBox.warning(self, "Test", "Per un confronto servono almeno due metodi installati.")
            return
        s = self._settings()
        dialog = TestDialog(self, s, s.quality)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._confirm_and_launch(dialog.chosen(), report=True, mesh=dialog.mesh.isChecked())

    def _start(self) -> None:
        self._confirm_and_launch(self._settings(), report=False, mesh=self.mesh_check.isChecked())

    def _make_report(self) -> None:
        self._launch(self._settings(), ["report"])

    def _confirm_and_launch(self, s: Settings, report: bool, mesh: bool) -> None:
        groups = self._groups(s, report, mesh)
        error = pipeline.validate(s, groups)
        if error:
            QMessageBox.warning(self, "Impossibile avviare", error)
            return
        if "sfm" in groups and pipeline.state(s)["sfm"] and not confirm(
                self, "Rifare l'allineamento?",
                f"Il progetto «{s.name}» ha già un allineamento.\n\nRifarlo richiede tempo e rende non più "
                "confrontabili (né visualizzabili) i modelli già creati; anche la mesh esistente viene cancellata.\n\n"
                "Rifare comunque l'allineamento?"):
            return
        if "mesh" in groups and config.count_images(s.photos) > 150 and s.mesh_size > 1000 and not confirm(
                self, "Mesh molto lenta",
                f"Con {config.count_images(s.photos)} foto a {s.mesh_size} px la mesh può richiedere molte ore "
                "(anche più di un giorno). Continuare?"):
            return
        self.show_result_when_done = not report
        self._launch(s, groups)

    def _launch(self, s: Settings, groups: List[str]) -> None:
        error = pipeline.validate(s, groups) or pipeline.acquire_lock(s.scene)
        if error:
            QMessageBox.warning(self, "Impossibile avviare", error)
            return
        self.locked_scene = s.scene
        self._close_viewer()
        s.save()
        self.open_report_when_done = "report" in groups
        self.log.clear()
        self.tabs.setCurrentWidget(self.run_tab)
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
        self.test_button.setEnabled(not running)
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
        if ok:
            self.redo_alignment.setChecked(False)
        if ok and self.show_result_when_done and self.table_runs:
            self.table.selectRow(0)  # il run appena creato: la tabella e' ordinata dal piu' recente
            self.tabs.setCurrentWidget(self.compare_tab)
        self.show_result_when_done = False
        if ok and self.open_report_when_done:
            report = config.WORK / "reports" / self._settings().name / "index.html"
            if report.exists():
                open_path(report)
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

    def _convert_model(self, name: str) -> None:
        run = self._selected()
        if run is not None and run.export_file is not None:
            self._convert(run.export_file, run.export_dir / name)

    def _convert_mesh(self) -> None:
        dense = self._settings().scene / "colmap" / "dense"
        self._convert(dense / "mesh-poisson.ply", dense / "mesh-poisson.glb")

    def _convert(self, source: Path, target: Path) -> None:
        if self.converter is not None and self.converter.isRunning():
            return
        self.formats_button.setEnabled(False)
        self.status.setText(f"Conversione in {target.name}…")
        self.converter = ConvertWorker(source, target, self)
        self.converter.done.connect(self._converted)
        self.converter.failed.connect(lambda message: (self.formats_button.setEnabled(True),
                                                       QMessageBox.critical(self, "Conversione", message)))
        self.converter.start()

    def _converted(self, target: str, count: int) -> None:
        self.formats_button.setEnabled(True)
        self.status.setText(f"Creato {Path(target).name}: {count:,} punti o vertici.".replace(",", "."))
        open_path(Path(target).parent)

    def _open_mesh(self) -> None:
        mesh = self._settings().scene / "colmap" / "dense" / "mesh-poisson.ply"
        meshlab = config.TOOLS / "meshlab" / "meshlab.exe"
        if meshlab.exists():
            QProcess.startDetached(str(meshlab), [str(mesh)])
        else:
            open_path(mesh)

    def _export_csv(self) -> None:
        project = self._settings().name
        path, _ = QFileDialog.getSaveFileName(self, "Esporta la tabella", str(config.WORK / f"confronto_{project}.csv"), "CSV (*.csv)")
        if path:
            runs.write_csv(project, Path(path))

    def _open_folder(self) -> None:
        scene = self._settings().scene
        open_path(scene if scene.exists() else config.WORK)

    def closeEvent(self, event) -> None:
        if self.runner.running():
            if not confirm(self, "Elaborazione in corso", "Interrompere l'elaborazione e chiudere?"):
                event.ignore()
                return
            self.runner.stop()
        for thread in (self.worker, self.converter):
            if thread is not None and thread.isRunning():
                thread.wait()
        if self.locked_scene is not None:
            pipeline.release_lock(self.locked_scene)
        self._close_viewer()
        event.accept()
