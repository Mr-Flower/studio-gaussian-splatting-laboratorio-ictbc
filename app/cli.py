"""Riga di comando: stessa pipeline dell'interfaccia grafica, per esecuzioni in serie e riproducibili.

Esempi:
  python -m app.cli run --project arco --photos D:\\foto --methods splatfacto nerfacto
  python -m app.cli run --project arco --steps eval export --methods splatfacto
  python -m app.cli report --project arco --csv confronto.csv
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QCoreApplication, QTimer

from . import __version__, config, pipeline, runs
from .config import GROUPS, METHODS, Settings
from .runner import Runner


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="esegue i passi della pipeline")
    run.add_argument("--project", required=True, help="nome del progetto (cartella in data/)")
    run.add_argument("--photos", help="cartella delle foto; se omessa usa quella salvata nel progetto")
    run.add_argument("--steps", nargs="+", choices=list(GROUPS), default=["sfm", "train", "eval", "export"],
                     help="passi da eseguire (predefiniti: sfm train eval export)")
    run.add_argument("--methods", nargs="+", choices=list(METHODS), help="metodi da allenare e confrontare")
    run.add_argument("--iterations", type=int, help="iterazioni di training")
    run.add_argument("--downscale", type=int, help="fattore di riduzione delle immagini (0 = automatico)")
    run.add_argument("--camera", choices=["single", "per_folder", "per_image"])
    run.add_argument("--matcher", choices=["exhaustive", "sequential"])
    run.add_argument("--mesh-size", type=int, help="lato massimo delle immagini per la mesh, in pixel")

    report = sub.add_parser("report", help="mostra la tabella di confronto dei run di un progetto")
    report.add_argument("--project", required=True)
    report.add_argument("--csv", type=Path, help="salva la tabella in un file CSV")
    return parser


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.load(args.project) or Settings(name=args.project)
    s.name = args.project
    if args.photos:
        s.photos = str(Path(args.photos).resolve())
    for field, value in (("methods", args.methods), ("iterations", args.iterations), ("downscale", args.downscale),
                         ("camera", args.camera), ("matcher", args.matcher), ("mesh_size", args.mesh_size)):
        if value is not None:
            setattr(s, field, value)
    return s


def _run(args: argparse.Namespace) -> int:
    missing = config.missing_components()
    if missing:
        print("Componenti mancanti:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    s = _settings(args)
    groups = [g for g in GROUPS if g in args.steps]  # nell'ordine di esecuzione
    error = pipeline.validate(s, groups) or pipeline.acquire_lock(s.scene)
    if error:
        print(error, file=sys.stderr)
        return 2
    s.save()

    app = QCoreApplication(sys.argv[:1])
    runner = Runner()
    result = {"ok": False}

    def finished(ok: bool, message: str) -> None:
        result["ok"] = ok
        print("\nCompletato." if ok else f"\nERRORE: {message}", flush=True)
        app.quit()

    last = {"percent": -1}

    def progress(value: float) -> None:
        percent = int(value * 100)
        if percent >= last["percent"] + 10:  # una riga ogni 10%: il dettaglio completo e' nel file di log
            last["percent"] = percent
            print(f"    {percent}%", flush=True)

    def started(index: int, total: int, label: str) -> None:
        last["percent"] = -1
        print(f"\n[{index + 1}/{total}] {label}", flush=True)

    runner.step_started.connect(started)
    runner.progress.connect(progress)
    runner.finished.connect(finished)
    signal.signal(signal.SIGINT, lambda *_: runner.stop())
    keepalive = QTimer()  # lascia girare l'interprete, altrimenti Ctrl+C non viene visto
    keepalive.start(300)
    keepalive.timeout.connect(lambda: None)
    try:
        QTimer.singleShot(0, lambda: runner.start(pipeline.build_steps(s, groups), s.scene / "pipeline.log"))
        app.exec()
    finally:
        pipeline.release_lock(s.scene)
    print(f"Log completo: {s.scene / 'pipeline.log'}")
    return 0 if result["ok"] else 1


def _report(args: argparse.Namespace) -> int:
    found = runs.list_runs(args.project)
    if not found:
        print(f"Nessun run per il progetto '{args.project}'.")
        return 1
    scene = config.ROOT / "data" / args.project
    for summary in (runs.alignment_summary(scene), runs.mesh_summary(scene)):
        if summary:
            print(summary)
    rows = [[str(runs.row(run)[key]) for key, _ in runs.COLUMNS] for run in found]
    header = [label for _, label in runs.COLUMNS]
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    for line in [header] + rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(line, widths)))
    if args.csv:
        runs.write_csv(args.project, args.csv)
        print(f"\nTabella salvata in {args.csv}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # la console di Windows puo' non avere tutti i caratteri
        stream.reconfigure(errors="replace")
    args = _parser().parse_args(argv)
    return _run(args) if args.command == "run" else _report(args)


if __name__ == "__main__":
    sys.exit(main())
