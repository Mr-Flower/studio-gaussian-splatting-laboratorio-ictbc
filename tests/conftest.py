import struct

import pytest

from app import config, pipeline

GB = 2**30


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Radice di progetto finta, con i componenti esterni presenti come file vuoti."""
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "TOOLS", tmp_path / "tools")
    monkeypatch.setattr(config, "VENV_SCRIPTS", tmp_path / ".venv" / "Scripts")
    # Memoria del computer e della scheda grafica: fissate, cosi' i test non dipendono dalla macchina.
    monkeypatch.setattr(pipeline, "hardware", lambda: (64 * GB, 48 * GB))
    for relative in ("tools/colmap-4.2.1/bin/colmap.exe", "tools/ffmpeg-9.0/bin/ffmpeg.exe"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True)
        path.touch()
    (tmp_path / ".venv" / "Scripts").mkdir(parents=True)
    for command in ("ns-train", "ns-eval", "ns-export", "ns-process-data", "ns-viewer"):
        (tmp_path / ".venv" / "Scripts" / f"{command}.exe").touch()
    return tmp_path


@pytest.fixture
def photos(tmp_path):
    folder = tmp_path / "foto"
    folder.mkdir()
    for i in range(5):
        (folder / f"img_{i}.jpg").touch()
    return folder


def make_sparse_model(colmap_dir, name, registered):
    model = colmap_dir / "sparse" / name
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(struct.pack("<Q", registered))
    return model


def make_run(root, project, method, timestamp, step=29999):
    run_dir = root / "outputs" / project / method / timestamp
    (run_dir / "nerfstudio_models").mkdir(parents=True)
    (run_dir / "config.yml").write_text("config")
    (run_dir / "nerfstudio_models" / f"step-{step:09d}.ckpt").write_bytes(b"x" * 2048)
    return run_dir
