"""Piccolo server locale che consegna i modelli a SuperSplat.

SuperSplat gira nel browser e puo' caricare un modello da un indirizzo (parametro ?load=): per
aprirvi un file del disco senza trascinarlo, il programma lo espone su 127.0.0.1, raggiungibile solo
da questo computer, e apre SuperSplat con quell'indirizzo. Vengono serviti solo i file registrati
con ModelServer.url, ognuno sotto un percorso non indovinabile.
"""
from __future__ import annotations

import secrets
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import quote, unquote

SUPERSPLAT_URL = "https://superspl.at/editor"


class ModelServer:
    def __init__(self) -> None:
        self.files: Dict[str, Path] = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _headers(self, status: int, length: int = 0) -> None:
                self.send_response(status)
                # La pagina di SuperSplat e' di un altro sito: il browser chiede questi permessi.
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "*")
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Cache-Control", "no-store")
                if status == 200:
                    self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(length))
                self.end_headers()

            def _file(self):
                token = unquote(self.path).strip("/").split("/")[0]
                path = owner.files.get(token)
                return path if path is not None and path.is_file() else None

            def do_OPTIONS(self) -> None:  # noqa: N802 - nome imposto da http.server
                self._headers(204)

            def do_HEAD(self) -> None:  # noqa: N802
                path = self._file()
                self._headers(200, path.stat().st_size) if path else self._headers(404)

            def do_GET(self) -> None:  # noqa: N802
                path = self._file()
                if path is None:
                    self._headers(404)
                    return
                self._headers(200, path.stat().st_size)
                try:
                    with open(path, "rb") as f:
                        shutil.copyfileobj(f, self.wfile, 1 << 20)
                except (ConnectionError, OSError):
                    pass  # il browser ha chiuso la pagina durante lo scaricamento

            def log_message(self, *args) -> None:  # niente righe di log: non c'e' console
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server.server_address[1]

    def url(self, path: Path) -> str:
        """Indirizzo locale da cui scaricare `path`."""
        token = next((t for t, p in self.files.items() if p == path), None) or secrets.token_urlsafe(12)
        self.files[token] = path
        return f"http://127.0.0.1:{self.port}/{token}/{quote(path.name)}"

    def supersplat(self, path: Path) -> str:
        """Indirizzo che apre SuperSplat con `path` gia' caricato."""
        return f"{SUPERSPLAT_URL}?load={quote(self.url(path), safe='')}&filename={quote(path.name, safe='')}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
