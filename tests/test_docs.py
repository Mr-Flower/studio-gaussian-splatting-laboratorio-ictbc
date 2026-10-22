import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted(ROOT.glob("PIANO_*.md"))]
# Caratteri di controllo ammessi: tab, a capo e ritorno a capo (file con terminazioni Windows).
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@pytest.mark.parametrize("document", DOCUMENTS, ids=lambda p: p.name)
def test_documents_have_no_control_characters(document):
    for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), start=1):
        found = CONTROL.search(line)
        assert found is None, f"{document.name}, riga {number}: carattere di controllo {found.group()!r}"
