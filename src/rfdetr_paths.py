"""
Konwencje ścieżek: repozytorium ``dermatoscopy_ai`` obok tego projektu (wspólny katalog nadrzędny).
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_SIBLING_NAME = "dermatoscopy_ai"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def sibling_dermatoscopy_ai() -> Path:
    """``<parent>/dermatoscopy_ai`` względem korzenia mini-projektu."""
    return repo_root().parent / DEFAULT_SIBLING_NAME


def ensure_sibling_on_syspath(path: Path | None = None) -> Path:
    """
    Dodaje katalog projektu ``dermatoscopy_ai`` na początek ``sys.path`` (import ``dermato_ai.*``).
    Zwraca ścieżkę (nie musi istnieć na maszynie CI — wtedy nic nie dodaje).
    """
    p = (path or sibling_dermatoscopy_ai()).resolve()
    if p.is_dir():
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return p
