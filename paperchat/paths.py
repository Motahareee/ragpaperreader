"""Resolve bundled resource paths that work both in dev mode (running from
the repo) and inside a PyInstaller-frozen app, where files are unpacked
under `sys._MEIPASS` instead of living next to the source."""
from __future__ import annotations

import sys
from pathlib import Path


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent
