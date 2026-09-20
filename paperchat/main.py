"""Entrypoint: opens a pywebview window rooted at the current (or given) folder."""
from __future__ import annotations

import sys
from pathlib import Path

import webview

from .api import Api
from .paths import resource_root


def _target_folder() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    if getattr(sys, "frozen", False):
        # Packaged macOS app: double-clicking from Finder does NOT set the
        # working directory to the enclosing folder, so we derive it from
        # where the .app bundle itself lives (the user drags/drops the app
        # into their papers folder and runs it from there).
        exe_path = Path(sys.executable).resolve()
        for parent in exe_path.parents:
            if parent.suffix == ".app":
                return parent.parent
        return exe_path.parent
    return Path.cwd()


def main():
    folder = _target_folder()
    api = Api(folder)

    ui_index = resource_root() / "paperchat" / "ui" / "index.html"
    window = webview.create_window(
        "PaperChat",
        url=str(ui_index),
        js_api=api,
        width=1280,
        height=820,
        min_size=(960, 640),
    )
    api.set_window(window)
    window.events.loaded += lambda: api.startup()

    webview.start()


if __name__ == "__main__":
    main()
