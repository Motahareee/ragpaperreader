# PyInstaller spec for PaperChat.app (macOS)
# Build with: pyinstaller build/paperchat.spec  (run from the repo root, on macOS)
from pathlib import Path

block_cipher = None
repo_root = Path(SPECPATH).resolve().parent

a = Analysis(
    ["run_paperchat.py"],
    pathex=[str(repo_root)],
    binaries=[],
    datas=[
        (str(repo_root / "paperchat" / "ui"), "paperchat/ui"),
        (str(repo_root / "models" / "embed-model.gguf"), "models"),
    ],
    hiddenimports=["llama_cpp"],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PaperChat",
    debug=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="PaperChat",
)

app = BUNDLE(
    coll,
    name="PaperChat.app",
    icon=None,
    bundle_identifier="com.paperchat.app",
    info_plist={
        "NSHighResolutionCapable": True,
        "LSMinimumSystemVersion": "11.0",
    },
)
