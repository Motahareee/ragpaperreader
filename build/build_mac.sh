#!/usr/bin/env bash
# Build PaperChat.app on macOS. Run from the repo root:
#   ./build/build_mac.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r build/requirements-build.txt

if [ ! -f "models/embed-model.gguf" ]; then
  echo "Fetching bundled embedding model..."
  python scripts/fetch_embed_model.py
fi

rm -rf build/build build/dist
pyinstaller --distpath build/dist --workpath build/build build/paperchat.spec

echo
echo "Done. App bundle at: build/dist/PaperChat.app"
echo "Copy PaperChat.app into a folder of PDFs and double-click it to run."
echo "Note: since it isn't notarized, the first launch may need right-click -> Open"
echo "to bypass Gatekeeper's 'unidentified developer' warning."
