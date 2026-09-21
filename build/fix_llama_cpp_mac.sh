#!/usr/bin/env bash
# Rebuilds llama-cpp-python without its bundled CLI tools (we only use the
# Python bindings), which sidesteps a common macOS build failure in
# vendor/llama.cpp/tools/mtmd/mtmd-helper.cpp.
#
# Run from the repo root, with the venv activated:
#   bash build/fix_llama_cpp_mac.sh
set -euo pipefail

CMAKE_ARGS="-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF" \
  pip install --force-reinstall --no-cache-dir llama-cpp-python

echo
echo "Done. Now re-run: pip install -r requirements.txt"
