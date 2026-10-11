#!/bin/bash
# Build the book. Usage: ./build.sh [--typst|--html|--docx|...]  (defaults to --typst)
# Needs only the project venv: node comes from Jupyter Book's own nodeenv, typst/pandoc from tools/bin.
cd "$(dirname "$0")"
export PATH="$PWD/tools/bin:$HOME/.local/share/jupyter-book/22.17.0/bin:$PATH"
export JB_ALLOW_NODEENV=1
exec ../.venv/bin/jupyter-book build "${@:---typst}"
