#!/bin/bash
# Data, models and run outputs for jupyter_book/ live on the Hugging Face Hub
# (private dataset repo). Paths there mirror this git repo, so a download lands
# every file where the code expects it.
#
#   download                 fetch everything into this checkout
#   upload                   push artifacts from this checkout (repo layout)
#   upload-legacy <dir>      one-time push from the old compem_book/ workspace
#                            (C1..C5, AppA, _runs, _kaggle at its top level)
#
# Needs: uv (for `uvx hf`), and `uvx hf auth login` with access to the repo.
set -euo pipefail

HF_REPO="${HF_REPO:-rb1189/research_extender-artifacts}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"          # git repo root (jupyter_book/_runs/..)

# Use at most 2 cores and low priority while hashing/uploading
RUN=(nice -n 10)
command -v taskset >/dev/null && RUN=(taskset -c 0,1 "${RUN[@]}")
hf() { "${RUN[@]}" uvx hf "$@"; }

# Never upload code (it lives in git), build output, caches, backups or
# anything quick to regenerate
NOT_DATA=(
  --exclude "*.py" --exclude "*.sh" --exclude "*.md" --exclude "*.yml" --exclude "*.yaml"
  --exclude "*.toml" --exclude "*.lock" --exclude "*.bib"
  --exclude "*.bak" --exclude "*.orig" --exclude "*.unexecuted" --exclude "*.pre_*"
  --exclude "*__pycache__*" --exclude "*.ipynb_checkpoints*" --exclude "*_build/*"
  --exclude "*dataset/grid_cache/*" --exclude "*coil_cnn_dataset_checkpoint.pkl"
  --exclude "*/_tmp_*" --exclude "_tmp_*" --exclude "*kernel_log_prev_boot.txt"
  --exclude "*/backup/*" --exclude "*.venv/*" --exclude ".cache/*"
)

push() {  # push <local dir> <path in repo> [extra args]
  local src="$1" dst="$2"; shift 2
  [ -e "$src" ] || { echo "skip $src (missing)"; return; }
  echo "=== $src -> $dst"
  hf upload "$HF_REPO" "$src" "$dst" --repo-type dataset --private \
    --commit-message "Upload $dst" "${NOT_DATA[@]}" "$@"
}

card() {  # dataset card shown on the Hub
  hf upload "$HF_REPO" "$HERE/hf_README.md" README.md --repo-type dataset --private \
    --commit-message "Update dataset card"
}

case "${1:-}" in
  download)
    hf download "$HF_REPO" --repo-type dataset --local-dir "$ROOT" \
      --exclude "README.md" --exclude ".gitattributes"
    echo "Artifacts are in place under $ROOT/jupyter_book"
    ;;
  upload)
    # Chapter folders: notebooks are code (in git); _runs/_kaggle notebooks
    # are executed outputs and are uploaded
    for c in Chp1 Chp2 Chp3 Chp4 Chp5 AppendixA; do
      push "$ROOT/jupyter_book/$c/content" "jupyter_book/$c/content" --exclude "*.ipynb"
    done
    push "$ROOT/jupyter_book/_runs" "jupyter_book/_runs"
    push "$ROOT/jupyter_book/_kaggle" "jupyter_book/_kaggle" --include "*.json"   # Kaggle metadata only; data is on Kaggle
    push "$ROOT/jupyter_book/_source" "jupyter_book/_source"   # original Notion export
    push "$ROOT/jupyter_book/_archive" "jupyter_book/_archive"  # files removed from chapters (lean-ups)
    card
    ;;
  upload-legacy)
    W="$(cd "${2:?usage: $0 upload-legacy <compem_book dir>}" && pwd)"
    for pair in C1:Chp1 C2:Chp2 C3:Chp3 C4:Chp4 C5:Chp5 AppA:AppendixA; do
      push "$W/${pair%%:*}" "jupyter_book/${pair##*:}/content" --exclude "*.ipynb"
    done
    push "$W/_runs" "jupyter_book/_runs"
    push "$W/_kaggle" "jupyter_book/_kaggle" --include "*.json"
    push "$W/compem_book_notion.zip" "jupyter_book/_source/compem_book_notion.zip"
    push "$W/_archive" "jupyter_book/_archive"
    card
    ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
