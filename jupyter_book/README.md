# CompEM book (`jupyter_book/`)

Chapters `Chp1`–`Chp5` and `AppendixA`; each chapter's notebooks and code are in
`ChpN/content/`. Code is in git. Data, trained models and run outputs (about 19 GB)
are on the Hugging Face Hub in the private dataset repo
[`rb1189/research_extender-artifacts`](https://huggingface.co/datasets/rb1189/research_extender-artifacts),
stored with the same paths as this repo.

Tested on Ubuntu 24.04, Python 3.12, FEMM 5 (femm-qt, 06 Oct 2026 build).

## Set up from scratch

### 1. Code and Python environment
```bash
git clone https://github.com/arbi11/research_extender
cd research_extender/jupyter_book
uv sync                      # Python 3.12 + locked packages, incl. the FEMM binding (third_party/pyfemm)
```

### 2. Data and models
```bash
uvx hf auth login            # account with access to rb1189/research_extender-artifacts
bash _runs/artifacts.sh download
```
Every file lands where the notebooks expect it, for example
`Chp3/content/data_gen/outputs/dataset_train/master_dataset.npz`. The `.gitignore`
keeps these files out of git.

### 3. FEMM 5 (only to re-run simulations)
The notebooks that only read data don't need FEMM. To generate new data, build
FEMM 5 from the source zip on https://femm.info/doku/doku.php?id=download5:
```bash
sudo apt install build-essential cmake ninja-build git unzip qt6-base-dev libqt6widgets6
unzip femm-qt-py-src_*.zip && cd femm-qt
./linux_build.sh && ./linux_build.sh install    # installs into ~/.local, not as root
```
Log out and back in once so `~/.local/bin` (`femm_gui`, `femm_headless`) is on `PATH`.

## Running
Run notebooks and scripts from their chapter's `content/` folder
(`uv run jupyter lab`, or `uv run python <script>.py`).

Chapter 3–5 scripts were written for FEMM 4.2 on Windows. Run them unchanged
through the wrapper, which gives each job its own FEMM port, maps
`"M-19 Steel"` to FEMM 5's `"M-19 14 mil"`, and reuses one FEMM session per job:
```bash
cd Chp3/content
uv run python ../../_runs/femm_port.py 48016 data_gen.run_dataset [args]
```
Parallel FEMM jobs need different ports. `Chp2/content/femm_solver.py` is already
ported to FEMM 5 and needs no wrapper.

## Results and status
`_runs/REPORT.md` has what was run and the numbers per chapter.
`_runs/c2_06b_ablations/RESULTS.md` has the PINN ablations.

## Updating the artifacts
After producing new data or models:
```bash
bash _runs/artifacts.sh upload
```
This uploads only data (never code), with the same paths. FEMM `.ans` files are
included, while the C2 grid cache and duplicate checkpoints are not
(`grid_cache.py` rebuilds the cache in minutes).
