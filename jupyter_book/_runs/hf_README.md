---
license: other
pretty_name: research_extender artifacts
---

# research_extender: CompEM book artifacts

Data, trained models and run outputs for the CompEM book in
[arbi11/research_extender](https://github.com/arbi11/research_extender).
Paths mirror that git repo (`jupyter_book/...`), so a download drops every file
where the code expects it. Code is not stored here.

## Restore
```bash
git clone https://github.com/arbi11/research_extender
cd research_extender/jupyter_book && uv sync
bash _runs/artifacts.sh download
```
See `jupyter_book/README.md` in the git repo for the full setup (FEMM 5 etc.).

## Contents
| Path | Contents |
|---|---|
| `jupyter_book/Chp2/content/` | C-core ANN data, CNN coil dataset (1000 samples), PINN coil samples, trained U-Nets and ANN, plots |
| `jupyter_book/Chp3/content/data_gen/` | IPM dataset (2,234 valid designs), FSCW set (60), phase 1–5 outputs, FEMM solutions, trained models, logs |
| `jupyter_book/Chp4/content/designs/` | GA and Q-learning runs (C-core, SynRM) |
| `jupyter_book/Chp5/content/designs/` | DQN and A2C runs |
| `jupyter_book/_runs/` | executed notebooks, run logs |
| `jupyter_book/_kaggle/` | Kaggle dataset/kernel metadata (the data itself is on Kaggle) |
| `jupyter_book/_source/` | original Notion export |

Not stored (quick to regenerate): the C2 grid cache (`grid_cache.py`), duplicate checkpoints, the Python environment.
