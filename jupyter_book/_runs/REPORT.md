# CompEM book: notebook test run (2026-10-09)

Source: Notion export `compem_book_notion.zip` → `source/notion/raw/`, copied unchanged into `C1/ … C5/ AppA/`.
Environment: uv project (`pyproject.toml`, `uv.lock`), Python 3.12, torch 2.14.1+cu130.
Executed copies of every notebook (with outputs and tracebacks) are in `_runs/executed/`.

Setup done (no code edits):
- The 4 Markdown pages (C2/01, C2/02, C4/00, C4/02) became one-cell Markdown notebooks.
- C3 `ml/` was moved to `data_gen/ml/` and empty `__init__.py` files were added, because the notebooks import `data_gen.ml.*`.
- matplotlib pinned to `<3.11`, because 3.11 removed `boxplot(labels=...)`, which C3/03 uses.
- The GPU isn't usable right now: `nvidia-smi` reports "Driver/library version mismatch" (the NVIDIA driver was updated without a reboot). Everything ran on CPU.

## Passed (18 of 38)
- AppA/00 (15 code cells)
- C1/01, 02, 03 (prose only)
- C2/01, 02, 07 (prose only); C2/04b, C2/05
- C3/03, C3/08
- C4/00, 01, 02 (prose); C4/03, C4/04
- C5/02, C5/08 (prose)

## Failed: need FEMM (pyFEMM can't find Wine/FEMM on Linux)
| Notebook | First failing cell | Error |
|---|---|---|
| C2/03_data_pipeline | 3 | `Exception: Wine binary not found` from `femm.openfemm()` |

## Failed: data files missing; the only scripts that make them need FEMM
| Notebook | Cell | Missing file | Generator (needs FEMM) |
|---|---|---|---|
| C2/04a_ann_fundamentals | 2 | `c_core_force_dataset.csv` | `generate_ann_data.py` |
| C2/04c_cnn_training | 1 | `dataset/coil_cnn_dataset.pkl` | `generate_cnn_training_data.py` |
| C2/06a_pinn_theory_formulation | 1 | `dataset/pinn/coil_n100/sample_0000.pkl` | `generate_pinns_data.py` (via `femm_solver`) |
| C2/06b_pinns_methodology | 1 | `dataset/pinn/coil_n100` | same |
| C3/02_data_generation | 1 | `data_gen/outputs/phase1_samples.csv` | `data_gen/run_phase1..4.py` |
| C3/04_model_implementation | 1 | `data_gen/outputs/dataset_train/master_dataset.npz` | `data_gen/run_dataset.py` |
| C3/05_transfer_learning | 1 | same `master_dataset.npz` | same |
| C3/06_uncertainty_quantification | 1 | `data_gen/outputs/phase5_map_train_arch/width4x/predictions.npz` | `ml/run_phase5_map.py` (needs the dataset above) |
| C3/07_results | 1 | `data_gen/outputs/phase5_envelope_train/predictions.npz` | `ml/run_phase5_envelope.py` (same) |
| C4/05_genetic_algorithms | 2 | `designs/ga/ga_logbook.json` | `ga_cCore.py` (env `seqTO_env` uses FEMM) |
| C4/06_mdp_implementation | 3 | `designs/q_improved/training_stats_improved.json` | Q-learning on `seqTO_env` (FEMM) |
| C4/07_results_and_future_directions | 1 | `designs/ga/ga_logbook.json` | `ga_cCore.py` |
| C5/04_deep_q_learning | 5 | `designs/dqn_run/training_stats.json` | `dqn_seqto.py` (env `seqTO_v2_env` uses FEMM) |
| C5/05_policy_methods_and_actor_critic | 1 | `designs/a2c_run/training_stats.json` | `a2c_seqto.py` (same) |
| C5/06_seqto_v2_environment_and_setup | 1 | `designs/env_probe/*.json` → `TypeError: 'NoneType' object is not subscriptable` | no generator script in the Notion export |
| C5/07_results_and_comparison | 1 | `designs/dqn_run/training_stats.json` | `dqn_seqto.py` |

## Failed: bugs in the Notion code (not fixed, per instructions)
| Notebook | Cell | Error | Cause |
|---|---|---|---|
| ~~C3/01_introduction~~ | 7 | `IndexError: index 43 is out of bounds for axis 0 with size 40` | **Not a bug** — the cell runs with the real phase-4 data (0 errors on 2026-10-11, see Chapter 3 section below). The error came from the placeholder grid used when the data was missing. |
| C5/01_introduction | 2 | `SyntaxError: unterminated string literal` | line 26 `boxstyle='round,pad=0.5,fc='yellow',alpha=0.7` has a misplaced quote |
| C5/03_function_approximators | 1 | `TypeError: loop of ufunc does not support argument 0 of type int` | line 34 `np.log10(3**2500)`: the integer is too big for numpy |

---

# Chapter 3 — full re-run, 2026-10-11 (all 8 notebooks execute with 0 errors)

Executed copies: `_runs/executed/C3__0[1-8]*.ipynb` (01:21–01:26, 4.5 min total). Logs: `_runs/c3_logs/` (FEMM sweep), `_runs/c3_ml_logs/` (training), `C3/data_gen/logs/`.

## How it was run on Linux / femm-qt (book code unchanged except one approved edit)
- FEMM shims at runtime only, in `_runs/femm_port.py`: one FEMM port per job; `"M-19 Steel"` (stock FEMM name) → femm-qt's `"M-19 14 mil"` (user's choice); `openfemm()` reuses the open session. Phase-1 open-circuit λ_d came out 0.0107 Wb vs the notebook's 0.010 Wb, so the material swap reproduces the book's physics.
- **One book-code edit, approved by the user:** `C3/data_gen/ml/train.py` — `device = "cuda" if available` (2 lines) and `model.cpu()` before each `return` (2 lines) so the unchanged predict/save code keeps working. Measured 1.12 s/epoch (CPU, 2 threads) → 0.071 s/epoch (MX450), **15×**; losses identical to all printed digits. Weights are saved as CPU tensors.
- Memory problem found and worked around: `data_gen/dataset_runner.py::aggregate()` (checkpoint every 10 designs) re-reads the whole master `.npz` once per existing design and keeps every copy alive → memory ∝ (designs)², ~39 GB at 1,000 designs; five workers were OOM-killed at 7–12 GB. Workaround `_runs/run_c3_chunked2.sh`: 25-design chunks, each in a fresh process against an empty scratch root, folders moved into `dataset_train/`, master rebuilt from `params.npz` at the end. Peak worker memory 267 MB afterwards.

## Datasets
| Dataset | Result | Book |
|---|---|---|
| Reference IPM, phases 1–4 (`outputs/phase*`) | done; fits R² 0.967–0.997 | — |
| IPM training set `dataset_train/master_dataset.npz` | **2,234 valid of 2,800 sampled** (20% rejected), 60×80 η grids, 4.8% NaN (outside envelope), η ≤ 0.990 | 2,511 of 2,800 |
| Power-factor maps `master_pf_dataset.npz` | 2,234 designs, PF ≤ 0.945 | — |
| FSCW set `dataset_fscw/master_dataset.npz` | 60 valid of 60 | 49 |
Notes: designs ≥ 372 were sampled as many 25-point LHS batches with fresh seeds (a well-spread random sample over the same box, not one stratified LHS; the book's docs sanction extending with new seeds). `N_max_rpm` is 8,000 for every IPM design (the config's speed cap), so the envelope net's N_max target is trivial.

## Training (12 runs, GPU, 19:47 → 01:21)
| Run | Ours | Book |
|---|---|---|
| η-map baseline, 5-fold OOF R² | **0.912** (RMSE 0.080) | 0.913 |
| η-map depth-1 / width-2× / width-4× | 0.964 / 0.962 / **0.971** | — / — / 0.973 |
| Torque envelope | T_max RMSE 0.38 Nm, MAPE 4.4% | — |
| FSCW standalone CV, baseline / width-4× | 0.751 / 0.737 (12 val designs per fold) | 0.835 / — |
| PF transfer sweep (54 runs) | reset-fine-tune beats scratch for n ≤ 50 (+0.05 at n=10); naive fine-tune fails (R² ≈ 0) | same pattern |
| PF transfer sweep, width-4× (54 runs) | reset-fine-tune beats scratch at every n (0.887 vs 0.881 at n=400); naive works at n ≤ 25 but plateaus at ~0.78 | — |
| FSCW transfer sweep (45 runs) | pretraining helps at every n (+0.10 at n=5); naive transfer works (+0.025 at n=25) | naive +0.04 at n=35 |
| FSCW transfer sweep, width-4× (45 runs) | scratch overfits badly at n ≤ 10; pretrained-reset +0.30 at n=10 | — |
Width-4× gives no lift on FSCW at the 60-design scale (standalone 0.737 vs 0.751) — one of the two outcomes C3/05 anticipated.

## Notebooks
All of C3/01–08: 0 error outputs (56 code cells). The C3/01 `IndexError` reported on 2026-10-09 was a missing-data artefact, not a code bug.
