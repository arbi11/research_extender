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

---

# Book build — whole-book PDF (2026-10-11)

`_book/_build/exports/compem_book.pdf` — 501 pages, 29.2 MB (final build 13:47), all 38 pages of C1–C5 + AppA with a numbered TOC, notebook outputs/figures, the thesis figures, and one consolidated bibliography at the end. Rebuild (≈3 min, no LaTeX needed):
```
cd _book && ../.venv/bin/python stage.py && ./build_pdf.sh
```
Toolchain: Jupyter Book 2.1.7 (MyST v1.11, installs its own Node 22) for the typst export, typst 0.14.1 in `_book/tools/bin` for the compile. 0.15 removed `diff` (MyST's `\partial`), 0.13 lacks `xrightarrow`. Jupyter Book 2 has no EPUB export; the user chose PDF only.

Staging (`_book/stage.py`) never edits the chapter folders: it copies the notebooks, swaps in error-free executed copies from `_runs/executed/` for notebooks that have no outputs, rewrites `{admonition}`→`{note}` and unwraps the `{grid}` in C4/02 (the typst exporter drops both silently), copies the thesis figures from `<project>/_static/figures/`, and draws grey placeholders only for figures that exist nowhere. `build_pdf.sh` then patches the exported typst: shims for LaTeX macros MyST passes through, `#link("")`, citations with keys missing from `references.bib` (rendered as red `[key]`), and links to missing/duplicated labels.

## Notebook execution status (fresh run 2026-10-11 13:43–13:47, CPU, 2 threads, inline backend, copies in `_runs/executed/`)
(A first run at 13:04 had `MPLBACKEND=Agg` set, which silently drops every `plt.show()` figure; the copies were redone without it.)
- Clean (0 errors): AppA/00; C2/04b; C3/01–08 (also re-verified today); C4/00–06; C5/02, 04, 08. C2/04a, 04c, 05, 06a, 06b carry the Kaggle-v2 outputs placed on 2026-10-11 00:22.
- Not executed: C1 (prose only), C2/01, 02, 07 (prose only), C2/03 (needs a FEMM session).
- Failing — missing data: C4/07 cell 4 `designs/synrm_diag/synrm_diag_results.json`; C5/05 cell 15 and C5/07 cells 2–12 `designs/{ppo_v2_run,a2c_v2_run}/training_stats.json`; C5/06 cell 8 `designs/env_probe/*.json`.
- Failing — code bugs in the Notion source: C5/01 cells 4 and 6 (`SyntaxError`, misplaced quotes), C5/03 cell 1 (`np.log10(3**2500)` overflows).
- C4 `q_learn_improved` (other session, `_runs/queue_c4b.log`) stopped at episode 150/200 at 11:23 without an rc line; `designs/q_improved/training_stats_improved.json` from that checkpoint is what C4/06 reads.

## Content gaps visible in the PDF
- Thesis figures: the Notion export carried none of the 36 files the notebooks reference as `../_static/figures/*` / `../content/designs/*`. 33 were recovered on 2026-10-11 from the user's repo `github.com/arbi11/research_extender` (`thesis_latex/Figures/`; the other session's clone under its scratchpad) into the new `compem_book/_static/figures/` (`v_design_cycle.png` is "The V-Design and Manufacturing Cycle.jpg" converted). Still missing anywhere: C4/04's `content/designs/step_34_trajectory.png`, `q_position/learning_curve_position.png`, `q_position/policy_position.png` — grey "FIGURE MISSING" boxes in the PDF.
- 16 citation keys are not in `C2/references.bib` (C1/01: unfccc2015paris; C4/01–05: mnih2015human, watkins1989learning, goldberg1989genetic, holland1992adaptation, lucchini2022topology, lee2021gta, nishanth2022review, otomo2021gabor, sato2015synrm, zhao2016topology_ga_synrm, barmada2021deep, bourdin2001filters, brown2022deep, guest2004achieving, khan2022reinforcement) — shown in red.
- C5/01 cites with `[[Author year]]` wiki links (24), rendered as plain text; AppA's table of contents links `#section1…8` have no targets.
- C5 headings carry their own numbers, so MyST's numbering doubles them ("5.1 1. Introduction…"). C2/06b cell 2 has a unicode middle dot inside `\text{}` (`T·mm`): fine in typst, a KaTeX error in the HTML build.

# Chapter 2 — 06b PINN rebuild and 03 data pipeline (2026-10-11 evening)

## 06b rebuilt on radial input features (user go 2026-10-11 ~19:55)
- Why: the placed Kaggle-v2 06b scored 55.3% NMSE while its text claimed ~7%. The ablations in `_runs/c2_06b_ablations/RESULTS.md` showed that the random Fourier features at wire scale overfit the 100 observations (159% data-only) and that a plain tanh net on (x/500, y/500) cannot represent the Poisson solution at the 14 mm wire (5.9–22% with physics, residual stuck at 0.015–0.03 for any λ), while adding r/500 and ln r as inputs gives 0.24%.
- Code: `C2/pinn_base.py` `PINN` gains `encoding='radial'` (default; inputs x/500, y/500, r/500, ln((r+1 mm)/500)) beside the old `'fourier'`, plus Xavier init (same as 06a's VanillaPINN). `C2/06b` cell 5 builds `ProperPINN(encoding='radial')`; cell 7 uses `trainer.lambda_physics` instead of a hard-coded 0.02; cell 8 error-map scale 0.3 → 0.05 of B_max. Losses, hard BC, near-wire collocation, cosine LR and clipping unchanged.
- Result (local MX450, 110 s, 3000 epochs, 33,601 params): **0.14% NMSE** (was 55.3%; analytic infinite-space wire field 0.9%; 06a vanilla 15.8%).
- Prose rewritten with measured numbers: 06b cells 0, 1, 3, 9, 10 (title, five-fixes table, "Fixes 3 and 4" section, results with the ablation table, summary) and 06a cells 2, 5, 9, 13 (scope; dataset facts: r_w = 14.4 mm, 8 of 100 observations within 100 mm holding 56% of the field energy, 0 of 1000 collocation points inside the wire; the five root causes now describe the data-only baseline 06a actually trains). Script `_runs/c2_06b_rebuild/rewrite_06_prose.py`; pre-edit copies in `_runs/c2_06b_rebuild/backup/`.
- Placed over `C2/06b_pinns_methodology.ipynb` (executed, 0 errors). `_runs/executed/C2__*` refreshed from the in-place notebooks (the Oct-9 copies with errors were stale).

## 03 data pipeline executed (0 errors, 3.1 min, `_runs/executed/C2__03_data_pipeline.ipynb`)
- The Oct-9 failure ("femm-qt is already connected to another client") was not the C3 port clash: cell 15 opens a FEMM session and never closes it, cell 17 calls openfemm() again. Windows pyFEMM starts a second instance; femm-qt refuses a second client on the same port. `FEMMSolver(debug_mode=True)` in cells 12/15/17 would also launch femm_gui on the desktop.
- Fixed without touching book code: `_runs/c2_04c_full/run_notebook.py --femm-shim` injects a first cell (removed before saving) that makes openfemm() reuse the open session and always run headless — the same shim as `_runs/femm_port.py` for the C3/C4/C5 scripts.
- Output: 14 code cells, 6 figures; 10 coil + 10 transformer + 10 IPM FEMM solves on 200×200 grids. The notebook writes no files, so nothing in C2/ changed.

PDF rebuilt 20:06 after these changes: `_book/_build/exports/compem_book.pdf`, 512 pages (C2/03 now included with outputs).

## Chapter 2 prose: thesis numbers replaced by measured ones (2026-10-11 ~22:30–23:00, user: "dont use numbers from any other source. Quote only what you did")
- Scope: every statement about *this chapter's own* experiments now quotes the notebooks and logs of this repo. Cited literature values (`{cite}`) and the historical narrative in 01/02 are left as attributed literature. Script `_runs/c2_prose_fix/rewrite_c2_measured.py`; pre-edit copies `_runs/c2_prose_fix/backup/`.
- Removed (thesis study, not run here): 45,000 field maps / 30k-10k-5k split, 35 architectures, 420–525 GPU-hours, GTX 1080 Ti, 12–17 h per model, 0.3–0.6 % NMSE, "3× from dilation", "5× from skip connections", 2.4 M-parameter 32-layer network with dropout 0.5, 1,000–10,000× speedup vs "2–8 h per FEA geometry", the ROI/dollar section, "PINNs reduce data by 20–40 %", "30–40 % faster transfer-learning convergence", "3–5× FEA speedup from CNN initialisation", 04a's RMSE 0.36 N / MAE 0.15 N / R² 0.98, and all references to a non-existent Section 4d.
- Replaced with (measured): coil dataset 1,000 FEMM solutions (700/201/99) in 34 min (2.1 s each); C-core 500 solutions in 32 min; 03 demo 3.8/4.3/5.9 s per coil/transformer/IPM design; U-Nets: baseline (`cnn_unet.py`, 7.76 M params, T4, 15.8 s/epoch) 2.48 % NMSE, dense `UNetField` (3.35 M, MX450, 43 s/epoch) 0.94 %, dilated (1,2,1,2) 0.17 % — 5.4× at equal size, val NMSE < 1 % at epoch 7 vs 44, best epochs 44/47/41 of 50, train/val within ×1.3, no dropout/weight decay; inference 16 ms per 256×256 map on the MX450 (159 ms on 2 CPU threads; baseline 23/245 ms) → ~130× vs the 2.1 s solve, 1,000 designs in 16 s vs 34 min, break-even ≈ 2,000 solves; 04a R² 0.990 / RMSE 1.66 N / MAE 1.02 N on 50 test designs (forces 0–81 N); 05 MC-dropout 90 % interval covered 52 % (synthetic task); 06a/06b 15.8 % → 0.14 % with the ablation table.
- Cells changed: 01[0], 02[0], 03[1,32], 04a[0,18], 04b[5,6,7,8,10], 04c[0,18,19,20,21,22], 05[0,9], 06a[0], 07[0–5]. 04c cell 19's thesis figure `Training_results3.png` dropped in favour of the notebook's own history plot. Markdown synced into `_runs/executed/C2__03` and `C2__04b` (the copies stage.py uses); the rest of `_runs/executed/C2__*` re-mirrored from the in-place notebooks.
- Inference timings come from a one-off benchmark on the MX450 (scratch script, 20 forward passes, batch 1 and 16) — not from a notebook cell.
- The user ran `_runs/c2_lean/archive_unreferenced.sh` on 2026-10-11 evening: C2/ is down to 2.2 GB, the legacy scripts and duplicate datasets sit in `_archive/C2_removed_2026-10-10/`, and `01_Intro.ipynb` is now `01_intro.ipynb`. `_book/myst.yml` was still pointing at `C2/01_Intro.ipynb` and the stale staged copy in `_book/C2/` got built instead of the rewritten intro (build 22:56, 510 pages); fixed the TOC entry, removed the stale copy and rebuilt.
- Final PDF after the prose fix: 22:57, 510 pages, 29.6 MB. Verified in the typeset sources (`_book/_build/typst/*.typ`): no "45,000", "35 network", "420 GPU", "0.3-0.6", "Section 4d" left in Chapter 2; "0.17%", "2.48%", "0.14%", "16 ms", "1.66 N" present. Still outside C2 (not edited, other owners): C1/01 and C1/02 say "FEA requires 2-8 hours per 2-D geometry", C1/03 names a GTX 1080 Ti as the hardware, C4/04 mentions 45,000 and C4/01 "2-8 hours".

## Chapter 1 prose: same rule applied (2026-10-11 ~23:30, user: "do the same for chapter 1")
- Script `_runs/c1_prose_fix/rewrite_c1_measured.py`, pre-edit copies `_runs/c1_prose_fix/backup/`. Markdown only.
- Removed: "FEA 1-2 / 2-8 hours per 2-D geometry, days for 3-D", "NN 0.01-0.1 s", "72,000–2,880,000×" and "10^3–10^6×" speedups, "10,000 designs × 2 h = 2.3 years", "ROI within 2-3 product cycles", "$500-$2,000 GPUs", "trained on a single NVIDIA 1080 Ti", "hours to days of training", C1/03's chapter summaries that promised "0.1-1 % NMSE with 180,000× speedup", "RNNs with attention" for Chapter 3 (the trained model is `MapDecoder`: MLP → transposed-conv decoder), "Monte Carlo Tree Search" for Chapter 4 (none exists; it runs tabular Q-learning + GA) and "policy gradient methods ... master the task" for Chapter 5 (DQN/A2C/PPO; outcomes still being produced by the other session, so only run durations are quoted), "Dissertation" → "Book".
- Replaced with measured: FEMM 2.1 s (coil), 5.9 s (IPM, C2 pipeline), 19–22 s per design (C3 performance maps; 2,462-design sweep ≈ 20 h wall with 2 workers, 2,234 valid of 2,800 sampled, 60 FSCW); U-Net 16 ms/map GPU, 159 ms CPU → ~130× / 13×; 1,000 designs 34 min vs 16 s; 100,000 predictions 27 min; C2 0.17 % NMSE, 52 % coverage, PINN 15.8 → 0.14 %; C3 R² 0.912 baseline / 0.971 width-4×, PF transfer +0.05 R² at n=10; C5 DQN 38,768 s and A2C 22,305 s per 200-episode run; hardware i7-11370H / 14 GB / MX450 2 GB + one Kaggle T4 run; Python 3.12, PyTorch 2.14, FEMM 5 (femm-qt).
- Left as background: C1/01's energy, EV and IE-class statistics and C1/02's design-era table (uncited, several `[CITATION NEEDED]` comments in the source); cited literature values.
- Noticed, not fixed (C3 belongs to me but was not asked): C3's own text says "RNN / GRU + attention" while `data_gen/ml/models.py` trains `MapDecoder` and `EnvelopeMLP`; same class of mismatch as C2 had.
- Also rewritten (after the script): the two cited sentences in C1/01 cell 2 and C1/02 cell 2 that stated "2-8 hours per 2-D FEA" and "72,000–2,880,000×" — now quote the measured FEMM times and keep the industrial-scale statement qualitative with its `{cite}`. PDF rebuilt 23:29, 512 pages, 29.8 MB; typeset C1 sources contain none of the removed strings and do contain "2.1 s", "19–22 s", "130", "i7-11370H", "0.912".

# Chapter 5 — finishing run (started 2026-10-11 23:40, user: "focus on finishing chapter 5")
State found: C5/01 (2 SyntaxErrors), C5/03 (int overflow) are Notion-source bugs; C5/05, 06, 07 need data that was never produced here: `designs/{ppo_run, ppo_v2_run, a2c_v2_run, sb3_ppo_run, sb3_a2c_run}/training_stats.json` and `designs/env_probe/{determinism,random_search,action_effect}_*.json` (the notebook text names `env_probe.py`, which did not exist). Existing: `dqn_run` (200 ep, 38,768 s) and `a2c_run` (200 ep, 22,305 s) from the other session's queue.
Done now:
- C5/01 cells 4, 6 fixed (quote bug; the radar chart of qualitative strings replaced by a table figure), C5/03 cell 1 fixed (3**2500 overflow → log10 arithmetic). Backups `_runs/c5_runs/backup/`. Executed clean into `_runs/executed/`.
- New book file `C5/env_probe.py` (determinism / random-search / action-effect / random-policy / cell-flips; JSON schema as the notebooks read it, plus `sec_per_solve`). Smoke: 2.1 s per FEMM solve, deterministic (identical force 220.943210 N twice).
- Two queues launched (2 FEMM workers = the core cap), scripts `_runs/c5_runs/queue_{A,B}.sh`, logs `_runs/data_logs/c5_*.log`, `.done` markers:
  A (port 48019): probes (5 + 200 + 100 solves, ~12 min) → ppo_v2 200 ep → sb3 PPO 6,800 steps, reward-scale 0.001 → random-policy 50 ep → cell-flips on the DQN best mask.
  B (port 48020): a2c_v2 200 ep → ppo_seqto (raw) 200 ep → sb3 A2C 6,800 steps, reward-scale 0.001.
  At the earlier pace (3.3 s per step for A2C, 6,800 steps per run) each run is ~6–7 h; expected finish of both queues ≈ 2026-10-12 late afternoon.
Still to do after the runs: re-execute C5/04–07 (04 only gains the random-search ceiling line), rewrite C5/06 §6.8.1 and C5/07 §7.4–7.8 from the measured probe/run numbers (the current text quotes thesis-era results: 1305 N DQN, 610 N random-search ceiling, 84 s per FEMM solve on "the Windows training box"), set `SEC_PER_SOLVE` in 07 cell 9 from the probe, drop the hard-coded 610 N in 05 cell 15, rebuild the PDF.
