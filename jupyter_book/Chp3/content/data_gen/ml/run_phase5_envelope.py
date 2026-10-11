"""Phase 5A driver: train an envelope MLP to predict (N_max, T_max).

Mirrors the structure of run_phase1.py / run_phase4.py: argparse, dual-handler
logging, exit-code conventions, and outputs under data_gen/outputs/.

Usage:
    python -m data_gen.ml.run_phase5_envelope --smoke
    python -m data_gen.ml.run_phase5_envelope --epochs 300 --folds 5
    python -m data_gen.ml.run_phase5_envelope --loo
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import (
    DatasetBundle,
    kfold_indices,
    load_dataset,
    loo_indices,
    standardise,
    unstandardise,
)
from .metrics import mae, mape, r2, rmse
from .models import EnvelopeMLP
from .plots import loss_curves, pred_vs_actual
from .train import predict, train_one_fold


HERE = Path(__file__).resolve().parent
DATA_GEN_ROOT = HERE.parent
LOG_DIR = DATA_GEN_ROOT / "logs"
DEFAULT_OUT_DIR = DATA_GEN_ROOT / "outputs" / "phase5_envelope"
DEFAULT_DATASET = DATA_GEN_ROOT / "outputs" / "dataset" / "master_dataset.npz"

TARGET_NAMES = ["N_max_rpm", "T_max_Nm"]


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase5_envelope_{ts}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_path


def _build_folds(n: int, args: argparse.Namespace) -> list:
    if args.loo:
        return loo_indices(n)
    return kfold_indices(n, k=args.folds, seed=args.seed)


def _stack_targets(bundle: DatasetBundle) -> np.ndarray:
    return np.stack([bundle.N_max_rpm, bundle.T_max_Nm], axis=1).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-out CV (overrides --folds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs, 1 fold, light output for wiring verification")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for CV metrics, weights, plots, and "
                             "predictions.npz. Defaults to outputs/phase5_envelope. "
                             "Use a different dir (e.g. outputs/phase5_envelope_train) "
                             "to isolate runs.")
    parser.add_argument("--hidden", type=int, nargs="+", default=[32, 32],
                        help="hidden layer widths for the MLP")
    args = parser.parse_args()
    out_dir = args.out_dir

    log_path = _setup_logging()
    log = logging.getLogger("run_phase5_envelope")
    log.info("=== Phase 5A: envelope-net training ===")
    log.info("Log file: %s", log_path)

    try:
        bundle = load_dataset(args.dataset)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    log.info("Loaded dataset: %d designs, params=%s",
             bundle.n_designs, bundle.param_names)

    if bundle.n_designs < 4:
        log.error("Need at least 4 designs to do CV; have %d", bundle.n_designs)
        return 3

    # Stack targets, standardise inputs and targets independently.
    X = bundle.design_params
    Y = _stack_targets(bundle)
    X_std, x_mean, x_std = standardise(X)
    Y_std, y_mean, y_std = standardise(Y)
    log.info("Input  stats: mean=%s, std=%s",
             np.round(x_mean, 3).tolist(), np.round(x_std, 3).tolist())
    log.info("Target stats: mean=%s, std=%s",
             np.round(y_mean, 3).tolist(), np.round(y_std, 3).tolist())

    # Smoke flag overrides training intensity.
    if args.smoke:
        args.epochs = 3
        folds = _build_folds(bundle.n_designs, argparse.Namespace(
            loo=False, folds=2, seed=args.seed))
        folds = folds[:1]  # only one fold for the smoke run
        log.info("[smoke] 1 fold, 3 epochs")
    else:
        folds = _build_folds(bundle.n_designs, args)
        log.info("CV: %d folds (%s)", len(folds), "loo" if args.loo else f"k={args.folds}")

    out_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    oof_pred = np.full_like(Y, np.nan, dtype=np.float32)
    oof_pred_std = np.full_like(Y_std, np.nan, dtype=np.float32)
    fold_id = np.full(bundle.n_designs, -1, dtype=np.int32)
    rows = []
    t0 = time.time()
    torch.manual_seed(args.seed)
    for k, (train_idx, val_idx) in enumerate(folds, start=1):
        model = EnvelopeMLP(input_dim=X.shape[1], hidden=tuple(args.hidden),
                            output_dim=Y.shape[1])
        log.info("Fold %d/%d: train=%d, val=%d (model params=%d)",
                 k, len(folds), len(train_idx), len(val_idx), model.n_params)
        if args.smoke:
            log.info("  shapes: train_x=%s, train_y=%s, val_x=%s, val_y=%s",
                     X_std[train_idx].shape, Y_std[train_idx].shape,
                     X_std[val_idx].shape, Y_std[val_idx].shape)
        hist = train_one_fold(
            model,
            X_std[train_idx], Y_std[train_idx],
            X_std[val_idx], Y_std[val_idx],
            epochs=args.epochs, lr=args.lr,
            log_every=max(1, args.epochs // 10),
            fold_name=f"f{k}",
        )
        histories.append(asdict(hist))

        # Out-of-fold predictions (back to physical units)
        pred_std = predict(model, X_std[val_idx])
        oof_pred_std[val_idx] = pred_std
        oof_pred[val_idx] = unstandardise(pred_std, y_mean, y_std)
        fold_id[val_idx] = k

        weights_path = out_dir / f"weights_fold{k}.pt"
        torch.save(model.state_dict(), weights_path)
        # Per-fold metrics in physical units
        true_phys = Y[val_idx]
        pred_phys = oof_pred[val_idx]
        per_tgt_rmse = rmse(pred_phys, true_phys, axis=0)
        per_tgt_mape = mape(pred_phys, true_phys, axis=0)
        per_tgt_r2 = r2(pred_phys, true_phys, axis=0)
        rows.append({
            "fold": k, "n_val": int(len(val_idx)),
            "rmse_N_max_rpm": float(per_tgt_rmse[0]),
            "rmse_T_max_Nm":  float(per_tgt_rmse[1]),
            "mape_N_max_rpm_%": float(per_tgt_mape[0]),
            "mape_T_max_Nm_%":  float(per_tgt_mape[1]),
            "final_train_loss": hist.train_loss[-1],
            "final_val_loss":   hist.val_loss[-1],
        })

    elapsed = time.time() - t0
    log.info("Training done in %.1f s", elapsed)

    # Aggregate metrics across all folds using out-of-fold preds (each design once).
    have_pred = fold_id >= 0
    if not np.any(have_pred):
        log.error("No predictions produced")
        return 4
    overall = {
        "fold": "OVERALL",
        "n_val": int(np.sum(have_pred)),
        "rmse_N_max_rpm": float(rmse(oof_pred[have_pred, 0], Y[have_pred, 0])),
        "rmse_T_max_Nm":  float(rmse(oof_pred[have_pred, 1], Y[have_pred, 1])),
        "mape_N_max_rpm_%": float(mape(oof_pred[have_pred, 0], Y[have_pred, 0])),
        "mape_T_max_Nm_%":  float(mape(oof_pred[have_pred, 1], Y[have_pred, 1])),
        "final_train_loss": float(np.mean([r["final_train_loss"] for r in rows])),
        "final_val_loss":   float(np.mean([r["final_val_loss"] for r in rows])),
    }
    # R^2 per target (OOF)
    r2_per = r2(oof_pred[have_pred], Y[have_pred], axis=0)
    log.info("OOF metrics (physical units):")
    log.info("  N_max: rmse=%.1f rpm   mape=%.2f%%   r2=%.3f",
             overall["rmse_N_max_rpm"], overall["mape_N_max_rpm_%"], float(r2_per[0]))
    log.info("  T_max: rmse=%.3f Nm    mape=%.2f%%   r2=%.3f",
             overall["rmse_T_max_Nm"],  overall["mape_T_max_Nm_%"],  float(r2_per[1]))

    # Write artefacts.
    rows.append(overall)
    pd.DataFrame(rows).to_csv(out_dir / "cv_metrics.csv", index=False)
    suffix = "_smoke" if args.smoke else ""
    loss_curves(histories, out_dir / f"loss_curves{suffix}.png",
                title=f"Phase 5A -- envelope-net  ({len(folds)} folds, {args.epochs} epochs)")
    pred_vs_actual(oof_pred[have_pred], Y[have_pred], TARGET_NAMES,
                   out_dir / f"pred_vs_actual{suffix}.png")
    np.savez(
        out_dir / "predictions.npz",
        oof_pred=oof_pred,
        oof_pred_std=oof_pred_std,
        oof_true=Y,
        fold_id=fold_id,
        target_names=np.array(TARGET_NAMES, dtype="U16"),
        input_mean=x_mean, input_std=x_std,
        target_mean=y_mean, target_std=y_std,
    )
    log.info("Wrote %s/{cv_metrics.csv, loss_curves%s.png, pred_vs_actual%s.png, predictions.npz}",
             out_dir, suffix, suffix)
    log.info("=== Phase 5A done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
