"""Phase 5B driver: train a decoder MLP+ConvTranspose to predict the
normalised efficiency map from the 6-dim design parameter vector.

Usage:
    python -m data_gen.ml.run_phase5_map --smoke
    python -m data_gen.ml.run_phase5_map --epochs 800 --folds 5
    python -m data_gen.ml.run_phase5_map --loo
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
)
from .metrics import masked_mae, masked_r2, masked_rmse
from .models import MAPDECODER_PRESETS, MapDecoder
from .plots import loss_curves, map_comparison
from .train import train_one_fold_masked


HERE = Path(__file__).resolve().parent
DATA_GEN_ROOT = HERE.parent
LOG_DIR = DATA_GEN_ROOT / "logs"
DEFAULT_OUT_DIR = DATA_GEN_ROOT / "outputs" / "phase5_map"
DEFAULT_DATASET = DATA_GEN_ROOT / "outputs" / "dataset" / "master_dataset.npz"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"phase5_map_{ts}.log"
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


def _predict_map(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32)
        return model(x_t).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--loo", action="store_true",
                        help="leave-one-out CV (overrides --folds)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true",
                        help="5 epochs, 1 fold, light output for wiring verification")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for CV metrics, weights, plots, and "
                             "predictions.npz. Defaults to outputs/phase5_map. "
                             "Use a different dir (e.g. outputs/phase5_map_train) "
                             "to isolate runs.")
    parser.add_argument("--n-show", type=int, default=4,
                        help="how many designs to show in the map comparison plot")
    parser.add_argument("--arch-preset", type=str, default="baseline",
                        choices=sorted(MAPDECODER_PRESETS.keys()),
                        help="MapDecoder architecture variant. "
                             "'baseline' = the original 157k-param topology; "
                             "width2x / width4x scale bottleneck channels; "
                             "depth1 inserts a same-resolution conv block "
                             "between the two ConvTranspose stages.")
    args = parser.parse_args()
    out_dir = args.out_dir
    arch_kwargs = MAPDECODER_PRESETS[args.arch_preset]

    log_path = _setup_logging()
    log = logging.getLogger("run_phase5_map")
    log.info("=== Phase 5B: map-net training ===")
    log.info("Log file: %s", log_path)

    try:
        bundle = load_dataset(args.dataset)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    log.info("Loaded dataset: %d designs, params=%s",
             bundle.n_designs, bundle.param_names)

    if bundle.n_designs < 4:
        log.error("Need at least 4 designs for CV; have %d", bundle.n_designs)
        return 3

    X = bundle.design_params
    Y = bundle.eta_norm_grids                # (N, 60, 80), NaN outside envelope
    M = bundle.eta_mask                       # (N, 60, 80) bool
    X_std, x_mean, x_std = standardise(X)
    log.info("Input  stats: mean=%s, std=%s",
             np.round(x_mean, 3).tolist(), np.round(x_std, 3).tolist())
    log.info("Envelope coverage: %.1f%% of cells inside (mean across designs)",
             float(M.mean()) * 100.0)

    if args.smoke:
        args.epochs = 5
        folds = _build_folds(bundle.n_designs, argparse.Namespace(
            loo=False, folds=2, seed=args.seed))
        folds = folds[:1]
        log.info("[smoke] 1 fold, 5 epochs")
    else:
        folds = _build_folds(bundle.n_designs, args)
        log.info("CV: %d folds (%s)", len(folds), "loo" if args.loo else f"k={args.folds}")

    out_dir.mkdir(parents=True, exist_ok=True)

    histories = []
    oof_pred = np.full_like(Y, np.nan, dtype=np.float32)
    fold_id = np.full(bundle.n_designs, -1, dtype=np.int32)
    rows = []
    t0 = time.time()
    torch.manual_seed(args.seed)

    log.info("Architecture preset: %s  (kwargs=%s)", args.arch_preset, arch_kwargs)
    for k, (train_idx, val_idx) in enumerate(folds, start=1):
        model = MapDecoder(input_dim=X.shape[1], **arch_kwargs)
        log.info("Fold %d/%d: train=%d, val=%d (model params=%d)",
                 k, len(folds), len(train_idx), len(val_idx), model.n_params)
        if args.smoke:
            log.info("  shapes: train_x=%s, train_y=%s, train_mask=%s",
                     X_std[train_idx].shape, Y[train_idx].shape, M[train_idx].shape)

        hist = train_one_fold_masked(
            model,
            X_std[train_idx], Y[train_idx], M[train_idx],
            X_std[val_idx], Y[val_idx], M[val_idx],
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            log_every=max(1, args.epochs // 10),
            fold_name=f"f{k}",
        )
        histories.append(asdict(hist))

        pred = _predict_map(model, X_std[val_idx])
        oof_pred[val_idx] = pred
        fold_id[val_idx] = k

        weights_path = out_dir / f"weights_fold{k}.pt"
        torch.save(model.state_dict(), weights_path)

        # Per-fold masked metrics in eta units
        true_v = Y[val_idx]
        mask_v = M[val_idx]
        rows.append({
            "fold": k,
            "n_val": int(len(val_idx)),
            "masked_rmse_eta": masked_rmse(pred, true_v, mask_v),
            "masked_mae_eta":  masked_mae(pred, true_v, mask_v),
            "masked_r2":       masked_r2(pred, true_v, mask_v),
            "final_train_loss": hist.train_loss[-1],
            "final_val_loss":   hist.val_loss[-1],
        })

    elapsed = time.time() - t0
    log.info("Training done in %.1f s", elapsed)

    have_pred = fold_id >= 0
    if not np.any(have_pred):
        log.error("No predictions produced")
        return 4

    rmse_overall = masked_rmse(oof_pred[have_pred], Y[have_pred], M[have_pred])
    mae_overall = masked_mae(oof_pred[have_pred], Y[have_pred], M[have_pred])
    r2_overall = masked_r2(oof_pred[have_pred], Y[have_pred], M[have_pred])
    log.info("OOF masked metrics: rmse=%.4f  mae=%.4f  r2=%.3f",
             rmse_overall, mae_overall, r2_overall)
    rows.append({
        "fold": "OVERALL",
        "n_val": int(np.sum(have_pred)),
        "masked_rmse_eta": rmse_overall,
        "masked_mae_eta":  mae_overall,
        "masked_r2":       r2_overall,
        "final_train_loss": float(np.mean([r["final_train_loss"] for r in rows])),
        "final_val_loss":   float(np.mean([r["final_val_loss"] for r in rows])),
    })
    pd.DataFrame(rows).to_csv(out_dir / "cv_metrics.csv", index=False)

    suffix = "_smoke" if args.smoke else ""
    loss_curves(histories, out_dir / f"loss_curves{suffix}.png",
                title=f"Phase 5B -- map-net  ({len(folds)} folds, {args.epochs} epochs)")
    map_comparison(
        oof_pred[have_pred], Y[have_pred], M[have_pred],
        out_dir / f"pred_vs_actual_maps{suffix}.png",
        n_designs_to_show=args.n_show,
        seed=args.seed,
    )
    np.savez(
        out_dir / "predictions.npz",
        oof_pred=oof_pred,
        oof_true=Y,
        mask=M,
        fold_id=fold_id,
        input_mean=x_mean, input_std=x_std,
    )
    log.info("Wrote %s/{cv_metrics.csv, loss_curves%s.png, pred_vs_actual_maps%s.png, predictions.npz}",
             out_dir, suffix, suffix)
    log.info("=== Phase 5B done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
