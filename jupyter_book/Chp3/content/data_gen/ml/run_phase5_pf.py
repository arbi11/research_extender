"""Phase 5C: train a MapDecoder to predict the power-factor map.

Supports both training from scratch and fine-tuning from a pretrained
η-`MapDecoder` checkpoint, with a configurable training-set size for the
sample-efficiency sweep used in Chapter 5 (transfer learning).

Usage:
    # From scratch on 50 designs:
    python -m data_gen.ml.run_phase5_pf --n-train 50 --seed 0

    # Fine-tune from the IPM η checkpoint:
    python -m data_gen.ml.run_phase5_pf --n-train 50 --seed 0 \
        --pretrained data_gen/outputs/phase5_map_train/weights_fold1.pt
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

from .data import standardise
from .metrics import masked_mae, masked_r2, masked_rmse
from .models import MAPDECODER_PRESETS, MapDecoder
from .train import train_one_fold_masked


HERE = Path(__file__).resolve().parent
DATA_GEN_ROOT = HERE.parent
LOG_DIR = DATA_GEN_ROOT / "logs"
DEFAULT_OUT_DIR = DATA_GEN_ROOT / "outputs" / "phase5_pf_train"
DEFAULT_DATASET = DATA_GEN_ROOT / "outputs" / "dataset_train" / "master_pf_dataset.npz"


def _setup_logging(log_name: str) -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{log_name}_{ts}.log"
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


def load_pf_master(npz_path: Path):
    z = np.load(npz_path, allow_pickle=False)
    X = np.asarray(z["design_params"], dtype=np.float32)
    Y = np.asarray(z["pf_norm_grids"], dtype=np.float32)
    M = ~np.isnan(Y)
    return X, Y, M, [str(s) for s in z["param_names"]]


def _predict_map(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.as_tensor(x, dtype=torch.float32)
        return model(x_t).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-train", type=int, default=100,
                        help="number of designs to use for training. The rest "
                             "become the held-out validation set.")
    parser.add_argument("--seed", type=int, default=0,
                        help="controls the train/val split AND the model init.")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="Path to a MapDecoder state_dict to load before "
                             "training. Without it, train from scratch.")
    parser.add_argument("--freeze", type=str, default="none",
                        choices=["none", "head", "decoder"],
                        help="freeze a subset of the pretrained weights "
                             "(only meaningful with --pretrained).")
    parser.add_argument("--reset-output", action="store_true",
                        help="Re-initialise the final ConvTranspose (decoder[2]) "
                             "after loading the pretrained checkpoint. Critical "
                             "when the pretrained and new targets have different "
                             "output distributions (e.g. eta around 0.7 vs PF "
                             "around 0.1) - keeps useful feature extractor, "
                             "lets the task-specific output head re-learn.")
    parser.add_argument("--run-id", type=str, default=None,
                        help="subdirectory under --out-dir. Auto-generated if absent.")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-epoch logging (used by the sweep driver).")
    parser.add_argument("--arch-preset", type=str, default="baseline",
                        choices=sorted(MAPDECODER_PRESETS.keys()),
                        help="MapDecoder architecture variant — MUST match the "
                             "architecture the --pretrained checkpoint was "
                             "trained with (default: baseline).")
    args = parser.parse_args()
    arch_kwargs = MAPDECODER_PRESETS[args.arch_preset]

    if args.quiet:
        # Send logs only to file, not stdout.
        log_path = _setup_logging("phase5_pf_quiet")
        # Detach stdout handler:
        for h in list(logging.getLogger().handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.getLogger().removeHandler(h)
    else:
        log_path = _setup_logging("phase5_pf")
    log = logging.getLogger("run_phase5_pf")
    log.info("=== Phase 5C: PF map training (n_train=%d, seed=%d, mode=%s) ===",
             args.n_train, args.seed,
             "pretrained" if args.pretrained else "scratch")
    log.info("Log file: %s", log_path)

    if not args.dataset.exists():
        log.error("Master PF dataset not found: %s", args.dataset)
        return 2

    X, Y, M, param_names = load_pf_master(args.dataset)
    n_designs = X.shape[0]
    log.info("Loaded %d designs from %s", n_designs, args.dataset)

    if args.n_train >= n_designs:
        log.error("n-train (%d) must be < n_designs (%d)", args.n_train, n_designs)
        return 3

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_designs)
    train_idx = np.sort(perm[: args.n_train])
    val_idx = np.sort(perm[args.n_train:])

    # Per-fold-style standardisation: stats computed on train rows only.
    X_train_std, x_mean, x_std = standardise(X[train_idx])
    X_val_std = (X[val_idx] - x_mean) / x_std

    log.info("train=%d, val=%d", len(train_idx), len(val_idx))
    log.info("input stats: mean=%s, std=%s",
             np.round(x_mean, 3).tolist(), np.round(x_std, 3).tolist())

    torch.manual_seed(args.seed)
    model = MapDecoder(input_dim=6, **arch_kwargs)
    log.info("Architecture preset: %s (kwargs=%s, params=%d)",
             args.arch_preset, arch_kwargs, model.n_params)

    if args.pretrained is not None:
        if not args.pretrained.exists():
            log.error("Pretrained checkpoint not found: %s", args.pretrained)
            return 4
        sd = torch.load(args.pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
        log.info("Loaded pretrained weights from %s", args.pretrained)

        if args.reset_output:
            # Re-initialise the final ConvTranspose layer to break the inherited
            # output-distribution bias from the pretraining task. Using
            # decoder[-1] (last layer) so this stays correct across architecture
            # presets (baseline / width-2x / width-4x all have the output
            # ConvTranspose at decoder[2], but depth+1 puts it at decoder[4]).
            out_layer = model.decoder[-1]
            torch.nn.init.kaiming_uniform_(out_layer.weight, a=5 ** 0.5)
            if out_layer.bias is not None:
                fan_in = out_layer.weight.shape[0] * out_layer.weight.shape[2] ** 2
                bound = 1 / fan_in ** 0.5
                torch.nn.init.uniform_(out_layer.bias, -bound, bound)
            log.info("Re-initialised final output ConvTranspose (decoder[2])")

        if args.freeze == "head":
            for p in model.head.parameters():
                p.requires_grad = False
            log.info("Froze model.head")
        elif args.freeze == "decoder":
            for p in model.decoder.parameters():
                p.requires_grad = False
            log.info("Froze model.decoder")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model: %d total params, %d trainable", model.n_params, trainable)

    t0 = time.time()
    hist = train_one_fold_masked(
        model,
        X_train_std, Y[train_idx], M[train_idx],
        X_val_std, Y[val_idx], M[val_idx],
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        log_every=max(1, args.epochs // 8),
        fold_name="pf",
    )
    elapsed = time.time() - t0

    val_pred = _predict_map(model, X_val_std)
    val_rmse = masked_rmse(val_pred, Y[val_idx], M[val_idx])
    val_mae = masked_mae(val_pred, Y[val_idx], M[val_idx])
    val_r2 = masked_r2(val_pred, Y[val_idx], M[val_idx])
    log.info("Val masked metrics: rmse=%.4f  mae=%.4f  r2=%.3f  (%.1f s)",
             val_rmse, val_mae, val_r2, elapsed)

    # Default run-id encodes the experimental knobs.
    if args.run_id is None:
        if args.pretrained:
            mode_tag = "pretrained-reset" if args.reset_output else "pretrained-naive"
            if args.freeze != "none":
                mode_tag = f"{mode_tag}_freeze-{args.freeze}"
        else:
            mode_tag = "scratch"
        args.run_id = f"n{args.n_train:04d}_s{args.seed}_{mode_tag}"

    out_dir = args.out_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    if args.pretrained:
        mode_str = "pretrained-reset" if args.reset_output else "pretrained-naive"
    else:
        mode_str = "scratch"
    pd.DataFrame([{
        "n_train": args.n_train,
        "seed": args.seed,
        "mode": mode_str,
        "freeze": args.freeze if args.pretrained else "n/a",
        "val_rmse": val_rmse,
        "val_mae": val_mae,
        "val_r2": val_r2,
        "epochs": args.epochs,
        "wall_s": elapsed,
        "pretrained_path": str(args.pretrained) if args.pretrained else "",
    }]).to_csv(out_dir / "metrics.csv", index=False)
    np.savez(
        out_dir / "predictions.npz",
        val_pred=val_pred,
        val_true=Y[val_idx],
        val_mask=M[val_idx],
        val_idx=val_idx,
        train_idx=train_idx,
        train_loss=np.asarray(hist.train_loss, dtype=np.float32),
        val_loss=np.asarray(hist.val_loss, dtype=np.float32),
        input_mean=x_mean, input_std=x_std,
    )
    log.info("Wrote %s/{weights.pt, metrics.csv, predictions.npz}", out_dir)
    log.info("=== Done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
