"""Post-hoc power-factor map computation.

Walks each design_NNN folder under a dataset root, reads the saved Phase 3
operating-point CSV, reconstructs the dq voltage from already-saved current and
flux, computes the power factor per operating point, interpolates onto the
dense (N, T) grid, and resamples to the common normalised axis. Aggregates the
per-design results into a master NPZ that mirrors `master_dataset.npz`'s schema
but with `pf_norm_grids` in place of `eta_norm_grids`.

No FEMM runs: PF is fully derivable from
    u_d = R_s * i_d - omega_e * lambda_q
    u_q = R_s * i_q + omega_e * lambda_d
    PF  = (u_d*i_d + u_q*i_q) / (|u| * |i|)

Usage (PowerShell):

    python -m data_gen.compute_pf_maps --dataset-root data_gen/outputs/dataset_train
    python -m data_gen.compute_pf_maps --dataset-root data_gen/outputs/dataset_train --n-designs 10  # smoke
"""

from __future__ import annotations
import argparse
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .dataset_runner import N_NORM_SPEED, N_NORM_TORQUE, _normalized_eta_grid
from .design_space import param_names
from .stage4_efficiency_map import interpolate_eta_map


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"compute_pf_{ts}.log"
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


def compute_pf_column(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Compute |power factor| per operating point from saved Phase 3 data.

    Reports the *displacement* power factor magnitude in [0, 1] — industry
    convention is to quote |PF| and report leading/lagging separately. The
    signed dq-power expression that follows from this codebase's Park
    convention is consistent within Phase 2 and Phase 3 but has the opposite
    sign from the textbook motoring formula (a side effect of the Park
    q-axis fix in plan_May22.md §6); the magnitude is unambiguous either way.
    """
    pole_pairs = cfg.geom.num_pole_pairs
    R_s = cfg.limits.R_s_ohm

    i_d = df["i_d_A"].to_numpy()
    i_q = df["i_q_A"].to_numpy()
    lam_d = df["lambda_d_Wb"].to_numpy()
    lam_q = df["lambda_q_Wb"].to_numpy()
    omega_e = pole_pairs * 2.0 * np.pi * df["N_rpm"].to_numpy() / 60.0

    u_d = R_s * i_d - omega_e * lam_q
    u_q = R_s * i_q + omega_e * lam_d

    u_mag = np.hypot(u_d, u_q)
    i_mag = np.hypot(i_d, i_q)
    denom = u_mag * i_mag
    pf_signed = np.where(denom > 1e-9,
                         (u_d * i_d + u_q * i_q) / np.where(denom > 1e-9, denom, 1.0),
                         0.0)
    pf = np.clip(np.abs(pf_signed), 0.0, 1.0)
    return pd.Series(pf, index=df.index, name="pf")


def process_one_design(design_dir: Path, cfg: Config) -> dict | None:
    """Compute the PF map for one design folder. Returns dict with:
        design_idx, params, N_max_rpm, T_max_Nm, pf_norm_grid.
    Returns None if Phase 3 / Phase 2 data is missing or interpolation fails.
    """
    log = logging.getLogger(__name__)
    p3 = design_dir / "phase3_eta_points.csv"
    p2 = design_dir / "phase2_trajectory.csv"
    pn = design_dir / "params.npz"
    if not p3.exists() or not p2.exists():
        log.warning("Skip %s (missing phase2 or phase3 CSV)", design_dir.name)
        return None

    try:
        points_df = pd.read_csv(p3)
        envelope_df = pd.read_csv(p2)
    except pd.errors.EmptyDataError:
        log.warning("Skip %s (unparseable CSV)", design_dir.name)
        return None
    if points_df.empty or envelope_df.empty:
        log.warning("Skip %s (empty CSV)", design_dir.name)
        return None

    points_df["pf"] = compute_pf_column(points_df, cfg)
    # Persist back to disk so users can inspect PF per operating point.
    points_df.to_csv(p3, index=False)

    try:
        grid = interpolate_eta_map(points_df, envelope_df, target_col="pf")
    except Exception:
        log.exception("Interpolation failed for %s", design_dir.name)
        return None

    try:
        pf_norm = _normalized_eta_grid(
            grid["eta_masked"],
            grid["N_grid_rpm"],
            grid["T_grid_Nm"],
            grid["envelope_T_at_N_grid"],
        )
    except Exception:
        log.exception("Normalisation failed for %s", design_dir.name)
        return None

    # Pull design metadata from params.npz if present (added recently); fall
    # back to NaN sentinels otherwise so this script still produces *some*
    # output.
    if pn.exists():
        z = np.load(pn, allow_pickle=False)
        return {
            "design_idx": int(z["design_idx"]),
            "params": np.asarray(z["params"], dtype=float),
            "N_max_rpm": float(z["N_max_rpm"]),
            "T_max_Nm": float(z["T_max_Nm"]),
            "pf_norm_grid": pf_norm.astype(np.float32),
        }
    # No params.npz - extract design idx from folder name and synthesise the rest.
    m = re.match(r"design_(\d+)$", design_dir.name)
    if not m:
        log.warning("Skip %s (cannot parse design idx)", design_dir.name)
        return None
    feas = envelope_df[envelope_df["regime"] != "infeasible"]
    return {
        "design_idx": int(m.group(1)),
        "params": np.full(len(param_names()), np.nan),  # filled in from existing master below
        "N_max_rpm": float(feas["N_rpm"].max()) if not feas.empty else float("nan"),
        "T_max_Nm": float(feas["T_em_Nm"].max()) if not feas.empty else float("nan"),
        "pf_norm_grid": pf_norm.astype(np.float32),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path,
                        default=HERE / "outputs" / "dataset_train",
                        help="Directory containing design_NNN/ subfolders. "
                             "Writes master_pf_dataset.npz here.")
    parser.add_argument("--n-designs", type=int, default=-1,
                        help="Only process the first N designs (smoke testing). "
                             "Default -1 = all.")
    parser.add_argument("--out-name", type=str, default="master_pf_dataset.npz",
                        help="Filename for the aggregated NPZ written under "
                             "--dataset-root.")
    args = parser.parse_args()

    log_path = _setup_logging()
    log = logging.getLogger("compute_pf_maps")
    log.info("=== Computing PF maps ===")
    log.info("Log file: %s", log_path)
    log.info("Dataset root: %s", args.dataset_root)

    cfg = Config()
    log.info("Using default Config: pole_pairs=%d, R_s=%.3f ohm",
             cfg.geom.num_pole_pairs, cfg.limits.R_s_ohm)

    if not args.dataset_root.exists():
        log.error("Dataset root not found: %s", args.dataset_root)
        return 2

    design_dirs = sorted(d for d in args.dataset_root.iterdir()
                         if d.is_dir() and re.match(r"design_(\d+)$", d.name))
    if args.n_designs > 0:
        design_dirs = design_dirs[: args.n_designs]
    log.info("Found %d design folders to process", len(design_dirs))

    # Load the existing master for grandfathered design params (older folders
    # lack params.npz but the master has their LHS vector).
    master_path = args.dataset_root / "master_dataset.npz"
    existing_params: dict[int, np.ndarray] = {}
    if master_path.exists():
        z = np.load(master_path, allow_pickle=False)
        for k, di in enumerate(z["design_idxs"].tolist()):
            existing_params[int(di)] = z["design_params"][k]
        log.info("Grandfathered params for %d designs from existing master",
                 len(existing_params))

    rows: list[dict] = []
    t0 = time.time()
    for k, d in enumerate(design_dirs, 1):
        r = process_one_design(d, cfg)
        if r is None:
            continue
        # Fill in params from the existing master if the per-design params.npz
        # was missing.
        if np.any(np.isnan(r["params"])) and r["design_idx"] in existing_params:
            r["params"] = existing_params[r["design_idx"]]
        rows.append(r)
        if k % 50 == 0 or k == len(design_dirs):
            log.info("  processed %d / %d  (%.1f s elapsed)",
                     k, len(design_dirs), time.time() - t0)

    if not rows:
        log.error("No PF maps produced")
        return 3

    # Drop rows still missing params (no master and no params.npz).
    rows_with_params = [r for r in rows if not np.any(np.isnan(r["params"]))]
    if len(rows_with_params) < len(rows):
        log.warning("Dropping %d designs with missing params",
                    len(rows) - len(rows_with_params))
    rows = rows_with_params

    # Sort by design_idx ascending so the output ordering matches master_dataset.npz.
    rows.sort(key=lambda r: r["design_idx"])
    n = len(rows)
    n_params = len(param_names())
    design_idxs = np.array([r["design_idx"] for r in rows], dtype=int)
    design_params = np.stack([r["params"] for r in rows]).astype(float)
    N_max_arr = np.array([r["N_max_rpm"] for r in rows], dtype=float)
    T_max_arr = np.array([r["T_max_Nm"] for r in rows], dtype=float)
    pf_arr = np.stack([r["pf_norm_grid"] for r in rows]).astype(np.float32)

    out_path = args.dataset_root / args.out_name
    np.savez(
        out_path,
        param_names=np.array(param_names(), dtype="U24"),
        design_idxs=design_idxs,
        design_params=design_params,
        N_max_rpm=N_max_arr,
        T_max_Nm=T_max_arr,
        N_norm_grid=np.linspace(0.0, 1.0, N_NORM_SPEED),
        T_norm_grid=np.linspace(0.0, 1.0, N_NORM_TORQUE),
        pf_norm_grids=pf_arr,
    )
    finite_pf = pf_arr[~np.isnan(pf_arr)]
    log.info("Wrote %s (%d designs)", out_path, n)
    log.info("PF range inside envelope: %.3f .. %.3f", finite_pf.min(), finite_pf.max())
    log.info("PF mean across all cells: %.3f", finite_pf.mean())
    log.info("=== Done in %.1f s ===", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
