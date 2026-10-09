"""Multi-design dataset orchestrator.

Runs Phases 1 -> 2 -> 3 -> 4 for one motor design in a single FEMM session and
saves the per-design outputs under a dedicated directory. Aggregates designs
into a master NPZ for the ML stage.

The per-design pipeline:
    Phase 1   build geometry + mesh + 9-point characterization sweep
              -> save samples CSV, fit lambda_d/lambda_q/T polynomials
    Phase 2   pure-Python trajectory solve on the fits
              -> save envelope CSV
    Phase 3   re-use FEMM session (geometry still loaded) for operating-point
              sweep with loss + eta
              -> save eta-points CSV
    Phase 4   interpolate eta scatter onto a dense grid; save NPZ + PNG

Master dataset (outputs/dataset/master_dataset.npz):
    param_names     (n_params,)         which geometry knobs varied
    design_params   (n_designs, n_params)  vector per design
    N_max_rpm       (n_designs,)         peak speed of envelope
    T_max_Nm        (n_designs,)         peak torque of envelope
    N_norm_grid     (n_norm_speed,)      common normalized speed axis [0, 1]
    T_norm_grid     (n_norm_torque,)     common normalized torque axis [0, 1]
    eta_norm_grids  (n_designs, n_norm_torque, n_norm_speed)
                                         efficiency maps on the common
                                         normalized grid (NaN outside envelope)
"""

from __future__ import annotations
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from .config import Config, FSCWGeometry, FSCWWindingPattern, IPMGeometry
from .control_strategies import TrajectorySolver, OperatingPoint
from .design_space import (
    DESIGN_PARAM_RANGES,
    FSCW_DESIGN_PARAM_RANGES,
    geometry_from_params,
    geometry_from_params_fscw,
    is_feasible,
    is_feasible_fscw,
    lhs_designs,
    lhs_designs_fscw,
    param_names,
    fscw_param_names,
)
from .flux_map_fit import fit_flux_maps, fit_torque, fit_quality
from .geometry_ipm import build_ipm
from .geometry_fscw import build_fscw
from .stage1_characterize import run_sparse_sweep
from .stage3_operating_points import run_phase3_sweep
from .stage4_efficiency_map import (
    interpolate_eta_map,
    save_grid_npz,
    plot_efficiency_map,
)


log = logging.getLogger(__name__)


N_NORM_SPEED = 80         # resolution of the common normalized-axis grid
N_NORM_TORQUE = 60


@dataclass
class DesignResult:
    design_idx: int
    params: np.ndarray
    N_max_rpm: float
    T_max_Nm: float
    eta_norm_grid: np.ndarray         # shape (N_NORM_TORQUE, N_NORM_SPEED)
    elapsed_s: float
    n_phase3_pts: int


def _close_femm_doc() -> None:
    try:
        import femm  # noqa: WPS433
        femm.mi_close()
    except Exception:
        # Not fatal; the next design will openfemm + newdocument anyway.
        log.debug("femm.mi_close() failed (probably nothing open)")


def _normalized_eta_grid(
    eta_masked: np.ndarray,
    N_grid: np.ndarray,
    T_grid: np.ndarray,
    envelope_T_at_N: np.ndarray,
) -> np.ndarray:
    """Resample a (T, N) eta grid onto a common (T/T_max(N), N/N_max) axis.

    Each design's eta-map is defined over (N, T) with N in [N_min_d, N_max_d]
    and T in [0, T_max(N)]. The paper normalises each design's speed and
    torque to [0, 1] before training the network. We do the same: produce a
    dense map on (n_norm_torque, n_norm_speed) where the axes are
    N_norm = N / N_max and T_norm = T / T_max(N). Outside the envelope is NaN.

    envelope_T_at_N may be supplied as a 1D array (preferred, shape (n_speed,))
    or as the 2D mesh that interpolate_eta_map happens to return; in the latter
    case the first row is used (all rows are identical by construction).
    """
    # Tolerate both 1D and 2D envelope input.
    env = np.asarray(envelope_T_at_N)
    if env.ndim == 2:
        env = env[0]

    # Interpolator for eta_masked on the original axes; replace NaN with 0
    # inside the interpolator and re-mask afterwards.
    eta_safe = np.where(np.isnan(eta_masked), 0.0, eta_masked)
    interp = RegularGridInterpolator(
        (T_grid, N_grid), eta_safe, bounds_error=False, fill_value=np.nan,
    )

    N_norm_axis = np.linspace(0.0, 1.0, N_NORM_SPEED)
    T_norm_axis = np.linspace(0.0, 1.0, N_NORM_TORQUE)

    N_max = float(N_grid.max())
    N_actual = N_norm_axis * N_max                              # (n_norm_speed,)
    T_env_at = np.interp(N_actual, N_grid, env)                 # (n_norm_speed,)

    T_actual = T_norm_axis[:, None] * T_env_at[None, :]         # (n_norm_torque, n_norm_speed)
    N_actual_2d = np.broadcast_to(N_actual[None, :], T_actual.shape)
    points = np.column_stack([T_actual.ravel(), N_actual_2d.ravel()])
    out = interp(points).reshape(T_actual.shape)

    # Columns at zero-envelope speeds are not meaningful.
    out[:, T_env_at <= 0] = np.nan
    return out


def run_one_design(
    design_idx: int,
    params: np.ndarray,
    out_dir: Path,
    n_phase3_speeds: int = 6,
    n_phase3_torques: int = 5,
    motor_type: str = "ipm",
) -> Optional[DesignResult]:
    """Run phases 1 -> 4 for one design and write artefacts under out_dir.

    `motor_type` selects between the IPM (default) and FSCW pipelines. The
    geometry builder and design-space helpers are swapped accordingly; the
    Phase 1-4 stages downstream are motor-agnostic.
    """
    t_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("=== Design %d at %s (motor=%s) ===", design_idx, out_dir.name, motor_type)

    if motor_type == "ipm":
        names = param_names()
        geom = geometry_from_params(params, IPMGeometry())
        ok, reason = is_feasible(geom)
        winding_cls = None  # use default WindingPattern from Config
        builder = build_ipm
        fem_filename = "ipm.fem"
    elif motor_type == "fscw":
        names = fscw_param_names()
        geom = geometry_from_params_fscw(params, FSCWGeometry())
        ok, reason = is_feasible_fscw(geom)
        winding_cls = FSCWWindingPattern
        builder = build_fscw
        fem_filename = "fscw.fem"
    else:
        raise ValueError(f"Unknown motor_type={motor_type!r}; use 'ipm' or 'fscw'")

    log.info("Params: %s",
             dict(zip(names, [float(round(x, 2)) for x in params])))

    if not ok:
        log.error("Design %d infeasible: %s", design_idx, reason)
        return None

    cfg_kwargs = {"geom": geom, "motor_type": motor_type}
    if winding_cls is not None:
        cfg_kwargs["winding"] = winding_cls()
    cfg = Config(**cfg_kwargs)

    fem_path = str(out_dir / fem_filename)
    try:
        builder(cfg, fem_path)
    except Exception:
        log.exception("Design %d geometry build failed", design_idx)
        _close_femm_doc()
        return None

    # Phase 1 sweep (also creates the mesh).
    try:
        df_samples = run_sparse_sweep(cfg)
    except Exception:
        log.exception("Design %d Phase 1 sweep failed", design_idx)
        _close_femm_doc()
        return None
    df_samples.to_csv(out_dir / "phase1_samples.csv", index=False)

    # Fits.
    lam_d_fit, lam_q_fit = fit_flux_maps(df_samples)
    torque_fit = fit_torque(df_samples)
    q_T = fit_quality(torque_fit, df_samples["i_d_A"], df_samples["i_q_A"], df_samples["torque_Nm"])
    log.info("Design %d torque fit: R^2=%.3f, RMSE=%.2f", design_idx, q_T["r2"], q_T["rmse"])
    np.savez(
        out_dir / "phase1_fit_coeffs.npz",
        lambda_d_coeffs=lam_d_fit.coeffs,
        lambda_q_coeffs=lam_q_fit.coeffs,
        torque_coeffs=torque_fit.coeffs,
    )

    # Phase 2: envelope.
    solver = TrajectorySolver(
        lam_d_fit=lam_d_fit, lam_q_fit=lam_q_fit, torque_fit=torque_fit,
        pole_pairs=cfg.geom.num_pole_pairs, limits=cfg.limits,
    )
    envelope: List[OperatingPoint] = solver.sweep_speed(n_speeds=80)
    envelope_df = pd.DataFrame([asdict(op) for op in envelope])
    envelope_df.to_csv(out_dir / "phase2_trajectory.csv", index=False)
    feas = envelope_df[envelope_df["regime"] != "infeasible"]
    if feas.empty:
        log.error("Design %d produced empty envelope", design_idx)
        _close_femm_doc()
        return None
    N_max = float(feas["N_rpm"].max())
    T_max = float(feas["T_em_Nm"].max())
    log.info("Design %d envelope: N_max=%.0f rpm, T_max=%.2f Nm", design_idx, N_max, T_max)

    # Phase 3: operating-point FEA (re-uses still-open FEMM doc).
    try:
        df3 = run_phase3_sweep(cfg, solver, envelope_df,
                               n_speeds=n_phase3_speeds, n_torques=n_phase3_torques)
    except Exception:
        log.exception("Design %d Phase 3 sweep failed", design_idx)
        _close_femm_doc()
        return None
    df3.to_csv(out_dir / "phase3_eta_points.csv", index=False)
    if df3.empty:
        log.error("Design %d Phase 3 returned no rows", design_idx)
        _close_femm_doc()
        return None

    # Phase 4: interpolation + plot + normalised grid. Catch failures here too
    # so one bad design can't kill the whole sweep.
    try:
        grid = interpolate_eta_map(df3, envelope_df)
        save_grid_npz(out_dir / "phase4_eta_grid.npz", grid)
        plot_efficiency_map(grid, df3, envelope_df, out_dir / "phase4_efficiency_map.png")
        eta_norm = _normalized_eta_grid(
            grid["eta_masked"],
            grid["N_grid_rpm"],
            grid["T_grid_Nm"],
            grid["envelope_T_at_N_grid"],
        )
    except Exception:
        log.exception("Design %d Phase 4 / normalisation failed", design_idx)
        _close_femm_doc()
        return None

    elapsed = time.time() - t_start
    finite = eta_norm[~np.isnan(eta_norm)]
    eta_lo = float(finite.min()) if finite.size else float("nan")
    eta_hi = float(finite.max()) if finite.size else float("nan")
    log.info("Design %d done in %.1f s, %d phase-3 pts, eta in [%.3f, %.3f]",
             design_idx, elapsed, len(df3), eta_lo, eta_hi)

    # Per-design self-describing snapshot. Lets `--rebuild-master` reconstruct
    # master_dataset.npz from disk without relying on any process state.
    try:
        np.savez(
            out_dir / "params.npz",
            design_idx=np.int64(design_idx),
            params=np.asarray(params, dtype=float),
            param_names=np.array(names, dtype="U24"),
            N_max_rpm=np.float64(N_max),
            T_max_Nm=np.float64(T_max),
            eta_norm_grid=eta_norm.astype(np.float32),
        )
    except Exception:
        log.exception("Failed to write per-design params.npz for design %d", design_idx)

    _close_femm_doc()
    return DesignResult(
        design_idx=design_idx,
        params=np.asarray(params, dtype=float),
        N_max_rpm=N_max,
        T_max_Nm=T_max,
        eta_norm_grid=eta_norm,
        elapsed_s=elapsed,
        n_phase3_pts=len(df3),
    )


def _detect_start_index(dataset_root: Path) -> int:
    """Return one past the highest existing design_NNN folder index, or 1 if none."""
    pat = re.compile(r"^design_(\d+)$")
    max_idx = 0
    if dataset_root.exists():
        for p in dataset_root.iterdir():
            if not p.is_dir():
                continue
            m = pat.match(p.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def aggregate(results: List[DesignResult], master_out: Path, append: bool = True,
              motor_type: str = "ipm") -> None:
    """Save per-design results into the master dataset.

    If `append` is True and `master_out` already exists, merge incoming results
    with the existing rows, deduplicating by design_idx (incoming wins). This
    is the path used by the dataset runner so that extending an existing sweep
    (--start) preserves earlier designs.

    If `append` is False, write a fresh master containing only `results`.
    """
    if not results and not (append and master_out.exists()):
        log.error("No design results to aggregate")
        return

    # Start with incoming (indexed by design_idx for dedup).
    incoming = {r.design_idx: r for r in results}

    # Load existing master if appending.
    existing_rows: dict[int, dict] = {}
    if append and master_out.exists():
        try:
            z = np.load(master_out, allow_pickle=False)
            for k, di in enumerate(z["design_idxs"].tolist()):
                if di in incoming:
                    continue  # incoming overrides
                existing_rows[int(di)] = {
                    "params": z["design_params"][k],
                    "N_max_rpm": float(z["N_max_rpm"][k]),
                    "T_max_Nm": float(z["T_max_Nm"][k]),
                    "eta_norm_grid": z["eta_norm_grids"][k],
                }
            log.info("Merging with %d existing designs from %s", len(existing_rows), master_out)
        except (OSError, KeyError) as exc:
            log.warning("Could not read existing master (%s); writing fresh", exc)
            existing_rows = {}

    # Stable order: by design_idx ascending.
    all_idxs = sorted(set(incoming) | set(existing_rows))
    n_designs = len(all_idxs)
    names = fscw_param_names() if motor_type == "fscw" else param_names()
    n_params = len(names)
    design_params = np.zeros((n_designs, n_params))
    N_max_arr = np.zeros(n_designs)
    T_max_arr = np.zeros(n_designs)
    eta_norm_arr = np.zeros((n_designs, N_NORM_TORQUE, N_NORM_SPEED))
    design_idxs = np.zeros(n_designs, dtype=int)
    for k, di in enumerate(all_idxs):
        if di in incoming:
            r = incoming[di]
            design_params[k] = r.params
            N_max_arr[k] = r.N_max_rpm
            T_max_arr[k] = r.T_max_Nm
            eta_norm_arr[k] = r.eta_norm_grid
        else:
            r = existing_rows[di]
            design_params[k] = r["params"]
            N_max_arr[k] = r["N_max_rpm"]
            T_max_arr[k] = r["T_max_Nm"]
            eta_norm_arr[k] = r["eta_norm_grid"]
        design_idxs[k] = di

    np.savez(
        master_out,
        param_names=np.array(names, dtype="U24"),
        design_idxs=design_idxs,
        design_params=design_params,
        N_max_rpm=N_max_arr,
        T_max_Nm=T_max_arr,
        N_norm_grid=np.linspace(0.0, 1.0, N_NORM_SPEED),
        T_norm_grid=np.linspace(0.0, 1.0, N_NORM_TORQUE),
        eta_norm_grids=eta_norm_arr,
    )
    log.info("Wrote master dataset (%d designs total; %d new) to %s",
             n_designs, len(incoming), master_out)


def rebuild_master_from_disk(dataset_root: Path, motor_type: str = "ipm") -> int:
    """Scan dataset_root for design_NNN/params.npz files and write a fresh
    master_dataset.npz from whatever is found. Designs that have an entry in
    an existing master_dataset.npz but no params.npz on disk (e.g. from a
    sweep that ran before per-design params were saved) are grandfathered in.

    Returns the number of designs in the rebuilt master.
    """
    master_path = dataset_root / "master_dataset.npz"
    pat = re.compile(r"^design_(\d+)$")
    names = fscw_param_names() if motor_type == "fscw" else param_names()
    n_params_expected = len(names)

    rows: dict[int, dict] = {}

    # First, harvest from per-design params.npz (new path).
    for p in sorted(dataset_root.iterdir() if dataset_root.exists() else []):
        if not p.is_dir():
            continue
        m = pat.match(p.name)
        if not m:
            continue
        params_path = p / "params.npz"
        if not params_path.exists():
            continue
        try:
            z = np.load(params_path, allow_pickle=False)
            di = int(z["design_idx"])
            rows[di] = {
                "params": np.asarray(z["params"], dtype=float),
                "N_max_rpm": float(z["N_max_rpm"]),
                "T_max_Nm": float(z["T_max_Nm"]),
                "eta_norm_grid": np.asarray(z["eta_norm_grid"]),
            }
        except (OSError, KeyError) as exc:
            log.warning("Skipping %s (%s)", params_path, exc)

    # Grandfather: keep rows from the existing master that aren't on disk yet.
    if master_path.exists():
        try:
            z = np.load(master_path, allow_pickle=False)
            for k, di in enumerate(z["design_idxs"].tolist()):
                if int(di) in rows:
                    continue
                rows[int(di)] = {
                    "params": np.asarray(z["design_params"][k], dtype=float),
                    "N_max_rpm": float(z["N_max_rpm"][k]),
                    "T_max_Nm": float(z["T_max_Nm"][k]),
                    "eta_norm_grid": np.asarray(z["eta_norm_grids"][k]),
                }
        except (OSError, KeyError) as exc:
            log.warning("Could not read existing master (%s)", exc)

    if not rows:
        log.error("No designs found to rebuild master from")
        return 0

    all_idxs = sorted(rows)
    n = len(all_idxs)
    design_params = np.zeros((n, n_params_expected))
    N_max_arr = np.zeros(n)
    T_max_arr = np.zeros(n)
    eta_norm_arr = np.zeros((n, N_NORM_TORQUE, N_NORM_SPEED))
    design_idxs = np.zeros(n, dtype=int)
    for k, di in enumerate(all_idxs):
        r = rows[di]
        design_params[k] = r["params"]
        N_max_arr[k] = r["N_max_rpm"]
        T_max_arr[k] = r["T_max_Nm"]
        eta_norm_arr[k] = r["eta_norm_grid"]
        design_idxs[k] = di

    np.savez(
        master_path,
        param_names=np.array(names, dtype="U24"),
        design_idxs=design_idxs,
        design_params=design_params,
        N_max_rpm=N_max_arr,
        T_max_Nm=T_max_arr,
        N_norm_grid=np.linspace(0.0, 1.0, N_NORM_SPEED),
        T_norm_grid=np.linspace(0.0, 1.0, N_NORM_TORQUE),
        eta_norm_grids=eta_norm_arr,
    )
    log.info("Rebuilt master (%d designs, idx %d..%d) -> %s",
             n, all_idxs[0], all_idxs[-1], master_path)
    return n


def run_dataset(
    n_designs: int,
    master_seed: int,
    dataset_root: Path,
    n_phase3_speeds: int = 6,
    n_phase3_torques: int = 5,
    checkpoint_every: int = 10,
    start: int = 1,
    motor_type: str = "ipm",
) -> List[DesignResult]:
    """Sample n_designs feasible designs and run the pipeline on each.

    Writes outputs/dataset/master_dataset.npz every `checkpoint_every` designs
    so a long sweep that is interrupted can still be analysed up to the last
    checkpoint. Designs that fail any phase are logged and skipped.

    `start` controls the first design index. Use `start=-1` to auto-detect:
    scans `dataset_root` for the highest existing `design_NNN` folder and
    starts one past it. This is how an interrupted or partial sweep is
    extended without overwriting the work already on disk.

    When `start > 1`, each new run uses a different LHS instance (the
    `lhs_designs` seed already depends on `master_seed`); the n_designs new
    samples are statistically independent of any prior run's samples, and
    the combined dataset covers the design space well even though it is
    no longer a single 'pure' LHS.
    """
    dataset_root.mkdir(parents=True, exist_ok=True)
    if start == -1:
        start = _detect_start_index(dataset_root)
        log.info("Auto-detected start index: %d", start)
    elif start < 1:
        raise ValueError(f"--start must be >= 1 or -1 (auto); got {start}")

    if motor_type == "ipm":
        samples = lhs_designs(n_designs=n_designs, seed=master_seed)
    elif motor_type == "fscw":
        samples = lhs_designs_fscw(n_designs=n_designs, seed=master_seed)
    else:
        raise ValueError(f"Unknown motor_type={motor_type!r}; use 'ipm' or 'fscw'")
    log.info("LHS (%s) produced %d feasible designs (writing design_%d .. design_%d)",
             motor_type, samples.shape[0], start, start + samples.shape[0] - 1)

    master_path = dataset_root / "master_dataset.npz"
    results: List[DesignResult] = []
    n_fail = 0
    for offset, row in enumerate(samples):
        k = start + offset
        out_dir = dataset_root / f"design_{k:03d}"
        try:
            r = run_one_design(
                k, row, out_dir,
                n_phase3_speeds=n_phase3_speeds,
                n_phase3_torques=n_phase3_torques,
                motor_type=motor_type,
            )
        except Exception:
            log.exception("Design %d unhandled exception", k)
            _close_femm_doc()
            r = None
        if r is not None:
            results.append(r)
        else:
            n_fail += 1

        # Periodic checkpoint — robust to interruption on long sweeps.
        finished_this_run = offset + 1
        if results and (finished_this_run % checkpoint_every == 0 or finished_this_run == len(samples)):
            try:
                aggregate(results, master_path, append=True, motor_type=motor_type)
                log.info("Checkpoint: %d/%d succeeded in this run, %d failed; master appended at %s",
                         len(results), finished_this_run, n_fail, master_path)
            except Exception:
                log.exception("Checkpoint aggregation failed at design %d", k)

    log.info("Dataset sweep complete: %d/%d designs succeeded in this run (%d failed)",
             len(results), len(samples), n_fail)
    return results
