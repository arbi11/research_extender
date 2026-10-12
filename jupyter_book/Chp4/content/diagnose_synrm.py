"""
diagnose_synrm.py  -  Per-angle torque diagnostics for the 5x5 SynRM SeqTO-v1 rotor

Reconstructed: the original script is not in the Notion export and no part of
its source survives. This version is rebuilt from what the book says about it:
  * seqTO_SynRM_env.calculate_reward() docstring: the diagnostics showed that
    per-angle torques at offsets >= 15 deg mech are insensitive to rotor
    topology, which is why the reward takes max() over the 6 rotor angles;
  * 07_results_and_future_directions.ipynb, which reads
    designs/synrm_diag/synrm_diag_results.json and expects the baselines below,
    each with a 'torques_abs' list of the 6 per-angle |torque| values.

Baselines (5x5 polar grid, rows = radial bands, cols = 18 deg angular sectors):
    A_empty            all air                                   (0 cells)
    B_full_iron        all 25 cells iron                         (25 cells)
    D_conv_2barrier    bands i=0, 2, 4 iron (test_conventional)  (15 cells)
    E_rib_j0_at_9deg   radial rib in sector j=0 (centre 9 deg)   (5 cells)
    F_rib_j2_at_45deg  radial rib in sector j=2 (d-axis, 45 deg) (5 cells)
    G_rib_j4_at_81deg  radial rib in sector j=4 (centre 81 deg)  (5 cells)

Each baseline is one calculate_reward() call (6 FEMM solves, one per rotor angle).

Usage
-----
    python diagnose_synrm.py
"""

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seqTO_SynRM_env as senv
except ImportError as e:
    print(f"ERROR: failed to import seqTO_SynRM_env: {e}")
    sys.exit(1)

OUT_DIR = os.path.join(senv.DESIGNS_DIR, 'synrm_diag')


def _rib(j):
    m = np.zeros((senv.DD_ROWS, senv.DD_COLS), dtype=int)
    m[:, j] = 1
    return m


def baselines():
    conv = np.zeros((senv.DD_ROWS, senv.DD_COLS), dtype=int)
    conv[0, :] = 1
    conv[2, :] = 1
    conv[4, :] = 1
    return {
        'A_empty':           np.zeros((senv.DD_ROWS, senv.DD_COLS), dtype=int),
        'B_full_iron':       np.ones((senv.DD_ROWS, senv.DD_COLS), dtype=int),
        'D_conv_2barrier':   conv,
        'E_rib_j0_at_9deg':  _rib(0),
        'F_rib_j2_at_45deg': _rib(2),
        'G_rib_j4_at_81deg': _rib(4),
    }


def main():
    senv.configure(5, 5)
    if not os.path.exists(senv.BASE_FEM_FILE):
        print(f"Base geometry not found.  Building {senv.BASE_FEM_FILE}...")
        senv.build_synrm_geometry()
    os.makedirs(OUT_DIR, exist_ok=True)

    angles = [float(a) for a in senv.ROTOR_ANGLES_MECH]
    print("=" * 72)
    print(f"  SynRM per-angle torque diagnostics  ({senv.DD_ROWS}x{senv.DD_COLS} grid)")
    print(f"  Rotor angles (mech deg): {angles}")
    print(f"  Base FEM: {senv.BASE_FEM_FILE}    Output: {OUT_DIR}")
    print("=" * 72)

    results = {'rotor_angles_mech_deg': angles, 'grid': [senv.DD_ROWS, senv.DD_COLS]}
    for name, mask in baselines().items():
        reward, torques = senv.calculate_reward(mask)
        results[name] = {
            'iron_cells':  int(mask.sum()),
            'mask':        mask.tolist(),
            'torques_abs': [float(t) for t in torques],
            'max_torque':  float(reward),
            'mean_torque': float(np.mean(torques)),
        }
        print(f"  {name:<20} iron={int(mask.sum()):2d}  max={reward:8.4f}  "
              f"per-angle={[round(float(t), 4) for t in torques]}", flush=True)

    # Topology sensitivity per angle: spread of |torque| across the baselines
    names = [k for k in results if isinstance(results[k], dict) and 'torques_abs' in results[k]]
    table = np.array([results[n]['torques_abs'] for n in names])
    spread = (table.max(axis=0) - table.min(axis=0)).tolist()
    results['per_angle_spread_across_baselines'] = spread

    with open(os.path.join(OUT_DIR, 'synrm_diag_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for n, row in zip(names, table):
        ax.plot(angles, row, marker='o', label=f"{n} ({results[n]['iron_cells']}c)")
    ax.set_xlabel('Rotor angle (mech deg)')
    ax.set_ylabel('|torque|  (weighted stress tensor, quarter model)')
    ax.set_title('SynRM per-angle torque for diagnostic baselines')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'synrm_diag_per_angle.png'), dpi=120)
    plt.close()

    print()
    print(f"  Spread of |torque| across baselines, per angle: {[round(s, 4) for s in spread]}")
    print(f"  Results -> {os.path.join(OUT_DIR, 'synrm_diag_results.json')}")


if __name__ == '__main__':
    main()
