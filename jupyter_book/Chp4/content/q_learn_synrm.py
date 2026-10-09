"""
q_learn_synrm.py  -  Tabular Q-learning for SynRM SeqTO-v1 (configurable grid)

Mirror of q_learn_improved.py adapted for the SynRM environment in
seqTO_SynRM_env.py.  Uses a compact 6-feature state representation that keeps
the Q-table tractable (3^6 = 729 max states) regardless of design-grid size,
while preserving enough Markov structure for convergent learning.

Grid size is selectable via --grid-size N for any N in [5, 10].  The env
caches a per-grid .fem geometry file (synrm_seqto_NxN.fem); first run at a
new grid size builds it, subsequent runs reuse it.

State : 6-feature tuple, each feature discretised to N_BINS=3 bins
        Features:
          1. pos_r_bin    - controller radial index (0..DD_ROWS-1)
          2. pos_c_bin    - controller angular index (0..DD_COLS-1)
          3. iron_count   - iron budget consumed so far (relative to K2)
          4. outer_iron   - fraction of iron in outer ~40% of radial rows
                            (cells adjacent to the air gap)
          5. left_iron    - fraction of iron in left ~40% of angular cols
                            (the q-axis-side where GA found its optimum)
          6. step_bin     - episode progress (early/mid/late)

State space: 3^6 = 729 states for ALL grid sizes -- the binning insulates the
state-space size from grid resolution.

Fitness  : seqTO_SynRM_env.calculate_reward()  (max |torque| across 6 angles)
Budget   : K1 = max(N_cols, 5)  <=  iron  <=  K2 = ceil(0.8 * N_cells)
           (overridable via --k1 / --k2)

Runtime expectations
--------------------
Each step is one calculate_reward() call = 6 FEMM solves ~= 60-90 s.
Default episodes=50 with max_steps=N*N gives:
   5x5  ->  50 * 25  ~= 1250 calls  ~= 21 hours
   7x7  ->  50 * 49  ~= 2450 calls  ~= 41 hours
   10x10 -> 50 * 100 ~= 5000 calls  ~= 83 hours

For practical SynRM runs, scale --episodes inversely with --grid-size, or
shorten --max-steps below N*N.  See --help for tuning knobs.

Usage
-----
    python q_learn_synrm.py                              # 5x5, 50 ep (default)
    python q_learn_synrm.py --grid-size 7                # 7x7
    python q_learn_synrm.py --grid-size 7 --episodes 20  # 7x7 smoke test
    python q_learn_synrm.py --grid-size 10 --max-steps 50 --episodes 30
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seqTO_SynRM_env as senv
except ImportError as e:
    print(f"ERROR: failed to import seqTO_SynRM_env: {e}")
    print("       Run this script from jupyter_book/Chp4/content/")
    sys.exit(1)


# ── Default hyperparameters (not grid-dependent) ─────────────────────────────
ALPHA          = 0.1
GAMMA          = 0.9
EPSILON_START  = 1.0
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.92
N_BINS         = 3
RANDOM_START   = True
CHECKPOINT_EVERY = 5

DEFAULT_EPISODES = 50

# Sparse Q-dict (filled during training)
Q = defaultdict(lambda: np.zeros(4))


def _bin(val, lo, hi, n):
    if hi <= lo:
        return 0
    return min(int((val - lo) * n / (hi - lo)), n - 1)


def _outer_row_count(dd_rows):
    """Number of rows treated as 'outer' for the state-feature slicing.

    At least 2 rows, but never more than ~40% of the design domain — keeps
    the feature meaningful regardless of grid resolution.
    """
    return max(2, dd_rows * 2 // 5)


def _left_col_count(dd_cols):
    """Number of cols treated as 'left / q-axis side' for state-feature slicing."""
    return max(2, dd_cols * 2 // 5)


def get_state(env):
    """Compact 6-feature state tuple computed from the SynRM env.

    Looks up DD_ROWS/DD_COLS/K2/MAX_STEPS via the senv module so it
    transparently respects whatever grid size configure() set.
    """
    dd_rows, dd_cols = senv.DD_ROWS, senv.DD_COLS
    k2, max_steps    = senv.K2, senv.MAX_STEPS

    iron       = env.iron_mask
    iron_count = int(np.sum(iron))

    r_bin    = _bin(env.pos_r, 0, dd_rows - 1, N_BINS)
    c_bin    = _bin(env.pos_c, 0, dd_cols - 1, N_BINS)
    iron_b   = _bin(iron_count, 0, k2, N_BINS)

    n_outer    = _outer_row_count(dd_rows)
    outer_iron = int(np.sum(iron[-n_outer:, :]))
    outer_frac = outer_iron / max(iron_count, 1)
    outer_b    = _bin(outer_frac, 0.0, 1.0, N_BINS)

    n_left    = _left_col_count(dd_cols)
    left_iron = int(np.sum(iron[:, :n_left]))
    left_frac = left_iron / max(iron_count, 1)
    left_b    = _bin(left_frac, 0.0, 1.0, N_BINS)

    step_b = _bin(env.step_count, 0, max_steps, N_BINS)

    return (r_bin, c_bin, iron_b, outer_b, left_b, step_b)


def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    return int(np.argmax(Q[state]))


def q_update(state, action, reward, next_state, done):
    target = reward if done else reward + GAMMA * np.max(Q[next_state])
    Q[state][action] += ALPHA * (target - Q[state][action])


# ── Per-episode snapshot helper ──────────────────────────────────────────────

def _plot_polar_mask(iron_mask, ep_num, total_reward, save_path):
    """Save a polar plot of the design grid (any DD_ROWS x DD_COLS)."""
    dd_rows, dd_cols = senv.DD_ROWS, senv.DD_COLS
    dr, dtheta       = senv.DR, senv.DTHETA
    r_hub, r_peri    = senv.R_HUB, senv.R_PERI_IN

    fig, ax = plt.subplots(figsize=(5.0, 5.0), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('E'); ax.set_theta_direction(1)
    ax.set_thetamin(0); ax.set_thetamax(90)
    ax.set_ylim(0, r_peri + 4)

    n_arc = 16
    for i in range(dd_rows):
        for j in range(dd_cols):
            r1 = r_hub + i * dr
            r2 = r_hub + (i + 1) * dr
            t1 = np.radians(j * dtheta)
            t2 = np.radians((j + 1) * dtheta)
            theta_arc  = np.linspace(t1, t2, n_arc)
            theta_poly = np.concatenate([theta_arc, theta_arc[::-1]])
            r_poly     = np.concatenate([np.full(n_arc, r1), np.full(n_arc, r2)])
            color = '#2C3E50' if iron_mask[i, j] else '#FEF9E7'
            ax.fill(theta_poly, r_poly, color=color, edgecolor='#7F8C8D', linewidth=0.4)

    ax.plot([np.radians(45)] * 2, [0, r_peri + 2], 'r--', lw=1.0, alpha=0.7)
    ax.text(np.radians(45), r_peri + 3.5, 'd-axis',
            ha='center', va='bottom', fontsize=8, color='red')
    iron_count = int(np.sum(iron_mask))
    ax.set_title(f'Ep {ep_num:03d}  -  iron: {iron_count}/{dd_rows*dd_cols}  -  '
                 f'cum reward: {total_reward:.2f}',
                 fontsize=10, pad=18)
    ax.set_yticks([r_hub, r_peri]); ax.set_yticklabels([f'{r_hub:.1f}', f'{r_peri:.1f}'], fontsize=7)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def _save_ep_snapshot(env, ep_num, total_reward, results_dir):
    ep_dir = os.path.join(results_dir, 'episodes')
    os.makedirs(ep_dir, exist_ok=True)
    _plot_polar_mask(env.iron_mask, ep_num, total_reward,
                     os.path.join(ep_dir, f'ep_{ep_num:03d}_mask.png'))


# ── Main training loop ───────────────────────────────────────────────────────

def train(n_episodes, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    dd_rows, dd_cols = senv.DD_ROWS, senv.DD_COLS
    k1, k2, max_steps = senv.K1, senv.K2, senv.MAX_STEPS

    print("=" * 72)
    print(f"  Tabular Q-learning  -  SynRM SeqTO-v1  ({dd_rows}x{dd_cols} grid)")
    print(f"  State space  : {N_BINS}^6 = {N_BINS ** 6} max states  (sparse dict)")
    print(f"  Features     : pos_r, pos_c, iron_frac, outer_iron, left_iron, step")
    print(f"  outer_iron   : last {_outer_row_count(dd_rows)} radial rows")
    print(f"  left_iron    : first {_left_col_count(dd_cols)} angular cols")
    print(f"  Random start : {RANDOM_START}")
    print(f"  Episodes     : {n_episodes}    Steps/ep : {max_steps}")
    print(f"  alpha={ALPHA}  gamma={GAMMA}  eps {EPSILON_START}->{EPSILON_MIN}  "
          f"(decay {EPSILON_DECAY})")
    print(f"  Budget       : K1={k1} <= iron <= K2={k2}  (of {dd_rows*dd_cols} cells)")
    print(f"  Output dir   : {results_dir}")
    print(f"  Base FEM     : {senv.BASE_FEM_FILE}")
    print("=" * 72)

    env             = senv.SynRMSeqTOEnv(max_steps=max_steps)
    epsilon         = EPSILON_START
    episode_rewards = []
    epsilon_history = []
    q_size_history  = []
    best_reward     = -float('inf')
    best_ep_num     = -1

    for ep in range(1, n_episodes + 1):
        env.reset()
        if RANDOM_START:
            env.pos_r = int(np.random.randint(0, dd_rows))
            env.pos_c = int(np.random.randint(0, dd_cols))
        state        = get_state(env)
        total_reward = 0.0

        for step in range(max_steps):
            action = choose_action(state, epsilon)
            _, reward, done, _ = env.step(action)
            next_state = get_state(env)
            q_update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        episode_rewards.append(float(total_reward))
        epsilon_history.append(float(epsilon))
        q_size_history.append(int(len(Q)))
        _save_ep_snapshot(env, ep, total_reward, results_dir)

        if total_reward > best_reward:
            best_reward = total_reward
            best_ep_num = ep

        if ep % CHECKPOINT_EVERY == 0 or ep == n_episodes:
            stats = {
                'episode_rewards': episode_rewards,
                'epsilon_history': epsilon_history,
                'q_size_history':  q_size_history,
                'hyperparams': {
                    'dd_rows':       dd_rows,
                    'dd_cols':       dd_cols,
                    'n_episodes':    n_episodes,
                    'max_steps':     max_steps,
                    'alpha':         ALPHA,
                    'gamma':         GAMMA,
                    'epsilon_start': EPSILON_START,
                    'epsilon_min':   EPSILON_MIN,
                    'epsilon_decay': EPSILON_DECAY,
                    'n_bins':        N_BINS,
                    'random_start':  RANDOM_START,
                    'state':         'compact_6feature',
                    'state_dim':     6,
                    'max_states':    N_BINS ** 6,
                    'k1':            k1,
                    'k2':            k2,
                    'outer_rows':    _outer_row_count(dd_rows),
                    'left_cols':     _left_col_count(dd_cols),
                    'best_episode':  best_ep_num,
                    'best_reward':   float(best_reward),
                },
            }
            with open(os.path.join(results_dir, 'training_stats_improved.json'), 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"  ep {ep:3d}/{n_episodes}  reward={total_reward:8.2f}  "
                  f"best={best_reward:8.2f} (ep {best_ep_num})  "
                  f"eps={epsilon:.3f}  |Q|={len(Q)}")

    np.savez(os.path.join(results_dir, 'q_table.npz'),
             keys=np.array(list(Q.keys()), dtype=object),
             values=np.array(list(Q.values())),
             epsilon=epsilon)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_ep_num}  reward = {best_reward:.4f}")
    print(f"  Final |Q| = {len(Q)} states  "
          f"({100*len(Q)/(N_BINS**6):.1f}% coverage of {N_BINS**6} max)")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)
    return best_reward, best_ep_num


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='Tabular Q-learning for SynRM SeqTO-v1 '
                                            '(configurable grid)')
    p.add_argument('--grid-size', type=int, default=5,
                   help='design domain size N for an N x N grid (default 5, range [5,10])')
    p.add_argument('--episodes',  type=int, default=DEFAULT_EPISODES,
                   help=f'training episodes (default {DEFAULT_EPISODES})')
    p.add_argument('--max-steps', type=int, default=None,
                   help='override per-episode step cap (default = grid_size^2)')
    p.add_argument('--k1', type=int, default=None,
                   help='override min iron cells (default = max(grid_size, 5))')
    p.add_argument('--k2', type=int, default=None,
                   help='override max iron cells (default = floor(0.8 * grid_size^2))')
    p.add_argument('--results-dir', type=str, default=None,
                   help=f'override output dir (default designs/q_synrm_NxN_improved)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed')
    return p.parse_args()


def main():
    args = _parse_args()
    if not (5 <= args.grid_size <= 10):
        print(f"WARNING: grid-size={args.grid_size} is outside the validated range [5, 10]")

    # Reconfigure the env BEFORE instantiation
    senv.configure(args.grid_size, args.grid_size,
                   max_steps=args.max_steps, k1=args.k1, k2=args.k2)

    np.random.seed(args.seed)

    results_dir = args.results_dir or os.path.join(
        senv.DESIGNS_DIR, f'q_synrm_{args.grid_size}x{args.grid_size}_improved'
    )
    train(args.episodes, results_dir)


if __name__ == '__main__':
    main()
