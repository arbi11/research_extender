"""
q_learn_improved.py  -  Tabular Q-learning for C-core SeqTO-v1

Reconstructed: the original script is not in the Notion export. This version is
rebuilt from the excerpts quoted in 06_mdp_implementation.ipynb (6-feature
state encoder, choose_action, q_update, train loop, hyperparameter table and
artifact list) and from q_learn_synrm.py, which describes itself as a mirror of
this script for the SynRM environment.

State : compact 6-feature tuple, each feature discretised to N_BINS=3 bins
          1. pos_r_bin     - controller row in the design window
          2. pos_c_bin     - controller column
          3. iron_frac_bin - iron cells deposited so far, relative to K2
          4. arm_iron_bin  - fraction of iron in armature-facing cols (19-25)
          5. vert_dist_bin - fraction of iron in the top half of the window
          6. step_bin      - episode progress
        3^6 = 729 max states.
Reward : seqTO_env.calculate_reward()  (mean |B| at 6 armature points x 1000),
         accumulated over the episode.

Artifacts (designs/q_improved/):
    training_stats_improved.json   per-episode reward, epsilon, Q-table size
    episodes/ep_NNN_mask.png       iron mask at the end of each episode
    episodes/ep_NNN_final.fem      FEMM model of the final step of each episode
    q_table.npz                    learned Q-table
    checkpoint.pkl                 full training state after the last finished
                                   episode (Q-table, epsilon, histories, RNG state)

Usage
-----
    python q_learn_improved.py                 # 200 episodes x 34 steps (default)
    python q_learn_improved.py --episodes 20   # smoke test
    python q_learn_improved.py --resume        # continue from checkpoint.pkl; the
                                               # result is identical to an
                                               # uninterrupted run
"""

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

try:
    import seqTO_env as cenv
    from seqTO_env import (CcoreSeqTOEnv, ROWS, COLS, COIL_LEN, COIL1_COLS, COIL2_COLS,
                           ARM_ROWS, ARM_COLS, DW_ROWS, DW_COLS, K1, K2, is_design_cell)
except ImportError as e:
    print(f"ERROR: failed to import seqTO_env: {e}")
    print("       Run this script from the Chapter 4 content directory")
    sys.exit(1)


# ── Hyperparameters (Section 6.6) ────────────────────────────────────────────
N_EPISODES     = 200
MAX_STEPS      = 34        # matches the GA chromosome length
ALPHA          = 0.1
GAMMA          = 0.9
EPSILON_START  = 1.0
EPSILON_MIN    = 0.05
EPSILON_DECAY  = 0.97
N_BINS         = 3
RANDOM_START   = True
CHECKPOINT_EVERY = 5

RESULTS_DIR = os.path.join(cenv.DESIGNS_DIR, 'q_improved')

R_RANGE = DW_ROWS[1] - DW_ROWS[0]                  # 15
C_RANGE = DW_COLS[1] - DW_COLS[0]                  # 21
ARM_ZONE_START = DW_COLS[0] + (2 * C_RANGE // 3)   # col 19 - armature-facing zone
TOP_HALF_END   = DW_ROWS[0] + R_RANGE // 2         # row 7  - top/bottom split

Q = defaultdict(lambda: np.zeros(4))


# ── State encoder (verbatim logic quoted in notebook 06) ─────────────────────

def _bin(val, lo, hi, n):
    """Map val in [lo, hi] to integer bin index 0..n-1."""
    if hi <= lo:
        return 0
    return min(int((val - lo) * n / (hi - lo)), n - 1)


def get_state(env):
    """Compact 6-feature state tuple computed from the C-core env."""
    iron_mask  = env.iron_mask
    iron_count = int(np.sum(iron_mask))
    r_bin      = _bin(env.pos_r, DW_ROWS[0], DW_ROWS[1] - 1, N_BINS)
    c_bin      = _bin(env.pos_c, DW_COLS[0], DW_COLS[1] - 1, N_BINS)
    iron_b     = _bin(iron_count, 0, K2, N_BINS)
    arm_iron   = int(np.sum(iron_mask[DW_ROWS[0]:DW_ROWS[1], ARM_ZONE_START:DW_COLS[1]]))
    arm_b      = _bin(arm_iron / max(iron_count, 1), 0.0, 1.0, N_BINS)
    top_iron   = int(np.sum(iron_mask[DW_ROWS[0]:TOP_HALF_END, DW_COLS[0]:DW_COLS[1]]))
    vert_b     = _bin(top_iron / max(iron_count, 1), 0.0, 1.0, N_BINS)
    step_b     = _bin(env.step_count, 0, MAX_STEPS, N_BINS)
    return (r_bin, c_bin, iron_b, arm_b, vert_b, step_b)


def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    return int(np.argmax(Q[state]))


def q_update(state, action, reward, next_state, done):
    target = reward if done else reward + GAMMA * np.max(Q[next_state])
    Q[state][action] += ALPHA * (target - Q[state][action])


def random_start(env):
    """Put the controller on a random valid cell of the design window."""
    while True:
        r = int(np.random.randint(DW_ROWS[0], DW_ROWS[1]))
        c = int(np.random.randint(DW_COLS[0], DW_COLS[1]))
        if env._check_position(r, c) == 0:
            env.pos_r, env.pos_c = r, c
            return


# ── Per-episode artifacts ────────────────────────────────────────────────────

CLRS = ['#D5D8DC', '#2980B9', '#E67E22', '#27AE60', '#2C3E50', '#FEF9E7', '#FAD7A0']


def _region_map(iron_mask):
    region = np.zeros((ROWS, COLS), dtype=int)
    for i in range(ROWS):
        for j in range(COLS):
            if i < COIL_LEN and COIL1_COLS[0] <= j < COIL1_COLS[1]:                    region[i, j] = 1
            elif i < COIL_LEN and COIL2_COLS[0] <= j < COIL2_COLS[1]:                  region[i, j] = 2
            elif ARM_ROWS[0] <= i < ARM_ROWS[1] and ARM_COLS[0] <= j < ARM_COLS[1]:    region[i, j] = 3
            elif is_design_cell(i, j):                                                  region[i, j] = 4 if iron_mask[i, j] else 5
            elif DW_ROWS[0] <= i < DW_ROWS[1] and DW_COLS[0] <= j < DW_COLS[1]:        region[i, j] = 6
    return region


def _save_ep_snapshot(env, ep_num, total_reward, results_dir):
    ep_dir = os.path.join(results_dir, 'episodes')
    os.makedirs(ep_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(_region_map(env.iron_mask), cmap=ListedColormap(CLRS), vmin=0, vmax=6,
              origin='upper', aspect='equal')
    ax.plot(env.pos_c, env.pos_r, 'rx', markersize=12, markeredgewidth=2)
    ax.set_title(f'Ep {ep_num:03d}  -  iron: {int(env.iron_mask.sum())} cells  -  '
                 f'cum reward: {total_reward:.0f}')
    ax.set_xlabel('column'); ax.set_ylabel('row')
    ax.legend(handles=[mpatches.Patch(fc='#2C3E50', label='Iron'),
                       mpatches.Patch(fc='#FEF9E7', ec='grey', lw=0.5, label='Air (design window)'),
                       mpatches.Patch(fc='#2980B9', label='Left coil'),
                       mpatches.Patch(fc='#E67E22', label='Right coil'),
                       mpatches.Patch(fc='#27AE60', label='Armature')],
              fontsize=7, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(ep_dir, f'ep_{ep_num:03d}_mask.png'), dpi=110, bbox_inches='tight')
    plt.close()

    # calculate_reward() saves each solve to _TMP_FEM; after the last step it
    # holds the episode's final topology, so keep a copy of it.
    if os.path.exists(cenv._TMP_FEM):
        shutil.copyfile(cenv._TMP_FEM, os.path.join(ep_dir, f'ep_{ep_num:03d}_final.fem'))


# ── Checkpoint / resume ──────────────────────────────────────────────────────

CHECKPOINT_FILE = 'checkpoint.pkl'


def _save_checkpoint(results_dir, ep, epsilon, episode_rewards, epsilon_history,
                     q_size_history, best_reward, best_ep_num):
    """Write the full training state after episode `ep` (atomic replace)."""
    state = {
        'episode':         ep,
        'epsilon':         epsilon,
        'q':               {k: v.copy() for k, v in Q.items()},
        'episode_rewards': episode_rewards,
        'epsilon_history': epsilon_history,
        'q_size_history':  q_size_history,
        'best_reward':     best_reward,
        'best_ep_num':     best_ep_num,
        'rng_state':       np.random.get_state(),
    }
    path = os.path.join(results_dir, CHECKPOINT_FILE)
    with open(path + '.tmp', 'wb') as f:
        pickle.dump(state, f)
    os.replace(path + '.tmp', path)


def _load_checkpoint(results_dir):
    path = os.path.join(results_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        print(f"ERROR: --resume given but {path} does not exist")
        sys.exit(1)
    with open(path, 'rb') as f:
        state = pickle.load(f)
    Q.clear()
    Q.update(state['q'])
    np.random.set_state(state['rng_state'])
    return state


# ── Training loop ────────────────────────────────────────────────────────────

def train(n_episodes=N_EPISODES, results_dir=RESULTS_DIR, resume=False):
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 72)
    print("  Tabular Q-learning  -  C-core SeqTO-v1")
    print(f"  State space  : {N_BINS}^6 = {N_BINS ** 6} max states  (sparse dict)")
    print(f"  Features     : pos_r, pos_c, iron_frac, arm_iron (cols >= {ARM_ZONE_START}), "
          f"top_iron (rows < {TOP_HALF_END}), step")
    print(f"  Random start : {RANDOM_START}")
    print(f"  Episodes     : {n_episodes}    Steps/ep : {MAX_STEPS}")
    print(f"  alpha={ALPHA}  gamma={GAMMA}  eps {EPSILON_START}->{EPSILON_MIN}  "
          f"(decay {EPSILON_DECAY})")
    print(f"  Budget       : K1={K1} <= iron <= K2={K2}")
    print(f"  Output dir   : {results_dir}")
    print("=" * 72)

    env             = CcoreSeqTOEnv(max_steps=MAX_STEPS)
    epsilon         = EPSILON_START
    episode_rewards = []
    epsilon_history = []
    q_size_history  = []
    best_reward     = -float('inf')
    best_ep_num     = -1
    start_ep        = 1

    if resume:
        ck              = _load_checkpoint(results_dir)
        epsilon         = ck['epsilon']
        episode_rewards = ck['episode_rewards']
        epsilon_history = ck['epsilon_history']
        q_size_history  = ck['q_size_history']
        best_reward     = ck['best_reward']
        best_ep_num     = ck['best_ep_num']
        start_ep        = ck['episode'] + 1
        print(f"  Resumed after episode {ck['episode']}  (|Q|={len(Q)}, eps={epsilon:.3f}, "
              f"best={best_reward:.2f} at ep {best_ep_num})", flush=True)

    for ep in range(start_ep, n_episodes + 1):
        env.reset()
        if RANDOM_START:
            random_start(env)
        state        = get_state(env)
        total_reward = 0.0

        for step in range(MAX_STEPS):
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

        _save_checkpoint(results_dir, ep, epsilon, episode_rewards, epsilon_history,
                         q_size_history, best_reward, best_ep_num)

        if ep % CHECKPOINT_EVERY == 0 or ep == n_episodes:
            stats = {
                'episode_rewards': episode_rewards,
                'epsilon_history': epsilon_history,
                'q_size_history':  q_size_history,
                'hyperparams': {
                    'n_episodes':    n_episodes,
                    'max_steps':     MAX_STEPS,
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
                    'k1':            K1,
                    'k2':            K2,
                    'best_episode':  best_ep_num,
                    'best_reward':   float(best_reward),
                },
            }
            with open(os.path.join(results_dir, 'training_stats_improved.json'), 'w') as f:
                json.dump(stats, f, indent=2)

            print(f"  ep {ep:3d}/{n_episodes}  reward={total_reward:9.2f}  "
                  f"best={best_reward:9.2f} (ep {best_ep_num})  "
                  f"eps={epsilon:.3f}  |Q|={len(Q)}  iron={int(env.iron_mask.sum())}", flush=True)

    np.savez(os.path.join(results_dir, 'q_table.npz'),
             keys=np.array(list(Q.keys()), dtype=object),
             values=np.array(list(Q.values())),
             epsilon=epsilon)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_ep_num}  reward = {best_reward:.4f}")
    print(f"  Final |Q| = {len(Q)} states  "
          f"({100 * len(Q) / (N_BINS ** 6):.1f}% coverage of {N_BINS ** 6} max)")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)
    return best_reward, best_ep_num


def _parse_args():
    p = argparse.ArgumentParser(description='Tabular Q-learning for C-core SeqTO-v1')
    p.add_argument('--episodes', type=int, default=N_EPISODES,
                   help=f'training episodes (default {N_EPISODES})')
    p.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                   help=f'output dir (default {RESULTS_DIR})')
    p.add_argument('--seed', type=int, default=42, help='RNG seed')
    p.add_argument('--resume', action='store_true',
                   help=f'continue from <results-dir>/{CHECKPOINT_FILE}')
    return p.parse_args()


def main():
    args = _parse_args()
    np.random.seed(args.seed)          # replaced by the saved RNG state on --resume
    train(args.episodes, args.results_dir, resume=args.resume)


if __name__ == '__main__':
    main()
