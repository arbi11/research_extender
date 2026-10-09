"""
dqn_seqto.py -- Deep Q-Network for SeqTO-v2 (C-core actuator).

Replaces Chp4's tabular q_learn_improved.py with a CNN-based Q-function over
the 3-channel [material, flux, boundary] observation produced by SeqTO-v2.
The action space (4 controller moves) and the chromosome-as-action-sequence
formulation are preserved from Chp4 (Flavour A).

Components
----------
* Replay buffer  : fixed-size deque of (s, a, r, s', done) transitions
* Q-network      : shared CNN backbone + 4-way linear Q-head (net_seqto.DQN)
* Target network : Polyak-averaged copy of Q-network (soft update each step)
* Exploration    : epsilon-greedy with multiplicative decay
* Loss           : SmoothL1 between Q(s, a) and r + gamma * max Q_target(s', .)

Outputs (in designs/dqn_run/)
-----------------------------
* training_stats.json   per-episode reward, loss, epsilon, force_N, |Q| size
* episodes/ep_NNN_mask.png  rendered iron mask after every episode
* policy.pt             saved torch state_dict for later replay
* best_mask.npy         iron mask of the highest-reward episode

Runtime
-------
Each step is one FEMM solve plus 18*35 = 630 mo_getb flux samples.  Expect
~1-3 seconds per step on a typical Windows box, so a 50-episode smoke run
of 34 steps is ~1-2 h; a 500-episode production run is ~12-24 h.

Usage
-----
    python dqn_seqto.py                                # 50 episodes (smoke)
    python dqn_seqto.py --episodes 500                 # production
    python dqn_seqto.py --reward-kind mean_b           # Chp4-compat reward
    python dqn_seqto.py --no-flux                      # skip flux channel sampling
"""

import argparse
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
except ImportError:
    print("ERROR: pytorch not installed.  Run: pip install torch")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from seqTO_v2_env import CcoreSeqTOv2Env, ROWS, COLS
from net_seqto    import DQN


# ── Hyperparameters (CLI overridable) ───────────────────────────────────────
ALPHA         = 3e-4         # Adam learning rate
GAMMA         = 0.99         # discount factor
EPSILON_START = 1.0
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.97         # multiplicative per episode
BATCH_SIZE    = 32
REPLAY_SIZE   = 10_000
WARMUP_STEPS  = 200          # collect this many transitions before any gradient step
TARGET_TAU    = 0.005        # Polyak smoothing for the target net
GRAD_CLIP     = 10.0


# ── Replay buffer ───────────────────────────────────────────────────────────

class ReplayBuffer:
    """Fixed-size buffer of (s, a, r, s', done) transitions."""

    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.stack(s).astype(np.float32),
            np.array(a,    dtype=np.int64),
            np.array(r,    dtype=np.float32),
            np.stack(s2).astype(np.float32),
            np.array(done, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ── Atomic write helpers ────────────────────────────────────────────────────

def _atomic_save(obj, path, mode='torch'):
    """Write to <path>.tmp then os.replace -> final.  Survives mid-write kills."""
    path = Path(path)
    tmp  = path.with_suffix(path.suffix + '.tmp')
    if mode == 'torch':
        torch.save(obj, tmp)
    else:
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=2)
    os.replace(tmp, path)


# ── Episode snapshot (matches Chp4 q_learn_improved.py output pattern) ──────

def _save_ep_mask(iron_mask, ep_num, reward, results_dir):
    """Render and save the final iron mask of one episode as PNG."""
    if not _MPL_AVAILABLE:
        return
    ep_dir = Path(results_dir) / 'episodes'
    ep_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(iron_mask, aspect='auto', cmap='Blues',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Ep {ep_num:03d}  -  iron: {int(iron_mask.sum())}/288  '
                 f'-  reward: {reward:.4f}')
    plt.tight_layout()
    plt.savefig(ep_dir / f'ep_{ep_num:03d}_mask.png', dpi=100, bbox_inches='tight')
    plt.close()


# ── Training driver ────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[dqn_seqto] device = {device}")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path  = results_dir / 'checkpoint.pt'
    stats_path = results_dir / 'training_stats.json'

    env = CcoreSeqTOv2Env(
        max_steps     = args.max_steps,
        reward_kind   = args.reward_kind,
        random_start  = True,
        seed          = args.seed,
        base_fem_file = args.base_fem_file,
        tmp_fem       = args.tmp_fem,
    )

    # Q-network and target network
    q_net      = DQN(in_channels=3, action_size=4).to(device)
    target_net = DQN(in_channels=3, action_size=4).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.alpha)
    replay    = ReplayBuffer(REPLAY_SIZE)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    epsilon = EPSILON_START
    total_steps  = 0
    episode_stats = []
    best_reward, best_episode = -float('inf'), -1
    best_mask = None
    start_ep = 1

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if args.resume:
        if not ckpt_path.exists():
            print(f"[dqn_seqto] --resume given but {ckpt_path} not found; aborting")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_args = ckpt['args']
        for key in ('reward_kind', 'max_steps', 'seed'):
            if ck_args[key] != getattr(args, key):
                print(f"[dqn_seqto] resume aborted: {key} mismatch "
                      f"(ckpt={ck_args[key]!r}, cli={getattr(args, key)!r})")
                sys.exit(1)
        if args.episodes <= ckpt['episode']:
            print(f"[dqn_seqto] resume aborted: --episodes={args.episodes} <= "
                  f"completed {ckpt['episode']}; pass a larger value to continue")
            sys.exit(1)

        q_net.load_state_dict(ckpt['q_net'])
        target_net.load_state_dict(ckpt['target_net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        for t in ckpt['replay']:
            replay.push(*t)
        epsilon      = ckpt['epsilon']
        total_steps  = ckpt['total_steps']
        best_reward  = ckpt['best_reward']
        best_episode = ckpt['best_episode']
        best_mask    = ckpt['best_mask']
        start_ep     = ckpt['episode'] + 1
        random.setstate(ckpt['rng']['python'])
        np.random.set_state(ckpt['rng']['numpy'])
        torch.set_rng_state(ckpt['rng']['torch'].cpu()
                            if hasattr(ckpt['rng']['torch'], 'cpu')
                            else ckpt['rng']['torch'])

        if stats_path.exists():
            existing = json.load(open(stats_path)).get('episode_stats', [])
            episode_stats = [e for e in existing if e['episode'] <= ckpt['episode']]
        print(f"[dqn_seqto] resumed from episode {ckpt['episode']} -> {start_ep}..{args.episodes}  "
              f"best={best_reward:.3f} (ep {best_episode})  replay={len(replay)}")

    print("=" * 72)
    print(f"  DQN on SeqTO-v2 C-core  (episodes {start_ep}..{args.episodes} x {args.max_steps} steps)")
    print(f"  Reward kind  : {args.reward_kind}    Random start: True   Double DQN: {args.double}")
    print(f"  alpha={args.alpha}  gamma={GAMMA}  eps {EPSILON_START}->{EPSILON_MIN}  "
          f"(decay {EPSILON_DECAY})")
    print(f"  batch={BATCH_SIZE}  replay={REPLAY_SIZE}  warmup={WARMUP_STEPS}  "
          f"target_tau={TARGET_TAU}")
    print(f"  Output dir   : {results_dir}    checkpoint_every={args.checkpoint_every}")
    print("=" * 72)

    for ep in range(start_ep, args.episodes + 1):
        obs        = env.reset()
        ep_reward  = 0.0
        ep_losses  = []
        ep_force   = 0.0

        for step in range(args.max_steps):
            # ── Epsilon-greedy action ────────────────────────────────────
            if np.random.random() < epsilon:
                action = int(np.random.randint(4))
            else:
                with torch.no_grad():
                    s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    q = q_net(s)
                    action = int(q.argmax(dim=1).item())

            next_obs, reward, done, info = env.step(action)
            replay.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            ep_force   = info.get('force', 0.0)
            total_steps += 1

            # ── Gradient step ────────────────────────────────────────────
            if total_steps > WARMUP_STEPS and len(replay) >= BATCH_SIZE:
                S, A, R, S2, D = replay.sample(BATCH_SIZE)
                S  = torch.from_numpy(S).to(device)
                A  = torch.from_numpy(A).to(device).unsqueeze(1)
                R  = torch.from_numpy(R).to(device)
                S2 = torch.from_numpy(S2).to(device)
                D  = torch.from_numpy(D).to(device)

                q_sa = q_net(S).gather(1, A).squeeze(1)
                with torch.no_grad():
                    if args.double:
                        a_next = q_net(S2).argmax(dim=1, keepdim=True)
                        q_next = target_net(S2).gather(1, a_next).squeeze(1)
                    else:
                        q_next = target_net(S2).max(dim=1).values
                    y      = R + GAMMA * q_next * (1 - D)
                loss = F.smooth_l1_loss(q_sa, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), GRAD_CLIP)
                optimizer.step()

                # Polyak soft update of target net
                with torch.no_grad():
                    for tp, p in zip(target_net.parameters(), q_net.parameters()):
                        tp.data.mul_(1.0 - TARGET_TAU).add_(TARGET_TAU * p.data)

                ep_losses.append(float(loss.item()))

            if done:
                break

        # ── End-of-episode bookkeeping ───────────────────────────────────
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        ep_loss_mean = float(np.mean(ep_losses)) if ep_losses else 0.0

        if ep_reward > best_reward:
            best_reward  = ep_reward
            best_episode = ep
            best_mask    = env.iron_mask.copy()

        episode_stats.append({
            'episode':    ep,
            'reward':     float(ep_reward),
            'force_N':    float(ep_force),
            'loss':       ep_loss_mean,
            'epsilon':    float(epsilon),
            'iron_count': int(env.iron_mask.sum()),
            'best_so_far': float(best_reward),
        })

        _save_ep_mask(env.iron_mask, ep, ep_reward, results_dir)

        if ep % args.log_every == 0 or ep == args.episodes:
            print(f"  ep {ep:3d}/{args.episodes}  "
                  f"reward={ep_reward:9.3f}  force={ep_force:+8.3f}N  "
                  f"loss={ep_loss_mean:7.4f}  eps={epsilon:.3f}  "
                  f"best={best_reward:9.3f} (ep {best_episode})")

            stats = {
                'episode_stats': episode_stats,
                'hyperparams': {
                    'episodes':      args.episodes,
                    'max_steps':     args.max_steps,
                    'alpha':         args.alpha,
                    'gamma':         GAMMA,
                    'epsilon_start': EPSILON_START,
                    'epsilon_min':   EPSILON_MIN,
                    'epsilon_decay': EPSILON_DECAY,
                    'batch_size':    BATCH_SIZE,
                    'replay_size':   REPLAY_SIZE,
                    'warmup_steps':  WARMUP_STEPS,
                    'target_tau':    TARGET_TAU,
                    'double_dqn':    args.double,
                    'reward_kind':   args.reward_kind,
                    'best_episode':  best_episode,
                    'best_reward':   float(best_reward),
                },
            }
            _atomic_save(stats, stats_path, mode='json')

        # ── Periodic checkpoint (atomic) ──────────────────────────────────────
        if ep % args.checkpoint_every == 0 or ep == args.episodes:
            _atomic_save({
                'args': {'reward_kind': args.reward_kind,
                         'max_steps':   args.max_steps,
                         'seed':        args.seed},
                'episode':      ep,
                'q_net':        q_net.state_dict(),
                'target_net':   target_net.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'replay':       list(replay.buf)[-2000:],
                'epsilon':      epsilon,
                'total_steps':  total_steps,
                'best_reward':  float(best_reward),
                'best_episode': best_episode,
                'best_mask':    best_mask,
                'rng': {
                    'python': random.getstate(),
                    'numpy':  np.random.get_state(),
                    'torch':  torch.get_rng_state(),
                },
            }, ckpt_path)

    # Final saves
    torch.save(q_net.state_dict(), results_dir / 'policy.pt')
    if best_mask is not None:
        np.save(results_dir / 'best_mask.npy', best_mask)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_episode}  "
          f"reward={best_reward:.4f}")
    print(f"  Final ε = {epsilon:.4f}  total steps = {total_steps}")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='DQN on SeqTO-v2 C-core')
    p.add_argument('--episodes',      type=int,   default=200,
                   help='training episodes (default 200)')
    p.add_argument('--max-steps',     type=int,   default=34,
                   help='steps per episode (default 34 = matches Chp4)')
    p.add_argument('--alpha',         type=float, default=ALPHA,
                   help='Adam learning rate')
    p.add_argument('--reward-kind',   choices=['force', 'mean_b'], default='force',
                   help='reward signal: force (Newtons, DRL target) or mean_b (Chp4 compat)')
    p.add_argument('--base-fem-file', type=str,   default='ccore_seqto_dqn.fem',
                   help='this run\'s base FEMM geometry (auto-built on first run if missing)')
    p.add_argument('--tmp-fem',       type=str,   default='_tmp_solve_v2_dqn.fem',
                   help='scratch FEMM file used by mi_analyse (distinct from a2c\'s '
                        'default so concurrent DQN + A2C runs don\'t collide)')
    p.add_argument('--results-dir',   type=str,   default='designs/dqn_run',
                   help='output directory for stats / masks / policy')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--log-every',     type=int,   default=1,
                   help='print + write training_stats.json every N episodes')
    p.add_argument('--checkpoint-every', type=int, default=5,
                   help='atomic checkpoint.pt every N episodes (default 5)')
    p.add_argument('--resume',        action='store_true',
                   help='resume from checkpoint.pt in --results-dir; '
                        'reward_kind/max_steps/seed must match the checkpoint')
    p.add_argument('--no-double', dest='double', action='store_false',
                   help='disable Double DQN (use vanilla max-target instead)')
    p.set_defaults(double=True)
    return p.parse_args()


def main():
    args = _parse_args()
    train(args)


if __name__ == '__main__':
    main()
