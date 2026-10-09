"""
a2c_seqto.py -- Advantage Actor-Critic for SeqTO-v2 (C-core actuator).

Policy-based counterpart to dqn_seqto.py.  Shares the same CNN backbone
(net_seqto.ActorCritic) but adds a value head and learns a stochastic policy
over the 4-action controller directly.

Algorithm: A2C (synchronous, single-environment).  Each episode produces a
rollout; advantages are computed from Monte-Carlo returns minus the value
baseline; gradients are taken once per episode.

Components
----------
* Actor-Critic net : shared CNN backbone + policy head (4 logits) + value head (1)
* Returns          : Monte-Carlo G_t = sum_{k>=t} gamma^{k-t} * r_k   (terminal-bootstrapped)
* Advantage        : A_t = G_t - V(s_t)
* Losses           : policy = -log pi(a|s) * A.detach()
                     value  = MSE(V(s), G)
                     entropy bonus = -beta * H(pi)

Outputs (in designs/a2c_run/)
-----------------------------
* training_stats.json   per-episode reward, policy_loss, value_loss, entropy, force_N
* episodes/ep_NNN_mask.png  rendered iron mask after every episode
* policy.pt             saved torch state_dict
* best_mask.npy         iron mask of the highest-reward episode

Runtime
-------
Same per-step cost as DQN (one FEMM solve + 630 flux samples).  Gradient is
computed once per episode (after rollout), so per-episode wall-clock is
roughly the same as DQN.  ~1-2 h for a 50-episode smoke run.

Usage
-----
    python a2c_seqto.py                                # 50 episodes (smoke)
    python a2c_seqto.py --episodes 500                 # production
    python a2c_seqto.py --reward-kind mean_b           # Chp4-compat reward
    python a2c_seqto.py --entropy-coef 0.0             # disable entropy bonus
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Categorical
except ImportError:
    print("ERROR: pytorch not installed.  Run: pip install torch")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from seqTO_v2_env import CcoreSeqTOv2Env, ROWS, COLS
from net_seqto    import ActorCritic


# ── Hyperparameters (CLI overridable) ───────────────────────────────────────
ALPHA              = 3e-4          # Adam learning rate
GAMMA              = 0.99          # discount factor
ENTROPY_COEF_START = 0.05          # entropy bonus at episode 1
ENTROPY_COEF_FINAL = 0.01          # entropy bonus once decay is finished
ENTROPY_DECAY_EPS  = 100           # linear decay over the first N episodes
VALUE_COEF         = 0.5           # weight on value-function loss
GRAD_CLIP          = 10.0


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


def _entropy_coef(ep, start, final, decay_eps):
    """Linear schedule from `start` at ep=1 to `final` at ep=decay_eps+1; flat after."""
    if decay_eps <= 0 or ep > decay_eps:
        return final
    return start + (final - start) * (ep - 1) / decay_eps


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

def _compute_returns(rewards, gamma):
    """Compute discounted Monte-Carlo returns G_t for each step.

    G_t = r_t + gamma * G_{t+1};  G_T = r_T  (terminal bootstrap = 0).
    """
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[a2c_seqto] device = {device}")

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

    net       = ActorCritic(in_channels=3, action_size=4).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.alpha)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episode_stats = []
    best_reward, best_episode = -float('inf'), -1
    best_mask = None
    start_ep  = 1

    # ── Resume from checkpoint ──────────────────────────────────────────────
    if args.resume:
        if not ckpt_path.exists():
            print(f"[a2c_seqto] --resume given but {ckpt_path} not found; aborting")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_args = ckpt['args']
        for key in ('reward_kind', 'max_steps', 'seed'):
            if ck_args[key] != getattr(args, key):
                print(f"[a2c_seqto] resume aborted: {key} mismatch "
                      f"(ckpt={ck_args[key]!r}, cli={getattr(args, key)!r})")
                sys.exit(1)
        if args.episodes <= ckpt['episode']:
            print(f"[a2c_seqto] resume aborted: --episodes={args.episodes} <= "
                  f"completed {ckpt['episode']}; pass a larger value to continue")
            sys.exit(1)

        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
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
        print(f"[a2c_seqto] resumed from episode {ckpt['episode']} -> {start_ep}..{args.episodes}  "
              f"best={best_reward:.3f} (ep {best_episode})")

    print("=" * 72)
    print(f"  A2C on SeqTO-v2 C-core  (episodes {start_ep}..{args.episodes} x {args.max_steps} steps)")
    print(f"  Reward kind  : {args.reward_kind}    Random start: True")
    print(f"  alpha={args.alpha}  gamma={GAMMA}  value_coef={VALUE_COEF}")
    print(f"  entropy_coef: {args.entropy_coef_start} -> {args.entropy_coef_final} "
          f"over {args.entropy_decay_eps} eps   advantage_norm={args.advantage_norm}")
    print(f"  Output dir   : {results_dir}    checkpoint_every={args.checkpoint_every}")
    print("=" * 72)

    for ep in range(start_ep, args.episodes + 1):
        obs       = env.reset()
        log_probs = []
        values    = []
        rewards   = []
        entropies = []
        ep_force  = 0.0

        for step in range(args.max_steps):
            s     = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            logits, v = net(s)
            dist  = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy  = dist.entropy()

            next_obs, reward, done, info = env.step(int(action.item()))

            log_probs.append(log_prob.squeeze())
            values.append(v.squeeze())
            rewards.append(reward)
            entropies.append(entropy.squeeze())
            ep_force = info.get('force', 0.0)
            obs = next_obs
            if done:
                break

        # ── Compute returns and advantages ─────────────────────────────────
        returns = _compute_returns(rewards, GAMMA)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t  = torch.stack(values)
        log_probs_t = torch.stack(log_probs)
        entropies_t = torch.stack(entropies)

        advantages = returns_t - values_t.detach()
        if args.advantage_norm and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ent_coef = _entropy_coef(ep, args.entropy_coef_start,
                                 args.entropy_coef_final, args.entropy_decay_eps)
        policy_loss = -(log_probs_t * advantages).mean()
        value_loss  = F.mse_loss(values_t, returns_t)
        entropy_b   = entropies_t.mean()
        loss        = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy_b

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
        optimizer.step()

        ep_reward = float(sum(rewards))
        if ep_reward > best_reward:
            best_reward  = ep_reward
            best_episode = ep
            best_mask    = env.iron_mask.copy()

        episode_stats.append({
            'episode':       ep,
            'reward':        ep_reward,
            'force_N':       float(ep_force),
            'policy_loss':   float(policy_loss.item()),
            'value_loss':    float(value_loss.item()),
            'entropy':       float(entropy_b.item()),
            'iron_count':    int(env.iron_mask.sum()),
            'best_so_far':   float(best_reward),
        })

        _save_ep_mask(env.iron_mask, ep, ep_reward, results_dir)

        if ep % args.log_every == 0 or ep == args.episodes:
            print(f"  ep {ep:3d}/{args.episodes}  "
                  f"reward={ep_reward:9.3f}  force={ep_force:+8.3f}N  "
                  f"pol={policy_loss.item():+7.4f}  val={value_loss.item():7.4f}  "
                  f"H={entropy_b.item():.3f} (β={ent_coef:.3f})  "
                  f"best={best_reward:9.3f} (ep {best_episode})")

            stats = {
                'episode_stats': episode_stats,
                'hyperparams': {
                    'episodes':            args.episodes,
                    'max_steps':           args.max_steps,
                    'alpha':               args.alpha,
                    'gamma':               GAMMA,
                    'entropy_coef_start':  args.entropy_coef_start,
                    'entropy_coef_final':  args.entropy_coef_final,
                    'entropy_decay_eps':   args.entropy_decay_eps,
                    'value_coef':          VALUE_COEF,
                    'advantage_norm':      args.advantage_norm,
                    'reward_kind':         args.reward_kind,
                    'best_episode':        best_episode,
                    'best_reward':         float(best_reward),
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
                'net':          net.state_dict(),
                'optimizer':    optimizer.state_dict(),
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
    torch.save(net.state_dict(), results_dir / 'policy.pt')
    if best_mask is not None:
        np.save(results_dir / 'best_mask.npy', best_mask)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_episode}  reward={best_reward:.4f}")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='A2C on SeqTO-v2 C-core')
    p.add_argument('--episodes',      type=int,   default=200,
                   help='training episodes (default 200)')
    p.add_argument('--max-steps',     type=int,   default=34,
                   help='steps per episode (default 34 = matches Chp4)')
    p.add_argument('--alpha',         type=float, default=ALPHA,
                   help='Adam learning rate')
    p.add_argument('--entropy-coef-start', type=float, default=ENTROPY_COEF_START,
                   help='entropy bonus at episode 1 (default 0.05)')
    p.add_argument('--entropy-coef-final', type=float, default=ENTROPY_COEF_FINAL,
                   help='entropy bonus once decay finishes (default 0.01)')
    p.add_argument('--entropy-decay-eps',  type=int,   default=ENTROPY_DECAY_EPS,
                   help='linearly decay entropy coef over the first N episodes (default 100)')
    p.add_argument('--no-advantage-norm', dest='advantage_norm', action='store_false',
                   help='disable per-batch advantage normalisation')
    p.set_defaults(advantage_norm=True)
    p.add_argument('--reward-kind',   choices=['force', 'mean_b'], default='force',
                   help='reward signal: force (Newtons, DRL target) or mean_b (Chp4 compat)')
    p.add_argument('--base-fem-file', type=str,   default='ccore_seqto_a2c.fem',
                   help='this run\'s base FEMM geometry (auto-built on first run if missing)')
    p.add_argument('--tmp-fem',       type=str,   default='_tmp_solve_v2_a2c.fem',
                   help='scratch FEMM file used by mi_analyse (distinct from dqn\'s '
                        'default so concurrent DQN + A2C runs don\'t collide)')
    p.add_argument('--results-dir',   type=str,   default='designs/a2c_run',
                   help='output directory for stats / masks / policy')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--log-every',     type=int,   default=1,
                   help='print + write training_stats.json every N episodes')
    p.add_argument('--checkpoint-every', type=int, default=5,
                   help='atomic checkpoint.pt every N episodes (default 5)')
    p.add_argument('--resume',        action='store_true',
                   help='resume from checkpoint.pt in --results-dir; '
                        'reward_kind/max_steps/seed must match the checkpoint')
    return p.parse_args()


def main():
    args = _parse_args()
    train(args)


if __name__ == '__main__':
    main()
