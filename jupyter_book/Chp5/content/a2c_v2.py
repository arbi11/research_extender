"""
a2c_v2.py -- Advantage Actor-Critic for SeqTO-v2 with a reward-scale knob.

Same algorithm as a2c_seqto.py (CNN actor-critic, MC returns, entropy schedule,
advantage normalisation, atomic checkpoint/resume).  The only behavioural
difference is a single `--reward-scale` multiplier applied to every step
reward before it enters returns / advantages.

Motivation
----------
On the raw force-reward env, returns are O(1e3-1e4).  The value-loss MSE
then sits at 1e5-1e7, which dominates the shared-backbone gradient under
GRAD_CLIP and starves the policy head.  Scaling rewards by a small constant
(e.g. 0.001 -> returns become O(1-20)) brings value loss into the same
order as the policy gradient, so the entropy bonus can actually keep the
distribution non-degenerate.

Outputs (in designs/a2c_v2_run/ by default)
-------------------------------------------
Same schema as a2c_seqto.py.  `hyperparams.reward_scale` records the
multiplier that was in effect, so cross-run comparisons are unambiguous.
NB: `reward` in episode_stats is the SCALED reward (what the algo actually
saw); multiply by 1 / reward_scale to recover env-native units.

Usage
-----
    python a2c_v2.py                                    # 200 episodes, scale 0.001
    python a2c_v2.py --episodes 500 --reward-scale 0.0005
    python a2c_v2.py --reward-scale 1.0                 # disable scaling (equivalent to a2c_seqto)
    python a2c_v2.py --episodes 400 --resume            # continue past previous cap
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
ALPHA              = 3e-4
GAMMA              = 0.99
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_FINAL = 0.01
ENTROPY_DECAY_EPS  = 100
VALUE_COEF         = 0.5
GRAD_CLIP          = 10.0
REWARD_SCALE       = 0.001          # default: rewards 700-21,000 -> 0.7-21


def _atomic_save(obj, path, mode='torch'):
    path = Path(path)
    tmp  = path.with_suffix(path.suffix + '.tmp')
    if mode == 'torch':
        torch.save(obj, tmp)
    else:
        with open(tmp, 'w') as f:
            json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _entropy_coef(ep, start, final, decay_eps):
    if decay_eps <= 0 or ep > decay_eps:
        return final
    return start + (final - start) * (ep - 1) / decay_eps


def _save_ep_mask(iron_mask, ep_num, reward, results_dir):
    if not _MPL_AVAILABLE:
        return
    ep_dir = Path(results_dir) / 'episodes'
    ep_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.imshow(iron_mask, aspect='auto', cmap='Blues',
              interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f'Ep {ep_num:03d}  -  iron: {int(iron_mask.sum())}/288  '
                 f'-  reward(scaled): {reward:.4f}')
    plt.tight_layout()
    plt.savefig(ep_dir / f'ep_{ep_num:03d}_mask.png', dpi=100, bbox_inches='tight')
    plt.close()


def _compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[a2c_v2] device = {device}")

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

    if args.resume:
        if not ckpt_path.exists():
            print(f"[a2c_v2] --resume given but {ckpt_path} not found; aborting")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_args = ckpt['args']
        for key in ('reward_kind', 'max_steps', 'seed', 'reward_scale'):
            if ck_args.get(key) != getattr(args, key):
                print(f"[a2c_v2] resume aborted: {key} mismatch "
                      f"(ckpt={ck_args.get(key)!r}, cli={getattr(args, key)!r})")
                sys.exit(1)
        if args.episodes <= ckpt['episode']:
            print(f"[a2c_v2] resume aborted: --episodes={args.episodes} <= "
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
        print(f"[a2c_v2] resumed from episode {ckpt['episode']} -> {start_ep}..{args.episodes}  "
              f"best={best_reward:.3f} (ep {best_episode})")

    print("=" * 72)
    print(f"  A2C-v2 on SeqTO-v2 C-core  (episodes {start_ep}..{args.episodes} x {args.max_steps} steps)")
    print(f"  Reward kind  : {args.reward_kind}    reward_scale={args.reward_scale}    Random start: True")
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
            reward = reward * args.reward_scale

            log_probs.append(log_prob.squeeze())
            values.append(v.squeeze())
            rewards.append(reward)
            entropies.append(entropy.squeeze())
            ep_force = info.get('force', 0.0)
            obs = next_obs
            if done:
                break

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
                    'reward_scale':        args.reward_scale,
                    'best_episode':        best_episode,
                    'best_reward':         float(best_reward),
                },
            }
            _atomic_save(stats, stats_path, mode='json')

        if ep % args.checkpoint_every == 0 or ep == args.episodes:
            _atomic_save({
                'args': {'reward_kind':  args.reward_kind,
                         'max_steps':    args.max_steps,
                         'seed':         args.seed,
                         'reward_scale': args.reward_scale},
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

    torch.save(net.state_dict(), results_dir / 'policy.pt')
    if best_mask is not None:
        np.save(results_dir / 'best_mask.npy', best_mask)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_episode}  "
          f"reward(scaled)={best_reward:.4f}  reward(raw)≈{best_reward/args.reward_scale:.1f}")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)


def _parse_args():
    p = argparse.ArgumentParser(description='A2C-v2 (reward-scaled) on SeqTO-v2 C-core')
    p.add_argument('--episodes',      type=int,   default=200)
    p.add_argument('--max-steps',     type=int,   default=34)
    p.add_argument('--alpha',         type=float, default=ALPHA)
    p.add_argument('--reward-scale',  type=float, default=REWARD_SCALE,
                   help='multiply every step reward by this constant '
                        '(default 0.001; pass 1.0 to disable)')
    p.add_argument('--entropy-coef-start', type=float, default=ENTROPY_COEF_START)
    p.add_argument('--entropy-coef-final', type=float, default=ENTROPY_COEF_FINAL)
    p.add_argument('--entropy-decay-eps',  type=int,   default=ENTROPY_DECAY_EPS)
    p.add_argument('--no-advantage-norm', dest='advantage_norm', action='store_false')
    p.set_defaults(advantage_norm=True)
    p.add_argument('--reward-kind',   choices=['force', 'mean_b'], default='force')
    p.add_argument('--base-fem-file', type=str,   default='ccore_seqto_a2c_v2.fem',
                   help='this run\'s base FEMM geometry (auto-built on first run if missing)')
    p.add_argument('--tmp-fem',       type=str,   default='_tmp_solve_v2_a2c_v2.fem',
                   help='scratch FEMM file (distinct so this run does not collide '
                        'with any concurrent a2c_seqto/dqn/ppo run)')
    p.add_argument('--results-dir',   type=str,   default='designs/a2c_v2_run')
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--log-every',     type=int,   default=1)
    p.add_argument('--checkpoint-every', type=int, default=5)
    p.add_argument('--resume',        action='store_true')
    return p.parse_args()


def main():
    args = _parse_args()
    train(args)


if __name__ == '__main__':
    main()
