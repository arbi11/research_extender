"""
ppo_seqto.py -- Proximal Policy Optimization for SeqTO-v2 (C-core actuator).

Same env + ActorCritic backbone as a2c_seqto.py; differences:

  * GAE-lambda advantages (lambda=0.95 default)
  * Clipped surrogate objective (eps=0.2 default)
  * K epochs of minibatch SGD per rollout (default K=4)
  * Per-batch advantage normalization
  * Entropy schedule (linear from start -> final over N episodes)

Rollouts are one episode long (max_steps=34), so a "rollout" is exactly one
episode and each update sees ~34 transitions; with K=4 epochs that's 4
gradient steps per episode (vs. A2C's 1).

Outputs (in designs/ppo_run/)
-----------------------------
* training_stats.json   per-episode reward, policy_loss, value_loss, entropy, kl, force_N
* episodes/ep_NNN_mask.png  rendered iron mask after every episode
* checkpoint.pt         atomic, written every --checkpoint-every episodes
* policy.pt             final state_dict
* best_mask.npy         iron mask of the highest-reward episode

Resume
------
    python ppo_seqto.py --episodes 200            # fresh run
    python ppo_seqto.py --episodes 400 --resume   # continue past previous cap
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
GAE_LAMBDA         = 0.95
CLIP_EPS           = 0.2
PPO_EPOCHS         = 4
MINIBATCH_SIZE     = 34            # one full episode by default
ENTROPY_COEF_START = 0.05
ENTROPY_COEF_FINAL = 0.01
ENTROPY_DECAY_EPS  = 100
VALUE_COEF         = 0.5
GRAD_CLIP          = 10.0


def _atomic_save(obj, path, mode='torch'):
    """Write to <path>.tmp then os.replace -> final."""
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
                 f'-  reward: {reward:.4f}')
    plt.tight_layout()
    plt.savefig(ep_dir / f'ep_{ep_num:03d}_mask.png', dpi=100, bbox_inches='tight')
    plt.close()


def _compute_gae(rewards, values, dones, gamma, lam):
    """Generalized Advantage Estimation.

    `values` has length T+1 (bootstrap value at the terminal state appended).
    Returns advantages (length T) and returns G_t = A_t + V(s_t).
    """
    T = len(rewards)
    advantages = [0.0] * T
    last_gae   = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t]
        delta    = rewards[t] + gamma * values[t + 1] * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = [a + v for a, v in zip(advantages, values[:-1])]
    return advantages, returns


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[ppo_seqto] device = {device}")

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
            print(f"[ppo_seqto] --resume given but {ckpt_path} not found; aborting")
            sys.exit(1)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        ck_args = ckpt['args']
        for key in ('reward_kind', 'max_steps', 'seed'):
            if ck_args[key] != getattr(args, key):
                print(f"[ppo_seqto] resume aborted: {key} mismatch "
                      f"(ckpt={ck_args[key]!r}, cli={getattr(args, key)!r})")
                sys.exit(1)
        if args.episodes <= ckpt['episode']:
            print(f"[ppo_seqto] resume aborted: --episodes={args.episodes} <= "
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
        print(f"[ppo_seqto] resumed from episode {ckpt['episode']} -> {start_ep}..{args.episodes}  "
              f"best={best_reward:.3f} (ep {best_episode})")

    print("=" * 72)
    print(f"  PPO on SeqTO-v2 C-core  (episodes {start_ep}..{args.episodes} x {args.max_steps} steps)")
    print(f"  Reward kind  : {args.reward_kind}    Random start: True")
    print(f"  alpha={args.alpha}  gamma={GAMMA}  gae_lambda={args.gae_lambda}  "
          f"clip={args.clip_eps}  K={args.ppo_epochs}  mb={args.minibatch_size}")
    print(f"  entropy_coef: {args.entropy_coef_start} -> {args.entropy_coef_final} "
          f"over {args.entropy_decay_eps} eps")
    print(f"  Output dir   : {results_dir}    checkpoint_every={args.checkpoint_every}")
    print("=" * 72)

    for ep in range(start_ep, args.episodes + 1):
        # ── Rollout: collect one episode ─────────────────────────────────────
        obs = env.reset()
        rollout_obs       = []
        rollout_actions   = []
        rollout_log_probs = []
        rollout_values    = []
        rollout_rewards   = []
        rollout_dones     = []
        ep_force          = 0.0

        for step in range(args.max_steps):
            s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                logits, v = net(s)
                dist  = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_obs, reward, done, info = env.step(int(action.item()))

            rollout_obs.append(obs)
            rollout_actions.append(int(action.item()))
            rollout_log_probs.append(float(log_prob.item()))
            rollout_values.append(float(v.item()))
            rollout_rewards.append(float(reward))
            rollout_dones.append(float(done))
            ep_force = info.get('force', 0.0)

            obs = next_obs
            if done:
                break

        # Bootstrap value at final state (0 if terminated by max_steps cap and
        # the env treats that as terminal; we set non_terminal via `dones`)
        with torch.no_grad():
            s = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            _, v_last = net(s)
            rollout_values.append(float(v_last.item()))

        advantages, returns = _compute_gae(
            rollout_rewards, rollout_values, rollout_dones,
            GAMMA, args.gae_lambda,
        )

        # ── PPO update: K epochs of minibatch SGD ────────────────────────────
        obs_t      = torch.from_numpy(np.stack(rollout_obs)).float().to(device)
        actions_t  = torch.tensor(rollout_actions, dtype=torch.int64, device=device)
        old_lp_t   = torch.tensor(rollout_log_probs, dtype=torch.float32, device=device)
        adv_t      = torch.tensor(advantages, dtype=torch.float32, device=device)
        ret_t      = torch.tensor(returns,    dtype=torch.float32, device=device)

        if args.advantage_norm and adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        ent_coef = _entropy_coef(ep, args.entropy_coef_start,
                                 args.entropy_coef_final, args.entropy_decay_eps)

        T  = obs_t.shape[0]
        mb = min(args.minibatch_size, T)
        pol_losses, val_losses, ent_means, kl_means, clip_fracs = [], [], [], [], []

        for _ in range(args.ppo_epochs):
            idx = np.arange(T)
            np.random.shuffle(idx)
            for start in range(0, T, mb):
                mb_idx = idx[start:start + mb]
                if len(mb_idx) < 2:
                    continue

                logits, v = net(obs_t[mb_idx])
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(actions_t[mb_idx])
                entropy  = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t[mb_idx])
                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss  = F.mse_loss(v.squeeze(-1), ret_t[mb_idx])
                loss        = policy_loss + VALUE_COEF * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
                optimizer.step()

                with torch.no_grad():
                    approx_kl  = (old_lp_t[mb_idx] - new_lp).mean()
                    clip_frac  = ((ratio - 1.0).abs() > args.clip_eps).float().mean()
                pol_losses.append(float(policy_loss.item()))
                val_losses.append(float(value_loss.item()))
                ent_means.append(float(entropy.item()))
                kl_means.append(float(approx_kl.item()))
                clip_fracs.append(float(clip_frac.item()))

        ep_reward = float(sum(rollout_rewards))
        if ep_reward > best_reward:
            best_reward  = ep_reward
            best_episode = ep
            best_mask    = env.iron_mask.copy()

        pol_loss_mean = float(np.mean(pol_losses)) if pol_losses else 0.0
        val_loss_mean = float(np.mean(val_losses)) if val_losses else 0.0
        ent_mean      = float(np.mean(ent_means))  if ent_means  else 0.0
        kl_mean       = float(np.mean(kl_means))   if kl_means   else 0.0
        clip_frac_m   = float(np.mean(clip_fracs)) if clip_fracs else 0.0

        episode_stats.append({
            'episode':       ep,
            'reward':        ep_reward,
            'force_N':       float(ep_force),
            'policy_loss':   pol_loss_mean,
            'value_loss':    val_loss_mean,
            'entropy':       ent_mean,
            'approx_kl':     kl_mean,
            'clip_frac':     clip_frac_m,
            'iron_count':    int(env.iron_mask.sum()),
            'best_so_far':   float(best_reward),
        })

        _save_ep_mask(env.iron_mask, ep, ep_reward, results_dir)

        if ep % args.log_every == 0 or ep == args.episodes:
            print(f"  ep {ep:3d}/{args.episodes}  "
                  f"reward={ep_reward:9.3f}  force={ep_force:+8.3f}N  "
                  f"pol={pol_loss_mean:+7.4f}  val={val_loss_mean:7.4f}  "
                  f"H={ent_mean:.3f} (β={ent_coef:.3f})  "
                  f"kl={kl_mean:+.4f}  clip={clip_frac_m:.2f}  "
                  f"best={best_reward:9.3f} (ep {best_episode})")

            stats = {
                'episode_stats': episode_stats,
                'hyperparams': {
                    'episodes':            args.episodes,
                    'max_steps':           args.max_steps,
                    'alpha':               args.alpha,
                    'gamma':               GAMMA,
                    'gae_lambda':          args.gae_lambda,
                    'clip_eps':            args.clip_eps,
                    'ppo_epochs':          args.ppo_epochs,
                    'minibatch_size':      args.minibatch_size,
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

    torch.save(net.state_dict(), results_dir / 'policy.pt')
    if best_mask is not None:
        np.save(results_dir / 'best_mask.npy', best_mask)

    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {best_episode}  reward={best_reward:.4f}")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)


def _parse_args():
    p = argparse.ArgumentParser(description='PPO on SeqTO-v2 C-core')
    p.add_argument('--episodes',      type=int,   default=200)
    p.add_argument('--max-steps',     type=int,   default=34)
    p.add_argument('--alpha',         type=float, default=ALPHA)
    p.add_argument('--gae-lambda',    type=float, default=GAE_LAMBDA)
    p.add_argument('--clip-eps',      type=float, default=CLIP_EPS)
    p.add_argument('--ppo-epochs',    type=int,   default=PPO_EPOCHS)
    p.add_argument('--minibatch-size', type=int,  default=MINIBATCH_SIZE)
    p.add_argument('--entropy-coef-start', type=float, default=ENTROPY_COEF_START)
    p.add_argument('--entropy-coef-final', type=float, default=ENTROPY_COEF_FINAL)
    p.add_argument('--entropy-decay-eps',  type=int,   default=ENTROPY_DECAY_EPS)
    p.add_argument('--no-advantage-norm', dest='advantage_norm', action='store_false')
    p.set_defaults(advantage_norm=True)
    p.add_argument('--reward-kind',   choices=['force', 'mean_b'], default='force')
    p.add_argument('--base-fem-file', type=str,   default='ccore_seqto_ppo.fem',
                   help='this run\'s base FEMM geometry (auto-built on first run if missing)')
    p.add_argument('--tmp-fem',       type=str,   default='_tmp_solve_v2_ppo.fem',
                   help='scratch FEMM file used by mi_analyse (distinct from dqn/a2c)')
    p.add_argument('--results-dir',   type=str,   default='designs/ppo_run')
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
