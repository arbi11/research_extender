"""
sb3_runner.py -- Stable-Baselines3 baselines (PPO / A2C / DQN) on SeqTO-v2.

Lets you run vetted reference implementations against the same env that our
custom dqn_seqto.py / a2c_seqto.py / ppo_seqto.py use, so you can tell
whether entropy collapse, value-loss explosions, etc. come from the
algorithm code or from the environment / reward scale.

Setup
-----
    pip install stable-baselines3 gymnasium

Usage
-----
    # PPO smoke
    python sb3_runner.py --algo ppo --timesteps 10000 --seed 42

    # A2C smoke with reward scaling (drops returns from ~1e4 -> ~1e1)
    python sb3_runner.py --algo a2c --timesteps 10000 --reward-scale 0.001

    # DQN baseline
    python sb3_runner.py --algo dqn --timesteps 10000

Outputs (in designs/sb3_<algo>_run/)
------------------------------------
* training_stats.json   per-episode reward, force_N, iron_count, best_so_far
* episodes/ep_NNN_mask.png
* policy.zip            SB3 model (use SB3 to reload, not torch.load)
* best_mask.npy

Notes
-----
* Uses MlpPolicy with a custom CNN feature extractor (SB3's default NatureCNN
  requires HxW >= 36; we have 18x35).  The extractor mirrors net_seqto.SeqToCNN
  so the network is roughly the same capacity as our custom scripts.
* SB3 uses VecEnv internally even for a single env, so we run via DummyVecEnv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
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

try:
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    print("ERROR: stable-baselines3 not installed.  Run: pip install stable-baselines3")
    sys.exit(1)

from seqTO_v2_gym import SeqToV2GymEnv


# ── Custom feature extractor (mirrors net_seqto.SeqToCNN) ───────────────────

class SeqToCNNExtractor(BaseFeaturesExtractor):
    """CNN backbone matching net_seqto.SeqToCNN, for SB3 MlpPolicy."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample   = torch.zeros(1, *observation_space.shape)
            flat_dim = self.conv(sample).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


# ── Episode-stats callback (writes our training_stats.json schema) ──────────

def _atomic_dump_json(obj, path):
    path = Path(path)
    tmp  = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


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


class EpisodeStatsCallback(BaseCallback):
    """Mirrors the training_stats.json schema our custom scripts emit."""

    def __init__(self, results_dir, hyperparams, save_every: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.stats_path  = self.results_dir / 'training_stats.json'
        self.save_every  = save_every
        self.hyperparams = dict(hyperparams)

        self.episode_stats = []
        self.best_reward   = -float('inf')
        self.best_episode  = -1
        self._ep_reward    = 0.0
        self._ep_force     = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones   = self.locals['dones']
        infos   = self.locals['infos']
        # Single env (DummyVecEnv index 0)
        self._ep_reward += float(rewards[0])
        if 'force' in infos[0]:
            self._ep_force = float(infos[0]['force'])

        if dones[0]:
            ep = len(self.episode_stats) + 1
            info = infos[0]
            final_mask   = info.get('final_iron_mask')
            iron_count   = int(info.get('final_iron_count', -1))

            if self._ep_reward > self.best_reward:
                self.best_reward  = self._ep_reward
                self.best_episode = ep
                if final_mask is not None:
                    np.save(self.results_dir / 'best_mask.npy', final_mask)

            self.episode_stats.append({
                'episode':     ep,
                'reward':      float(self._ep_reward),
                'force_N':     float(self._ep_force),
                'iron_count':  iron_count,
                'best_so_far': float(self.best_reward),
            })

            if final_mask is not None:
                _save_ep_mask(final_mask, ep, self._ep_reward, self.results_dir)

            if self.verbose:
                print(f"  ep {ep:3d}  reward={self._ep_reward:9.3f}  "
                      f"force={self._ep_force:+8.3f}N  iron={iron_count}  "
                      f"best={self.best_reward:9.3f} (ep {self.best_episode})")

            if ep % self.save_every == 0:
                self._dump()

            self._ep_reward = 0.0
            self._ep_force  = 0.0

        return True

    def _on_training_end(self) -> None:
        self._dump()

    def _dump(self):
        _atomic_dump_json({
            'episode_stats': self.episode_stats,
            'hyperparams': {
                **self.hyperparams,
                'best_episode': self.best_episode,
                'best_reward':  float(self.best_reward),
            },
        }, self.stats_path)


# ── Algo construction ───────────────────────────────────────────────────────

def _make_model(algo, env, seed, device):
    """Build an SB3 model with sensible defaults and our custom feature extractor."""
    policy_kwargs = {
        'features_extractor_class':  SeqToCNNExtractor,
        'features_extractor_kwargs': {'features_dim': 256},
        # SB3 image-normalisation assumes uint8 [0,255]; our obs are already
        # in [0, 1] float32, so disable it.
        'normalize_images': False,
    }

    if algo == 'ppo':
        return PPO(
            'MlpPolicy', env,
            policy_kwargs = policy_kwargs,
            n_steps       = 64,        # rollout length per update
            batch_size    = 64,
            n_epochs      = 4,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            learning_rate = 3e-4,
            seed          = seed,
            device        = device,
            verbose       = 1,
        )
    if algo == 'a2c':
        return A2C(
            'MlpPolicy', env,
            policy_kwargs = policy_kwargs,
            n_steps       = 34,        # ~one episode per rollout
            gamma         = 0.99,
            gae_lambda    = 1.0,       # A2C in SB3 defaults to 1.0 (MC returns)
            ent_coef      = 0.01,
            vf_coef       = 0.5,
            learning_rate = 7e-4,
            seed          = seed,
            device        = device,
            verbose       = 1,
        )
    if algo == 'dqn':
        return DQN(
            'MlpPolicy', env,
            policy_kwargs       = policy_kwargs,
            buffer_size         = 10_000,
            learning_starts     = 200,
            batch_size          = 32,
            gamma               = 0.99,
            learning_rate       = 3e-4,
            target_update_interval = 100,
            exploration_fraction = 0.5,
            exploration_final_eps = 0.05,
            seed                = seed,
            device              = device,
            verbose             = 1,
        )
    raise ValueError(f"unknown algo {algo!r}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def _make_env():
        return SeqToV2GymEnv(
            max_steps     = args.max_steps,
            reward_kind   = args.reward_kind,
            random_start  = True,
            seed          = args.seed,
            base_fem_file = args.base_fem_file,
            tmp_fem       = args.tmp_fem,
            reward_scale  = args.reward_scale,
        )

    env = DummyVecEnv([_make_env])

    model = _make_model(args.algo, env, args.seed, args.device)

    hyperparams = {
        'algo':         args.algo,
        'timesteps':    args.timesteps,
        'max_steps':    args.max_steps,
        'reward_kind':  args.reward_kind,
        'reward_scale': args.reward_scale,
        'seed':         args.seed,
        'device':       args.device,
    }
    cb = EpisodeStatsCallback(
        results_dir = results_dir,
        hyperparams = hyperparams,
        save_every  = args.log_every,
        verbose     = 1,
    )

    print("=" * 72)
    print(f"  SB3 {args.algo.upper()} on SeqTO-v2 C-core")
    print(f"  timesteps = {args.timesteps}    max_steps/ep = {args.max_steps}")
    print(f"  reward_kind = {args.reward_kind}    reward_scale = {args.reward_scale}")
    print(f"  Output dir = {results_dir}")
    print("=" * 72)

    model.learn(total_timesteps=args.timesteps, callback=cb)

    model.save(results_dir / 'policy.zip')
    print()
    print("=" * 72)
    print(f"  Training complete.  Best episode: {cb.best_episode}  "
          f"reward={cb.best_reward:.4f}")
    print(f"  Artifacts -> {results_dir}/")
    print("=" * 72)


def _parse_args():
    p = argparse.ArgumentParser(description='SB3 baselines on SeqTO-v2 C-core')
    p.add_argument('--algo',         choices=['ppo', 'a2c', 'dqn'], required=True)
    p.add_argument('--timesteps',    type=int,   default=10_000,
                   help='total environment steps (default 10k ~= 290 episodes)')
    p.add_argument('--max-steps',    type=int,   default=34)
    p.add_argument('--reward-kind',  choices=['force', 'mean_b'], default='force')
    p.add_argument('--reward-scale', type=float, default=1.0,
                   help='multiply every reward by this constant.  Try 0.001 '
                        'to drop force-rewards from ~1e4 -> ~1e1.')
    p.add_argument('--base-fem-file', type=str,  default=None,
                   help='base FEMM geometry (default: ccore_seqto_sb3_<algo>.fem)')
    p.add_argument('--tmp-fem',       type=str,  default=None,
                   help='scratch FEMM file (default: _tmp_solve_v2_sb3_<algo>.fem)')
    p.add_argument('--results-dir',   type=str,  default=None,
                   help='default: designs/sb3_<algo>_run')
    p.add_argument('--seed',          type=int,  default=42)
    p.add_argument('--log-every',     type=int,  default=5,
                   help='write training_stats.json every N episodes')
    p.add_argument('--device',        choices=['cpu', 'cuda', 'auto'], default='cpu')
    args = p.parse_args()

    if args.results_dir is None:
        args.results_dir = f'designs/sb3_{args.algo}_run'
    if args.base_fem_file is None:
        args.base_fem_file = f'ccore_seqto_sb3_{args.algo}.fem'
    if args.tmp_fem is None:
        args.tmp_fem = f'_tmp_solve_v2_sb3_{args.algo}.fem'
    return args


if __name__ == '__main__':
    main()
