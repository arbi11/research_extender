"""
seqTO_v2_gym.py -- Gymnasium adapter around CcoreSeqTOv2Env.

Lets the SeqTO-v2 C-core env be consumed by libraries that expect the
Gymnasium API (Stable-Baselines3, CleanRL, Tianshou, ...).  The original
CcoreSeqTOv2Env returns the older 4-tuple (obs, reward, done, info); this
wrapper:

  * declares observation_space (Box, 3x18x35 float32) and action_space (Discrete 4)
  * converts the step() return to the Gymnasium 5-tuple
        (obs, reward, terminated, truncated, info)
    `truncated` is set when the episode ended because we hit max_steps;
    `terminated` for any other end-state (in practice rare here -- only when
    iron_count > K2).
  * forwards a `reward_scale` multiplier (constant) -- useful for testing
    whether the env's raw return magnitude is the cause of value-loss /
    entropy-collapse pathologies.
  * stashes the final iron_mask in info['final_iron_mask'] on the terminal
    step, so a logging callback can render it AFTER the auto-reset.

Composition (not inheritance) is used so seqTO_v2_env.py stays untouched.
"""

from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(
        "gymnasium is required for the gym adapter.  Install with: "
        "pip install gymnasium"
    ) from e

from seqTO_v2_env import CcoreSeqTOv2Env, ROWS, COLS


class SeqToV2GymEnv(gym.Env):
    """Gymnasium wrapper for CcoreSeqTOv2Env."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int  = 34,
        reward_kind: str = 'force',
        random_start: bool = True,
        seed: int = 42,
        base_fem_file: str = 'ccore_seqto_sb3.fem',
        tmp_fem: str       = '_tmp_solve_v2_sb3.fem',
        reward_scale: float = 1.0,
    ):
        super().__init__()
        self._inner = CcoreSeqTOv2Env(
            max_steps     = max_steps,
            reward_kind   = reward_kind,
            random_start  = random_start,
            seed          = seed,
            base_fem_file = base_fem_file,
            tmp_fem       = tmp_fem,
        )
        self.action_space      = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, ROWS, COLS), dtype=np.float32,
        )
        self._max_steps    = max_steps
        self._reward_scale = float(reward_scale)
        self._steps        = 0

    # ── Gymnasium API ───────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs = self._inner.reset()
        self._steps = 0
        return obs.astype(np.float32, copy=False), {}

    def step(self, action):
        obs, reward, done, info = self._inner.step(int(action))
        self._steps += 1

        truncated  = bool(done) and self._steps >= self._max_steps
        terminated = bool(done) and not truncated

        if terminated or truncated:
            info = dict(info)
            info['final_iron_mask']  = self._inner.iron_mask.copy()
            info['final_iron_count'] = int(self._inner.iron_mask.sum())
            if 'force' not in info:
                info['force'] = 0.0

        return (
            obs.astype(np.float32, copy=False),
            float(reward) * self._reward_scale,
            terminated,
            truncated,
            info,
        )

    # ── Convenience pass-through ────────────────────────────────────────────

    @property
    def iron_mask(self):
        return self._inner.iron_mask
