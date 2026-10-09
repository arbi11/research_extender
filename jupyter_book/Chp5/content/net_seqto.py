"""
net_seqto.py -- Shared CNN backbone + DQN / Actor-Critic heads for SeqTO-v2.

Used by dqn_seqto.py and a2c_seqto.py.  The backbone is shared so both
algorithms see identical feature dimensions, which makes ablation studies
(e.g. "same network, different objective") clean.

Input shape : [B, 3, 18, 35]   (material, flux, boundary)
Spatial flow:
    Conv1 (3  -> 32, k3 s1 p1)  -> [B, 32, 18, 35]
    Conv2 (32 -> 64, k3 s2 p1)  -> [B, 64,  9, 18]
    Conv3 (64 -> 64, k3 s2 p1)  -> [B, 64,  5,  9]
    Flatten + Linear (2880 -> 256)
Output      : [B, 256] feature vector

Heads:
    DQNHead         : Linear(256 -> 4)               # Q(s, a)
    ActorCriticHead : Linear(256 -> 4) + Linear(256 -> 1)
"""

import torch
import torch.nn as nn


class SeqToCNN(nn.Module):
    """Shared convolutional feature extractor for 18x35 multi-channel observations."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        # After 3 conv layers on (18, 35): spatial = (5, 9)
        self._flat_dim = 64 * 5 * 9
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._flat_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class DQN(nn.Module):
    """Deep Q-Network: shared backbone + linear Q-head."""

    def __init__(self, in_channels: int = 3, action_size: int = 4, feature_dim: int = 256):
        super().__init__()
        self.backbone = SeqToCNN(in_channels, feature_dim)
        self.q_head   = nn.Linear(feature_dim, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_head(self.backbone(x))


class ActorCritic(nn.Module):
    """Actor-Critic: shared backbone + categorical policy head + value head."""

    def __init__(self, in_channels: int = 3, action_size: int = 4, feature_dim: int = 256):
        super().__init__()
        self.backbone    = SeqToCNN(in_channels, feature_dim)
        self.policy_head = nn.Linear(feature_dim, action_size)
        self.value_head  = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor):
        """Returns (policy_logits [B, A], value [B, 1])."""
        h = self.backbone(x)
        return self.policy_head(h), self.value_head(h)


def _smoke_test():
    """Verify forward-pass shapes for both heads."""
    print("=" * 60)
    print("  net_seqto smoke test")
    print("=" * 60)
    dummy = torch.zeros(4, 3, 18, 35)   # batch of 4 observations

    backbone = SeqToCNN()
    feats    = backbone(dummy)
    print(f"  SeqToCNN          : {tuple(dummy.shape)} -> {tuple(feats.shape)}  "
          f"(feature_dim={backbone.feature_dim})")
    assert feats.shape == (4, 256)

    dqn = DQN()
    q   = dqn(dummy)
    print(f"  DQN.q_head        : {tuple(dummy.shape)} -> {tuple(q.shape)}")
    assert q.shape == (4, 4)

    ac  = ActorCritic()
    pi, v = ac(dummy)
    print(f"  ActorCritic       : {tuple(dummy.shape)} -> pi {tuple(pi.shape)}, v {tuple(v.shape)}")
    assert pi.shape == (4, 4)
    assert v.shape  == (4, 1)

    # Parameter counts
    n_back = sum(p.numel() for p in backbone.parameters())
    n_dqn  = sum(p.numel() for p in dqn.parameters())
    n_ac   = sum(p.numel() for p in ac.parameters())
    print(f"  Param counts      : backbone {n_back:,d}  DQN {n_dqn:,d}  A2C {n_ac:,d}")
    print("Smoke test PASSED.")


if __name__ == '__main__':
    _smoke_test()
