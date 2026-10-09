"""Neural-net architectures for Phase 5.

Phase 5A: EnvelopeMLP (6 -> 2 regression).
Phase 5B: MapDecoder (6 -> 60x80 efficiency-map grid).
"""

from __future__ import annotations
from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class EnvelopeMLP(nn.Module):
    """Tiny MLP predicting (N_max_rpm, T_max_Nm) from the 6-dim design vector.

    Architecture: [6] -> [hidden_0] -> ReLU -> ... -> [hidden_-1] -> ReLU -> [2]
    With the default hidden=(32, 32) this is ~1.2 k parameters.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden: Sequence[int] = (32, 32),
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MapDecoder(nn.Module):
    """Decoder predicting a (60, 80) normalised efficiency map from a 6-dim
    design vector.

    Default architecture (kept bit-exact identical to the original baseline so
    existing checkpoints in phase5_map_train/ keep loading):
        Linear(6 -> 64) -> ReLU
        Linear(64 -> bottleneck_c * bottleneck_h * bottleneck_w) -> ReLU
        reshape -> (B, bottleneck_c, bottleneck_h, bottleneck_w)
        ConvTranspose2d (bottleneck_c -> decoder_channels[0], k=4, s=2, p=1) -> ReLU
        [optional Conv2d (k=3) -> ReLU if intermediate_conv is True]
        ConvTranspose2d (decoder_channels[0] -> decoder_channels[1], k=4, s=2, p=1)
        bilinear resize to (out_h, out_w) = (60, 80)
        sigmoid

    Defaults: bottleneck_shape=(8, 15, 20), decoder_channels=(4, 1),
    intermediate_conv=False. ~157 k parameters.

    Architecture-sweep variants set bottleneck channels and decoder_channels
    higher (width sweep), or set intermediate_conv=True to insert a same-
    resolution conv block (depth+1 variant).
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden: int = 64,
        bottleneck_shape: Tuple[int, int, int] = (8, 15, 20),
        decoder_channels: Tuple[int, ...] = (4, 1),
        intermediate_conv: bool = False,
        out_shape: Tuple[int, int] = (60, 80),
    ) -> None:
        super().__init__()
        if len(decoder_channels) != 2:
            raise ValueError(
                "decoder_channels must be a 2-tuple (c_mid, c_out); got "
                f"length {len(decoder_channels)}: {decoder_channels}"
            )
        self.bottleneck_shape = bottleneck_shape
        self.decoder_channels = tuple(decoder_channels)
        self.intermediate_conv = bool(intermediate_conv)
        self.out_shape = out_shape

        c, h, w = bottleneck_shape
        c_mid, c_out = self.decoder_channels
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, c * h * w),
            nn.ReLU(inplace=True),
        )

        decoder_layers: list[nn.Module] = [
            nn.ConvTranspose2d(c, c_mid, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        ]
        if self.intermediate_conv:
            decoder_layers.extend([
                nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ])
        decoder_layers.append(
            nn.ConvTranspose2d(c_mid, c_out, kernel_size=4, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.head(x)
        c, h, w = self.bottleneck_shape
        z = z.view(z.size(0), c, h, w)
        y = self.decoder(z)
        if y.shape[-2:] != self.out_shape:
            y = F.interpolate(y, size=self.out_shape, mode="bilinear",
                              align_corners=False)
        y = torch.sigmoid(y)
        return y.squeeze(1)  # (B, out_h, out_w)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# Architecture presets for the MapDecoder sweep (§4.7.2 in the chapter).
# Keys map to constructor kwargs; the baseline is identical to MapDecoder defaults.
MAPDECODER_PRESETS: dict[str, dict] = {
    "baseline": dict(
        bottleneck_shape=(8, 15, 20),
        decoder_channels=(4, 1),
        intermediate_conv=False,
    ),
    "width2x": dict(
        bottleneck_shape=(16, 15, 20),
        decoder_channels=(8, 1),
        intermediate_conv=False,
    ),
    "width4x": dict(
        bottleneck_shape=(32, 15, 20),
        decoder_channels=(16, 1),
        intermediate_conv=False,
    ),
    "depth1": dict(
        bottleneck_shape=(8, 15, 20),
        decoder_channels=(4, 1),
        intermediate_conv=True,
    ),
}
