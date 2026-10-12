"""
U-Net for B-field prediction + one-epoch train/validate helpers (used by 04c_cnn_training.ipynb).

Input channels: material mask (0 air / 1 iron / 2 copper / 3 magnet), excitation map.
Output channel: |B| (normalised by the notebook).
"""

import torch
import torch.nn as nn
from tqdm import tqdm


class UNet(nn.Module):
    """
    U-Net architecture for B-field prediction.

    Encoder-decoder with skip connections for spatial detail preservation.
    """

    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features

        # Encoder (downsampling path)
        self.enc1 = self._block(in_channels, features)      # 256 -> 256
        self.pool1 = nn.MaxPool2d(2, 2)                     # 256 -> 128

        self.enc2 = self._block(features, features * 2)     # 128 -> 128
        self.pool2 = nn.MaxPool2d(2, 2)                     # 128 -> 64

        self.enc3 = self._block(features * 2, features * 4) # 64 -> 64
        self.pool3 = nn.MaxPool2d(2, 2)                     # 64 -> 32

        self.enc4 = self._block(features * 4, features * 8) # 32 -> 32
        self.pool4 = nn.MaxPool2d(2, 2)                     # 32 -> 16

        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16)  # 16 -> 16

        # Decoder (upsampling path with skip connections)
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.dec4 = self._block(features * 16, features * 8)  # Concat: 16 + 8

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = self._block(features * 2, features)

        # Output layer
        self.out_conv = nn.Conv2d(features, out_channels, 1)

    def _block(self, in_channels, features):
        """Double convolution block with BatchNorm and ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out_conv(dec1)


# ============================================================================
# Training and Validation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0

    for X, Y in tqdm(dataloader, desc="Training", leave=False):
        X, Y = X.to(device), Y.to(device)

        # Forward pass
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * X.size(0)

    return epoch_loss / len(dataloader.dataset)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            Y_pred = model(X)
            loss = criterion(Y_pred, Y)
            epoch_loss += loss.item() * X.size(0)

    return epoch_loss / len(dataloader.dataset)
