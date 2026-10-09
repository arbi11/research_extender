"""
CNN Model Architectures for Uncertainty Quantification

Provides modular U-Net architectures for:
- Standard regression (mean-only prediction)
- Heteroscedastic regression (mean + variance prediction)

Key components:
- UNetFeatures: Encoder-decoder that outputs feature maps
- UNet: Standard UNet (features + final 1x1 conv)
- HeteroscedasticUNet: Outputs both mean (μ) and log-variance (log σ²)
"""

import torch
import torch.nn as nn


# ============================================================================
# U-Net Feature Extractor
# ============================================================================

class UNetFeatures(nn.Module):
    """
    U-Net encoder-decoder that outputs feature maps before final convolution.

    Architecture:
        - Encoder: 4 downsampling stages (32 → 64 → 128 → 256 features)
        - Bottleneck: 512 features
        - Decoder: 4 upsampling stages with skip connections
        - Output: 32-channel feature maps (before final 1x1 conv)

    This modular design allows:
    - Standard UNet: features → 1x1 conv → single output
    - Heteroscedastic UNet: features → two 1x1 convs → (mean, log-variance)
    """

    def __init__(self, in_channels=2, init_features=32):
        """
        Args:
            in_channels: Number of input channels (default: 2 for [geometry, excitation])
            init_features: Number of features in first encoder block (default: 32)
        """
        super(UNetFeatures, self).__init__()

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
        self.dec4 = self._block(features * 16, features * 8)  # Concat: 256 + 256 -> 256

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.dec3 = self._block(features * 8, features * 4)

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.dec2 = self._block(features * 4, features * 2)

        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.dec1 = self._block(features * 2, features)

        # Store init_features for downstream use
        self.init_features = init_features

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
        """
        Args:
            x: (B, in_channels, H, W) input tensor

        Returns:
            (B, init_features, H, W) feature maps
        """
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

        # Return feature maps (B, init_features, H, W)
        return dec1


# ============================================================================
# Standard U-Net (Mean-Only Prediction)
# ============================================================================

class UNet(nn.Module):
    """
    Standard U-Net for deterministic regression.

    Outputs single-channel prediction (mean B-field magnitude).
    Compatible with existing pretrained checkpoints.
    """

    def __init__(self, in_channels=2, out_channels=1, init_features=32):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            init_features: Number of features in first encoder block
        """
        super(UNet, self).__init__()

        self.features = UNetFeatures(in_channels, init_features)
        self.out_conv = nn.Conv2d(init_features, out_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) input tensor

        Returns:
            (B, out_channels, H, W) predicted field
        """
        feats = self.features(x)
        return self.out_conv(feats)


# ============================================================================
# Heteroscedastic U-Net (Mean + Variance Prediction)
# ============================================================================

class HeteroscedasticUNet(nn.Module):
    """
    U-Net for heteroscedastic regression (predicts mean AND uncertainty).

    Architecture:
        features (32 channels) → split into:
            - mean_head (1x1 conv) → μ (mean field)
            - logvar_head (1x1 conv) → log σ² (log-variance)

    Why log-variance?
        - Ensures σ² > 0 via exp() transform
        - Numerically stable (avoids underflow/overflow)
        - Standard in heteroscedastic regression literature

    Usage:
        model = HeteroscedasticUNet(in_channels=2, init_features=32)
        mu, log_var = model(x)

        # For uncertainty decomposition with MC Dropout:
        results = model.predict_with_uncertainty(x, n_samples=50)
    """

    def __init__(self, in_channels=2, init_features=32, logvar_init=-3.0):
        """
        Args:
            in_channels: Number of input channels
            init_features: Number of features in encoder (default: 32)
            logvar_init: Initial log-variance bias (default: -3.0 → σ ≈ 0.05)
        """
        super(HeteroscedasticUNet, self).__init__()

        self.features = UNetFeatures(in_channels, init_features)

        # Separate heads for mean and log-variance
        self.mean_head = nn.Conv2d(init_features, 1, 1)
        self.logvar_head = nn.Conv2d(init_features, 1, 1)

        # Initialize log-variance head to small negative value
        # (corresponds to initial σ ≈ exp(-3/2) ≈ 0.22)
        with torch.no_grad():
            self.logvar_head.bias.fill_(logvar_init)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) input tensor

        Returns:
            mu: (B, 1, H, W) predicted mean field
            log_var: (B, 1, H, W) predicted log-variance
        """
        feats = self.features(x)  # (B, init_features, H, W)

        mu = self.mean_head(feats)
        log_var = self.logvar_head(feats)

        # Clamp log-variance for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=5)

        return mu, log_var

    def predict_with_uncertainty(self, x, n_samples=50):
        """
        MC Dropout inference for BOTH epistemic and aleatoric uncertainty.

        Epistemic uncertainty: Model disagreement (variance of means across MC samples)
        Aleatoric uncertainty: Data noise (average of predicted variances)

        Args:
            x: (B, in_channels, H, W) input tensor
            n_samples: Number of MC dropout samples (T)

        Returns:
            dict with:
                - mu_pred: (B, 1, H, W) mean prediction
                - std_epistemic: (B, 1, H, W) epistemic uncertainty (√var_epistemic)
                - std_aleatoric: (B, 1, H, W) aleatoric uncertainty (√var_aleatoric)
                - std_total: (B, 1, H, W) total uncertainty (√(var_epi + var_alea))
        """
        # Enable dropout (but NOT BatchNorm - see enable_dropout_only helper)
        self.train()

        mus = []
        log_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                mu, log_var = self(x)
                mus.append(mu)
                log_vars.append(log_var)

        mus = torch.stack(mus)  # (T, B, 1, H, W)
        log_vars = torch.stack(log_vars)  # (T, B, 1, H, W)

        # Mean prediction (average over MC samples)
        mu_pred = mus.mean(dim=0)  # (B, 1, H, W)

        # Epistemic variance (model disagreement)
        var_epistemic = mus.var(dim=0, unbiased=False)  # (B, 1, H, W)

        # Aleatoric variance (average predicted noise)
        var_aleatoric = torch.exp(log_vars).mean(dim=0)  # (B, 1, H, W)

        # Total predictive variance
        var_total = var_epistemic + var_aleatoric

        return {
            'mu_pred': mu_pred,
            'std_epistemic': torch.sqrt(var_epistemic + 1e-8),
            'std_aleatoric': torch.sqrt(var_aleatoric + 1e-8),
            'std_total': torch.sqrt(var_total + 1e-8),
        }


# ============================================================================
# Loss Functions
# ============================================================================

def heteroscedastic_nll_loss(mu, log_var, y_true):
    """
    Gaussian negative log-likelihood for heteroscedastic regression.

    Loss = 1/2 * exp(-s) * (y - μ)² + 1/2 * s

    where s = log σ² is the predicted log-variance.

    Intuition:
        - First term: Weighted MSE (errors in high-uncertainty regions count less)
        - Second term: Penalty for inflating variance (prevents trivial solution)

    Args:
        mu: (B, 1, H, W) predicted mean
        log_var: (B, 1, H, W) predicted log-variance
        y_true: (B, 1, H, W) ground truth

    Returns:
        scalar loss
    """
    # Ensure numerical stability
    log_var = torch.clamp(log_var, min=-10, max=5)

    # Compute per-pixel loss
    inv_var = torch.exp(-log_var)  # 1/σ²
    sq_err = (y_true - mu) ** 2

    # NLL = 0.5 * (σ⁻² * err² + log σ²)
    nll = 0.5 * (inv_var * sq_err + log_var)

    # Return mean loss over all pixels and batch
    return nll.mean()


# ============================================================================
# Helper Functions
# ============================================================================

def enable_dropout_only(model):
    """
    Set model to eval mode but keep ONLY dropout layers in train mode.

    Critical for MC Dropout: We want stochasticity from dropout,
    but NOT from BatchNorm (which would add unwanted noise).

    Usage:
        enable_dropout_only(model)
        for _ in range(n_samples):
            pred = model(x)  # Dropout active, BatchNorm frozen

    Args:
        model: PyTorch model with dropout layers
    """
    model.eval()  # Set all layers to eval mode

    # Re-enable training mode for dropout layers only
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


# ============================================================================
# Module Testing (run this file directly to test)
# ============================================================================

if __name__ == "__main__":
    print("Testing cnn_model.py components...")

    # Test UNetFeatures
    print("\n1. Testing UNetFeatures...")
    unet_feats = UNetFeatures(in_channels=2, init_features=32)
    x = torch.randn(2, 2, 256, 256)
    feats = unet_feats(x)
    print(f"   Input: {x.shape} → Features: {feats.shape}")
    assert feats.shape == (2, 32, 256, 256), "UNetFeatures output shape mismatch!"

    # Test UNet
    print("\n2. Testing UNet...")
    unet = UNet(in_channels=2, out_channels=1, init_features=32)
    y = unet(x)
    print(f"   Input: {x.shape} → Output: {y.shape}")
    assert y.shape == (2, 1, 256, 256), "UNet output shape mismatch!"

    # Test HeteroscedasticUNet
    print("\n3. Testing HeteroscedasticUNet...")
    hetero_unet = HeteroscedasticUNet(in_channels=2, init_features=32, logvar_init=-3.0)
    mu, log_var = hetero_unet(x)
    print(f"   Input: {x.shape} → mu: {mu.shape}, log_var: {log_var.shape}")
    assert mu.shape == (2, 1, 256, 256), "HeteroscedasticUNet mu shape mismatch!"
    assert log_var.shape == (2, 1, 256, 256), "HeteroscedasticUNet log_var shape mismatch!"

    # Test predict_with_uncertainty
    print("\n4. Testing predict_with_uncertainty...")
    results = hetero_unet.predict_with_uncertainty(x, n_samples=10)
    print(f"   mu_pred: {results['mu_pred'].shape}")
    print(f"   std_epistemic: {results['std_epistemic'].shape}")
    print(f"   std_aleatoric: {results['std_aleatoric'].shape}")
    print(f"   std_total: {results['std_total'].shape}")

    # Test loss function
    print("\n5. Testing heteroscedastic_nll_loss...")
    y_true = torch.randn(2, 1, 256, 256)
    loss = heteroscedastic_nll_loss(mu, log_var, y_true)
    print(f"   Loss: {loss.item():.6f}")

    # Test enable_dropout_only
    print("\n6. Testing enable_dropout_only...")
    hetero_unet.eval()
    print(f"   Model in eval mode")
    enable_dropout_only(hetero_unet)
    # Check that dropout is in train mode
    dropout_in_train = any(m.training for m in hetero_unet.modules() if isinstance(m, nn.Dropout))
    # Check that BatchNorm is in eval mode
    bn_in_eval = all(not m.training for m in hetero_unet.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"   Dropout in train mode: {dropout_in_train}")
    print(f"   BatchNorm in eval mode: {bn_in_eval}")

    print("\n✅ All tests passed!")
