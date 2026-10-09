"""
CNN Training Script for B-Field Prediction

Trains a U-Net CNN to predict magnetic field (B_magnitude) from geometry and excitation.

Input Channels:
    - Channel 0: Material mask (0=air, 1=iron, 2=copper, 3=magnet)
    - Channel 1: Source current density (A/m²)

Output Channel:
    - B_magnitude (Tesla)

Usage:
    python train_cnn.py --problem-type coil --epochs 100 --batch-size 16
    python train_cnn.py --problem-type ipm --epochs 150 --batch-size 32 --learning-rate 5e-4
    python train_cnn.py --problem-type transformer --epochs 25 --batch-size 128
    
"""

import argparse
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from femm_solver import MeshInterpolator

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# ============================================================================
# Dataset Class
# ============================================================================

class FEMMDataset(Dataset):
    """
    PyTorch Dataset for FEM mesh data → CNN tensors

    Converts unstructured triangular mesh to uniform grids for CNN training.
    """

    def __init__(self, samples, resolution=256, bounds=None):
        """
        Args:
            samples: List of sample dicts from generate_cnn_training_data.py
            resolution: Grid resolution (e.g., 256 for 256×256)
            bounds: (xmin, xmax, ymin, ymax) in mm - ensures consistent spatial mapping
        """
        self.samples = samples
        self.resolution = resolution
        self.bounds = bounds

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        mesh_data = sample['mesh_data']

        # Interpolate mesh to uniform grid
        interp = MeshInterpolator(mesh_data)
        grid = interp.interpolate_to_grid(
            resolution=self.resolution,
            bounds=self.bounds
        )

        xx, yy = grid['grid_coords']

        # Channel 0: Material mask (geometry)
        material_mask = interp.generate_semantic_mask(xx, yy).astype(np.float32)

        # Channel 1: Excitation (source current density)
        source_current = grid['source_current_density'].astype(np.float32)

        # Target: B magnitude
        B_magnitude = grid['B_magnitude'].astype(np.float32)

        # Stack into tensors
        X = np.stack([material_mask, source_current], axis=0)  # (2, H, W)
        Y = B_magnitude[None, :, :]  # (1, H, W)

        return torch.from_numpy(X), torch.from_numpy(Y)


# ============================================================================
# Global Bounds Computation
# ============================================================================

def compute_global_bounds(samples, margin=1.1):
    """
    Compute fixed spatial bounds that contain all samples.

    Ensures consistent pixel→physical mapping across all samples.

    Args:
        samples: List of sample dicts
        margin: Expansion factor (1.1 = 10% margin)

    Returns:
        (xmin, xmax, ymin, ymax) in mm
    """
    all_xmin, all_xmax = [], []
    all_ymin, all_ymax = [], []

    print("Computing global bounds from all samples...")
    for sample in tqdm(samples, desc="Analyzing bounds"):
        coords = sample['mesh_data'].node_coords
        all_xmin.append(coords[:, 0].min())
        all_xmax.append(coords[:, 0].max())
        all_ymin.append(coords[:, 1].min())
        all_ymax.append(coords[:, 1].max())

    xmin = min(all_xmin) * margin if min(all_xmin) > 0 else min(all_xmin) / margin
    xmax = max(all_xmax) * margin
    ymin = min(all_ymin) * margin if min(all_ymin) > 0 else min(all_ymin) / margin
    ymax = max(all_ymax) * margin

    # Make square domain (ensures consistent aspect ratio)
    x_range = xmax - xmin
    y_range = ymax - ymin
    max_range = max(x_range, y_range)

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2

    xmin = x_center - max_range / 2
    xmax = x_center + max_range / 2
    ymin = y_center - max_range / 2
    ymax = y_center + max_range / 2

    return (xmin, xmax, ymin, ymax)


# ============================================================================
# U-Net Architecture
# ============================================================================

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


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_training_curves(history, problem_type, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title(f'{problem_type.upper()} - Training History', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{problem_type}_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves to {output_dir}/{problem_type}_training_curves.png")


def visualize_predictions(model, test_dataset, device, problem_type, output_dir, num_samples=5):
    """Visualize CNN predictions vs ground truth"""
    model.eval()

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

    # Handle case of single sample
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        X, Y_true = test_dataset[i]
        X_input = X.unsqueeze(0).to(device)

        with torch.no_grad():
            Y_pred = model(X_input)

        X = X.cpu().numpy()
        Y_true = Y_true.cpu().numpy()[0]
        Y_pred = Y_pred.cpu().numpy()[0, 0]

        # Plot material mask
        axes[i, 0].imshow(X[0], cmap='viridis', interpolation='nearest')
        axes[i, 0].set_title('Material Mask', fontsize=11)
        axes[i, 0].axis('off')

        # Plot ground truth
        im1 = axes[i, 1].imshow(Y_true, cmap='hot', interpolation='bilinear')
        axes[i, 1].set_title('Ground Truth |B|', fontsize=11)
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        # Plot prediction
        im2 = axes[i, 2].imshow(Y_pred, cmap='hot', interpolation='bilinear')
        axes[i, 2].set_title('CNN Prediction |B|', fontsize=11)
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle(f'{problem_type.upper()} - CNN Predictions', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{problem_type}_predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved predictions to {output_dir}/{problem_type}_predictions.png")


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CNN for B-field prediction from FEM data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--problem-type', type=str, required=True,
                       choices=['coil', 'transformer', 'ipm', 'c_core'],
                       help='Type of electromagnetic problem')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to dataset pickle (default: dataset/{problem_type}_cnn_dataset.pkl)')
    parser.add_argument('--resolution', type=int, default=256,
                       help='Grid resolution (256 for 256×256)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (32 for 8-12GB GPU, 64+ for 16GB+ GPU)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.10,
                       help='Test split ratio')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Output directory for models and plots')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='DataLoader workers (0=main process only)')

    return parser.parse_args()


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    args = parse_args()

    print("=" * 80)
    print(f"CNN TRAINING: {args.problem_type.upper()}")
    print("=" * 80)

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load dataset
    if args.dataset_path is None:
        dataset_path = Path(f'dataset/{args.problem_type}_cnn_dataset.pkl')
    else:
        dataset_path = Path(args.dataset_path)

    print(f"\nLoading dataset: {dataset_path}")
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    # Handle both formats: new (list of samples) and old checkpoint (dict with 'samples' key)
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
        print(f"  Loaded checkpoint format")
    elif isinstance(data, list):
        samples = data
    else:
        raise ValueError(f"Unexpected dataset format: {type(data)}")

    print(f"  Loaded {len(samples)} samples")

    # Compute global bounds
    print("\nComputing global bounds...")
    bounds = compute_global_bounds(samples)
    print(f"  Global bounds (mm): xmin={bounds[0]:.2f}, xmax={bounds[1]:.2f}, "
          f"ymin={bounds[2]:.2f}, ymax={bounds[3]:.2f}")

    # Train/val/test split
    print("\nSplitting dataset...")
    train_samples, test_samples = train_test_split(
        samples, test_size=args.test_split, random_state=args.seed
    )
    train_samples, val_samples = train_test_split(
        train_samples,
        test_size=args.val_split / (1 - args.test_split),
        random_state=args.seed
    )

    print(f"  Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  Val:   {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"  Test:  {len(test_samples)} samples ({len(test_samples)/len(samples)*100:.1f}%)")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = FEMMDataset(train_samples, resolution=args.resolution, bounds=bounds)
    val_dataset = FEMMDataset(val_samples, resolution=args.resolution, bounds=bounds)
    test_dataset = FEMMDataset(test_samples, resolution=args.resolution, bounds=bounds)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True if DEVICE.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True if DEVICE.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True if DEVICE.type == 'cuda' else False
    )

    # Create model
    print("\nInitializing model...")
    model = UNet(in_channels=2, out_channels=1, init_features=32).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining configuration:")
    print(f"  Loss: MSE")
    print(f"  Optimizer: Adam (lr={args.learning_rate})")
    print(f"  Scheduler: CosineAnnealingLR")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Device: {DEVICE}")

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'bounds': bounds,
                'resolution': args.resolution,
                'problem_type': args.problem_type
            }, output_dir / f'{args.problem_type}_best_model.pth')

    # Test evaluation
    print("\n" + "=" * 80)
    print("TEST EVALUATION")
    print("=" * 80)

    test_loss = validate_epoch(model, test_loader, criterion, DEVICE)
    test_rmse = np.sqrt(test_loss)

    print(f"Test MSE:  {test_loss:.6f}")
    print(f"Test RMSE: {test_rmse:.6f} T")

    # Save training history
    with open(output_dir / f'{args.problem_type}_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    # Save training summary
    summary = {
        'problem_type': args.problem_type,
        'resolution': args.resolution,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'num_samples': len(samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'best_val_loss': float(best_val_loss),
        'test_loss': float(test_loss),
        'test_rmse': float(test_rmse),
        'total_params': total_params,
        'bounds': bounds
    }

    with open(output_dir / f'{args.problem_type}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_training_curves(history, args.problem_type, output_dir)
    visualize_predictions(model, test_dataset, DEVICE, args.problem_type, output_dir, num_samples=min(5, len(test_dataset)))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best model: {output_dir}/{args.problem_type}_best_model.pth")
    print(f"History:    {output_dir}/{args.problem_type}_history.pkl")
    print(f"Summary:    {output_dir}/{args.problem_type}_summary.json")
    print(f"Plots:      {output_dir}/{args.problem_type}_*.png")
    print("=" * 80)


if __name__ == '__main__':
    main()
