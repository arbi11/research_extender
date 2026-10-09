"""
PINN Training Script

Train physics-informed neural networks for electromagnetic field prediction.
Uses sparse FEMM observations + physics constraints (Maxwell's equations).

Key features:
- Data-driven loss on sparse observations
- Physics-driven loss via automatic differentiation
- Combined loss: L_total = L_data + λ * L_physics
- Validation against full FEMM solution

Author: Generated for FEMM-PINN integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import json


# Problem type to PINN data directory mapping
PROBLEM_TYPE_MAP = {
    'simple_wire': 'coil',      # simple_wire uses coil PINN data
    'c_core': 'c_core',         # c_core uses c_core PINN data
    'ipm_motor': 'ipm_motor',   # ipm_motor uses ipm_motor PINN data
    'transformer': 'transformer' # transformer uses transformer PINN data
}

# Problem type to B-field scale mapping
# Based on ACTUAL field magnitudes in datasets (not theoretical expectations)
# Goal: Scale outputs to ~0.1-10 range for optimal network learning
B_SCALE_MAP = {
    'simple_wire': 1e6,     # Actual: ~0.00001 T → scaled to ~10
    'c_core': 1e3,          # Expected: ~0.001 T → scaled to ~1
    'ipm_motor': 1e2,       # Actual: ~0.02 T → scaled to ~2
    'transformer': 1e3      # Actual: ~0.0004 T → scaled to ~0.4
}


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for electromagnetic fields.

    Architecture: MLP with tanh activations
    Input: (x, y, params) — spatial coordinates + physical parameters
    Output: A_z magnetic vector potential [T·mm]

    Conditioning on physical parameters (e.g. current I, wire_radius r_w) allows
    ONE network to generalise across an entire family of configurations.  Without
    this, a network that only sees (x,y) cannot distinguish a 10 A wire from a
    100 A wire and the training loss becomes incoherent across samples.

    Input encoding:
      - (x, y): random Fourier features [sin(Bx), cos(Bx)] to overcome spectral
                bias (Tancik et al. 2020).  Fourier matrix B is fixed at init.
      - params: concatenated raw (after [0,1] normalisation); NO Fourier encoding —
                they are scalar scalings, not spatial coordinates.

    Ref: Beltrán-Pulido et al. (2022) IEEE Trans. Energy Conv., arxiv 2202.04041.
    """

    # Per-problem parameter normalisation bounds: (min, max) for each parameter
    # in the order they appear in sample['parameters'].
    PARAM_BOUNDS = {
        'coil':        {'current': (10.0, 100.0), 'wire_radius': (0.005, 0.05)},
        'transformer': {'primary_turns': (100, 500), 'secondary_turns': (100, 500),
                        'core_area': (1e-4, 1e-3), 'frequency': (50, 400),
                        'primary_voltage': (100, 240)},
        'ipm_motor':   {'current_amplitude': (10, 100), 'magnet_strength': (1.0, 1.4),
                        'rotor_position': (0, 360), 'air_gap': (5e-4, 2e-3)},
    }

    def __init__(self, hidden_layers=[64, 128, 128, 64], normalize=True, b_scale=1e6,
                 n_fourier=16, problem_type='coil'):
        """
        Args:
            hidden_layers: MLP hidden layer widths
            normalize: (kept for backward compat; Fourier encoding replaces coord normalisation)
            b_scale: B-field scaling factor (problem-specific)
            n_fourier: Number of Fourier frequency pairs for (x,y) encoding
            problem_type: Key into PARAM_BOUNDS; determines n_params and normalisation
        """
        super().__init__()

        self.normalize = normalize
        self.coord_scale = 500.0  # Domain is [-500, 500] mm
        self.B_scale = b_scale
        self.n_fourier = n_fourier
        self.problem_type = problem_type

        # Fourier feature matrix B ∈ R^{n_fourier × 2}: fixed random frequencies
        # Input (x_norm, y_norm) maps to [sin(Bx), cos(Bx)] ∈ R^{2*n_fourier}
        torch.manual_seed(0)
        B_mat = torch.randn(n_fourier, 2) * 1.0  # σ=1 — moderate frequency bandwidth (Tancik 2020)
        self.register_buffer('B_mat', B_mat)

        # Geometry-aware Fourier encoding: second set of features using coordinates
        # normalised by wire_radius (x/r_w, y/r_w).  This lets the network capture
        # fine-scale structure near the wire regardless of absolute wire size.
        # Only used for 'coil' since that is the only problem type with a wire_radius param.
        self.use_wire_scale = (problem_type == 'coil')
        if self.use_wire_scale:
            B_mat_wire = torch.randn(n_fourier, 2) * 1.0
            self.register_buffer('B_mat_wire', B_mat_wire)

        # Determine number of physical parameters for this problem type
        bounds = self.PARAM_BOUNDS.get(problem_type, {})
        self.param_names = list(bounds.keys())
        self.param_bounds = list(bounds.values())
        n_params = len(self.param_names)

        # Network input dimension:
        #   domain-scale Fourier: 2*n_fourier
        #   wire-scale Fourier (coil only): 2*n_fourier
        #   physical parameters: n_params
        wire_feat_dim = 2 * n_fourier if self.use_wire_scale else 0
        in_dim = 2 * n_fourier + wire_feat_dim + n_params
        layers = []
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  # → A_z

        self.net = nn.Sequential(*layers)

    def _encode_coords(self, x, y, params_dict=None):
        """
        Random Fourier features for (x, y). x,y in mm.

        For coil problems, also adds wire-scale features using coordinates
        normalised by wire_radius (x/r_w_mm, y/r_w_mm).  This gives the network
        a scale-invariant view of the wire geometry regardless of r_w.
        """
        x_norm = x / self.coord_scale   # → [-1, 1]
        y_norm = y / self.coord_scale
        xy = torch.cat([x_norm, y_norm], dim=-1)       # (N, 2)
        proj = xy @ self.B_mat.T                        # (N, n_fourier)
        domain_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*n_fourier)

        if not self.use_wire_scale:
            return domain_feat

        # Wire-scale features: normalise by r_w so near-wire structure is always visible
        if params_dict is not None and 'wire_radius' in params_dict:
            r_w_mm = float(params_dict['wire_radius']) * 1000.0  # m → mm
        else:
            r_w_mm = 27.5  # fallback: midpoint of [5, 50] mm range

        r_w_mm = max(r_w_mm, 1.0)  # guard against division by zero
        # Clamp to ±10 wire-radii so small wires don't generate extreme coordinates
        # (e.g. r_w=5mm, x=500mm → x/r_w=100; sin(100) oscillates wildly and makes
        # training unstable).  Fields beyond 10 r_w are smooth and captured by domain features.
        xy_wire = torch.cat([x / r_w_mm, y / r_w_mm], dim=-1).clamp(-10.0, 10.0)
        proj_wire = xy_wire @ self.B_mat_wire.T
        wire_feat = torch.cat([torch.sin(proj_wire), torch.cos(proj_wire)], dim=-1)

        return torch.cat([domain_feat, wire_feat], dim=-1)  # (N, 4*n_fourier)

    def _encode_params(self, params_dict, n_points, device):
        """
        Normalise physical parameters to [0, 1] and broadcast to (n_points, n_params).

        params_dict: dict from sample['parameters']
        """
        if not self.param_names:
            return torch.zeros(n_points, 0, device=device)
        vecs = []
        for name, (lo, hi) in zip(self.param_names, self.param_bounds):
            val = float(params_dict.get(name, 0.0))
            norm_val = (val - lo) / (hi - lo + 1e-30)
            vecs.append(torch.full((n_points, 1), norm_val, dtype=torch.float32, device=device))
        return torch.cat(vecs, dim=-1)  # (n_points, n_params)

    def forward(self, x, y, params_dict=None):
        """
        Forward pass: (x, y, params) → A_z.

        Args:
            x: (N, 1) tensor of x-coordinates [mm], requires_grad may be True
            y: (N, 1) tensor of y-coordinates [mm]
            params_dict: dict from sample['parameters'], e.g. {'current': 50.0, ...}
                         If None, parameter inputs are all zeros (single-sample mode).
        """
        device = x.device
        feat = self._encode_coords(x, y, params_dict)              # (N, 2*n_fourier) or (N, 4*n_fourier)
        p = self._encode_params(params_dict, x.shape[0], device) if params_dict else \
            torch.zeros(x.shape[0], len(self.param_names), device=device)
        inp = torch.cat([feat, p], dim=-1)
        return self.net(inp)

    def compute_B_field(self, x, y, params_dict=None):
        """
        Derive B-field from A_z via automatic differentiation.

        For 2D magnetostatics: B = ∇×A = (∂A_z/∂y, -∂A_z/∂x, 0)

        This automatically satisfies ∇·B = 0 (Gauss's law) by construction,
        since ∇·(∇×A) = 0 for any vector field A.

        Args:
            x: X-coordinates tensor (batch_size, 1)
            y: Y-coordinates tensor (batch_size, 1)

        Returns:
            Tuple of (Bx, By) magnetic flux density components
        """
        # Enable gradient computation
        x = x.clone().detach().requires_grad_(True)
        y = y.clone().detach().requires_grad_(True)

        # Forward pass: get A_z (pass params_dict so network is conditioned correctly)
        Az = self.forward(x, y, params_dict)

        # Compute derivatives: B = ∇×A
        dAz_dx = torch.autograd.grad(
            Az, x, grad_outputs=torch.ones_like(Az),
            create_graph=True, retain_graph=True
        )[0]

        dAz_dy = torch.autograd.grad(
            Az, y, grad_outputs=torch.ones_like(Az),
            create_graph=True, retain_graph=True
        )[0]

        # B-field components from curl
        Bx = dAz_dy   # ∂A_z/∂y
        By = -dAz_dx  # -∂A_z/∂x

        return Bx, By


class PINNDataset(Dataset):
    """
    Load PINN data generated by generate_pinns_data.py.
    """

    def __init__(self, data_dir):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing sample_*.pkl files
        """
        self.samples = []
        data_path = Path(data_dir)

        if not data_path.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

        # Load all pickle files
        pkl_files = sorted(data_path.glob('sample_*.pkl'))
        if len(pkl_files) == 0:
            raise ValueError(f"No sample files found in {data_dir}")

        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                self.samples.append(pickle.load(f))

        print(f"Loaded {len(self.samples)} samples from {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class PINNTrainer:
    """
    Training pipeline for physics-informed neural networks.
    """

    def __init__(self, model, lambda_physics=1.0, lambda_bc=10.0, lr=1e-3, device=None):
        """
        Initialize trainer.

        Args:
            model: PINN model
            lambda_physics: Weight for physics loss
            lambda_bc: Weight for boundary condition loss
            lr: Learning rate
            device: Torch device (auto-detected if None)
        """
        self.model = model
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc

        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training history
        self.history = {
            'train_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'bc_loss': [],
            'validation_nmse': []
        }

    def _interpolate_sources(self, source_grid, grid_coords, colloc_points):
        """
        Interpolate FEMM source data (J_z, μ) to collocation points.

        Args:
            source_grid: (H, W) grid from FEMM
            grid_coords: (xx, yy) meshgrid coordinates
            colloc_points: (x_colloc, y_colloc) arrays

        Returns:
            source_values: (N_colloc, 1) interpolated values
        """
        from scipy.interpolate import RegularGridInterpolator

        xx, yy = grid_coords
        # Handle tensor inputs
        if isinstance(xx, torch.Tensor):
            if xx.dim() == 3:
                xx = xx.squeeze(0)
            xx = xx.numpy()
        if isinstance(yy, torch.Tensor):
            if yy.dim() == 3:
                yy = yy.squeeze(0)
            yy = yy.numpy()
        if isinstance(source_grid, torch.Tensor):
            if source_grid.dim() == 3:
                source_grid = source_grid.squeeze(0)
            source_grid = source_grid.numpy()

        x_grid = xx[0, :]  # 1D x coordinates
        y_grid = yy[:, 0]  # 1D y coordinates

        # Create interpolator
        interpolator = RegularGridInterpolator(
            (y_grid, x_grid), source_grid,
            method='linear', bounds_error=False, fill_value=0.0
        )

        # Interpolate to collocation points
        x_colloc, y_colloc = colloc_points
        points = np.column_stack([y_colloc.flatten(), x_colloc.flatten()])
        values = interpolator(points)

        return values.reshape(-1, 1)

    def compute_data_loss(self, sample):
        """
        Compute MSE loss on sparse B-field observations.

        Compares derived B-field (from A_z) with FEMM observations.

        Args:
            sample: Data sample dict

        Returns:
            Data loss (scalar tensor)
        """
        obs = sample['observations']

        # Convert to tensors (handling both numpy and tensor inputs)
        coords = obs['coords']
        if isinstance(coords, torch.Tensor):
            # Remove batch dimension if present
            if coords.dim() == 3:
                coords = coords.squeeze(0)
            coords = coords.numpy()
        coords = coords.astype(np.float32)

        x = torch.from_numpy(coords[:, 0:1]).to(self.device)
        y = torch.from_numpy(coords[:, 1:2]).to(self.device)

        Bx = obs['Bx']
        if isinstance(Bx, torch.Tensor):
            if Bx.dim() == 2:
                Bx = Bx.squeeze(0)
            Bx = Bx.numpy()
        Bx_true = torch.from_numpy(Bx.astype(np.float32)).unsqueeze(1).to(self.device)
        Bx_true = Bx_true * self.model.B_scale  # Tesla → scaled

        By = obs['By']
        if isinstance(By, torch.Tensor):
            if By.dim() == 2:
                By = By.squeeze(0)
            By = By.numpy()
        By_true = torch.from_numpy(By.astype(np.float32)).unsqueeze(1).to(self.device)
        By_true = By_true * self.model.B_scale  # Tesla → scaled

        # Derive B-field from A_z via autodiff (pass physical params for conditioning)
        params = sample.get('parameters', None)
        Bx_pred, By_pred = self.model.compute_B_field(x, y, params)

        # Scale predicted B to match scaled observations
        Bx_pred = Bx_pred * self.model.B_scale
        By_pred = By_pred * self.model.B_scale


        # MSE loss
        loss = F.mse_loss(Bx_pred, Bx_true) + F.mse_loss(By_pred, By_true)

        return loss

    def compute_physics_loss(self, sample):
        """
        Compute physics residual loss: Ampère's law for magnetostatics.

        Enforces: ∇²A_z = -μ₀ J_z (Poisson equation)

        This is the fundamental PDE for 2D magnetostatics with source current J_z.
        Note: ∇·B = 0 (Gauss's law) is automatically satisfied by B = ∇×A construction.

        Args:
            sample: Data sample dict

        Returns:
            Physics loss (scalar tensor)
        """
        # Get collocation points
        domain = sample['collocation']['domain']
        if isinstance(domain, torch.Tensor):
            # Remove batch dimension if present
            if domain.dim() == 3:
                domain = domain.squeeze(0)
            domain = domain.numpy()
        domain = domain.astype(np.float32)

        x = torch.from_numpy(domain[:, 0:1]).to(self.device)
        y = torch.from_numpy(domain[:, 1:2]).to(self.device)

        # Enable gradient computation
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)

        # Get source current density J_z from FEMM (interpolate to collocation points)
        J_z = self._interpolate_sources(
            sample['femm_sources']['J_z'],
            sample['validation']['grid_coords'],
            (x.cpu().detach().numpy(), y.cpu().detach().numpy())
        )
        J_z = torch.from_numpy(J_z.astype(np.float32)).to(self.device)

        # Forward pass: predict A_z (conditioned on physical parameters)
        params = sample.get('parameters', None)
        Az = self.model(x, y, params)

        # Compute first derivatives
        dAz_dx = torch.autograd.grad(
            Az, x, grad_outputs=torch.ones_like(Az),
            create_graph=True, retain_graph=True
        )[0]

        dAz_dy = torch.autograd.grad(
            Az, y, grad_outputs=torch.ones_like(Az),
            create_graph=True, retain_graph=True
        )[0]

        # Compute second derivatives (Laplacian: ∇²A_z = ∂²A_z/∂x² + ∂²A_z/∂y²)
        d2Az_dx2 = torch.autograd.grad(
            dAz_dx, x, grad_outputs=torch.ones_like(dAz_dx),
            create_graph=True, retain_graph=True
        )[0]

        d2Az_dy2 = torch.autograd.grad(
            dAz_dy, y, grad_outputs=torch.ones_like(dAz_dy),
            create_graph=True, retain_graph=True
        )[0]

        laplacian_Az = d2Az_dx2 + d2Az_dy2

        # Ampère's law residual: ∇²A_z + μ₀ J_z = 0
        mu_0 = 4 * np.pi * 1e-7  # Permeability of free space [H/m]

        # Unit conversion note:
        # Network output A_z has units [T·mm] (from data_loss: ∂A_z/∂y_mm = Bx [T]).
        # The correct Poisson equation in mm-network coordinates is:
        #   laplacian_Az * 1e3 + μ₀ J_z = 0  (both sides in [T/m])
        # because ∂²A_z_SI/∂x_m² = laplacian_Az_mm × 1e3 (exact, verified by chain rule).
        #
        # HOWEVER, with coord_scale_factor=1e3, physics_raw ≈ 1e-6 at initialization,
        # requiring lambda ≈ 1e9 to balance with data_loss ≈ 1600 — impractical.
        # Using 1e6 gives physics_raw ≈ 4e-3 (practical range), but the Laplacian term
        # then dominates μ₀J_z by ~1000×, making the constraint enforce ∇²A_z≈0 (Laplace)
        # rather than the correct Poisson equation. Physics loss should be used with
        # lambda_physics=0 (data_only) until a proper normalization scheme is implemented.
        coord_scale_factor = 1e6  # practical scale (NOT the correct SI 1e3); see note above

        # Ampère residual (do NOT multiply by B_scale — was Bug 4; amplified by 1e12)
        ampere_residual = (laplacian_Az * coord_scale_factor +
                          mu_0 * J_z)

        # Physics loss: MSE of residual
        physics_loss = torch.mean(ampere_residual ** 2)

        return physics_loss

    def compute_boundary_loss(self, sample):
        """
        Enforce boundary conditions on A_z.

        For far-field boundary: A_z → 0 (magnetic potential decays far from sources)

        Args:
            sample: Data sample dict

        Returns:
            Boundary condition loss (scalar tensor)
        """
        # Get boundary collocation points
        boundary = sample['collocation']['boundary']
        if isinstance(boundary, torch.Tensor):
            if boundary.dim() == 3:
                boundary = boundary.squeeze(0)
            boundary = boundary.numpy()
        boundary = boundary.astype(np.float32)

        x_bc = torch.from_numpy(boundary[:, 0:1]).to(self.device)
        y_bc = torch.from_numpy(boundary[:, 1:2]).to(self.device)

        # Predict A_z at boundary (conditioned on physical parameters)
        params = sample.get('parameters', None)
        Az_bc = self.model(x_bc, y_bc, params)

        # Boundary condition: A_z = 0 at far boundary
        # This is physically reasonable since we're far from sources
        Az_target = torch.zeros_like(Az_bc)

        # Bug fix: do NOT multiply by B_scale² here.
        # The data loss operates on B = dAz/dy (Tesla), not on A_z directly.
        # Scaling A_z by B_scale² inflated bc_loss by ~1e12, completely
        # dominating the optimizer and preventing any learning.
        # lambda_bc in the trainer controls the relative weight instead.
        bc_loss = F.mse_loss(Az_bc, Az_target)

        return bc_loss

    def train_epoch(self, dataloader):
        """
        Train for one epoch with all loss components.

        Args:
            dataloader: Training data loader

        Returns:
            Tuple of (total_loss, data_loss, physics_loss, bc_loss)
        """
        self.model.train()

        total_loss_sum = 0
        data_loss_sum = 0
        physics_loss_sum = 0
        bc_loss_sum = 0
        n_batches = 0

        for batch in dataloader:
            # Note: batch is a list with one sample (batch_size=1)
            sample = batch[0] if isinstance(batch, list) else batch

            self.optimizer.zero_grad()

            # Three loss components
            L_data = self.compute_data_loss(sample)
            L_physics = self.compute_physics_loss(sample)
            L_bc = self.compute_boundary_loss(sample)

            # Combined loss with weights
            L_total = L_data + self.lambda_physics * L_physics + self.lambda_bc * L_bc

            # Backpropagation
            L_total.backward()
            self.optimizer.step()

            # Accumulate losses
            total_loss_sum += L_total.item()
            data_loss_sum += L_data.item()
            physics_loss_sum += L_physics.item()
            bc_loss_sum += L_bc.item()
            n_batches += 1

        # Average losses
        avg_total = total_loss_sum / n_batches
        avg_data = data_loss_sum / n_batches
        avg_physics = physics_loss_sum / n_batches
        avg_bc = bc_loss_sum / n_batches

        return avg_total, avg_data, avg_physics, avg_bc

    def validate(self, sample):
        """
        Validate against full FEMM solution.

        Computes normalized mean squared error (NMSE) as percentage.

        Args:
            sample: Data sample dict with validation fields

        Returns:
            NMSE percentage
        """
        self.model.eval()

        # Get validation grid (can do this without gradients)
        val = sample['validation']
        xx, yy = val['grid_coords']
        Bx_femm = val['Bx_femm']
        By_femm = val['By_femm']

        # Convert to numpy if needed and squeeze batch dimension
        if isinstance(xx, torch.Tensor):
            if xx.dim() == 3:
                xx = xx.squeeze(0)
            xx = xx.numpy()
        if isinstance(yy, torch.Tensor):
            if yy.dim() == 3:
                yy = yy.squeeze(0)
            yy = yy.numpy()
        if isinstance(Bx_femm, torch.Tensor):
            if Bx_femm.dim() == 3:
                Bx_femm = Bx_femm.squeeze(0)
            Bx_femm = Bx_femm.numpy()
        if isinstance(By_femm, torch.Tensor):
            if By_femm.dim() == 3:
                By_femm = By_femm.squeeze(0)
            By_femm = By_femm.numpy()

        # Flatten grid
        x_flat = torch.from_numpy(xx.flatten().astype(np.float32)[:, None]).to(self.device)
        y_flat = torch.from_numpy(yy.flatten().astype(np.float32)[:, None]).to(self.device)

        # Derive B-field from A_z (need gradients enabled for autodiff)
        # Use torch.enable_grad() to temporarily enable gradients during validation
        params = sample.get('parameters', None)
        with torch.enable_grad():
            x_flat = x_flat.requires_grad_(True)
            y_flat = y_flat.requires_grad_(True)
            Bx_pred, By_pred = self.model.compute_B_field(x_flat, y_flat, params)

        # Convert to numpy
        Bx_pred = Bx_pred.detach().cpu().numpy().flatten()
        By_pred = By_pred.detach().cpu().numpy().flatten()

        # Bug fix: do NOT divide by B_scale here.
        # Training minimises MSE(Bx_raw * B_scale, Bx_true * B_scale), which is
        # equivalent to MSE(Bx_raw, Bx_true) — both sides scale identically so
        # the network learns to output B in Tesla directly (not B_scale × Tesla).
        # Dividing by B_scale made every prediction 1e6× too small, guaranteeing
        # NMSE = 100% regardless of training quality.

        # Reshape to grid
        grid_shape = Bx_femm.shape
        Bx_pred = Bx_pred.reshape(grid_shape)
        By_pred = By_pred.reshape(grid_shape)

        # Compute NMSE
        mse = np.mean((Bx_pred - Bx_femm)**2 + (By_pred - By_femm)**2)
        variance = np.mean(Bx_femm**2 + By_femm**2)

        if variance > 0:
            nmse = (mse / variance) * 100  # Percentage
        else:
            nmse = 0.0

        return nmse

    def train(self, train_dataset, epochs=1000, val_interval=100, save_path=None):
        """
        Full training loop.

        Args:
            train_dataset: PINNDataset instance
            epochs: Number of training epochs
            val_interval: Validation every N epochs
            save_path: Path to save model checkpoints

        Returns:
            Training history dict
        """
        dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        print(f"\n{'='*60}")
        print(f"Training PINN (A_z formulation)")
        print(f"{'='*60}")
        print(f"Formulation: Magnetic vector potential (A_z)")
        print(f"Physics: Ampère's law (∇²A_z = -μ₀J_z)")
        print(f"Epochs: {epochs}")
        print(f"Lambda physics: {self.lambda_physics}")
        print(f"Lambda BC: {self.lambda_bc}")
        print(f"Samples: {len(train_dataset)}")
        print(f"{'='*60}\n")

        for epoch in tqdm(range(epochs), desc="Training"):
            # Train one epoch
            train_loss, data_loss, physics_loss, bc_loss = self.train_epoch(dataloader)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['data_loss'].append(data_loss)
            self.history['physics_loss'].append(physics_loss)
            self.history['bc_loss'].append(bc_loss)

            # Validation
            if (epoch + 1) % val_interval == 0:
                # Validate on first sample
                nmse = self.validate(train_dataset[0])
                self.history['validation_nmse'].append(nmse)

                # Print progress
                print(f'\nEpoch {epoch+1}/{epochs}:')
                print(f'  Loss: {train_loss:.4e} (Data: {data_loss:.4e}, Physics: {physics_loss:.4e}, BC: {bc_loss:.4e})')
                print(f'  NMSE: {nmse:.2f}%')

                # Save checkpoint
                if save_path is not None:
                    checkpoint_path = Path(save_path) / f'checkpoint_epoch_{epoch+1}.pth'
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'history': self.history,
                        'nmse': nmse
                    }, checkpoint_path)

        return self.history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Train PINN on FEMM data'
    )
    parser.add_argument(
        '--problem-type',
        type=str,
        required=True,
        choices=['simple_wire', 'c_core', 'ipm_motor', 'transformer'],
        help='Type of electromagnetic problem'
    )
    parser.add_argument(
        '--n-obs',
        type=int,
        default=100,
        help='Number of sparse observations per sample (default: 100)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Number of training epochs (default: 1000)'
    )
    parser.add_argument(
        '--lambda-physics',
        type=float,
        default=1.0,
        help='Weight for physics loss (default: 1.0)'
    )
    parser.add_argument(
        '--lambda-bc',
        type=float,
        default=10.0,
        help='Weight for boundary condition loss (default: 10.0)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--hidden-layers',
        nargs='+',
        type=int,
        default=[64, 128, 128, 64],
        help='Hidden layer dimensions (default: 64 128 128 64)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/pinn',
        help='Output directory for model and results (default: results/pinn)'
    )
    parser.add_argument(
        '--val-interval',
        type=int,
        default=100,
        help='Validation interval in epochs (default: 100)'
    )

    args = parser.parse_args()

    # Map problem type to PINN data directory
    pinn_prefix = PROBLEM_TYPE_MAP[args.problem_type]
    data_dir = Path(f'dataset/pinn/{pinn_prefix}_n{args.n_obs}')

    # Check if PINN data exists
    if not data_dir.exists() or len(list(data_dir.glob('sample_*.pkl'))) == 0:
        raise FileNotFoundError(
            f"\nPINN training data not found at: {data_dir}\n\n"
            f"Please generate PINN data first using:\n"
            f"  python generate_pinns_data.py \\\n"
            f"    --problem-type {pinn_prefix} \\\n"
            f"    --n-samples 100 \\\n"
            f"    --obs-counts {args.n_obs}"
        )

    print(f"Loading PINN data from: {data_dir}")

    print(f"""
{'='*60}
PINN Training Configuration
{'='*60}
Problem type: {args.problem_type}
Data directory: {data_dir}
N observations: {args.n_obs}
Epochs: {args.epochs}
Lambda physics: {args.lambda_physics}
Lambda BC: {args.lambda_bc}
Learning rate: {args.lr}
Hidden layers: {args.hidden_layers}
Output directory: {args.output_dir}
{'='*60}
    """)

    # Load dataset
    try:
        dataset = PINNDataset(str(data_dir))
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Get problem-specific B-field scaling
    b_scale = B_SCALE_MAP[args.problem_type]
    print(f"\nUsing B-field scale: {b_scale:.0e} (problem-specific normalization)")

    # Create model
    # problem_type key for parameter conditioning (coil / transformer / ipm_motor)
    pinn_problem_type = pinn_prefix  # e.g. 'coil', 'transformer', 'ipm_motor'
    model = PINN(hidden_layers=args.hidden_layers, normalize=True, b_scale=b_scale,
                 problem_type=pinn_problem_type)
    print(f"\nModel architecture:")
    print(model)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {n_params:,}")

    # Create trainer
    trainer = PINNTrainer(
        model,
        lambda_physics=args.lambda_physics,
        lambda_bc=args.lambda_bc,
        lr=args.lr
    )

    # Train
    try:
        history = trainer.train(
            dataset,
            epochs=args.epochs,
            val_interval=args.val_interval,
            save_path=args.output_dir
        )

        # Save final model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / 'pinn_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f'\nModel saved to {model_path}')

        # Save history
        history_path = output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f'History saved to {history_path}')

        # Print final results
        if len(history['validation_nmse']) > 0:
            final_nmse = history['validation_nmse'][-1]
            print(f"\n{'='*60}")
            print(f"Training Complete!")
            print(f"{'='*60}")
            print(f"Final NMSE: {final_nmse:.2f}%")
            print(f"{'='*60}\n")

    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Stopping training...")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
