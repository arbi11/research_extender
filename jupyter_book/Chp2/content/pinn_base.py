"""
PINN building blocks used by 06b_pinns_methodology.ipynb.

- PINN: tanh MLP on an input encoding of (x, y) -- 'radial' (x/500, y/500, r/500, ln r: the
  coordinates the 2-D Poisson solution is built from; used by 06b) or the earlier 'fourier'
  (random Fourier features at domain and wire scale) -- with optional physical-parameter
  conditioning; B = curl(A_z) via autograd (compute_B_field).
- PINNDataset: loads the sample_*.pkl files written by generate_pinns_data.py.
- PINNTrainer: device/optimizer setup, the plain data / physics / boundary losses and the
  full-grid validation (NMSE). The notebook subclasses it with wire-scale nondimensional losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pickle
from pathlib import Path


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

    Input encoding (encoding=):
      - 'radial' (default): [x/500, y/500, r/500, ln((r + 1 mm)/500)].  The field of a
                 current-carrying wire is a function of r and the 2-D Green's function of
                 the Poisson equation is ln r, so the network receives the coordinates the
                 solution is built from and only has to learn the smooth correction imposed
                 by the A_z = 0 walls.  A plain tanh MLP on (x/500, y/500) alone cannot
                 resolve the 1/r field of a 14 mm wire in a 1000 mm box.
      - 'fourier': random Fourier features [sin(Bx), cos(Bx)] at domain scale (x/500) and,
                 for the coil, at wire scale (x/r_w) (Tancik et al. 2020).  Kept for
                 reference: with 100 sparse observations the wire-scale features overfit
                 (data loss ~1e-6 while the field away from the points is wrong).
      - params: concatenated raw (after [0,1] normalisation); no coordinate encoding —
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
                 n_fourier=16, problem_type='coil', encoding='radial'):
        """
        Args:
            hidden_layers: MLP hidden layer widths
            normalize: (kept for backward compat; the input encoding replaces coord normalisation)
            b_scale: B-field scaling factor (problem-specific)
            n_fourier: Number of Fourier frequency pairs for (x,y) encoding ('fourier' only)
            problem_type: Key into PARAM_BOUNDS; determines n_params and normalisation
            encoding: 'radial' (x/500, y/500, r/500, ln r) or 'fourier' (random Fourier features)
        """
        super().__init__()
        if encoding not in ('radial', 'fourier'):
            raise ValueError(f"encoding must be 'radial' or 'fourier', got {encoding!r}")

        self.normalize = normalize
        self.coord_scale = 500.0  # Domain is [-500, 500] mm
        self.B_scale = b_scale
        self.n_fourier = n_fourier
        self.problem_type = problem_type
        self.encoding = encoding

        if encoding == 'fourier':
            # Fourier feature matrix B ∈ R^{n_fourier × 2}: fixed random frequencies
            # Input (x_norm, y_norm) maps to [sin(Bx), cos(Bx)] ∈ R^{2*n_fourier}
            torch.manual_seed(0)
            B_mat = torch.randn(n_fourier, 2) * 1.0  # σ=1 — moderate frequency bandwidth (Tancik 2020)
            self.register_buffer('B_mat', B_mat)

            # Geometry-aware Fourier encoding: second set of features using coordinates
            # normalised by wire_radius (x/r_w, y/r_w).
            # Only used for 'coil' since that is the only problem type with a wire_radius param.
            self.use_wire_scale = (problem_type == 'coil')
            if self.use_wire_scale:
                B_mat_wire = torch.randn(n_fourier, 2) * 1.0
                self.register_buffer('B_mat_wire', B_mat_wire)
            coord_dim = 2 * n_fourier + (2 * n_fourier if self.use_wire_scale else 0)
        else:
            self.use_wire_scale = False
            coord_dim = 4   # x/500, y/500, r/500, ln((r + 1 mm)/500)

        # Determine number of physical parameters for this problem type
        bounds = self.PARAM_BOUNDS.get(problem_type, {})
        self.param_names = list(bounds.keys())
        self.param_bounds = list(bounds.values())
        n_params = len(self.param_names)

        # Network input: coordinate features + physical parameters
        in_dim = coord_dim + n_params
        layers = []
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.Tanh())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  # → A_z

        self.net = nn.Sequential(*layers)
        # Xavier init (Glorot & Bengio 2010), the standard choice for tanh MLPs and the same
        # init as the VanillaPINN of 06a, so the two notebooks differ only in what 06b adds.
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode_coords(self, x, y, params_dict=None):
        """
        Coordinate features for (x, y) in mm.

        'radial':  [x/500, y/500, r/500, ln((r + 1 mm)/500)].  r and ln r are the coordinates
                   the wire field and the 2-D Poisson Green's function depend on; the +1 mm
                   keeps the log finite on the axis.
        'fourier': random Fourier features at domain scale and, for coil problems, at wire
                   scale (coordinates normalised by r_w, clamped to ±10 r_w).
        """
        x_norm = x / self.coord_scale   # → [-1, 1]
        y_norm = y / self.coord_scale

        if self.encoding == 'radial':
            r = torch.sqrt(x**2 + y**2 + 1e-6)   # +1e-6 mm²: finite second derivatives at r = 0
            return torch.cat([x_norm, y_norm, r / self.coord_scale,
                              torch.log((r + 1.0) / self.coord_scale)], dim=-1)   # (N, 4)

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
        feat = self._encode_coords(x, y, params_dict)              # (N, 4) radial, or (N, 2*n_fourier)/(N, 4*n_fourier) fourier
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
