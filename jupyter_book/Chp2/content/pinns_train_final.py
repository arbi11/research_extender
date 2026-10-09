"""
pinn_train_final.py
===================
Trains PINN on a single fixed sample and produces publication-quality plots:
  1. Loss and NMSE curves (training history)
  2. Field reconstruction comparison: FEMM vs PINN vs absolute error
  3. Scatter plot: predicted vs true B-field magnitude

This is the definitive experiment for the notebook (06b_pinn_implementation_results.ipynb).
Uses coil_n100, sample 0, data_only, 3000 epochs.

Run from the content/ directory:
    python3 pinn_train_final.py

Results → results/pinn_final/
"""

import warnings, json, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

_src = open("train_pinns.py").read()
exec(_src.split("def main():")[0], globals())


class FixedPINNTrainer(PINNTrainer):
    """Boundary loss without B_scale² bug."""
    def compute_boundary_loss(self, sample):
        boundary = sample["collocation"]["boundary"]
        if isinstance(boundary, torch.Tensor):
            if boundary.dim() == 3:
                boundary = boundary.squeeze(0)
            boundary = boundary.numpy()
        boundary = boundary.astype(np.float32)
        x_bc = torch.from_numpy(boundary[:, 0:1]).to(self.device)
        y_bc = torch.from_numpy(boundary[:, 1:2]).to(self.device)
        Az_bc = self.model(x_bc, y_bc)
        return F.mse_loss(Az_bc, torch.zeros_like(Az_bc))


def plot_field_comparison(sample, trainer, out_dir):
    """Generate FEMM vs PINN 2×3 comparison plot."""
    val = sample["validation"]

    Bx_femm = val["Bx_femm"]
    By_femm = val["By_femm"]
    if isinstance(Bx_femm, torch.Tensor):
        Bx_femm = Bx_femm.squeeze(0).numpy()
        By_femm = By_femm.squeeze(0).numpy()

    B_femm = np.sqrt(Bx_femm**2 + By_femm**2)

    grid_coords = val["grid_coords"]
    if isinstance(grid_coords, torch.Tensor):
        grid_coords = grid_coords.squeeze(0)
        xx = grid_coords[0].numpy()
        yy = grid_coords[1].numpy()
    else:
        xx, yy = grid_coords

    # Predict on full grid
    x_flat = torch.from_numpy(xx.flatten().astype(np.float32)[:, None])
    y_flat = torch.from_numpy(yy.flatten().astype(np.float32)[:, None])

    with torch.enable_grad():
        x_flat = x_flat.requires_grad_(True)
        y_flat = y_flat.requires_grad_(True)
        Bx_pred, By_pred = trainer.model.compute_B_field(x_flat, y_flat)

    Bx_pred = Bx_pred.detach().numpy().reshape(Bx_femm.shape)
    By_pred = By_pred.detach().numpy().reshape(By_femm.shape)
    B_pred = np.sqrt(Bx_pred**2 + By_pred**2)

    # Relative error
    B_max = B_femm.max()
    err_Bx = np.abs(Bx_pred - Bx_femm) / (B_max + 1e-30)
    err_By = np.abs(By_pred - By_femm) / (B_max + 1e-30)
    err_B  = np.abs(B_pred  - B_femm)  / (B_max + 1e-30)

    params = sample["parameters"]
    I = params["current"]
    r_w = params["wire_radius"] * 1000  # mm

    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    extent = [xx.min(), xx.max(), yy.min(), yy.max()]
    kw_femm = dict(origin="lower", extent=extent, aspect="auto", cmap="RdBu_r")
    kw_err  = dict(origin="lower", extent=extent, aspect="auto", cmap="hot_r")

    vmax_B  = B_femm.max()
    vmax_Bx = np.abs(Bx_femm).max()
    vmax_By = np.abs(By_femm).max()

    # Row 0: FEMM ground truth
    ax00 = fig.add_subplot(gs[0, 0])
    im = ax00.imshow(B_femm, vmin=0, vmax=vmax_B, **kw_femm)
    plt.colorbar(im, ax=ax00, label="|B| [T]")
    ax00.set_title(f"FEMM |B|  (I={I:.1f}A, r_w={r_w:.1f}mm)")
    ax00.set_xlabel("x [mm]"); ax00.set_ylabel("y [mm]")

    ax01 = fig.add_subplot(gs[0, 1])
    im = ax01.imshow(B_pred, vmin=0, vmax=vmax_B, **kw_femm)
    plt.colorbar(im, ax=ax01, label="|B| [T]")
    ax01.set_title("PINN |B|")
    ax01.set_xlabel("x [mm]")

    ax02 = fig.add_subplot(gs[0, 2])
    im = ax02.imshow(err_B, vmin=0, vmax=0.3, **kw_err)
    plt.colorbar(im, ax=ax02, label="|error| / B_max")
    ax02.set_title("Relative error |B|")
    ax02.set_xlabel("x [mm]")

    # Mark observation points
    obs = sample["observations"]
    obs_coords = obs["coords"]
    if isinstance(obs_coords, torch.Tensor):
        obs_coords = obs_coords.squeeze(0).numpy()
    for ax in [ax00, ax01, ax02]:
        ax.scatter(obs_coords[:, 0], obs_coords[:, 1],
                   s=4, c="lime", alpha=0.7, label="observations")

    # Row 1: Bx comparison
    ax10 = fig.add_subplot(gs[1, 0])
    im = ax10.imshow(Bx_femm, vmin=-vmax_Bx, vmax=vmax_Bx, **kw_femm)
    plt.colorbar(im, ax=ax10, label="Bx [T]")
    ax10.set_title("FEMM Bx")
    ax10.set_xlabel("x [mm]"); ax10.set_ylabel("y [mm]")

    ax11 = fig.add_subplot(gs[1, 1])
    im = ax11.imshow(Bx_pred, vmin=-vmax_Bx, vmax=vmax_Bx, **kw_femm)
    plt.colorbar(im, ax=ax11, label="Bx [T]")
    ax11.set_title("PINN Bx")
    ax11.set_xlabel("x [mm]")

    ax12 = fig.add_subplot(gs[1, 2])
    im = ax12.imshow(err_Bx, vmin=0, vmax=0.3, **kw_err)
    plt.colorbar(im, ax=ax12, label="|error| / B_max")
    ax12.set_title("Relative error Bx")
    ax12.set_xlabel("x [mm]")

    nmse = np.mean((Bx_pred - Bx_femm)**2 + (By_pred - By_femm)**2) / \
           (np.mean(Bx_femm**2 + By_femm**2) + 1e-30) * 100
    fig.suptitle(f"PINN field reconstruction — coil (I={I:.1f}A, r_w={r_w:.1f}mm)\n"
                 f"NMSE = {nmse:.1f}%  |  {obs_coords.shape[0]} sparse observations",
                 fontsize=13, fontweight="bold")

    path = out_dir / "field_comparison.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Field comparison → {path}")
    return B_femm, B_pred, Bx_femm, Bx_pred


def plot_training_history(history, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["data"]) + 1)
    ax1.semilogy(epochs, history["data"], "r-", lw=2, label="data loss")
    if any(v > 0 for v in history.get("physics", [])):
        ax1.semilogy(epochs, history["physics"], "g-", lw=1.5, label="physics loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["nmse_epoch"], history["nmse"], "b-o", lw=2, ms=4)
    ax2.axhline(100, color="gray", ls="--", lw=1, alpha=0.5, label="100% (random baseline)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("NMSE (%)")
    ax2.set_title("Validation NMSE over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / "training_history.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Training history → {path}")


def main():
    out_dir = Path("results/pinn_final")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    n_epochs = 3000
    print_every = 500

    print(f"Loading dataset ...")
    dataset = PINNDataset("dataset/pinn/coil_n100")
    sample0 = dataset[0]
    params = sample0["parameters"]
    n_obs = sample0["observations"]["coords"].shape[0]
    print(f"  Sample 0: I={params['current']:.1f}A  r_w={params['wire_radius']*1000:.1f}mm  n_obs={n_obs}")

    model = PINN(hidden_layers=[64, 128, 128, 64], normalize=True, b_scale=1e6)
    trainer = FixedPINNTrainer(model, lambda_physics=0.0, lambda_bc=0.0, lr=1e-3)
    loader = DataLoader(Subset(dataset, [0]), batch_size=1, shuffle=False)

    history = {"data": [], "physics": [], "bc": [], "total": [], "nmse": [], "nmse_epoch": []}
    t0 = time.time()
    print(f"\nTraining {n_epochs} epochs (single sample, data-only)...")

    for ep in range(n_epochs):
        total, data, phys, bc = trainer.train_epoch(loader)
        history["data"].append(data)
        history["physics"].append(phys)
        history["bc"].append(bc)
        history["total"].append(total)

        if (ep + 1) % 100 == 0:
            nmse = trainer.validate(sample0)
            history["nmse"].append(nmse)
            history["nmse_epoch"].append(ep + 1)
            if (ep + 1) % print_every == 0:
                print(f"  ep {ep+1:5d} | data={data:.3e}  NMSE={nmse:.1f}%")

    final_nmse = trainer.validate(sample0)
    print(f"\n  → Final NMSE: {final_nmse:.2f}%  ({time.time()-t0:.0f}s)")

    # Save model
    torch.save(model.state_dict(), out_dir / "pinn_model.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump({**history, "final_nmse": final_nmse,
                   "params": params, "n_epochs": n_epochs}, f, indent=2)

    # Plots
    plot_training_history(history, out_dir)
    plot_field_comparison(sample0, trainer, out_dir)

    print(f"\nDone. Final NMSE: {final_nmse:.2f}%")


if __name__ == "__main__":
    main()
