"""Run notebook 04c's dense or dilated U-Net training headlessly (--variant).

Executes the notebook's own cells verbatim (cell 3 data, cell 6 model, cell 8 training utils),
then the dense branch of cell 16 (UNetField, dilations (1,1,1,1), Adam 1e-3, 50 epochs),
saves the weights where cell 16 expects them (C2/cnn_dense_unet.pth) and the history as JSON.
Extras printed here (not in the notebook): reference losses, test metrics in Tesla, per-sample
relative error, a prediction figure under _runs/c2_04c_dense/.
"""
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument('--epochs', type=int, default=50)
ap.add_argument('--variant', choices=['dense', 'dilated'], default='dense',
                help='dense = encoder dilations (1,1,1,1); dilated = (1,2,1,2), as in cell 16')
ap.add_argument('--check-only', action='store_true', help='run the data cell only and exit')
ap.add_argument('--out', default=None)
args = ap.parse_args()
DIL = {'dense': (1, 1, 1, 1), 'dilated': (1, 2, 1, 2)}[args.variant]
WEIGHTS = f'cnn_{args.variant}_unet.pth'
args.out = args.out or f'/home/akhan/code/compem_book/_runs/c2_04c_{args.variant}'
os.makedirs(args.out, exist_ok=True)

C2 = '/home/akhan/code/compem_book/C2'
os.chdir(C2); sys.path.insert(0, C2)
import numpy as np, torch
torch.manual_seed(42); np.random.seed(42)

nb = json.load(open('04c_cnn_training.ipynb', encoding='utf-8'))
cell = lambda i: ''.join(nb['cells'][i]['source'])
g = {'__name__': '__nb04c__'}
t0 = time.time(); exec(compile(cell(3), '04c cell 3', 'exec'), g); print(f"[runner] cell 3 (data) done in {time.time()-t0:.1f}s")
exec(compile(cell(6), '04c cell 6', 'exec'), g)
exec(compile(cell(8), '04c cell 8', 'exec'), g)

GRIDS, B_max, train_idx, val_idx, test_idx = g['GRIDS'], g['B_max'], g['train_idx'], g['val_idx'], g['test_idx']
train_ds, val_ds, test_ds = g['train_ds'], g['val_ds'], g['test_ds']
print(f"[runner] split {len(train_ds)}/{len(val_ds)}/{len(test_ds)}, B_max={B_max:.6e} T, BATCH_SIZE={g['BATCH_SIZE']}")

# excitation channel really carries the current now?
X0, Y0 = train_ds[0]
nz = (X0[1] != 0).sum().item()
print(f"[runner] sample 0: mask labels {torch.unique(X0[0]).tolist()}, excitation nonzero pixels {nz}, value {X0[1][X0[1]!=0].mean().item() if nz else 0:.4f}")
if args.check_only:
    sys.exit(0)

# reference losses in the notebook's normalised units (|B| / B_max), validation split
mean_train = np.zeros(GRIDS.B.shape[1:], dtype=np.float64)
for i in train_idx: mean_train += GRIDS.B[int(i)]
mean_train /= (len(train_idx) * B_max)
se0 = sem = 0.0
for i in val_idx:
    y = np.asarray(GRIDS.B[int(i)], dtype=np.float64) / B_max
    se0 += np.mean(y ** 2); sem += np.mean((y - mean_train) ** 2)
mse_zero, mse_mean = se0 / len(val_idx), sem / len(val_idx)
print(f"[runner] reference val MSE (normalised): predict-zero {mse_zero:.6e}, predict-mean-train-field {mse_mean:.6e}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = g['UNetField'](in_channels=2, out_channels=1, base=32, encoder_dilations=DIL)
print(f"[runner] {args.variant} UNetField dilations {DIL}, params {sum(p.numel() for p in model.parameters())/1e6:.2f}M, device {device}, epochs {args.epochs}")
t0 = time.time()
model, hist = g['train_model'](model, g['train_loader'], g['val_loader'], device=device,
                               epochs=args.epochs, lr=1e-3, verbose_every=1)
print(f"[runner] training done in {(time.time()-t0)/60:.1f} min; best val {min(hist['val_loss']):.6e} at epoch {int(np.argmin(hist['val_loss']))+1}")
torch.save(model.state_dict(), WEIGHTS)
os.makedirs('models', exist_ok=True)
json.dump(hist, open(f'models/cnn_{args.variant}_unet_history.json', 'w'), indent=2)
print(f"[runner] saved {WEIGHTS} and models/cnn_{args.variant}_unet_history.json")

# test metrics (same formulas as cell 20) + Tesla + per-sample relative error
model.eval(); mse_sum = n = 0; rows = []; preds = []
with torch.no_grad():
    for Xb, Yb in g['test_loader']:
        P = model(Xb.to(device)).cpu()
        mse_sum += torch.sum((P - Yb) ** 2).item(); n += Yb.numel()
        for j in range(len(P)):
            t, p = Yb[j, 0].numpy(), P[j, 0].numpy()
            rows.append((t.max(), np.sqrt(np.mean((p - t) ** 2))))
            if len(preds) < 4: preds.append((Xb[j].numpy(), t, p))
mse = mse_sum / n; nmse = g['evaluate_metrics'](model, g['test_loader'], device=device)['NMSE']
r = np.array(rows)
print(f"{args.variant.capitalize()} U-Net | Test MSE: {mse:.6e} | RMSE: {np.sqrt(mse):.6e} | NMSE: {nmse:.6e}   (RMSE {np.sqrt(mse)*B_max:.3e} T)")
print(f"[runner] per-sample RMSE/peak: median {np.median(r[:,1]/r[:,0])*100:.2f}%, max {np.max(r[:,1]/r[:,0])*100:.2f}%, "
      f"<5%: {(r[:,1]/r[:,0] < 0.05).sum()}/{len(r)}, <10%: {(r[:,1]/r[:,0] < 0.10).sum()}/{len(r)}")
json.dump({'test_mse_norm': mse, 'test_rmse_norm': float(np.sqrt(mse)), 'test_nmse': nmse, 'test_rmse_T': float(np.sqrt(mse) * B_max),
           'B_max': B_max, 'val_mse_zero': mse_zero, 'val_mse_mean_field': mse_mean, 'best_val': min(hist['val_loss']),
           'epochs': args.epochs, 'batch_size': g['BATCH_SIZE'], 'variant': args.variant, 'encoder_dilations': list(DIL)}, open(f'{args.out}/summary.json', 'w'), indent=2)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(len(preds), 4, figsize=(16, 4 * len(preds)))
for k, (X, t, p) in enumerate(preds):
    for c, (img, ttl) in enumerate([(X[0], 'material labels'), (X[1], 'excitation channel'), (t * B_max, 'true |B| [T]'), (p * B_max, f'{args.variant} U-Net |B| [T]')]):
        im = ax[k, c].imshow(img, origin='lower', **({'vmin': 0, 'vmax': t.max() * B_max} if c >= 2 else {})); ax[k, c].set_title(ttl); plt.colorbar(im, ax=ax[k, c], fraction=0.046)
plt.tight_layout(); plt.savefig(f'{args.out}/predictions.png', dpi=80)
print(f"[runner] wrote {args.out}/summary.json and predictions.png")
