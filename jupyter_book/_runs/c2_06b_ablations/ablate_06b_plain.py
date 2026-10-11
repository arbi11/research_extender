"""06b diagnostic, part 2: plain tanh MLP (06a's VanillaPINN input scaling, no Fourier features) + hard Dirichlet BC,
trained with 06b's ProperPINNTrainer losses.  C: lambda_physics=0.02, D: lambda_physics=0 (data only)."""
import sys, os, json, time
os.chdir('/home/akhan/code/compem_book/C2'); sys.path.insert(0, '.')
import torch, torch.nn as nn, torch.nn.functional as F; torch.set_num_threads(1)
variant, E = sys.argv[1], int(sys.argv[2])
nb = json.load(open('06b_pinns_methodology.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g = {'__name__': '__ablate__'}
for i in range(3): exec(compile(cells[i], f'cell{i+1}', 'exec'), g)
ProperPINNTrainer, sample = g['ProperPINNTrainer'], g['sample']

class PlainHardBCPINN(nn.Module):
    """(x/500, y/500) -> tanh MLP -> A_z, times D(x,y) so A_z = 0 on the domain walls. No Fourier features."""
    def __init__(self, hidden=(64, 128, 128, 64)):
        super().__init__(); self.coord_scale = 500.0
        dims = [2, *hidden, 1]; layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2: layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, y, params_dict=None):
        xn, yn = x / self.coord_scale, y / self.coord_scale
        D = F.relu(1.0 - xn**2) * F.relu(1.0 - yn**2)
        return D * self.net(torch.cat([xn, yn], dim=-1))
    def compute_B_field(self, x, y, params_dict=None):
        xr = x.clone().requires_grad_(True); yr = y.clone().requires_grad_(True)
        Az = self.forward(xr, yr, params_dict); ones = torch.ones_like(Az)
        dAz_dx = torch.autograd.grad(Az, xr, ones, create_graph=True, retain_graph=True)[0]
        dAz_dy = torch.autograd.grad(Az, yr, ones, create_graph=True, retain_graph=True)[0]
        return dAz_dy, -dAz_dx

torch.manual_seed(42)
model = PlainHardBCPINN(); lam = {'C': 0.02, 'D': 0.0, 'E': 1.0, 'F': 10.0}[variant]
tr = ProperPINNTrainer(model, lambda_physics=lam, lr=1e-3, n_epochs=E)
print(f'[{variant}] model=PlainHardBCPINN lambda_physics={lam} epochs={E} params={sum(p.numel() for p in model.parameters())}', flush=True)
t0 = time.time()
for ep in range(E):
    total, data, phys = tr.train_epoch_proper(sample)
    if (ep+1) % 250 == 0 or ep == 0:
        nmse, _, _ = tr.validate(sample)
        print(f'[{variant}] ep {ep+1:5d} data={data:.3e} phys={phys:.3e} NMSE={nmse:.1f}% ({time.time()-t0:.0f}s)', flush=True)
nmse, _, _ = tr.validate(sample)
print(f'[{variant}] FINAL NMSE={nmse:.2f}% after {E} epochs ({time.time()-t0:.0f}s)', flush=True)
