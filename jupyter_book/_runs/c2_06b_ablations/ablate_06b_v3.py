"""06b diagnostic, part 3 (CPU). Uses 06b's own data/physics losses (ProperPINNTrainer from the notebook cells 1-3).
G  : plain tanh net + hard BC, inputs (x/500, y/500, r/500, ln((r+1mm)/500))        lambda_physics = argv[2]
H  : A = A_part(r) [analytic wire, nondimensional] + plain tanh net(x/500, y/500);  lambda_physics = argv[2],
     soft BC A=0 on the 100 wall points with weight 1 (the particular solution is not zero on the walls)
Reports NMSE on the full +-600 grid (notebook metric) and inside the +-500 FEMM box."""
import sys, os, json, time
os.chdir('/home/akhan/code/compem_book/C2'); sys.path.insert(0, '.')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F; torch.set_num_threads(1)
variant, lam, E = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
nb = json.load(open('06b_pinns_methodology.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g = {'__name__': '__ablate__'}
for i in range(3): exec(compile(cells[i], f'cell{i+1}', 'exec'), g)
ProperPINNTrainer, sample, xx, yy, Bx_femm, By_femm = g['ProperPINNTrainer'], g['sample'], g['xx'], g['yy'], g['Bx_femm'], g['By_femm']
r_w_mm = float(sample['parameters']['wire_radius']) * 1000.0

def mlp(n_in, hidden=(64, 128, 128, 64)):
    dims = [n_in, *hidden, 1]; layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims) - 2: layers.append(nn.Tanh())
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return net

class Base(nn.Module):
    def compute_B_field(self, x, y, params_dict=None):
        xr = x.clone().requires_grad_(True); yr = y.clone().requires_grad_(True)
        Az = self.forward(xr, yr, params_dict); ones = torch.ones_like(Az)
        dAz_dx = torch.autograd.grad(Az, xr, ones, create_graph=True, retain_graph=True)[0]
        dAz_dy = torch.autograd.grad(Az, yr, ones, create_graph=True, retain_graph=True)[0]
        return dAz_dy, -dAz_dx

class RadialHardBC(Base):
    def __init__(self): super().__init__(); self.net = mlp(4)
    def forward(self, x, y, params_dict=None):
        xn, yn = x / 500.0, y / 500.0; r = torch.sqrt(x**2 + y**2 + 1e-6)
        feats = torch.cat([xn, yn, r / 500.0, torch.log((r + 1.0) / 500.0)], dim=-1)
        D = F.relu(1.0 - xn**2) * F.relu(1.0 - yn**2)
        return D * self.net(feats)

class AnalyticPlusCorrection(Base):
    """A_hat = A_part(r) + N(x,y).  A_part in units of mu0 I/(2 pi): -ln(r/R0) outside, -(r^2/r_w^2 - 1)/2 - ln(r_w/R0) inside
    -> r_w^2 lap A_part + 2*1_inside = 0 exactly; B = curl A_part is the infinite-space wire field."""
    def __init__(self, r_w_mm, R0=500.0): super().__init__(); self.net = mlp(2); self.r_w = r_w_mm; self.R0 = R0
    def part(self, r):
        inside = r < self.r_w
        return torch.where(inside, -0.5 * (r**2 / self.r_w**2 - 1.0) - np.log(self.r_w / self.R0), -torch.log(torch.clamp(r, min=1e-3) / self.R0))
    def forward(self, x, y, params_dict=None):
        r = torch.sqrt(x**2 + y**2 + 1e-9)
        return self.part(r) + self.net(torch.cat([x / 500.0, y / 500.0], dim=-1))

torch.manual_seed(42)
model = RadialHardBC() if variant == 'G' else AnalyticPlusCorrection(r_w_mm)
tr = ProperPINNTrainer(model, lambda_physics=lam, lr=1e-3, n_epochs=E)
bnd = sample['collocation']['boundary'].astype(np.float32); xb = torch.from_numpy(bnd[:, 0:1]); yb = torch.from_numpy(bnd[:, 1:2])
box = (np.abs(xx) <= 500) & (np.abs(yy) <= 500)
def box_nmse(Bxp, Byp):
    e = (Bxp - Bx_femm)**2 + (Byp - By_femm)**2; v = Bx_femm**2 + By_femm**2
    return e[box].sum() / v[box].sum() * 100.0
tag = f'{variant}{lam:g}'
print(f'[{tag}] model={type(model).__name__} lambda_physics={lam} epochs={E} params={sum(p.numel() for p in model.parameters())}', flush=True)
t0 = time.time()
for ep in range(E):
    model.train(); tr.optimizer.zero_grad()
    L_data = tr.compute_data_loss(sample); L_phys = tr.compute_physics_loss(sample)
    L = L_data + lam * L_phys
    if variant == 'H':
        L_bc = torch.mean(model(xb, yb)**2); L = L + 1.0 * L_bc
    else:
        L_bc = torch.tensor(0.0)
    L.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); tr.optimizer.step(); tr.scheduler.step()
    if (ep + 1) % 500 == 0 or ep == 0:
        nmse, Bxp, Byp = tr.validate(sample)
        print(f'[{tag}] ep {ep+1:5d} data={L_data.item():.3e} phys={L_phys.item():.3e} bc={L_bc.item():.3e} NMSE={nmse:.2f}% box={box_nmse(Bxp, Byp):.2f}% ({time.time()-t0:.0f}s)', flush=True)
nmse, Bxp, Byp = tr.validate(sample)
print(f'[{tag}] FINAL NMSE={nmse:.2f}% (full grid)  {box_nmse(Bxp, Byp):.2f}% (inside box) after {E} epochs ({time.time()-t0:.0f}s)', flush=True)
