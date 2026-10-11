"""06b diagnostic: exec notebook cells 1-3 verbatim (fixed notebook), then train an ablation variant on CPU.
A = ProperPINN, lambda_physics=0 (data only)      -> is the physics loss hurting?
B = base PINN (no hard-BC multiplier), lambda=0.02 -> is the hard Dirichlet BC hurting?
C = ProperPINN, lambda=0.02 (reference, same as the notebook, shorter schedule)
"""
import sys, os, json, time
os.chdir('/home/akhan/code/compem_book/C2'); sys.path.insert(0, '.')
import torch; torch.set_num_threads(1)
variant, E = sys.argv[1], int(sys.argv[2])
nb = json.load(open('06b_pinns_methodology.ipynb'))
cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g = {'__name__': '__ablate__'}
for i in range(3): exec(compile(cells[i], f'cell{i+1}', 'exec'), g)
PINN, ProperPINN, ProperPINNTrainer, sample = g['PINN'], g['ProperPINN'], g['ProperPINNTrainer'], g['sample']
torch.manual_seed(42)
if variant == 'A':   model = ProperPINN(hidden_layers=[64,128,128,64], normalize=True, b_scale=1.0, n_fourier=16, problem_type='coil'); lam = 0.0
elif variant == 'B': model = PINN(hidden_layers=[64,128,128,64], normalize=True, b_scale=1.0, n_fourier=16, problem_type='coil'); lam = 0.02
else:                model = ProperPINN(hidden_layers=[64,128,128,64], normalize=True, b_scale=1.0, n_fourier=16, problem_type='coil'); lam = 0.02
tr = ProperPINNTrainer(model, lambda_physics=lam, lr=1e-3, n_epochs=E)
print(f'[{variant}] model={type(model).__name__} lambda_physics={lam} epochs={E}', flush=True)
t0 = time.time()
for ep in range(E):
    total, data, phys = tr.train_epoch_proper(sample)
    if (ep+1) % 250 == 0 or ep == 0:
        nmse, _, _ = tr.validate(sample)
        print(f'[{variant}] ep {ep+1:5d} data={data:.3e} phys={phys:.3e} NMSE={nmse:.1f}% ({time.time()-t0:.0f}s)', flush=True)
nmse, Bx, By = tr.validate(sample)
print(f'[{variant}] FINAL NMSE={nmse:.2f}% after {E} epochs ({time.time()-t0:.0f}s)', flush=True)
