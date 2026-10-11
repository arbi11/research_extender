"""Kaggle runner (CPU kernel): execute C2 notebooks 04a, 05, 06a, 06b end to end.

Inputs: dataset compem-c2-pinn-n100 (the 100 coil PINN sample pickles, flat) and compem-c2-nb-aux
(the four notebooks, train_pinns.py, c_core_force_dataset.csv, run_notebook.py).
CPU on purpose: 06b's ProperPINNTrainer.validate builds CPU tensors, so the notebook as written only runs
on a CPU-only machine; 04a/05/06a take < 1 min each anyway.
Outputs (/kaggle/working): <nb>_executed.ipynb per notebook, out/c_core_force_model.pth (04a), driver.log.
Args: optional notebook file names (default: all four) and driver flags such as --smoke.
Set KAGGLE_ROOT to run the same layout locally (dry run).
"""
import glob, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))
AUX = Path(next(glob.iglob(str(ROOT / 'input/**/run_notebook.py'), recursive=True))).parent
PINN = Path(next(glob.iglob(str(ROOT / 'input/**/sample_0000.pkl'), recursive=True))).parent
W = ROOT / 'working/C2'
OUT = ROOT / 'working/out'
for d in (W / 'dataset/pinn', OUT):
    d.mkdir(parents=True, exist_ok=True)
for f in AUX.iterdir():
    if f.is_file():
        shutil.copy(f, W / f.name)
link = W / 'dataset/pinn/coil_n100'
link.exists() or os.symlink(PINN, link)
print('AUX', AUX, '| PINN', PINN, f'({len(list(PINN.glob("sample_*.pkl")))} pickles) | W', W, flush=True)
subprocess.call([sys.executable, '-c', 'import os, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "cpus", os.cpu_count())'])

flags = [a for a in sys.argv[1:] if a.startswith('--')]
nbs = [a for a in sys.argv[1:] if not a.startswith('--')] or [
    '04a_ann_fundamentals.ipynb', '05_uncertainty_quantification.ipynb',
    '06a_pinn_theory_formulation.ipynb', '06b_pinns_methodology.ipynb']
rcs = {}
t_all = time.time()
with open(ROOT / 'working/driver.log', 'w') as log:
    for nb in nbs:
        stem = os.path.splitext(nb)[0]
        env = dict(os.environ, C2_DIR=str(W), NB_OUT=str(ROOT / f'working/{stem}_executed.ipynb'), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', str(os.cpu_count() or 4)))
        t0 = time.time()
        print(f'=== {nb} ===', flush=True); log.write(f'=== {nb} ===\n')
        p = subprocess.Popen([sys.executable, str(W / 'run_notebook.py'), '--nb', nb, *flags], cwd=W, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line); log.flush()
            if 'Warning' not in line and line.strip():
                print(line.rstrip(), flush=True)
        rcs[nb] = p.wait()
        print(f'driver rc={rcs[nb]} for {nb} in {(time.time()-t0)/60:.1f} min', flush=True)
print(f'all notebooks in {(time.time()-t_all)/60:.1f} min: {rcs}', flush=True)

for f in ('c_core_force_model.pth',):
    (W / f).exists() and shutil.copy(W / f, OUT / f)
shutil.rmtree(W, ignore_errors=True)   # keep the saved output small
rc = max(rcs.values()) if rcs else 1
print('ALL_DONE rc=%d' % rc, flush=True)
sys.exit(rc)
