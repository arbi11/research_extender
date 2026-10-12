"""Kaggle runner (T4): execute the Chapter 2 notebooks that train models, end to end.

Inputs: compem-c2-train (dataset/coil_cnn_dataset.pkl + metadata), compem-c2-pinn-n100 (the 100 coil PINN
sample pickles, flat), compem-c2-aux (notebooks, cnn_unet.py, pinn_base.py, grid_cache.py, femm_solver.py,
femm.py import shim, cnn_result_analysis.py, dense/dilated U-Net weights + histories, c_core csv, run_notebook.py).
Outputs (/kaggle/working): <nb>_executed.ipynb per notebook, out/ (c_core_force_model.pth, models/coil_baseline_unet_04c*,
analysis_notebook/*), driver.log.  Args: optional notebook file names (default: all five) and driver flags (--smoke).
Set KAGGLE_ROOT to run the same layout locally (dry run).
"""
import glob, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))
def find(name):
    hits = glob.glob(str(ROOT / f'input/**/{name}'), recursive=True)
    return Path(hits[0]).parent if hits else None
AUX, DATA, PINN = find('run_notebook.py'), find('coil_cnn_dataset.pkl'), find('sample_0000.pkl')
W, OUT = ROOT / 'working/C2', ROOT / 'working/out'
for d in (W / 'dataset/pinn', W / 'models', OUT):
    d.mkdir(parents=True, exist_ok=True)
for f in AUX.iterdir():
    if f.is_file():
        shutil.copy(f, W / f.name)
for h in ('cnn_dense_unet_history.json', 'cnn_dilated_unet_history.json'):
    shutil.move(str(W / h), str(W / 'models' / h))
if DATA:
    for f in ('coil_cnn_dataset.pkl', 'coil_cnn_dataset_metadata.json'):
        (W / 'dataset' / f).exists() or os.symlink(DATA / f, W / 'dataset' / f)
if PINN:
    (W / 'dataset/pinn/coil_n100').exists() or os.symlink(PINN, W / 'dataset/pinn/coil_n100')
print('AUX', AUX, '| DATA', DATA, '| PINN', PINN, '| W', W, flush=True)
subprocess.call([sys.executable, '-c', 'import os, torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", "cpus", os.cpu_count())'])

flags = [a for a in sys.argv[1:] if a.startswith('--')]
nbs = [a for a in sys.argv[1:] if not a.startswith('--')] or [
    '04a_ann_fundamentals.ipynb', '05_uncertainty_quantification.ipynb', '06a_pinn_theory_formulation.ipynb',
    '06b_pinns_methodology.ipynb', '04c_cnn_training.ipynb']
rcs, t_all = {}, time.time()
with open(ROOT / 'working/driver.log', 'w') as log:
    for nb in nbs:
        stem = os.path.splitext(nb)[0]
        env = dict(os.environ, C2_DIR=str(W), NB_OUT=str(ROOT / f'working/{stem}_executed.ipynb'), PYTHONUNBUFFERED='1',
                   OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', str(os.cpu_count() or 4)))
        t0 = time.time(); print(f'=== {nb} ===', flush=True); log.write(f'=== {nb} ===\n')
        p = subprocess.Popen([sys.executable, str(W / 'run_notebook.py'), '--nb', nb, *flags], cwd=W, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            log.write(line); log.flush()
            if 'Warning' not in line and 'frozen modules' not in line and 'PYDEVD' not in line and line.strip():
                print(line.rstrip(), flush=True)
        rcs[nb] = p.wait()
        print(f'driver rc={rcs[nb]} for {nb} in {(time.time()-t0)/60:.1f} min', flush=True)
print(f'all notebooks in {(time.time()-t_all)/60:.1f} min: {rcs}', flush=True)

(W / 'c_core_force_model.pth').exists() and shutil.copy(W / 'c_core_force_model.pth', OUT / 'c_core_force_model.pth')
for sub, pat in (('models', 'models/coil_baseline_unet_04c*'), ('analysis_notebook', 'analysis_notebook/*')):
    for f in glob.glob(str(W / pat)):
        (OUT / sub).mkdir(exist_ok=True); shutil.copy(f, OUT / sub / Path(f).name)
shutil.rmtree(W, ignore_errors=True)   # keep the saved output small (grid cache 0.6 GB, data symlinks, copies)
rc = max(rcs.values()) if rcs else 1
print('ALL_DONE rc=%d' % rc, flush=True); sys.exit(rc)
