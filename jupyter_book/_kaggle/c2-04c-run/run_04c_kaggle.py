"""Kaggle runner: execute C2/04c_cnn_training.ipynb end to end on a GPU.

Inputs: dataset compem-c2-train (dataset/coil_cnn_dataset.pkl + metadata) and compem-c2-04c-aux
(current notebook, python modules, trained dense/dilated U-Net weights + histories, run_notebook.py).
Outputs (/kaggle/working): 04c_executed.ipynb, out/ (baseline checkpoint + history, analysis plots), driver.log.
Set KAGGLE_ROOT to run the same layout locally (dry run).
"""
import glob, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(os.environ.get('KAGGLE_ROOT', '/kaggle'))
AUX = Path(next(glob.iglob(str(ROOT / 'input/**/grid_cache.py'), recursive=True))).parent
DATA = Path(next(glob.iglob(str(ROOT / 'input/**/coil_cnn_dataset.pkl'), recursive=True))).parent
W = ROOT / 'working/C2'
OUT = ROOT / 'working/out'
for d in (W / 'dataset', W / 'models', OUT):
    d.mkdir(parents=True, exist_ok=True)
for f in AUX.iterdir():
    if f.is_file():
        shutil.copy(f, W / f.name)
for h in ('cnn_dense_unet_history.json', 'cnn_dilated_unet_history.json'):
    shutil.move(str(W / h), str(W / 'models' / h))
for f in ('coil_cnn_dataset.pkl', 'coil_cnn_dataset_metadata.json'):
    (W / 'dataset' / f).exists() or os.symlink(DATA / f, W / 'dataset' / f)
print('AUX', AUX, '| DATA', DATA, '| W', W, flush=True)
subprocess.call([sys.executable, '-c', 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")'])
subprocess.call(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])

env = dict(os.environ, C2_DIR=str(W), NB_OUT=str(ROOT / 'working/04c_executed.ipynb'), PYTHONUNBUFFERED='1',
           OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '4'))
t0 = time.time()
with open(ROOT / 'working/driver.log', 'w') as log:
    p = subprocess.Popen([sys.executable, str(W / 'run_notebook.py'), *sys.argv[1:]], cwd=W, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log.write(line); log.flush()
        if 'Warning' not in line and line.strip():
            print(line.rstrip(), flush=True)
    rc = p.wait()
print(f'driver rc={rc} in {(time.time()-t0)/60:.1f} min', flush=True)

for f in glob.glob(str(W / 'models/coil_baseline_unet_04c*')) + glob.glob(str(W / 'analysis_notebook/*')):
    shutil.copy(f, OUT / Path(f).name)
shutil.rmtree(W, ignore_errors=True)   # keep the saved output small (grid cache 1.1 GB, data symlinks, copies)
print('ALL_DONE rc=%d' % rc, flush=True)
sys.exit(rc)
