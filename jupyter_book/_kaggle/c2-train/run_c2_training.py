# Kaggle runner for Chapter 2 training. Runs the unchanged Notion training scripts
# from the compem-c2-train dataset, then trains 04c's dense/dilated U-Nets by executing
# 04c with cell 7's USE_PRETRAINED_MODEL toggle set to False (the code path the notebook provides).
import json, os, shutil, subprocess, sys, time
from pathlib import Path

SRC = next(Path('/kaggle/input').glob('**/train_cnn.py')).parent
WORK = Path('/kaggle/working/C2')
WORK.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.name == 'dataset':
        (WORK / 'dataset').exists() or os.symlink(f, WORK / 'dataset')
    elif f.is_file():
        shutil.copy(f, WORK / f.name)
LOGS = Path('/kaggle/working/logs'); LOGS.mkdir(exist_ok=True)

def run(name, cmd, env_extra=None):
    env = dict(os.environ, PYTHONUNBUFFERED='1', **(env_extra or {}))
    t = time.time()
    with open(LOGS / f'{name}.log', 'w') as log:
        rc = subprocess.call(cmd, cwd=WORK, stdout=log, stderr=subprocess.STDOUT, env=env)
    print(f'{name}: rc={rc} in {time.time() - t:.0f}s', flush=True)
    return rc

# Log GPU use every 30 s for the whole run.
gpu_log = open(LOGS / 'nvidia_smi.csv', 'w')
subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total',
                  '--format=csv', '-l', '30'], stdout=gpu_log, stderr=subprocess.STDOUT)

subprocess.call([sys.executable, '-c', 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'])

# --num-workers 4: the dataset rebuilds each 256x256 grid from the FEA mesh per sample on CPU,
# so a single worker leaves the GPU idle. Kaggle T4 sessions have 4 vCPUs.
run('train_cnn', [sys.executable, 'train_cnn.py', '--problem-type', 'coil', '--num-workers', '4'])
run('train_pinns', [sys.executable, 'train_pinns.py', '--problem-type', 'simple_wire'])
# pinns_train_final.py mixes CPU and GPU tensors in its final plot, so it runs on CPU.
run('pinns_train_final', [sys.executable, 'pinns_train_final.py'], {'CUDA_VISIBLE_DEVICES': ''})

nb = json.load(open(WORK / '04c_cnn_training.ipynb'))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
src = ''.join(code[6]['source'])
assert 'USE_PRETRAINED_MODEL = True' in src
code[6]['source'] = src.replace('USE_PRETRAINED_MODEL = True', 'USE_PRETRAINED_MODEL = False', 1)
json.dump(nb, open(WORK / '04c_train_unets.ipynb', 'w'), indent=1)
run('04c_train_unets', [sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute',
                        '--allow-errors', '--ExecutePreprocessor.timeout=-1',
                        '--output', '04c_train_unets.executed.ipynb', '04c_train_unets.ipynb'])

os.unlink(WORK / 'dataset')  # keep the 1.6 GB input out of the saved output
print('ALL_DONE', flush=True)
