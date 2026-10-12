"""Execute a C2 notebook (default 04c_cnn_training.ipynb) headlessly with per-cell logging (nbclient).

--smoke   inject a cell before the baseline-training cell that swaps the loaders for tiny subsets,
          so every cell runs in ~2 min (checks code paths, not results); output goes to --out only.
--inplace write the executed notebook (with outputs) back over the book notebook when no cell errored.
"""
import argparse, os, re, sys, time
import nbformat
from nbclient import NotebookClient

C2 = os.environ.get('C2_DIR', '/home/akhan/code/compem_book/C2')   # Kaggle sets C2_DIR=/kaggle/working/C2
ap = argparse.ArgumentParser()
ap.add_argument('--nb', default='04c_cnn_training.ipynb', help='notebook file name inside C2_DIR')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--out', default=os.environ.get('NB_OUT'))
ap.add_argument('--inplace', action='store_true')
args = ap.parse_args()
NB = f'{C2}/{args.nb}'
if args.out is None:
    args.out = f"/home/akhan/code/compem_book/_runs/c2_04c_full/{os.path.splitext(args.nb)[0].split('_')[0]}_executed.ipynb"

nb = nbformat.read(NB, as_version=4)
k = None
if args.smoke and not any(c.cell_type == 'code' and 'from train_cnn import UNet' in c.source for c in nb.cells):
    # generic smoke: cap the epoch literals (N_EPOCHS = 3000 / num_epochs = 200) so the loops run in seconds
    n_sub = 0
    for c in nb.cells:
        if c.cell_type == 'code':
            c.source, n = re.subn(r'^(\s*(?:N_EPOCHS|num_epochs)\s*=\s*)\d+', r'\g<1>5', c.source, flags=re.M)
            n_sub += n
    print(f"[driver] smoke mode: capped {n_sub} epoch literal(s) to 5 (not saved in place)")
elif args.smoke:
    # find the baseline-training cell (the one that imports train_cnn's UNet) and inject before it
    k = next(i for i, c in enumerate(nb.cells) if c.cell_type == 'code' and 'from train_cnn import UNet' in c.source)
    nb.cells.insert(k, nbformat.v4.new_code_cell(
        "from torch.utils.data import Subset\n"
        "train_loader = DataLoader(Subset(train_ds, range(8)), batch_size=2, shuffle=True)\n"
        "val_loader   = DataLoader(Subset(val_ds, range(4)), batch_size=2)\n"
        "test_loader  = DataLoader(Subset(test_ds, range(4)), batch_size=2)\n"
        "print('[smoke] tiny loaders injected')\n"))
    print(f"[driver] smoke mode: injected tiny-loader cell at index {k}")

t_start = {}
def on_start(cell, cell_index):
    t_start[cell_index] = time.time()
    if cell.cell_type == 'code':
        first = next((l for l in cell.source.splitlines() if l.strip() and not l.strip().startswith('#')), '')[:80]
        print(f"[driver] {time.strftime('%H:%M:%S')} cell {cell_index:2d} start | {first}", flush=True)
def on_done(cell, cell_index, execute_reply):
    if cell.cell_type != 'code':
        return
    dt = time.time() - t_start.get(cell_index, time.time())
    errs = [o for o in cell.get('outputs', []) if o.get('output_type') == 'error']
    text = ''.join(o.get('text', '') for o in cell.get('outputs', []) if o.get('output_type') == 'stream')
    text = text.replace('\r', '\n'); lines = [l for l in text.splitlines() if l.strip() and 'it/s]' not in l and 's/it]' not in l]
    print(f"[driver] {time.strftime('%H:%M:%S')} cell {cell_index:2d} done in {dt:7.1f}s | {len(errs)} errors", flush=True)
    for l in lines[-12:]:
        print(f"    {l[:160]}", flush=True)
    for e in errs:
        print(f"    ERROR {e.get('ename')}: {str(e.get('evalue'))[:300]}", flush=True)

client = NotebookClient(nb, timeout=None, kernel_name='python3', allow_errors=True,
                        resources={'metadata': {'path': C2}}, on_cell_start=on_start, on_cell_executed=on_done)
t0 = time.time()
client.execute()
n_err = sum(1 for c in nb.cells if c.cell_type == 'code' for o in c.get('outputs', []) if o.get('output_type') == 'error')
if k is not None:
    del nb.cells[k]   # do not keep the injected cell in any saved copy
os.makedirs(os.path.dirname(args.out), exist_ok=True)
nbformat.write(nb, args.out)
print(f"[driver] finished in {(time.time()-t0)/60:.1f} min, {n_err} cell errors, wrote {args.out}")
if args.inplace and not args.smoke:
    if n_err == 0:
        nbformat.write(nb, NB); print(f"[driver] wrote executed notebook in place: {NB}")
    else:
        print("[driver] NOT writing in place because of cell errors")
sys.exit(1 if n_err else 0)
