"""Precompute femm_solver.MeshInterpolator outputs once per CNN sample.

Runs the book's own MeshInterpolator (unchanged) on every sample of a dataset pickle for each
requested bounds box, and stores exactly the fields the training code reads:
B_magnitude and source_current_density (float64, as returned) and the semantic mask (uint8).
Usage (from C2/): python ../_runs/gridcache/build_grid_cache.py <dataset.pkl> <out_dir> <workers>
"""
import hashlib, json, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.getcwd())
from femm_solver import MeshInterpolator  # noqa: E402

RES = 256
FIELDS = ('B_magnitude', 'source_current_density')


def mesh_key(mesh):
    return hashlib.sha1(np.ascontiguousarray(mesh.node_coords).tobytes()).hexdigest()


def work(args):
    i, mesh, bounds_list = args
    t = time.perf_counter()
    interp = MeshInterpolator(mesh)              # build interpolators once, reuse for every bounds
    out = []
    for b in bounds_list:
        g = interp.interpolate_to_grid(resolution=RES, bounds=b)
        xx, yy = g['grid_coords']
        out.append((g['B_magnitude'], g['source_current_density'], interp.generate_semantic_mask(xx, yy)))
    return i, mesh_key(mesh), out, time.perf_counter() - t


def main():
    pkl, out_dir, workers = sys.argv[1], sys.argv[2], int(sys.argv[3])
    sys.argv = ['x']
    import train_cnn  # for the book's compute_global_bounds
    samples = pickle.load(open(pkl, 'rb'))
    samples = samples if isinstance(samples, list) else samples['samples']
    bounds_list = [tuple(float(v) for v in train_cnn.compute_global_bounds(samples)),   # train_cnn.py
                   (-500.0, 500.0, -500.0, 500.0)]                                     # 04c (metadata bounds_mm)
    n = len(samples)
    os.makedirs(out_dir, exist_ok=True)
    arrs = []
    for k in range(len(bounds_list)):
        d = os.path.join(out_dir, f'bounds{k}')
        os.makedirs(d, exist_ok=True)
        arrs.append({f: np.lib.format.open_memmap(os.path.join(d, f + '.npy'), 'w+', np.float64, (n, RES, RES))
                     for f in FIELDS})
        arrs[k]['mask'] = np.lib.format.open_memmap(os.path.join(d, 'mask.npy'), 'w+', np.uint8, (n, RES, RES))
    keys = [None] * n
    t0 = time.time(); busy = 0.0
    with Pool(workers) as pool:
        jobs = ((i, s['mesh_data'], bounds_list) for i, s in enumerate(samples))
        for done, (i, key, out, dt) in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            keys[i] = key; busy += dt
            for k, (B, J, m) in enumerate(out):
                arrs[k]['B_magnitude'][i] = B; arrs[k]['source_current_density'][i] = J; arrs[k]['mask'][i] = m
            if done % 25 == 0 or done == n:
                el = time.time() - t0
                print(f'{time.strftime("%H:%M:%S")} {done}/{n} samples | {el/60:.1f} min elapsed | '
                      f'{busy/done:.2f} s/sample/worker | ETA {(n-done)*el/done/60:.1f} min', flush=True)
    for a in arrs:
        for v in a.values(): v.flush()
    json.dump({'resolution': RES, 'bounds': bounds_list, 'keys': keys, 'fields': list(FIELDS) + ['mask']},
              open(os.path.join(out_dir, 'index.json'), 'w'))
    print('CACHE_DONE', flush=True)


if __name__ == '__main__':
    main()
