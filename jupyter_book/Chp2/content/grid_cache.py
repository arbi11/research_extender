"""
grid_cache.py - precompute and cache the FEM-mesh -> uniform-grid conversion for the CNN code.

Why
---
train_cnn.py and 04c_cnn_training.ipynb convert every FEM mesh to a 256x256 grid inside
Dataset.__getitem__, i.e. once per sample per epoch (2-3 s each on this machine), while the
GPU step costs ~0.07 s per sample. Training is therefore ~97 % CPU-bound. The conversion is
deterministic, so it is done once here, stored, and keyed by a hash of
(dataset file content, resolution, bounds, converter version). A later run that finds a
matching cache loads it in well under a second; a mismatch (new dataset, other resolution or
bounds) builds a new cache next to the old one.

Numerics
--------
Identical to femm_solver.MeshInterpolator + generate_semantic_mask:
  * the same scipy LinearNDInterpolator (one Delaunay over the mesh nodes, fill_value=0),
    evaluated once for the three columns [Bx, By, source_current_density] instead of once
    per stored field (14 Delaunays);
  * the same matplotlib TrapezoidMapTriFinder for the material mask, called on the whole
    grid at once instead of 65,536 single-point Python calls.
Verified bit-identical on the coil dataset (see _runs/gridcache_verify.log).

Stored per cache: B.npy float32 (|B|, T), J.npy float32 (source current density, A/m^2),
mask.npy uint8 (0 air, 1 iron, 2 copper, 3 magnet), all shaped (N, R, R), plus meta.json.
float32 is what the training tensors are built from, so nothing is lost.

Library use
-----------
    from grid_cache import GridCache
    gc = GridCache.load_or_build(samples, 'dataset/coil_cnn_dataset.pkl', 256, bounds)
    B, J, mask = gc.B[i], gc.J[i], gc.mask[i]        # memory-mapped (N, R, R) arrays

CLI use (prebuild, e.g. on 2 cores)
-----------------------------------
    python grid_cache.py --dataset dataset/coil_cnn_dataset.pkl --bounds train_cnn --workers 2
    python grid_cache.py --dataset dataset/coil_cnn_dataset.pkl --bounds metadata  --workers 2
    python grid_cache.py --dataset dataset/coil_cnn_dataset.pkl --bounds -500 500 -500 500
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.tri import Triangulation
from scipy.interpolate import LinearNDInterpolator

CONVERTER_VERSION = 1            # bump if the conversion maths/stored fields change
FIELDS = ('Bx', 'By', 'source_current_density')
DEFAULT_ROOT = 'dataset/grid_cache'


# ----------------------------------------------------------------------------- conversion
def make_grid(resolution, bounds):
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    return np.meshgrid(x, y)


def mesh_to_grids(mesh, xx, yy):
    """One mesh -> (B_magnitude float32, source_current_density float32, mask uint8)."""
    names = list(mesh.field_names)
    cols = [names.index(f) for f in FIELDS]
    vals = np.ascontiguousarray(mesh.node_fields[:, cols])
    out = LinearNDInterpolator(mesh.node_coords, vals, fill_value=0.0)(xx, yy)   # (R, R, 3)
    B = np.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2)
    J = out[..., 2]
    tri = Triangulation(mesh.node_coords[:, 0], mesh.node_coords[:, 1], mesh.elements)
    idx = tri.get_trifinder()(xx.ravel(), yy.ravel())
    mask = np.zeros(idx.shape, dtype=np.uint8)
    inside = idx >= 0
    mask[inside] = np.asarray(mesh.element_materials)[idx[inside]]
    return B.astype(np.float32), J.astype(np.float32), mask.reshape(xx.shape)


# ----------------------------------------------------------------------------- hashing
def _fingerprint_file(path, root):
    """blake2b of the dataset file's content. Cached in <root>/fingerprints.json by
    (path, size, mtime) so repeat runs do not re-read the 1 GB file."""
    path = Path(path).resolve()
    st = path.stat()
    key = f'{path}|{st.st_size}|{st.st_mtime_ns}'
    side = Path(root) / 'fingerprints.json'
    table = {}
    if side.exists():
        try:
            table = json.loads(side.read_text())
        except Exception:
            table = {}
    if key in table:
        return table[key]
    h = hashlib.blake2b(digest_size=16)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    table[key] = h.hexdigest()
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(table, indent=1))
    return table[key]


def _as_samples(data):
    if isinstance(data, dict) and 'samples' in data:
        return data['samples']
    if isinstance(data, list):
        return data
    raise ValueError(f'Unexpected dataset format: {type(data)}')


def cache_key(fingerprint, resolution, bounds, n_samples):
    spec = [CONVERTER_VERSION, fingerprint, int(resolution),
            [round(float(b), 6) for b in bounds], int(n_samples), list(FIELDS)]
    return hashlib.sha256(json.dumps(spec).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- worker
_W = {}


def _init_worker(samples, xx, yy):
    _W['samples'], _W['xx'], _W['yy'] = samples, xx, yy


def _work(i):
    return i, mesh_to_grids(_W['samples'][i]['mesh_data'], _W['xx'], _W['yy'])


# ----------------------------------------------------------------------------- cache object
class GridCache:
    def __init__(self, directory, meta, B, J, mask):
        self.directory, self.meta, self.B, self.J, self.mask = Path(directory), meta, B, J, mask

    def __len__(self):
        return self.B.shape[0]

    def __repr__(self):
        return (f'GridCache({self.directory}, n={len(self)}, res={self.meta["resolution"]}, '
                f'bounds={self.meta["bounds"]})')

    # -- lookup --------------------------------------------------------------
    @staticmethod
    def directory_for(dataset_path, resolution, bounds, n_samples, root=DEFAULT_ROOT):
        fp = _fingerprint_file(dataset_path, root)
        return Path(root) / f'r{int(resolution)}_{cache_key(fp, resolution, bounds, n_samples)}', fp

    @classmethod
    def load(cls, directory, expected_meta=None):
        directory = Path(directory)
        meta_path = directory / 'meta.json'
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        if expected_meta:
            for k, v in expected_meta.items():
                if k == 'bounds':
                    if not np.allclose(meta[k], v, rtol=0, atol=1e-9):
                        return None
                elif meta.get(k) != v:
                    return None
        arrs = {f: np.load(directory / f'{f}.npy', mmap_mode='r') for f in ('B', 'J', 'mask')}
        n, r = meta['n_samples'], meta['resolution']
        if any(a.shape != (n, r, r) for a in arrs.values()):
            return None
        return cls(directory, meta, arrs['B'], arrs['J'], arrs['mask'])

    @classmethod
    def load_or_build(cls, samples, dataset_path, resolution, bounds, workers=2,
                      root=DEFAULT_ROOT, verbose=True):
        """Return the cache for (dataset file, resolution, bounds); build it if missing."""
        samples = _as_samples(samples)
        bounds = tuple(float(b) for b in bounds)
        t0 = time.perf_counter()
        directory, fp = cls.directory_for(dataset_path, resolution, bounds, len(samples), root)
        sample_ids = [int(s.get('sample_id', i)) for i, s in enumerate(samples)]
        expected = {'converter_version': CONVERTER_VERSION, 'fingerprint': fp,
                    'resolution': int(resolution), 'bounds': list(bounds),
                    'n_samples': len(samples), 'sample_ids': sample_ids}
        gc = cls.load(directory, expected)
        if gc is not None:
            if verbose:
                print(f'[grid_cache] HIT  {directory}  ({len(gc)} samples, '
                      f'{time.perf_counter() - t0:.2f}s to verify + map)', flush=True)
            return gc
        if verbose:
            print(f'[grid_cache] MISS {directory}  -> building {len(samples)} grids '
                  f'at {resolution}x{resolution} with {workers} worker(s)', flush=True)
        return cls.build(samples, dataset_path, resolution, bounds, directory, expected,
                         workers=workers, verbose=verbose)

    # -- build ---------------------------------------------------------------
    @classmethod
    def build(cls, samples, dataset_path, resolution, bounds, directory, meta,
              workers=2, verbose=True):
        directory = Path(directory)
        tmp = directory.with_name(directory.name + f'.building-{os.getpid()}')
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        n, R = len(samples), int(resolution)
        xx, yy = make_grid(R, bounds)
        B = np.lib.format.open_memmap(tmp / 'B.npy', 'w+', np.float32, (n, R, R))
        J = np.lib.format.open_memmap(tmp / 'J.npy', 'w+', np.float32, (n, R, R))
        M = np.lib.format.open_memmap(tmp / 'mask.npy', 'w+', np.uint8, (n, R, R))
        t0 = time.time()
        try:
            from tqdm import tqdm
        except ImportError:  # pragma: no cover
            tqdm = lambda it, **k: it  # noqa: E731
        ctx = mp.get_context('fork')
        with ctx.Pool(workers, initializer=_init_worker, initargs=(samples, xx, yy)) as pool:
            for i, (b, j, m) in tqdm(pool.imap_unordered(_work, range(n), chunksize=4),
                                     total=n, desc='Building grid cache', disable=not verbose):
                B[i], J[i], M[i] = b, j, m
        for a in (B, J, M):
            a.flush()
        del B, J, M
        meta = dict(meta, dataset_path=str(Path(dataset_path).resolve()), fields=list(FIELDS),
                    dtypes={'B': 'float32', 'J': 'float32', 'mask': 'uint8'},
                    built_at=time.strftime('%Y-%m-%dT%H:%M:%S'), build_seconds=round(time.time() - t0, 1),
                    workers=workers)
        (tmp / 'meta.json').write_text(json.dumps(meta, indent=1))
        if directory.exists():
            shutil.rmtree(directory)
        os.replace(tmp, directory)          # atomic: a half-built cache is never visible
        if verbose:
            print(f'[grid_cache] built {directory} in {meta["build_seconds"]}s', flush=True)
        return cls.load(directory)


# ----------------------------------------------------------------------------- CLI
def _resolve_bounds(args, samples, dataset_path):
    b = args.bounds
    if len(b) == 4:
        return tuple(float(v) for v in b)
    if b == ['train_cnn']:
        import train_cnn                    # the book's own compute_global_bounds()
        return tuple(float(v) for v in train_cnn.compute_global_bounds(samples))
    if b == ['metadata']:
        meta_path = Path(str(dataset_path).replace('.pkl', '_metadata.json'))
        meta = json.loads(meta_path.read_text())
        return tuple(float(v) for v in (meta.get('bounds_mm') or meta['bounds']))
    raise SystemExit('--bounds must be "train_cnn", "metadata" or four numbers xmin xmax ymin ymax')


def main():
    p = argparse.ArgumentParser(description='Prebuild the mesh->grid cache for a CNN dataset.')
    p.add_argument('--dataset', required=True, help='e.g. dataset/coil_cnn_dataset.pkl')
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--bounds', nargs='+', default=['train_cnn'],
                   help='"train_cnn" (compute_global_bounds), "metadata" (bounds_mm from the '
                        'metadata json) or xmin xmax ymin ymax in mm')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--root', default=DEFAULT_ROOT)
    args = p.parse_args()
    sys.argv = sys.argv[:1]                 # keep train_cnn's argparse quiet if imported
    t = time.perf_counter()
    with open(args.dataset, 'rb') as f:
        samples = _as_samples(pickle.load(f))
    print(f'Loaded {len(samples)} samples in {time.perf_counter() - t:.1f}s')
    bounds = _resolve_bounds(args, samples, args.dataset)
    print(f'Bounds (mm): {bounds}')
    gc = GridCache.load_or_build(samples, args.dataset, args.resolution, bounds,
                                 workers=args.workers, root=args.root)
    print(gc)


if __name__ == '__main__':
    main()
