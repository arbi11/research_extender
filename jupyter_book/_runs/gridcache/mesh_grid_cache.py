"""Serve precomputed MeshInterpolator grids to unchanged book code.

Importing this module patches femm_solver.MeshInterpolator in place:
  * __init__ no longer builds the ~8 per-field LinearNDInterpolators up front (built lazily on a miss)
  * interpolate_to_grid() returns the cached B_magnitude / source_current_density for a known mesh,
    resolution and bounds; any other field is computed for real on first access
  * generate_semantic_mask() returns the cached mask for the same grid
Anything not in the cache falls through to the original code. Cache dir: $COMPEM_GRID_CACHE.
"""
import atexit, hashlib, json, os
import numpy as np
import femm_solver

_DIR = os.environ.get('COMPEM_GRID_CACHE', 'dataset/grid_cache')
_idx = json.load(open(os.path.join(_DIR, 'index.json')))
_RES = _idx['resolution']
_BOUNDS = [tuple(b) for b in _idx['bounds']]
_ROW = {k: i for i, k in enumerate(_idx['keys'])}
_ARR = [{f: np.load(os.path.join(_DIR, f'bounds{k}', f + '.npy'), mmap_mode='r') for f in _idx['fields']}
        for k in range(len(_BOUNDS))]
STATS = {'grid_hit': 0, 'grid_miss': 0, 'mask_hit': 0, 'mask_miss': 0, 'field_fallback': 0}

MI = femm_solver.MeshInterpolator
_orig_build, _orig_interp, _orig_mask = MI._build_interpolators, MI.interpolate_to_grid, MI.generate_semantic_mask


def _bounds_id(bounds):
    if bounds is None:
        return None
    for k, b in enumerate(_BOUNDS):
        if np.allclose(b, bounds, rtol=0, atol=1e-9):
            return k
    return None


def _init(self, mesh_data):
    self.mesh = mesh_data
    self._cache_row = _ROW.get(hashlib.sha1(np.ascontiguousarray(mesh_data.node_coords).tobytes()).hexdigest())
    self._cache_grid = None


def _getattr(self, name):
    if name == 'interpolators':          # built only when real interpolation is needed
        _orig_build(self)
        return self.__dict__['interpolators']
    raise AttributeError(name)


class _Grid(dict):
    def __init__(self, interp, resolution, bounds, *a):
        super().__init__(*a); self._args = (interp, resolution, bounds)

    def __missing__(self, key):
        STATS['field_fallback'] += 1
        interp, res, bounds = self._args
        full = _orig_interp(interp, res, bounds)
        for k, v in full.items():
            self.setdefault(k, v)
        return self[key]


def _interp(self, resolution=800, bounds=None):
    k = _bounds_id(bounds)
    if self._cache_row is None or k is None or resolution != _RES:
        STATS['grid_miss'] += 1
        return _orig_interp(self, resolution, bounds)
    STATS['grid_hit'] += 1
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[2], bounds[3], resolution)
    xx, yy = np.meshgrid(x, y)
    a, r = _ARR[k], self._cache_row
    g = _Grid(self, resolution, bounds, {'grid_coords': (xx, yy),
                                         'B_magnitude': np.array(a['B_magnitude'][r]),
                                         'source_current_density': np.array(a['source_current_density'][r])})
    self._cache_grid = (k, xx, yy)
    return g


def _mask(self, xx, yy):
    c = self._cache_grid
    if c is not None and xx.shape == c[1].shape and np.array_equal(xx, c[1]) and np.array_equal(yy, c[2]):
        STATS['mask_hit'] += 1
        return np.array(_ARR[c[0]]['mask'][self._cache_row])
    STATS['mask_miss'] += 1
    return _orig_mask(self, xx, yy)


MI.__init__, MI.__getattr__ = _init, _getattr
MI.interpolate_to_grid, MI.generate_semantic_mask = _interp, _mask
atexit.register(lambda: print(f'[grid-cache] {STATS}', flush=True))
print(f'[grid-cache] serving {len(_ROW)} meshes x {len(_BOUNDS)} bounds from {_DIR}', flush=True)
