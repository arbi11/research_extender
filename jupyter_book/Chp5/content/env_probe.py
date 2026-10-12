"""
env_probe.py -- cheap diagnostics of the SeqTO-v2 C-core environment (Section 6.8).

Each probe calls FEMM directly through the same `seqTO_v2_env` machinery the RL
scripts use, so its numbers are comparable with the training runs.  Raw results
are written to designs/env_probe/<probe>_<timestamp>.json; the notebooks pick the
latest file per probe.

Usage (from C5/):
    python env_probe.py determinism   --n 5
    python env_probe.py random-search --n 200 --densities 0.1 0.3 0.5 0.7
    python env_probe.py action-effect --reps 25
    python env_probe.py random-policy --episodes 50
    python env_probe.py cell-flips    --from-best designs/dqn_run/best_mask.npy

Every JSON carries `sec_per_solve` (wall-clock per FEMM solve measured during the
probe) so later cost estimates can use the real figure for this machine.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from seqTO_v2_env import (CcoreSeqTOv2Env, _solve_and_extract, build_base_geometry,
                          is_design_cell, ACTIONS, ROWS, COLS, K2, MAX_STEPS_DEFAULT)

PROBE_DIR = Path('designs/env_probe')
BASE_FEM  = 'ccore_seqto_probe.fem'
TMP_FEM   = '_tmp_solve_v2_probe.fem'

DESIGN_CELLS = [(i, j) for i in range(ROWS) for j in range(COLS) if is_design_cell(i, j)]


def _ensure_geometry():
    if not os.path.exists(BASE_FEM):
        build_base_geometry(BASE_FEM)


def _solve(mask):
    """Force (N) for an arbitrary iron mask, timed."""
    t0 = time.perf_counter()
    _, force_N, _ = _solve_and_extract(mask, BASE_FEM, TMP_FEM)
    return float(force_N), time.perf_counter() - t0


def _random_mask(rng, n_iron):
    mask = np.zeros((ROWS, COLS), dtype=np.int32)
    idx = rng.choice(len(DESIGN_CELLS), size=int(n_iron), replace=False)
    for k in idx:
        i, j = DESIGN_CELLS[k]
        mask[i, j] = 1
    return mask


def _write(stem, obj):
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBE_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")
    return path


def _stats(values):
    v = np.asarray(values, dtype=float)
    return {'n': int(v.size), 'mean': float(v.mean()), 'std': float(v.std()),
            'min': float(v.min()), 'median': float(np.median(v)), 'max': float(v.max())}


# ── probes ───────────────────────────────────────────────────────────────────

def probe_determinism(args):
    """Same mask solved n times: is FEMM's force reading reproducible?"""
    _ensure_geometry()
    rng = np.random.default_rng(args.seed)
    mask = _random_mask(rng, round(0.5 * K2))
    forces, times = [], []
    for k in range(args.n):
        f, dt = _solve(mask)
        forces.append(f); times.append(dt)
        print(f"  solve {k + 1}/{args.n}: force = {f:.6f} N  ({dt:.1f} s)")
    out = {
        'probe': 'determinism', 'n': args.n, 'seed': args.seed,
        'mask_iron_count': int(mask.sum()),
        'forces_N': forces,
        'reward_std': float(np.std(forces)),
        'all_equal': bool(len(set(forces)) == 1),
        'sec_per_solve': float(np.mean(times)),
    }
    print(f"  reward_std = {out['reward_std']:.3e}   all_equal = {out['all_equal']}")
    _write('determinism', out)


def probe_random_search(args):
    """Random masks at several iron densities: the ceiling a trivial method reaches."""
    _ensure_geometry()
    rng = np.random.default_rng(args.seed)
    per_density = args.n // len(args.densities)
    samples, times = [], []
    for d in args.densities:
        n_iron = round(d * K2)
        for k in range(per_density):
            mask = _random_mask(rng, n_iron)
            f, dt = _solve(mask)
            times.append(dt)
            samples.append({'density': float(d), 'iron_count': int(mask.sum()), 'force_N': f})
            if (k + 1) % 10 == 0 or k == 0:
                print(f"  density {d:.1f}  sample {k + 1}/{per_density}: force = {f:.1f} N  ({dt:.1f} s)")
    summary = _stats([s['force_N'] for s in samples])
    by_d = {str(d): _stats([s['force_N'] for s in samples if s['density'] == d]) for d in args.densities}
    out = {
        'probe': 'random_search', 'seed': args.seed, 'densities': list(map(float, args.densities)),
        'n_per_density': per_density, 'iron_budget_K2': K2,
        'samples': samples, 'summary': summary, 'by_density': by_d,
        'sec_per_solve': float(np.mean(times)),
    }
    print(f"  n={summary['n']}  mean={summary['mean']:.1f} N  max={summary['max']:.1f} N  "
          f"({out['sec_per_solve']:.1f} s/solve)")
    for d, s in by_d.items():
        print(f"    density {d}: mean={s['mean']:.1f}  median={s['median']:.1f}  max={s['max']:.1f} N")
    _write('random_search', out)


def probe_action_effect(args):
    """Each of the 4 actions as the first step from random starts: are single-step rewards informative?"""
    env = CcoreSeqTOv2Env(max_steps=MAX_STEPS_DEFAULT, random_start=True, seed=args.seed,
                          base_fem_file=BASE_FEM, tmp_fem=TMP_FEM)
    per_action, times = {}, []
    for a in range(4):
        name = ACTIONS[a][0]
        samples = []
        for rep in range(args.reps):
            env.reset()
            start = (int(env.pos_r), int(env.pos_c))
            t0 = time.perf_counter()
            _, reward, _, info = env.step(a)
            times.append(time.perf_counter() - t0)
            samples.append({'start': start, 'force_N': float(info['force']),
                            'iron_count': int(info['iron_count'])})
        forces = [s['force_N'] for s in samples]
        per_action[str(a)] = {'name': name, 'samples': samples, **_stats(forces)}
        print(f"  action {a} {name:5s}: mean={np.mean(forces):6.2f}  std={np.std(forces):5.2f}  "
              f"range=[{min(forces):.1f}, {max(forces):.1f}] N  ({args.reps} reps)")
    out = {'probe': 'action_effect', 'reps': args.reps, 'seed': args.seed,
           'per_action': per_action, 'sec_per_solve': float(np.mean(times))}
    _write('action_effect', out)


def probe_random_policy(args):
    """Uniformly random action sequences through the env: the no-learning floor."""
    env = CcoreSeqTOv2Env(max_steps=args.max_steps, random_start=True, seed=args.seed,
                          base_fem_file=BASE_FEM, tmp_fem=TMP_FEM)
    rng = np.random.default_rng(args.seed)
    episodes, times = [], []
    for ep in range(1, args.episodes + 1):
        env.reset()
        forces, total = [], 0.0
        done = False
        while not done:
            t0 = time.perf_counter()
            _, reward, done, info = env.step(int(rng.integers(4)))
            times.append(time.perf_counter() - t0)
            forces.append(float(info['force'])); total += float(reward)
        episodes.append({'episode': ep, 'reward': total, 'final_force_N': forces[-1],
                         'max_force_N': float(max(forces)), 'steps': len(forces),
                         'iron_count': int(info['iron_count'])})
        print(f"  ep {ep:3d}/{args.episodes}: reward={total:9.1f}  final force={forces[-1]:7.1f} N  "
              f"max step force={max(forces):7.1f} N  iron={info['iron_count']}")
    out = {'probe': 'random_policy', 'episodes': args.episodes, 'max_steps': args.max_steps,
           'seed': args.seed, 'per_episode': episodes,
           'summary_final_force': _stats([e['final_force_N'] for e in episodes]),
           'summary_max_force': _stats([e['max_force_N'] for e in episodes]),
           'summary_reward': _stats([e['reward'] for e in episodes]),
           'sec_per_solve': float(np.mean(times))}
    _write('random_policy', out)


def probe_cell_flips(args):
    """Flip every design cell of a given mask one at a time: the local reward landscape."""
    _ensure_geometry()
    mask = np.load(args.from_best).astype(np.int32)
    base, dt0 = _solve(mask)
    print(f"  base mask: {int(mask.sum())} iron cells, force = {base:.1f} N")
    flips, times = [], [dt0]
    for k, (i, j) in enumerate(DESIGN_CELLS):
        m = mask.copy(); m[i, j] = 1 - m[i, j]
        f, dt = _solve(m); times.append(dt)
        flips.append({'cell': [i, j], 'to_iron': int(m[i, j]), 'force_N': f, 'delta_N': f - base})
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(DESIGN_CELLS)} flips  (last delta {f - base:+.1f} N)")
    deltas = [x['delta_N'] for x in flips]
    out = {'probe': 'cell_flips', 'source_mask': args.from_best, 'base_force_N': base,
           'base_iron_count': int(mask.sum()), 'flips': flips, 'summary_delta': _stats(deltas),
           'n_improving': int(sum(d > 0 for d in deltas)), 'sec_per_solve': float(np.mean(times))}
    print(f"  {out['n_improving']}/{len(flips)} single flips improve the force; "
          f"best delta {max(deltas):+.1f} N, worst {min(deltas):+.1f} N")
    _write('cell_flips', out)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='SeqTO-v2 environment probes (Section 6.8)')
    sub = p.add_subparsers(dest='probe', required=True)

    s = sub.add_parser('determinism');   s.add_argument('--n', type=int, default=5)
    s.add_argument('--seed', type=int, default=0); s.set_defaults(func=probe_determinism)

    s = sub.add_parser('random-search'); s.add_argument('--n', type=int, default=200)
    s.add_argument('--densities', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7])
    s.add_argument('--seed', type=int, default=0); s.set_defaults(func=probe_random_search)

    s = sub.add_parser('action-effect'); s.add_argument('--reps', type=int, default=25)
    s.add_argument('--seed', type=int, default=0); s.set_defaults(func=probe_action_effect)

    s = sub.add_parser('random-policy'); s.add_argument('--episodes', type=int, default=50)
    s.add_argument('--max-steps', type=int, default=MAX_STEPS_DEFAULT)
    s.add_argument('--seed', type=int, default=0); s.set_defaults(func=probe_random_policy)

    s = sub.add_parser('cell-flips');    s.add_argument('--from-best', type=str, required=True)
    s.set_defaults(func=probe_cell_flips)

    args = p.parse_args()
    print('=' * 64); print(f"  env_probe: {args.probe}"); print('=' * 64)
    t0 = time.time()
    args.func(args)
    print(f"  done in {time.time() - t0:.0f} s")


if __name__ == '__main__':
    main()
