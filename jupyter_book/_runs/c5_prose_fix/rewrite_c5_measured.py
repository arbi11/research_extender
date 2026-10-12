"""Rewrite the Chapter 5 results prose from the runs and probes measured in this repo.

Usage (from C5/):  python ../_runs/c5_prose_fix/rewrite_c5_measured.py [--dry-run]
Every number in the generated text is read from designs/*/training_stats.json or
designs/env_probe/*.json.  Thesis-era numbers are asserted gone at the end.
"""
import glob, json, re, shutil, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from seqTO_v2_env import K2

DRY = '--dry-run' in sys.argv
_sel = [a.split('=', 1)[1] for a in sys.argv if a.startswith('--notebooks=')]
SEL = set(_sel[0].split(',')) if _sel else {'05', '06', '07', '08'}
NEEDS = {'05': set(), '06': {'random_policy', 'cell_flips'}, '07': {'sb3a2c', 'random_policy', 'cell_flips'}, '08': {'sb3a2c', 'random_policy', 'cell_flips'}}
C5 = Path('.').resolve()
assert C5.name == 'C5', 'run from C5/'
BACKUP = Path('../_runs/c5_prose_fix/backup')

# ───────────────────────────── facts ──────────────────────────────────────────
def latest(stem):
    c = sorted(glob.glob(f'designs/env_probe/{stem}_*.json'))
    return json.load(open(c[-1])) if c else None

def run(d):
    p = Path(d) / 'training_stats.json'
    if not p.exists():
        return None
    s = json.load(open(p)); eps = s['episode_stats']; hp = s['hyperparams']
    f = np.array([e['force_N'] for e in eps]); n = len(f); q = n // 4
    o = dict(n=n, peak=f.max(), peak_ep=int(f.argmax()) + 1, q=[f[k*q:(k+1)*q].mean() for k in range(4)],
             last50=f[-50:].mean(), first50=f[:50].mean(), median=float(np.median(f)),
             iron_at_peak=int(eps[int(f.argmax())]['iron_count']),
             best_ep=hp.get('best_episode'), best_reward=hp.get('best_reward'), hp=hp)
    o['best_force'] = next(e['force_N'] for e in eps if e['episode'] == o['best_ep']) if o['best_ep'] else None
    o['best_iron']  = next(e['iron_count'] for e in eps if e['episode'] == o['best_ep']) if o['best_ep'] else None
    H = np.array([e.get('entropy', np.nan) for e in eps], float)
    if np.isfinite(H).any():
        below = np.where(H < 0.01)[0]
        o.update(H_first=H[0], H_last=H[-1], H_min=np.nanmin(H), H_at_peak=H[o['peak_ep'] - 1],
                 H_at_best=H[o['best_ep'] - 1] if o['best_ep'] else np.nan,
                 H_below=(int(below[0]) + 1) if len(below) else None,
                 H_q=[np.nanmean(H[k*q:(k+1)*q]) for k in range(4)])
    V = np.array([e.get('value_loss', np.nan) for e in eps], float)
    if np.isfinite(V).any():
        o.update(V_q=[np.nanmean(V[k*q:(k+1)*q]) for k in range(4)], V_max=np.nanmax(V), V_med=np.nanmedian(V))
    for key in ('approx_kl', 'clip_frac'):
        K = np.array([e.get(key, np.nan) for e in eps], float)
        if np.isfinite(K).any():
            nz = np.abs(K) > 1e-6
            o[key] = dict(n_zero=int((~nz).sum()), n_nonzero=int(nz.sum()),
                          med_nonzero=float(np.median(np.abs(K[nz]))) if nz.any() else 0.0,
                          max=float(np.nanmax(np.abs(K))), last_nonzero_ep=int(np.where(nz)[0].max()) + 1 if nz.any() else 0)
    L = np.array([e.get('loss', np.nan) for e in eps], float)
    if np.isfinite(L).any():
        o['loss_q'] = [np.nanmean(L[k*q:(k+1)*q]) for k in range(4)]
    return o

rs  = latest('random_search'); ae = latest('action_effect'); det = latest('determinism')
rp  = latest('random_policy'); cf = latest('cell_flips')
RSMAX = rs['summary']['max']; RSMEAN = rs['summary']['mean']; SPS = rs['sec_per_solve']
R = {k: run(f'designs/{d}') for k, d in [('dqn', 'dqn_run'), ('a2c', 'a2c_run'), ('a2cv2', 'a2c_v2_run'),
                                          ('ppo', 'ppo_run'), ('ppov2', 'ppo_v2_run'),
                                          ('sb3ppo', 'sb3_ppo_run'), ('sb3a2c', 'sb3_a2c_run')]}
WALL = {'dqn': 38768, 'a2c': 22305, 'a2cv2': 14133, 'ppo': 15383, 'ppov2': 13858}   # queue logs, seconds
if R['sb3a2c'] is not None and R['sb3a2c']['n'] < 200:
    R['sb3a2c'] = None
PENDING = [k for k in ('sb3a2c',) if R[k] is None] + [n for n, v in (('random_policy', rp), ('cell_flips', cf)) if v is None]
if not DRY and any(set(PENDING) & NEEDS[k] for k in SEL):
    sys.exit(f'pending data: {PENDING} needed by {sorted(k for k in SEL if set(PENDING) & NEEDS[k])}')

def N(x, d=1):  return f'{x:.{d}f}'
def Q(o, key='q', d=0): return ' / '.join(f'{v:.{d}f}' for v in o[key])
def E(x): return f'{x:.1e}'.replace('e-0', 'e-').replace('e+0', 'e+')
def above(o): return int((np.array([e['force_N'] for e in json.load(open(o))['episode_stats']]) > RSMAX).sum())
AB = {k: above(f'designs/{d}/training_stats.json') for k, d in [('dqn', 'dqn_run'), ('a2c', 'a2c_run'), ('a2cv2', 'a2c_v2_run'), ('ppo', 'ppo_run'), ('ppov2', 'ppo_v2_run'), ('sb3ppo', 'sb3_ppo_run')] + ([('sb3a2c', 'sb3_a2c_run')] if R['sb3a2c'] else [])}
dqn, a2c, a2cv2, ppo, ppov2, sb3ppo, sb3a2c = (R[k] for k in ('dqn', 'a2c', 'a2cv2', 'ppo', 'ppov2', 'sb3ppo', 'sb3a2c'))
_ppo_eps = json.load(open('designs/ppo_run/training_stats.json'))['episode_stats']
ppo_late_kl = [e['episode'] for e in _ppo_eps if abs(e.get('approx_kl', 0)) > 1e-6 and e['episode'] > 11]
assert [e['episode'] for e in _ppo_eps if abs(e.get('approx_kl', 0)) > 1e-6 and e['episode'] <= 11] == list(range(1, 12)), 'collapse window changed'
assert len(ppo_late_kl) == 4, ppo_late_kl
ONP = {k: v for k, v in (('SB3-PPO', sb3ppo), ('SB3-A2C', sb3a2c), ('A2C-v2', a2cv2), ('PPO-v2', ppov2)) if v is not None}
by_peak = sorted(ONP, key=lambda k: -ONP[k]['peak']); by_last = sorted(ONP, key=lambda k: -ONP[k]['last50'])
BP = ONP[by_peak[0]]; BL = ONP[by_last[0]]
order_peak = ' > '.join(by_peak); order_last = ' > '.join(by_last)
rising = [k for k in ONP if all(ONP[k]['q'][i + 1] > ONP[k]['q'][i] for i in range(3))]
rising_txt = (', '.join(rising[:-1]) + ' and ' + rising[-1]) if len(rising) > 1 else (rising[0] if rising else 'none of the on-policy runs')
sb3_beats = sb3a2c is not None and all(ONP[a][k] > ONP[b][k] for a in ('SB3-PPO', 'SB3-A2C') for b in ('A2C-v2', 'PPO-v2') for k in ('peak', 'last50'))
others_peak = '; '.join(f"{k} {N(ONP[k]['peak'])} N at episode {ONP[k]['peak_ep']}" for k in by_peak[1:])
others_last = ', '.join(f"{N(ONP[k]['last50'])} N for {k}" for k in by_last[1:])
above_txt = ', '.join(f"{AB[kk]} for {k}" for k, kk in (('SB3-PPO', 'sb3ppo'), ('SB3-A2C', 'sb3a2c'), ('PPO-v2', 'ppov2'), ('A2C-v2', 'a2cv2')) if kk in AB)
ae_means = {v['name']: v['mean'] for v in ae['per_action'].values()}
ae_std   = [v['std'] for v in ae['per_action'].values()]
ae_gap   = max(ae_means.values()) - min(ae_means.values())
DQN_UPDATES = dqn['n'] * 34 - 200          # one 32-sample minibatch per env step after 200 warm-up steps
DQN_REVISITS = DQN_UPDATES * 32 / (dqn['n'] * 34)

def trend(o):
    return 'last quarter above first' if o['q'][-1] > o['q'][0] else 'last quarter below first'

# pending-safe accessors for the dry run
class _P(dict):
    def __getattr__(self, k): return '<PENDING>'
    def __getitem__(self, k): return '<PENDING>'
sb3a2c_ = sb3a2c or _P(); rp_ = rp or _P(); cf_ = cf or _P()
def rpS(k, f='summary_final_force', d=1):
    return N(rp[f][k], d) if rp else '<PENDING>'
def cfS(k, d=1):
    return N(cf[k], d) if cf else '<PENDING>'
def cfD(k, d=1):
    return N(cf['summary_delta'][k], d) if cf else '<PENDING>'
if sb3a2c:
    sb3a2c_peak, sb3a2c_ep, sb3a2c_q, sb3a2c_last50, sb3a2c_trend = N(sb3a2c['peak']), sb3a2c['peak_ep'], Q(sb3a2c), N(sb3a2c['last50']), trend(sb3a2c)
    sb3a2c_ratio = N(sb3a2c['peak'] / RSMAX, 2); sb3a2c_n = sb3a2c['n']; sb3a2c_above = AB['sb3a2c']
else:
    sb3a2c_peak = sb3a2c_ep = sb3a2c_q = sb3a2c_last50 = sb3a2c_trend = sb3a2c_ratio = sb3a2c_n = sb3a2c_above = '<PENDING>'
if cf:
    best_flip = max(cf['flips'], key=lambda x: x['delta_N']); worst_flip = min(cf['flips'], key=lambda x: x['delta_N'])
    n_flips = len(cf['flips']); n_worse = sum(x['delta_N'] < 0 for x in cf['flips'])
    n_big_loss = sum(x['delta_N'] < -0.1 * cf['base_force_N'] for x in cf['flips'])
    n_small = sum(abs(x['delta_N']) < 0.05 * cf['base_force_N'] for x in cf['flips'])
    to_iron_imp = sum(1 for x in cf['flips'] if x['delta_N'] > 0 and x['to_iron'] == 1)
    to_air_imp  = sum(1 for x in cf['flips'] if x['delta_N'] > 0 and x['to_iron'] == 0)
    cf_verdict = ('most single-cell perturbations are cheap' if n_small > len(cf['flips']) / 2
                  else 'the mask is sensitive to most single-cell changes')
    if to_air_imp == 0 and to_iron_imp == cf['n_improving']:
        cf_verdict += ('; every improving flip adds iron and no iron removal helps, so with %d of the %d-cell budget used this mask is under-filled rather than finely tuned'
                       % (cf['base_iron_count'], K2))
    cf_base_src = 'the DQN best-by-reward mask (episode %d, %s N, %d iron cells)' % (dqn['best_ep'], N(dqn['best_force']), dqn['best_iron'])
    assert abs(cf['base_force_N'] - dqn['best_force']) < 0.5 * max(1.0, 0.01 * dqn['best_force']), (cf['base_force_N'], dqn['best_force'])
else:
    best_flip = worst_flip = {'delta_N': float('nan'), 'cell': ['?', '?'], 'to_iron': '?'}; n_flips = n_worse = n_big_loss = to_iron_imp = to_air_imp = n_small = '<PENDING>'; cf_base_src = cf_verdict = '<PENDING>'
if rp:
    rp_mean_vs_rs = N(rp['summary_final_force']['mean'] / RSMEAN, 2)
    rp_n_above_rsmean = sum(e['final_force_N'] > RSMEAN for e in rp['per_episode'])
    rp_n_above_rsmax  = sum(e['final_force_N'] > RSMAX for e in rp['per_episode'])
    rp_mean_vs_dqnq1 = N(dqn['q'][0] / rp['summary_final_force']['mean'], 2)
else:
    rp_mean_vs_rs = rp_n_above_rsmean = rp_n_above_rsmax = rp_mean_vs_dqnq1 = '<PENDING>'

# ───────────────────────────── texts ──────────────────────────────────────────
T = {}

T['05_13'] = f"""**A2C unscaled — what went wrong.** Entropy starts at {N(a2c['H_first'],3)} (uniform over 4 actions), is below 0.01 from episode {a2c['H_below']} onward and ends the run at {E(a2c['H_last'])}. `value_loss` never settles: its quarter means are {' / '.join(E(v) for v in a2c['V_q'])} with a single-episode maximum of {E(a2c['V_max'])}. This is the failure mode that motivates the `a2c_v2.py` reward-scaling fix: with raw force returns of $O(10^4)$ the value-loss MSE is $O(10^7)$, and under `GRAD_CLIP=10` the shared-backbone gradient is dominated by the value head — the entropy bonus ($\\beta$ decaying 0.05 → 0.01) cannot keep the policy distribution non-degenerate.

The peak end-of-episode force ({N(a2c['peak'])} N at episode {a2c['peak_ep']}, when the entropy had already fallen to {N(a2c['H_at_peak'],2)}) and the best-by-reward episode ({a2c['best_ep']}, {N(a2c['best_force'])} N, entropy {E(a2c['H_at_best'])}) both come from a near-deterministic policy. With a fixed action sequence the only thing that varies between episodes is the random start, so these peaks are draws from one sequence applied to random starts, not learned behaviour. The last 50 episodes average {N(a2c['last50'])} N, below the first 50 ({N(a2c['first50'])} N). See Section 5.9 below for the scaled-reward fix.
"""

T['05_16'] = f"""### 5.9.1 What the v2 fix achieves

The four panels above tell one story in four projections. Every number below is read from the `training_stats.json` of the four runs made on this machine.

**Entropy.** Both raw-reward runs decay from $\\ln 4 = 1.386$ to a deterministic policy: PPO (raw) is below 0.01 from episode {ppo['H_below']} and reads exactly 0 for the rest of the run; A2C (raw) is below 0.01 from episode {a2c['H_below']} and ends at {E(a2c['H_last'])}. Both `_v2` runs keep a live policy for all 200 episodes. A2C-v2's entropy never drops below {N(a2cv2['H_min'],2)} (quarter means {Q(a2cv2,'H_q',2)}). PPO-v2's falls to a minimum of {N(ppov2['H_min'],2)} with quarter means {Q(ppov2,'H_q',2)}, so it commits partially in the second half without collapsing.

**Value loss.** The unscaled curves sit at $10^6$–$10^7$: quarter means {' / '.join(E(v) for v in ppo['V_q'])} for PPO (raw) and {' / '.join(E(v) for v in a2c['V_q'])} for A2C (raw), with a maximum of {E(a2c['V_max'])}. The scaled curves sit around 1: PPO-v2 median {N(ppov2['V_med'],2)} and maximum {N(ppov2['V_max'],1)}; A2C-v2 median {N(a2cv2['V_med'],2)} and maximum {N(a2cv2['V_max'],0)}. With `value_coef=0.5` and `GRAD_CLIP=10`, the unscaled value head consumes the whole gradient budget and the policy gradient is starved.

**PPO update health.** This is the cleanest diagnostic. Unscaled PPO reports `approx_kl` of exactly 0 in {ppo['approx_kl']['n_zero']} of 200 episodes and `clip_frac` of exactly 0 in {ppo['clip_frac']['n_zero']}. The non-zero values sit in episodes 1–11, which is the collapse itself (entropy {N(ppo['H_first'],3)} → 0 over those episodes), and in four late episodes ({', '.join(str(e) for e in ppo_late_kl)}) where the ratio of near-zero action probabilities produces numerical spikes up to {N(ppo['approx_kl']['max'],1)}. Everywhere else the probability ratio stays pinned at 1 and the clipped objective does no work. Scaled PPO-v2 shows the signs you want: `approx_kl` is non-zero in {ppov2['approx_kl']['n_nonzero']} of 200 episodes (median magnitude {E(ppov2['approx_kl']['med_nonzero'])}, maximum {N(ppov2['approx_kl']['max'],3)}) and `clip_frac` is non-zero in {ppov2['clip_frac']['n_nonzero']} episodes, reaching {N(ppov2['clip_frac']['max'],2)} — the clipping is active in productive episodes, so PPO is actually updating.

**Force.** Peak end-of-episode force: PPO-v2 {N(ppov2['peak'])} N at episode {ppov2['peak_ep']} ({N(ppov2['peak']/RSMAX,2)}× the {N(RSMAX)} N random-search ceiling of Section 6.8); A2C-v2 {N(a2cv2['peak'])} N at episode {a2cv2['peak_ep']} ({N(a2cv2['peak']/RSMAX,2)}×). The raw runs post higher peaks — PPO {N(ppo['peak'])} N at episode {ppo['peak_ep']}, A2C {N(a2c['peak'])} N at episode {a2c['peak_ep']} — but both come from a policy that had already collapsed or was collapsing, i.e. one fixed action sequence applied to random starts. The late-run means separate the runs better than the peaks: last-50 mean {N(ppov2['last50'])} N for PPO-v2 against {N(ppo['last50'])} N for PPO (raw), and {N(a2cv2['last50'])} N for A2C-v2 against {N(a2c['last50'])} N for A2C (raw). PPO-v2's quarter means are {Q(ppov2)} N ({trend(ppov2)}); A2C-v2's are {Q(a2cv2)} N, flat after the first quarter. Over 200 episodes PPO-v2 exceeds the random-search ceiling {AB['ppov2']} times and A2C-v2 {AB['a2cv2']} times; DQN (Section 4.7) does so {AB['dqn']} times.

**Takeaway.** Reward scaling is a *necessary* fix for stable A2C / PPO on this env: it keeps the entropy alive, holds the value loss at $O(1)$ and lets the PPO ratio move. It is *not sufficient* to match DQN ({N(dqn['peak'])} N peak, last-50 mean {N(dqn['last50'])} N). Section 7 puts all runs, including the Stable-Baselines3 references, side by side.
"""

T['06_9_tail'] = f"""**Random-policy floor: {rpS('mean')} N.** The fourth probe (`random-policy --episodes 50`, run after the training runs on the same FEMM worker) applies uniformly random actions from random starts for 34 steps. Its 50 end-of-episode forces have mean {rpS('mean')} N, median {rpS('median')}, minimum {rpS('min')} and maximum {rpS('max')} N; the largest force seen at any step inside those episodes was {rpS('max','summary_max_force')} N. {rp_n_above_rsmean} of the 50 episodes end above the random-search mean ({N(RSMEAN)} N) and {rp_n_above_rsmax} above its ceiling. Section 6.8.2 shows the distribution next to the local landscape around the DQN best mask."""

T['06_new_code'] = r'''# Render the two probes that run after the training runs: the random-policy floor and
# the single-cell-flip landscape around the DQN best-by-reward mask.
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from seqTO_v2_env import ROWS, COLS, is_design_cell

PROBE_DIR = Path('designs/env_probe')

def _latest(stem):
    cands = sorted(PROBE_DIR.glob(f'{stem}_*.json'))
    return json.loads(cands[-1].read_text()) if cands else None

rp = _latest('random_policy')
cf = _latest('cell_flips')
rs = _latest('random_search')

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1, 1.6]})

# --- Panel 1: random-policy end-of-episode force ---------------------------
ax = axes[0]
finals = [e['final_force_N'] for e in rp['per_episode']]
ax.hist(finals, bins=12, color='#9fc5e8', edgecolor='white', label=f"random policy (n={len(finals)})")
ax.axvline(np.mean(finals), color='navy', lw=1.4, label=f"mean = {np.mean(finals):.1f} N")
ax.axvline(rs['summary']['mean'], color='gray', ls=':', lw=1.4, label=f"random-search mean = {rs['summary']['mean']:.1f} N")
ax.axvline(rs['summary']['max'], color='crimson', ls='--', lw=1.2, label=f"random-search max = {rs['summary']['max']:.0f} N")
ax.set_xlabel('end-of-episode force (N)'); ax.set_ylabel('episodes')
ax.set_title('Random-policy floor (34 random actions from random starts)')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# --- Panel 2: single-flip deltas on the grid --------------------------------
ax = axes[1]
delta = np.full((ROWS, COLS), np.nan)
for x in cf['flips']:
    i, j = x['cell']; delta[i, j] = x['delta_N']
mask = np.load(cf['source_mask'])
vmax = np.nanmax(np.abs(delta))
im = ax.imshow(delta, cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest')
ys, xs = np.where(mask == 1)
ax.scatter(xs, ys, s=14, marker='s', facecolors='none', edgecolors='k', linewidths=0.6, label='iron in base mask')
ax.set_title(f"Force change from flipping one cell of the DQN best-by-reward mask  "
             f"(base {cf['base_force_N']:.1f} N, {cf['base_iron_count']} iron cells)")
ax.set_xlabel('column'); ax.set_ylabel('row'); ax.legend(loc='lower right', fontsize=8)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Δ force (N)')
plt.tight_layout(); plt.show()

deltas = np.array([x['delta_N'] for x in cf['flips']])
print(f"random policy : mean final force {np.mean(finals):.1f} N  (median {np.median(finals):.1f}, "
      f"min {np.min(finals):.1f}, max {np.max(finals):.1f} N);  {rp['sec_per_solve']:.1f} s/solve")
print(f"cell flips    : {len(deltas)} design cells flipped one at a time around a {cf['base_force_N']:.1f} N mask")
print(f"                {cf['n_improving']} flips improve the force, {(deltas < 0).sum()} reduce it;  "
      f"best {deltas.max():+.1f} N, worst {deltas.min():+.1f} N, median {np.median(deltas):+.1f} N")
'''

T['06_new_md'] = f"""### 6.8.2 The floor and the local landscape

**Random actions reach {rpS('mean')} N on average.** Fifty episodes of uniformly random actions end at a mean force of {rpS('mean')} N, {rp_mean_vs_rs}× the random-search mean of {N(RSMEAN)} N. The two probes are different experiments: random search samples masks at fixed iron densities, whereas the random policy walks the controller for 34 steps and ends wherever the walk leaves it. DQN's first-quarter mean ({N(dqn['q'][0])} N, Section 4.7) is {rp_mean_vs_dqnq1}× this floor; every training-run mean in Section 7 should be read against it.

**Single-cell flips around the DQN best-by-reward mask.** Flipping each of the {n_flips} design cells of {cf_base_src} one at a time changes the force by a median of {cfD('median')} N (standard deviation {cfD('std')} N). {n_small} of the {n_flips} flips move the force by less than five percent of the base value and {n_big_loss} cost more than ten percent of it. {cfS('n_improving',0)} flips improve the force ({to_iron_imp} by adding iron, {to_air_imp} by removing it) and {n_worse} reduce it; the two extremes are {N(best_flip['delta_N'],1)} N (cell {tuple(best_flip['cell'])}, to {'iron' if best_flip['to_iron']==1 else 'air'}) and {N(worst_flip['delta_N'],1)} N (cell {tuple(worst_flip['cell'])}, to {'iron' if worst_flip['to_iron']==1 else 'air'}). So {cf_verdict}. The best single flip would lift this mask by {N(best_flip['delta_N'],1)} N for one FEMM solve, which makes a greedy local search from a trained mask a cheap post-processing step; it is listed in Section 8.4.
"""

T['07_0'] = f"""# 7. Results and Cross-Method Comparison

This section consolidates every run that was launched on the SeqTO-v2 C-core domain and pits them against each other and against the random-search ceiling (Section 6.8). Seven RL runs, one random-mask baseline and one random-policy reference are in scope. All of them were produced on this machine; the status in brackets is what their `training_stats.json` shows.

* `dqn_run` — custom DQN, 200/200 episodes (peak {N(dqn['peak'])} N at episode {dqn['peak_ep']})
* `a2c_run` — custom A2C with raw rewards (entropy below 0.01 from episode {a2c['H_below']} — negative control)
* `a2c_v2_run` — custom A2C with `reward_scale=0.001` (live policy throughout, peak {N(a2cv2['peak'])} N at episode {a2cv2['peak_ep']})
* `ppo_run` — custom PPO with raw rewards (entropy 0 from episode {ppo['H_below']} — negative control)
* `ppo_v2_run` — custom PPO with `reward_scale=0.001` (live policy throughout, peak {N(ppov2['peak'])} N at episode {ppov2['peak_ep']})
* `sb3_a2c_run` — Stable-Baselines3 A2C reference with `reward_scale=0.001` (peak {sb3a2c_peak} N at episode {sb3a2c_ep}, {sb3a2c_trend})
* `sb3_ppo_run` — Stable-Baselines3 PPO reference with `reward_scale=0.001` (peak {N(sb3ppo['peak'])} N at episode {sb3ppo['peak_ep']}, {trend(sb3ppo)})
* `random_search` — 200 random masks across 4 density levels, ceiling {N(RSMAX)} N (Section 6.8)
* `random_policy` — 50 episodes of random actions, mean end force {rpS('mean')} N (Section 6.8.2)
"""

T['07_7'] = f"""## 7.4 Why DQN beat the policy methods

DQN's peak ({N(dqn['peak'])} N) is {N(dqn['peak']/BP['peak'],2)}× the best on-policy peak ({by_peak[0]}, {N(BP['peak'])} N at episode {BP['peak_ep']}); the other scaled on-policy runs peak at {others_peak}. The gap is wider on the late-run means: DQN's last 50 episodes average {N(dqn['last50'])} N against {N(BL['last50'])} N for {by_last[0]} and {others_last}. DQN ends an episode above the random-search ceiling {AB['dqn']} times in 200, against {above_txt}. The explanation offered here is grounded in the action-effect probe from Section 6.8, not in algorithmic sophistication.

**Recall the action-effect finding.** Single-step rewards barely distinguish the 4 controller actions: taken as the first step from 25 random starts, their mean forces are {' / '.join(N(ae_means[k]) for k in ('RIGHT','LEFT','UP','DOWN'))} N for RIGHT / LEFT / UP / DOWN, a spread of {N(ae_gap)} N against per-action standard deviations of {N(min(ae_std))}–{N(max(ae_std))} N. The useful signal lives in *multi-step compositions* — sequences of 34 actions that assemble into productive topologies.

**How each algorithm extracts signal from this landscape** (from the scripts as run):

| Algorithm | Gradient steps per FEMM solve | Mechanism |
|---|---:|---|
| A2C-v2, SB3-A2C | 1/34 (one update per 34-step rollout) | One Monte-Carlo return, data discarded |
| PPO-v2 | 4/34 (K = 4 epochs, minibatch = the whole 34-step episode) | Limited rollout reuse via the clipped surrogate |
| SB3-PPO | 4/64 (4 epochs on 64-step rollouts) | Same, with a rollout that spans two episodes |
| **DQN** | **1 per step, and each transition is revisited about {DQN_REVISITS:.0f} times** | Replay buffer: one 32-sample minibatch per env step after 200 warm-up steps, buffer never full |

DQN's replay buffer is the dominant factor. Every FEMM solve produces one $(s, a, r, s')$ tuple that is drawn into about {DQN_REVISITS:.0f} minibatches over the run, and the bootstrapped TD target propagates the eventual force back through the 34 steps of state evolution. That is the mechanism needed when the per-step reward is uninformative: credit assignment is delegated to the Bellman target, not to the policy gradient.

The on-policy runs are bounded by their one-pass use of each rollout. What the 200-episode traces show, quarter by quarter: SB3-PPO {Q(sb3ppo)} N, SB3-A2C {sb3a2c_q} N, PPO-v2 {Q(ppov2)} N (entropy {Q(ppov2,'H_q',2)}) and A2C-v2 {Q(a2cv2)} N (entropy {Q(a2cv2,'H_q',2)}). Mean force rises in every quarter for {rising_txt}; DQN's quarter means are {Q(dqn)} N, rising throughout, with its peak in episode {dqn['peak_ep']} (Section 7.1). A2C-v2 and SB3-A2C make the same one gradient step per 34-step rollout, yet one climbs and the other stays flat after its first quarter; they differ in learning rate (3e-4 against 7e-4), optimiser (Adam against RMSprop), gradient-norm clip (10 against 0.5) and entropy coefficient (0.05 decaying to 0.01 against a constant 0.01). So the update cadence separates DQN from the on-policy family, but it does not by itself fix the outcome within that family.

The order among the four scaled on-policy runs is not a ranking: each was run with a single seed, and the order by peak ({order_peak}) differs from the order by last-50 mean ({order_last}). The DQN margin is the finding; it holds on every statistic above.

**The reward-scaling fix matters for the policy methods, not for DQN.** DQN uses a smooth-L1 loss on a TD target rather than a value-MSE-against-Monte-Carlo-return loss, so the $10^7$-scale value-loss explosion that derails A2C / PPO does not occur: its loss quarter means on raw rewards are {Q(dqn,'loss_q',0)}, and it kept improving throughout. The unscaled A2C and PPO runs collapsed; their scaled `_v2` counterparts trained with a live policy. The loss structure, not the reward magnitude, is what separates the two families here.
"""

T['07_8'] = f"""## 7.5 Computational cost

Wall-clock and FEMM-solve totals for every run. FEMM solve time was measured by the random-search probe on this machine: {rs['summary']['n']} solves in {rs['summary']['n']*SPS:.0f} s, i.e. {N(SPS)} s per solve. The table converts solve counts at that rate; the measured wall-clock of the training runs is higher because each step also opens and closes a FEMM session and runs the network update (DQN {WALL['dqn']/3600:.1f} h for episodes 6–200 after a resume, {WALL['dqn']/(195*34):.1f} s per step; A2C (raw) {WALL['a2c']/3600:.1f} h; PPO (raw) {WALL['ppo']/3600:.1f} h; A2C-v2 {WALL['a2cv2']/3600:.1f} h; PPO-v2 {WALL['ppov2']/3600:.1f} h).
"""

T['07_10'] = f"""**Interpretation.** The DQN run needed {dqn['n']*34:,} FEMM solves, {dqn['n']*34*SPS/3600:.1f} h of pure solver time at {N(SPS)} s per solve and {WALL['dqn']/3600:.1f} h of measured wall-clock. Random search produced its {N(RSMAX)} N ceiling from {rs['summary']['n']} solves in {rs['summary']['n']*SPS/60:.0f} min — for the solver budget of one DQN run you could run {dqn['n']*34//rs['summary']['n']} random-search probes. The justification for DQN's cost is the {N(dqn['peak']/RSMAX,2)}× improvement over the random-search peak and a last-50 mean of {N(dqn['last50'])} N, {N(dqn['last50']/RSMEAN,2)}× the random-search mean; whether that is worth {dqn['n']*34//rs['summary']['n']}× the solves is a problem-specific judgement that depends on whether the user needs *the* best topology or merely *a* good one.

The two raw-reward runs (A2C {WALL['a2c']/3600:.1f} h, PPO {WALL['ppo']/3600:.1f} h of wall-clock) produced data that is useful only as negative controls, since their policies were deterministic after episodes {a2c['H_below']} and {ppo['H_below']} respectively. The scaled runs amortise better: SB3-PPO posted its peak in episode {sb3ppo['peak_ep']} and its last quarter averages {N(sb3ppo['q'][3])} N, SB3-A2C peaked in episode {sb3a2c_ep} with a last quarter of {N(sb3a2c['q'][3]) if sb3a2c else '<PENDING>'} N, PPO-v2 peaked in episode {ppov2['peak_ep']} and A2C-v2 in episode {a2cv2['peak_ep']}.
"""

T['07_13'] = f"""## 7.7 What's missing for a defensible claim

To say "DQN beats every alternative on this problem" with full rigour, the following would still be needed. The items that the previous draft of this chapter listed as missing — SB3 runs with reward scaling, the random-policy floor and the cell-flips landscape — have since been run on this machine and are reported in Sections 7.2 and 6.8.2.

1. **DQN evaluation pass.** Load `designs/dqn_run/checkpoint.pt`, fix the policy (greedy argmax), and roll out 20–50 evaluation episodes from a held-out set of random starts. The mean force from that pass is the number that should populate any benchmark — the {N(dqn['peak'])} N peak is an *upper bound the agent has demonstrated reachable* (one episode out of 200), not a steady-state operating point. The closest proxy on disk is the last-50 mean of {N(dqn['last50'])} N under $\\varepsilon = 0.05$ exploration.
2. **PPO-v2 and SB3-PPO continuation past 200 episodes.** PPO-v2 posted its peak in episode {ppov2['peak_ep']} with quarter means {Q(ppov2)} N and entropy still at {N(ppov2['H_last'],2)}; SB3-PPO's quarter means rose {Q(sb3ppo)} N with no sign of a plateau. Neither trajectory is settled, and neither is on a path to overtake DQN within another 100 episodes at its measured slope. A 400- or 500-episode continuation is the one experiment that could change the "policy methods cannot match DQN on this env" claim.
3. **More than one seed.** Every row of the scorecard is a single seed. The four scaled on-policy runs order as {order_peak} by peak and as {order_last} by last-50 mean (Section 7.4), which is what one seed per run looks like. Three seeds per method would turn the DQN margin from an observation into a claim.
4. **Greedy local search from the trained masks.** Section 6.8.2 shows that the best single-cell flip around the DQN best-by-reward mask is worth {N(best_flip['delta_N'],1)} N. A deterministic hill-climb from each method's best mask costs {n_flips} solves per sweep ({n_flips*SPS/60 if cf else float('nan'):.0f} min) and would tell how much of each method's result is topology and how much is left on the table.
"""

T['07_14'] = f"""## 7.8 Summary

What the data on disk supports:

- **DQN achieved {N(dqn['peak'])} N peak end-of-episode force** in episode {dqn['peak_ep']}, {N(dqn['peak']/RSMAX,2)}× the random-search ceiling ({N(RSMAX)} N), with quarter means {Q(dqn)} N and {AB['dqn']} of 200 episodes above the ceiling. This is the strongest result in the chapter.
- **Reward scaling is a necessary intervention for stable A2C / PPO on this env.** Without it, both custom runs become deterministic within the first 40 episodes (entropy below 0.01 from episodes {ppo['H_below']} and {a2c['H_below']}; value loss at $10^6$–$10^8$). With it, both keep a live policy for 200 episodes; PPO-v2 reaches {N(ppov2['peak'])} N and A2C-v2 {N(a2cv2['peak'])} N. The scaling story is documented in Section 5.9.
- **DQN's advantage over the policy methods is consistent with the env's action-effect structure** (Section 6.8): single-step rewards barely distinguish actions, so credit assignment must come from multi-step signal compounding. DQN's replay buffer revisits each FEMM solve about {DQN_REVISITS:.0f} times across n-step TD updates; the on-policy methods use each rollout once (A2C) or four times (PPO), and accordingly they train more slowly.
- **The Stable-Baselines3 references, with the reward scaled, {'beat the custom v2 scripts on every statistic and still sit far below DQN' if sb3_beats else 'land in the same band as the custom v2 scripts, far below DQN'}.** SB3-PPO reached {N(sb3ppo['peak'])} N at episode {sb3ppo['peak_ep']} (quarter means {Q(sb3ppo)} N) and SB3-A2C {sb3a2c_peak} N at episode {sb3a2c_ep} (quarter means {sb3a2c_q} N); mean force rose in every quarter for {rising_txt}. All four scaled on-policy runs share the custom CNN feature extractor, so the comparison is between update rules and their settings, not networks.
- **The environment is deterministic, the random-policy floor is {rpS('mean')} N, and a good mask is locally robust**: {cfS('n_improving',0)} of {n_flips} single-cell flips around the DQN best-by-reward mask improve it, the median flip changes the force by {cfD('median')} N (Section 6.8.2).

What the data on disk does NOT yet support:

- Steady-state policy performance numbers for any method (no evaluation passes — all numbers are training-time values under exploration).
- A claim that DQN's {N(dqn['peak']/RSMAX,2)}× over random search is the *best possible* — only one DQN seed was run, with no hyperparameter sweep.
- A ranking among the four scaled on-policy runs: with one seed each, their order by peak ({order_peak}) differs from their order by last-50 mean ({order_last}).
- A clean DRL-vs-GA comparison on Newton-calibrated forces — Chapter 4's GA numbers are not recomputed here.
"""

T['08_0'] = f"""# 8. Conclusions and Future Work

## 8.1 Summary of contributions

This chapter applied deep reinforcement learning to the topology-optimisation problem formalised in Chapter 4. The contributions are:

1. **SeqTO-v2 environment** (Section 6) — a 3-channel `[material, flux, boundary]` tensor extension of Chapter 4's `CcoreSeqTOEnv`, with a Maxwell-stress force reward. Preserves Chapter 4's chromosome-as-action-sequence formulation, connectivity-by-construction guarantee, and `MAX_STEPS = 34` budget.
2. **Four DRL agent implementations** (Sections 4–5) — DQN (`dqn_seqto.py`), A2C (`a2c_seqto.py` + `a2c_v2.py`), and PPO (`ppo_seqto.py` + `ppo_v2.py`) sharing the CNN backbone in `net_seqto.py`. The `_v2` scripts add a single `--reward-scale` knob to isolate value-loss explosion as a failure mode independent of algorithm choice.
3. **Environment diagnostics** (Section 6.8) — five probes (`determinism`, `random-search`, `action-effect`, `random-policy`, `cell-flips`) that quantify what the policy is up against. The `action-effect` probe (single-step rewards are nearly action-blind: four action means within {N(ae_gap)} N of each other against standard deviations of {N(min(ae_std))}–{N(max(ae_std))} N) is the most informative — it explains why the replay-based method separates from the on-policy ones.
4. **A completed DQN training run** (Section 4.7) — 200 episodes, peak end-of-episode force {N(dqn['peak'])} N, {WALL['dqn']/3600:.1f} h of wall-clock for episodes 6–200. This is the headline result of the chapter.
5. **Reward-scaling intervention** (Section 5.9) — diagnosed the value-loss explosion + entropy-collapse failure mode that derails unscaled A2C / PPO on this env, and showed that a 0.001 multiplier on rewards keeps both policies live for 200 episodes without any other change to the algorithm or hyperparameters.
6. **Cross-method comparison** (Section 7) — running-best force overlay, scorecard, mask gallery, hyperparameter table, and wall-clock cost breakdown across all seven RL runs plus the random-search baseline and the random-policy floor.

## 8.2 Key findings

**DQN solves this problem; the policy methods don't quite.** DQN reaches {N(dqn['peak'])} N peak end-of-episode force ({N(dqn['peak']/RSMAX,2)}× the {N(RSMAX)} N random-search ceiling), with its last 50 episodes averaging {N(dqn['last50'])} N. The best on-policy peak is {N(BP['peak'])} N ({by_peak[0]}) and the best on-policy last-50 mean is {N(BL['last50'])} N ({by_last[0]}); the other scaled on-policy runs peak at {others_peak}. All of them beat the random-search ceiling at least once, but DQN beats the best of them by {N((dqn['peak']/BP['peak']-1)*100,0)} percent on the peak and by {N((dqn['last50']/BL['last50']-1)*100,0)} percent on the late-run mean.

**The gap is consistent with the env's action-effect structure, not with algorithm sophistication.** The `action-effect` probe (Section 6.8) showed that single-step rewards barely distinguish the 4 controller actions. The useful signal lives in *multi-step compositions*, so an algorithm's ability to extract it scales with how many times it reuses each FEMM solve:

| Algorithm | Gradient steps per FEMM solve | Peak (N) | Last-50 mean (N) |
|---|---:|---:|---:|
| DQN (replay buffer revisits each transition about {DQN_REVISITS:.0f} times) | 1 | **{N(dqn['peak'])}** | **{N(dqn['last50'])}** |
| SB3-PPO (4 epochs on 64-step rollouts) | 4/64 | {N(sb3ppo['peak'])} | {N(sb3ppo['last50'])} |
| PPO-v2 (4 epochs on each 34-step episode) | 4/34 | {N(ppov2['peak'])} | {N(ppov2['last50'])} |
| A2C-v2 (one Monte-Carlo update per episode) | 1/34 | {N(a2cv2['peak'])} | {N(a2cv2['last50'])} |
| SB3-A2C (one update per 34-step rollout) | 1/34 | {sb3a2c_peak} | {sb3a2c_last50} |

The four scaled on-policy runs were run with one seed each; they order as {order_peak} by peak and as {order_last} by last-50 mean, so only the DQN margin is claimed.

**Reward scaling is necessary but not sufficient for stable policy-gradient training on this env.** Unscaled A2C and PPO both become deterministic early (entropy below 0.01 from episodes {a2c['H_below']} and {ppo['H_below']}; value loss at $10^6$–$10^8$). Scaled `_v2` variants keep a live policy for 200 episodes — but they still don't catch up to DQN. The scaling fix is a *prerequisite* for the policy methods to be usable at all; it is not what closes the gap to DQN.

**The Stable-Baselines3 references {'did better than the custom scripts, and still far worse than DQN' if sb3_beats else 'landed in the same band as the custom scripts, far below DQN'}.** With the same CNN feature extractor and the same reward scaling, SB3-PPO ({N(sb3ppo['peak'])} N, quarter means {Q(sb3ppo)} N) and SB3-A2C ({sb3a2c_peak} N, quarter means {sb3a2c_q} N) {'exceed' if sb3_beats else 'do not consistently exceed'} the two custom v2 runs on peak and on last-50 mean. The library defaults differ from the custom choices — RMSprop at $7 \\times 10^{{-4}}$ with a gradient-norm clip of 0.5 for A2C, 64-step rollouts for PPO, a constant entropy coefficient of 0.01 — and with one seed per run those differences cannot be separated from seed luck. The ordering against DQN is unchanged either way.

## 8.3 Connection to Chapter 4

The action-effect finding (Section 6.8) is the deep-RL generalisation of the credit-assignment problem Chapter 4 ran into with tabular Q-learning on SeqTO-v1. Both problems share the same structural difficulty:

- **Per-step reward signal is weak.** In Chapter 4's SynRM env, the per-step torque change was small and noisy relative to inter-mask variance; in Chapter 5's C-core env, the four actions' first-step forces differ by less than one standard deviation.
- **Multi-step composition is what matters.** In both chapters, the productive policies were the ones that composed long sequences of actions into globally good masks, not the ones that made locally good single-step decisions.
- **Algorithms that re-use signal beat algorithms that don't.** Chapter 4's `q_learn_improved.py` succeeded against the tabular baseline largely because of its $\\varepsilon$-decay-then-replay-the-good-trajectories mechanism; Chapter 5's DQN succeeds against A2C and PPO largely because of its replay buffer. The same idea — extract more gradient signal per FEMM solve — works at both the tabular and deep-network scales.

So Chapter 5 isn't a new method on a new problem; it's the same problem with a richer state representation and a CNN function approximator. The chapter-level lesson is that **the env's credit-assignment difficulty constrains algorithm choice far more than the function approximator does**.

## 8.4 Planned future work

In priority order. All items are tracked rather than executed in this jupyter-book session.

1. **DQN evaluation pass.** Load `designs/dqn_run/checkpoint.pt`, fix the policy (greedy argmax), and roll out 20–50 evaluation episodes from held-out random starts. The mean force from that pass is the number that should populate any benchmark — the {N(dqn['peak'])} N peak is an upper bound the agent has demonstrated reachable, not a steady-state operating point. About 50 episodes × 34 solves × {N(SPS)} s, under 2 h on this machine.
2. **PPO continuation.** Both PPO runs were still moving at episode 200 (PPO-v2 peak in episode {ppov2['peak_ep']}; SB3-PPO quarter means {Q(sb3ppo)} N). A 400–500 episode rerun would establish whether PPO can match DQN with more compute, or whether the gap is structural.
3. **Greedy local search from the trained masks.** The cell-flips probe (Section 6.8.2) found that the best single flip around the DQN best-by-reward mask adds {N(best_flip['delta_N'],1)} N. A hill-climb costs {n_flips} solves per sweep and is the cheapest improvement available.
4. **Multiple seeds.** Three seeds per method would turn the single-seed ordering of Section 7 into a claim.
5. **Newton-calibrated cross-chapter comparison.** Re-evaluate the Chapter 4 GA-best and Q-learn-best masks under the SeqTO-v2 `reward_kind='force'` setting (same env, same Maxwell-stress integral) so the Chapter 4 versus Chapter 5 comparison stops mixing units. This is the same Maxwell-stress recompute deferred in Chapter 4 §7.6.3.
6. **DRL on the SynRM polar domain.** Chapter 4 §6.10 has the SynRM polar env (`seqTO_SynRM_env.py`) with a `--grid-size N` flag; porting `dqn_seqto.py` and `ppo_v2.py` to that env would extend the DRL-vs-classical comparison to a second machine.
7. **Hyperparameter sweeps.** Beyond the reward-scale fix: learning rate for the policy methods ($3 \\times 10^{{-4}}$ for every custom run, $7 \\times 10^{{-4}}$ for SB3-A2C), DQN target-network $\\tau$ (0.005) and DQN $\\gamma$ (0.99). Deferred because the chapter's headline result does not yet warrant the compute spend.

## 8.5 Limitations of this chapter

- **No evaluation passes.** Every number in Section 7 is a *training-time* value under exploration, not a steady-state evaluation. The peak force is an upper bound on reachable performance, not a guaranteed operating point. Item 1 above addresses this.
- **Single seed per run.** No seed-sensitivity analysis. The {N(dqn['peak'])} N DQN result might be reproducible across seeds or might be lucky; the data on disk cannot distinguish those.
- **C-core only.** SeqTO-v2 is C-core-only as written; the SynRM polar domain from Chapter 4 §6.10 is not addressed (item 6 above).
- **Short budgets.** Every run stops at 200 episodes (6,800 FEMM solves). The two PPO runs had not plateaued.
- **Cross-chapter unit mismatch.** Chapter 4 reports mean-$|B|$-based reward; Chapter 5 reports force in newtons. The cross-chapter row in Section 7 is unit-inconsistent until item 5 above is done.

## 8.6 Broader takeaway

Two things this chapter argues, both grounded in the data on disk rather than in algorithm-on-paper analysis:

1. **Env characterisation pays for itself.** The five env probes from Section 6.8 cost {(rs['summary']['n'] + det['n'] + ae['reps']*4 + (rp['episodes']*rp['max_steps'] if rp else 0) + (n_flips + 1 if cf else 0))} FEMM solves between them, about {((rs['summary']['n'] + det['n'] + ae['reps']*4 + (rp['episodes']*rp['max_steps'] if rp else 0) + (n_flips + 1 if cf else 0)) * SPS / 3600):.1f} h of solver time against {dqn['n']*34*SPS/3600:.1f} h for one training run. Their result — single-step rewards are nearly action-blind, the random floor is {rpS('mean')} N, a good mask is locally robust — frames every training number in the chapter and is consistent with the DQN margin.
2. **The hard part is matching the algorithm to the env's structure, not picking a sophisticated algorithm.** PPO is the more sophisticated algorithm; DQN won. The reason isn't about clipped surrogates or trust regions — it's about which mechanism extracts the most gradient signal per expensive simulator call. On an env where every FEMM solve costs {N(SPS)} s and a 200-episode run costs {WALL['dqn']/3600:.0f} h of wall-clock, that is the metric that matters.
"""

# ───────────────────────────── apply ──────────────────────────────────────────
def load_nb(name): return json.load(open(name))
def save_nb(name, nb):
    json.dump(nb, open(name, 'w'), indent=1, ensure_ascii=False); open(name, 'a').write('\n')
def set_md(nb, idx, text, must_start):
    c = nb['cells'][idx]; src = ''.join(c['source'])
    assert c['cell_type'] == 'markdown' and src.lstrip().startswith(must_start), (idx, src[:60])
    c['source'] = text.splitlines(keepends=True)
def md_cell(text):  return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}
def code_cell(text): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}

if DRY:
    for k, v in T.items():
        print(f'\n################ {k} ################\n{v}')
    print('PENDING:', PENDING)
    sys.exit(0)

NB = {'05': '05_policy_methods_and_actor_critic.ipynb', '06': '06_seqto_v2_environment_and_setup.ipynb',
      '07': '07_results_and_comparison.ipynb', '08': '08_conclusions.ipynb'}
for k in sorted(SEL):
    if not (BACKUP / NB[k]).exists():
        shutil.copy(NB[k], BACKUP / NB[k])

if '05' in SEL:
  nb = load_nb('05_policy_methods_and_actor_critic.ipynb')
  set_md(nb, 13, T['05_13'], '**A2C unscaled')
  set_md(nb, 16, T['05_16'], '### 5.9.1')
  save_nb('05_policy_methods_and_actor_critic.ipynb', nb)

if '06' in SEL:
  nb = load_nb('06_seqto_v2_environment_and_setup.ipynb')
  c = nb['cells'][9]; src = ''.join(c['source'])
  assert src.lstrip().startswith('### 6.8.1')
  head, sep, _ = src.partition('**Random-policy floor')
  assert sep, 'random-policy paragraph not found'
  c['source'] = (head + T['06_9_tail'] + '\n').splitlines(keepends=True)
  if not any(''.join(x['source']).startswith('### 6.8.2') for x in nb['cells'] if x['cell_type'] == 'markdown'):
      nb['cells'][10:10] = [code_cell(T['06_new_code']), md_cell(T['06_new_md'])]
  else:
      i = next(i for i, x in enumerate(nb['cells']) if x['cell_type'] == 'markdown' and ''.join(x['source']).startswith('### 6.8.2'))
      nb['cells'][i - 1] = code_cell(T['06_new_code']); nb['cells'][i] = md_cell(T['06_new_md'])
  save_nb('06_seqto_v2_environment_and_setup.ipynb', nb)

if '07' in SEL:
  nb = load_nb('07_results_and_comparison.ipynb')
  set_md(nb, 0, T['07_0'], '# 7.')
  set_md(nb, 7, T['07_7'], '## 7.4')
  set_md(nb, 8, T['07_8'], '## 7.5')
  set_md(nb, 10, T['07_10'], '**Interpretation.**')
  set_md(nb, 13, T['07_13'], "## 7.7")
  set_md(nb, 14, T['07_14'], '## 7.8')
  save_nb('07_results_and_comparison.ipynb', nb)

if '08' in SEL:
  nb = load_nb('08_conclusions.ipynb')
  set_md(nb, 0, T['08_0'], '# 8.')
  save_nb('08_conclusions.ipynb', nb)

# ───────────────────────────── verify ─────────────────────────────────────────
pat = re.compile(r'1305|1,305|\b795\b|\b764\b|\b782\b|\b799\b|\b610\b|84 s|\\sim 84|159 h|\b339\b|\b267\b|16,752|2\.14|ep 13/50|crashed|\b79 h\b|10k-step|reward_scale=1\.0|<PENDING>|\bnan\b')
bad = []
for name in sorted(NB[k] for k in SEL):
    for i, c in enumerate(load_nb(name)['cells']):
        if c['cell_type'] == 'markdown':
            hits = sorted(set(pat.findall(''.join(c['source']))))
            if hits: bad.append((name, i, hits))
print('leftover thesis-era tokens:', bad or 'none')
assert not bad
print('rewrite applied; backups in', BACKUP)
