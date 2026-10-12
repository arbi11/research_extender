"""Print the measured facts of every Chapter 5 run on disk (for the results prose). Usage: python summarize_runs.py [C5 dir]"""
import json, sys, glob, os
import numpy as np
C5 = sys.argv[1] if len(sys.argv) > 1 else '/home/akhan/code/compem_book/C5'
rs_files = sorted(glob.glob(f'{C5}/designs/env_probe/random_search_*.json'))
rs = json.load(open(rs_files[-1])) if rs_files else None
rs_max = rs['summary']['max'] if rs else float('nan')
print(f"random-search ceiling {rs_max:.1f} N, mean {rs['summary']['mean']:.1f} N, {rs['sec_per_solve']:.2f} s/solve" if rs else "no probe")
secs = {}
for log in glob.glob('/home/akhan/code/compem_book/_runs/c5_runs/queue_*.log'):
    for line in open(log):
        if line.startswith('rc='):
            p = line.split(); secs[p[2]] = int(p[1].split('=')[1])
for d in sorted(glob.glob(f'{C5}/designs/*_run')):
    name = os.path.basename(d); p = f'{d}/training_stats.json'
    if not os.path.exists(p): print(f"{name}: no training_stats.json"); continue
    s = json.load(open(p)); eps = s['episode_stats']; hp = s['hyperparams']
    f = np.array([e['force_N'] for e in eps]); n = len(f); q = max(n // 4, 1)
    quarters = [f[i*q:(i+1)*q].mean() for i in range(4)] if n >= 4 else [f.mean()]
    ent = [e.get('entropy') for e in eps if e.get('entropy') is not None]
    vl = [e.get('value_loss') for e in eps if e.get('value_loss') is not None]
    best_ep = hp.get('best_episode'); best_r = hp.get('best_reward')
    print(f"\n{name}: n={n}/{hp.get('episodes', hp.get('timesteps', '?'))}  reward_scale={hp.get('reward_scale', 1.0)}")
    print(f"  peak force {f.max():.1f} N at ep {f.argmax()+1} ({f.max()/rs_max:.2f}x random-search max); episodes above RS max: {(f > rs_max).sum()}")
    print(f"  quarter means (N): " + ' -> '.join(f'{x:.0f}' for x in quarters) + f";  first-10 {f[:10].mean():.0f}, last-10 {f[-10:].mean():.0f}, last-50 {f[-50:].mean():.0f}")
    if best_ep: print(f"  best-by-reward ep {best_ep}: reward {best_r:.3f}, force {f[best_ep-1]:.1f} N, iron {eps[best_ep-1]['iron_count']}")
    if ent: print(f"  entropy: first {ent[0]:.3f}, quarters " + ' -> '.join(f'{np.mean(ent[i*q:(i+1)*q]):.2f}' for i in range(4)) + f", last {ent[-1]:.3g} (log4 = 1.386)")
    if vl: print(f"  value_loss: quarters " + ' -> '.join(f'{np.mean(vl[i*q:(i+1)*q]):.3g}' for i in range(4)) + f", max {max(vl):.3g}")
    if 'approx_kl' in eps[-1]: print(f"  PPO: mean approx_kl {np.mean([e['approx_kl'] for e in eps]):.4f}, mean clip_frac {np.mean([e['clip_frac'] for e in eps]):.3f}")
    if 'epsilon' in eps[-1]: print(f"  DQN: epsilon end {eps[-1]['epsilon']:.3f}, loss quarters " + ' -> '.join(f'{np.nanmean([e["loss"] for e in eps[i*q:(i+1)*q]]):.3g}' for i in range(4)))
    key = {'a2c_v2_run': 'c5_a2c_v2', 'ppo_v2_run': 'c5_ppo_v2', 'ppo_run': 'c5_ppo_raw', 'sb3_ppo_run': 'c5_sb3_ppo', 'sb3_a2c_run': 'c5_sb3_a2c', 'dqn_run': 'c5_dqn', 'a2c_run': 'c5_a2c'}.get(name)
    if key in secs: print(f"  wall {secs[key]} s = {secs[key]/3600:.1f} h ({secs[key]/(n*34):.1f} s per step)")
