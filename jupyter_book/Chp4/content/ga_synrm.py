"""
ga_synrm.py — Genetic Algorithm for SynRM SeqTO-v1 (5×5 polar rotor grid) using DEAP.

Mirror of ga_cCore.py adapted for the SynRM environment in seqTO_SynRM_env.py.

Chromosome : action sequence  a = [a_1, ..., a_m]
              a_i in {0=RIGHT, 1=LEFT, 2=UP, 3=DOWN}
              Polar semantics:
                RIGHT / LEFT  increment / decrement angular index j (sector 0°..90°)
                UP    / DOWN  increment / decrement radial  index i (R_HUB..R_PERI_IN)
Fitness    : seqTO_SynRM_env.calculate_reward(iron_mask) — max |torque| across 6
              rotor angles (topology figure-of-merit, see env docstring; NOT
              calibrated reluctance torque in N·m).
Budget     : senv.K1 ≤ iron_count ≤ senv.K2  (5 ≤ iron ≤ 20; hard-zero outside)

Evaluation strategy
-------------------
Like ga_cCore: build the topology from the action sequence in pure Python, then
call calculate_reward() once per individual.  Each call internally performs 6
FEMM rotor-angle solves, so per-individual cost is ~60–120 s.

Defaults are smaller than ga_cCore because each FEMM evaluation here is ~6×
more expensive (rotor-angle sweep) and the design space (4^15 ≈ 10^9) is much
smaller than the C-core's (4^34 ≈ 10^20).

Requirements
------------
    pip install deap numpy matplotlib
    pyfemm + FEMM 4.2  (Windows only, for FEA reward)

Usage
-----
    python ga_synrm.py                          # defaults: seq-len=15, pop=20, n-gen=40
    python ga_synrm.py --pop 30 --n-gen 60     # larger run
    python ga_synrm.py --no-plot               # batch / headless

Outputs (in designs/synrm_ga/):
    ga_logbook.json    — full per-generation statistics
    ga_best_mask.npy   — best 5×5 iron mask
    ga_results.png     — convergence curve + polar topology of the best individual
"""

import os
import sys
import json
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from deap import base, creator, tools
except ImportError:
    print("ERROR: deap not installed.  Run: pip install deap")
    sys.exit(1)


# ── Import env module (attribute access so seqTO_SynRM_env.configure() ──
#   changes propagate to all references below)
import seqTO_SynRM_env as senv

# Grid-size-independent constant — convenient shorthand
ACTIONS = senv.ACTIONS

# DESIGNS_DIR is set per-run by main() once the grid size is known
DESIGNS_DIR = os.path.join(os.path.dirname(__file__), 'designs', 'synrm_ga')


# ── Helpers (pure Python, no FEMM) ─────────────────────────────────────────

def is_in_grid(i, j):
    return 0 <= i < senv.DD_ROWS and 0 <= j < senv.DD_COLS


def build_topology(action_seq):
    """
    Execute action sequence in pure Python; return iron_mask (5×5 int array).

    Mirrors SynRMSeqTOEnv.step() exactly:
      - Initial controller position is (0, 0) (innermost / most-CW cell).
      - Each action attempts a move; if the new (i, j) is in the 5×5 grid
        the controller moves there.  A 1×1 iron cell is then deposited at
        the controller's CURRENT position (whether or not the move was
        accepted).  This way an out-of-bounds move still deposits at the
        old position — same behaviour as the env.
    """
    iron_mask = np.zeros((senv.DD_ROWS, senv.DD_COLS), dtype=int)
    pos_r, pos_c = 0, 0

    for action in action_seq:
        _, dr, dc = ACTIONS[int(action)]
        new_r, new_c = pos_r + dr, pos_c + dc
        if is_in_grid(new_r, new_c):
            pos_r, pos_c = new_r, new_c
        # is_in_grid(pos_r, pos_c) is guaranteed true here (pos always in-grid)
        iron_mask[pos_r, pos_c] = 1

    return iron_mask


# ── Fitness evaluation ──────────────────────────────────────────────────────

def evaluate(individual):
    """
    1. Build topology (pure Python).
    2. Apply budget hard-zero filter — no FEMM call wasted on infeasible designs.
    3. Otherwise call seqTO_SynRM_env.calculate_reward() — single FEMM 6-angle sweep.
    """
    iron_mask  = build_topology(individual)
    iron_count = int(np.sum(iron_mask))

    if not (senv.K1 <= iron_count <= senv.K2):
        return (0.0,)

    # Lazy import — keeps this module importable even without pyfemm
    from seqTO_SynRM_env import calculate_reward
    reward, _ = calculate_reward(iron_mask)
    return (reward,)


# ── DEAP setup ──────────────────────────────────────────────────────────────

def _biased_action():
    """
    Bias toward RIGHT (angular sweep) and UP (radially outward) so random
    walks starting at (0,0) explore the design domain instead of bouncing
    off the inner / CW boundaries.
    """
    return random.choices([0, 1, 2, 3], weights=[30, 20, 30, 20])[0]


def _make_sweep_individual(seq_len):
    """
    Boustrophedon (snake) sweep across the 5×5 grid — a deterministic seed
    that reliably deposits ≥ 12 iron cells if seq_len ≥ 15.  Always
    injected into gen-0 so the population starts with at least one feasible
    individual.

    Pattern (for seq_len=15):
      RIGHT×4, UP, LEFT×4, UP, RIGHT×4, UP   → 15 actions, 15 unique cells
    """
    actions = []
    going_right = True
    while len(actions) < seq_len:
        for _ in range(senv.DD_COLS - 1):
            actions.append(0 if going_right else 1)     # RIGHT or LEFT
            if len(actions) >= seq_len: break
        if len(actions) >= seq_len: break
        going_right = not going_right
        actions.append(2)                                # UP one row
    return creator.Individual(actions[:seq_len])


def _setup_deap(seq_len):
    """Register DEAP primitives.  Idempotent."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("attr_action", _biased_action)
    tb.register("individual",  tools.initRepeat,
                creator.Individual, tb.attr_action, n=seq_len)
    tb.register("population",  tools.initRepeat, list, tb.individual)
    tb.register("evaluate",    evaluate)
    tb.register("mate",        tools.cxTwoPoint)
    tb.register("mutate",      tools.mutUniformInt,
                low=0, up=3, indpb=max(0.05, 1.0 / seq_len))
    tb.register("select",      tools.selTournament, tournsize=3)
    return tb


# ── Main GA loop ─────────────────────────────────────────────────────────────

def run_ga(seq_len=None, pop_size=20, n_gen=40,
           cxpb=0.8, mutpb=0.3, hof_size=3, stall_limit=12,
           seed=42, verbose=True):
    """Run GA with stall-based early termination.  Returns (pop, logbook, hof).

    seq_len=None (the default) resolves to senv.MAX_STEPS at call time so
    that senv.configure() updates are respected.
    """
    if seq_len is None:
        seq_len = senv.MAX_STEPS
    random.seed(seed)
    np.random.seed(seed)

    tb  = _setup_deap(seq_len)
    hof = tools.HallOfFame(hof_size)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0]
                             if ind.fitness.valid else float('nan'))
    stats.register("max",  lambda xs: np.nanmax(xs))
    stats.register("mean", lambda xs: np.nanmean(xs))
    stats.register("min",  lambda xs: np.nanmin(xs))
    stats.register("std",  lambda xs: np.nanstd(xs))

    logbook        = tools.Logbook()
    logbook.header = ['gen', 'nevals', 'max', 'mean', 'min', 'std']

    # ── Evaluate initial population ─────────────────────────────────────────
    pop = tb.population(n=pop_size - 1)
    pop.insert(0, _make_sweep_individual(seq_len))      # always one good seed

    iron_counts = [int(np.sum(build_topology(ind))) for ind in pop]
    feasible    = sum(senv.K1 <= c <= senv.K2 for c in iron_counts)
    if verbose:
        print(f"  [gen 0 init]  iron cells: min={min(iron_counts)}, "
              f"mean={np.mean(iron_counts):.0f}, max={max(iron_counts)}, "
              f"feasible={feasible}/{len(pop)}")

    fitnesses = list(map(tb.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    hof.update(pop)

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop), **record)
    if verbose:
        print(logbook.stream)

    best_fitness = hof[0].fitness.values[0]
    stall_count  = 0

    # ── Generational loop ───────────────────────────────────────────────────
    for gen in range(1, n_gen + 1):

        # Selection
        offspring = tb.select(pop, len(pop) - hof_size)
        offspring = list(map(tb.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                tb.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                tb.mutate(mutant)
                del mutant.fitness.values

        # Elitism: inject Hall-of-Fame back into population
        elite = [tb.clone(ind) for ind in hof]
        offspring.extend(elite)

        # Evaluate only individuals with invalid (new / mutated) fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(tb.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid), **record)
        if verbose:
            print(logbook.stream)

        # Stall detection
        current_best = hof[0].fitness.values[0]
        if current_best > best_fitness + 1e-6:
            best_fitness = current_best
            stall_count  = 0
        else:
            stall_count += 1

        if stall_count >= stall_limit:
            if verbose:
                print(f"\nStall criterion met at generation {gen} "
                      f"({stall_limit} gens without improvement).  Stopping.")
            break

    return pop, logbook, hof


# ── Results ─────────────────────────────────────────────────────────────────

def save_results(logbook, hof, seq_len):
    """Save logbook JSON, best iron_mask, and summary print."""
    os.makedirs(DESIGNS_DIR, exist_ok=True)

    lb_path = os.path.join(DESIGNS_DIR, 'ga_logbook.json')
    with open(lb_path, 'w') as f:
        json.dump([dict(r) for r in logbook], f, indent=2)
    print(f"\nLogbook saved → {lb_path}")

    best_mask = build_topology(hof[0])
    np.save(os.path.join(DESIGNS_DIR, 'ga_best_mask.npy'), best_mask)

    iron_count = int(np.sum(best_mask))
    print(f"\nBest individual:")
    print(f"  Action sequence : {list(hof[0])}")
    print(f"  Iron cells      : {iron_count}  (senv.K1={senv.K1}, senv.K2={senv.K2})")
    print(f"  Fitness         : {hof[0].fitness.values[0]:.4f}")
    print(f"  Best mask saved → {os.path.join(DESIGNS_DIR, 'ga_best_mask.npy')}")


def plot_results(logbook, hof, save=True):
    """Two-panel figure: convergence (left) + best topology in polar coords (right)."""
    gens    = logbook.select('gen')
    maxfit  = logbook.select('max')
    meanfit = logbook.select('mean')
    minfit  = logbook.select('min')

    fig     = plt.figure(figsize=(14, 6))
    ax_conv = fig.add_subplot(1, 2, 1)
    ax_topo = fig.add_subplot(1, 2, 2, projection='polar')

    # ── Left: convergence curve ────────────────────────────────────────────
    ax_conv.plot(gens, maxfit,  'crimson',   lw=2,   label='Best')
    ax_conv.plot(gens, meanfit, 'steelblue', lw=1.5, label='Mean')
    ax_conv.fill_between(gens, minfit, maxfit, alpha=0.15, color='steelblue')
    ax_conv.set_xlabel('Generation')
    ax_conv.set_ylabel('Fitness  (max |torque| across 6 rotor angles)')
    ax_conv.set_title('GA Convergence\n(SynRM SeqTO-v1, 5×5 polar grid)')
    ax_conv.legend()
    ax_conv.grid(alpha=0.3)

    # ── Right: best topology in polar projection ───────────────────────────
    best_mask = build_topology(hof[0])
    ax_topo.set_theta_zero_location('E')
    ax_topo.set_theta_direction(1)
    ax_topo.set_thetamin(0)
    ax_topo.set_thetamax(90)
    ax_topo.set_ylim(0, senv.R_PERI_IN + 4)

    n_arc = 16
    for i in range(senv.DD_ROWS):
        for j in range(senv.DD_COLS):
            r1 = senv.R_HUB + i * senv.DR
            r2 = senv.R_HUB + (i + 1) * senv.DR
            t1 = np.radians(j * senv.DTHETA)
            t2 = np.radians((j + 1) * senv.DTHETA)
            theta_arc  = np.linspace(t1, t2, n_arc)
            theta_poly = np.concatenate([theta_arc, theta_arc[::-1]])
            r_poly     = np.concatenate([np.full(n_arc, r1), np.full(n_arc, r2)])
            color = '#2C3E50' if best_mask[i, j] else '#FEF9E7'
            ax_topo.fill(theta_poly, r_poly,
                         color=color, edgecolor='#7F8C8D', linewidth=0.4)

    # d-axis marker at 45° mech
    ax_topo.plot([np.radians(45), np.radians(45)], [0, senv.R_PERI_IN + 2],
                 'r--', linewidth=1.0, alpha=0.7)
    ax_topo.text(np.radians(45), senv.R_PERI_IN + 3.5, 'd-axis',
                 ha='center', va='bottom', fontsize=8, color='red')

    iron_count = int(np.sum(best_mask))
    ax_topo.set_title(f'Best topology  (fitness={hof[0].fitness.values[0]:.4f})\n'
                      f'iron cells = {iron_count} / 25',
                      fontsize=11, pad=18)
    ax_topo.set_yticks([senv.R_HUB, senv.R_PERI_IN])
    ax_topo.set_yticklabels([f'{senv.R_HUB:.1f}', f'{senv.R_PERI_IN:.1f}'], fontsize=7)
    ax_topo.grid(True, alpha=0.4)
    ax_topo.legend(handles=[
        mpatches.Patch(fc='#2C3E50',                label='Iron (GA-placed)'),
        mpatches.Patch(fc='#FEF9E7', ec='grey', lw=0.5, label='Air (design domain)'),
    ], fontsize=8, loc='lower right')

    plt.suptitle('SynRM SeqTO-v1 — Genetic Algorithm (DEAP)', fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(DESIGNS_DIR, 'ga_results.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved → {path}")
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='GA for SynRM SeqTO-v1 (DEAP, configurable grid)')
    p.add_argument('--grid-size', type=int,   default=5,
                   help='design domain size N for an N x N polar grid (default 5, range [5,10])')
    p.add_argument('--seq-len',   type=int,   default=None,
                   help='chromosome length (default = grid_size^2 from env after configure)')
    p.add_argument('--pop',       type=int,   default=20,
                   help='population size (smaller than C-core: each eval is 6 FEMM solves)')
    p.add_argument('--n-gen',     type=int,   default=40, help='max generations')
    p.add_argument('--cxpb',      type=float, default=0.8, help='crossover probability')
    p.add_argument('--mutpb',     type=float, default=0.3, help='per-individual mutation prob')
    p.add_argument('--stall',     type=int,   default=12,  help='stall-termination patience')
    p.add_argument('--hof',       type=int,   default=3,   help='Hall-of-Fame size')
    p.add_argument('--seed',      type=int,   default=42,  help='random seed')
    p.add_argument('--no-plot',   action='store_true',     help='skip matplotlib output')
    return p.parse_args()


def main():
    global DESIGNS_DIR
    args = _parse_args()

    if not (5 <= args.grid_size <= 10):
        print(f"WARNING: grid-size={args.grid_size} is outside the validated range [5, 10]")

    # Reconfigure the env BEFORE any other env-attribute access
    senv.configure(args.grid_size, args.grid_size)

    # Resolve grid-dependent defaults now that configure() has run
    if args.seq_len is None:
        args.seq_len = senv.MAX_STEPS

    # Per-grid output dir
    DESIGNS_DIR = os.path.join(os.path.dirname(__file__),
                               'designs',
                               f'synrm_ga_{args.grid_size}x{args.grid_size}')

    print("═" * 64)
    print("  SynRM SeqTO-v1  ·  Genetic Algorithm (DEAP)")
    print("═" * 64)
    print(f"  Grid size         : {senv.DD_ROWS}×{senv.DD_COLS}")
    print(f"  Chromosome length : {args.seq_len} directional actions")
    print(f"  Population size   : {args.pop}")
    print(f"  Max generations   : {args.n_gen}  (stall limit: {args.stall})")
    print(f"  Crossover / mut   : {args.cxpb} / {args.mutpb}")
    print(f"  Random seed       : {args.seed}")
    print(f"  Design domain     : r={senv.R_HUB:.1f}–{senv.R_PERI_IN:.1f} mm, θ=0°–90°  "
          f"(DR={senv.DR:.3f} mm, DTHETA={senv.DTHETA:.1f}°)")
    print(f"  Budget            : K1={senv.K1} ≤ iron ≤ K2={senv.K2}  "
          f"(of {senv.DD_ROWS*senv.DD_COLS} cells)")
    print(f"  Output dir        : {DESIGNS_DIR}")
    print(f"  Fitness           : max |torque| over 6 rotor angles "
          f"(seqTO_SynRM_env.calculate_reward)")
    print("─" * 64)

    pop, logbook, hof = run_ga(
        seq_len     = args.seq_len,
        pop_size    = args.pop,
        n_gen       = args.n_gen,
        cxpb        = args.cxpb,
        mutpb       = args.mutpb,
        hof_size    = args.hof,
        stall_limit = args.stall,
        seed        = args.seed,
    )

    save_results(logbook, hof, args.seq_len)

    if not args.no_plot:
        plot_results(logbook, hof, save=True)


if __name__ == '__main__':
    main()
