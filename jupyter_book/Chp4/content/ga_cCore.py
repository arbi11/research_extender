"""
ga_cCore.py — Genetic Algorithm for C-core SeqTO-v1 using DEAP.

Chromosome : action sequence  a = [a_1, ..., a_m],  a_i in {0=RIGHT, 1=LEFT, 2=UP, 3=DOWN}
Fitness    : mean |B| at 6 armature sample points × 1000  (single FEMM FEA per individual)
Budget     : K1 <= iron_count <= K2  (soft penalty outside this range)

Evaluation strategy
-------------------
Unlike the Q-learning agent (which calls FEMM at every step), the GA builds the full
topology from the action sequence in pure Python, then calls calculate_reward() once.
This gives 1 FEA call per individual per generation instead of m calls — a significant
reduction in evaluation budget.

Requirements
------------
    pip install deap numpy matplotlib
    pyfemm + FEMM 4.2 (Windows only, for FEA reward)

Usage
-----
    python ga_cCore.py                          # defaults: seq-len=34, pop=50
    python ga_cCore.py --seq-len 34 --pop 50   # explicit defaults
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("ERROR: deap not installed.  Run: pip install deap")
    sys.exit(1)


# ── Constants (must match seqTO_env.py) ────────────────────────────────────
ROWS, COLS     = 18, 35
COIL_LEN       = 9
COIL1_COLS     = (2, 5)
COIL2_COLS     = (11, 14)
ARM_ROWS       = (0, 15)
ARM_COLS       = (27, 33)
DW_ROWS        = (0, ARM_ROWS[1])    # (0, 15)
DW_COLS        = (5, 26)
K1, K2         = 80, 180
ACTIONS        = {
    0: ('RIGHT',  0, +1),
    1: ('LEFT',   0, -1),
    2: ('UP',    -1,  0),
    3: ('DOWN',  +1,  0),
}
DESIGNS_DIR    = os.path.join(os.path.dirname(__file__), 'designs', 'ga')


# ── Helpers (pure Python, no FEMM) ─────────────────────────────────────────

def is_design_cell(i, j):
    in_dw    = DW_ROWS[0] <= i < DW_ROWS[1] and DW_COLS[0] <= j < DW_COLS[1]
    in_coil2 = i < COIL_LEN and COIL2_COLS[0] <= j < COIL2_COLS[1]
    return in_dw and not in_coil2


def build_topology(action_seq):
    """
    Execute action sequence in pure Python; return iron_mask (18×35 int array).
    Mirrors CcoreSeqTOEnv.step() logic exactly — no FEMM required.
    """
    iron_mask = np.zeros((ROWS, COLS), dtype=int)
    pos_r = DW_ROWS[0] + 1
    pos_c = DW_COLS[0] + 1

    for action in action_seq:
        _, dr, dc = ACTIONS[int(action)]
        new_r, new_c = pos_r + dr, pos_c + dc

        in_dw    = DW_ROWS[0] <= new_r < DW_ROWS[1] and DW_COLS[0] <= new_c < DW_COLS[1]
        in_coil2 = new_r < COIL_LEN and COIL2_COLS[0] <= new_c < COIL2_COLS[1]
        if in_dw and not in_coil2:
            pos_r, pos_c = new_r, new_c
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ri, ci = pos_r + di, pos_c + dj
                    if is_design_cell(ri, ci):
                        iron_mask[ri, ci] = 1

    return iron_mask


# ── Fitness evaluation ──────────────────────────────────────────────────────

def evaluate(individual):
    """
    1. Build topology (pure Python).
    2. Apply budget soft penalty.
    3. Call calculate_reward() once (single FEMM FEA).
    Returns (fitness,) as required by DEAP.
    """
    iron_mask  = build_topology(individual)
    iron_count = int(np.sum(iron_mask))

    # Hard zero for out-of-budget topologies — any valid FEMM reward must beat 0.0
    if not (K1 <= iron_count <= K2):
        return (0.0,)

    # Lazy import — keeps this module importable even without pyfemm
    from seqTO_env import calculate_reward
    reward, _, _ = calculate_reward(iron_mask)
    return (reward,)


# ── DEAP setup ──────────────────────────────────────────────────────────────

def _biased_action():
    """Prefer RIGHT (0) and DOWN (3) so random walks sweep the domain instead of backtracking."""
    return random.choices([0, 1, 2, 3], weights=[35, 15, 15, 35])[0]


def _make_sweep_individual(seq_len):
    """
    Boustrophedon (snake) sweep across the design window — a deterministic seed
    that reliably deposits 80+ iron cells.  One copy is always injected into gen-0.
    """
    actions = []
    going_right = True
    step = 0
    while len(actions) < seq_len:
        # Horizontal run (~16 steps to cross the 21-col window with a 3×3 brush)
        for _ in range(16):
            actions.append(0 if going_right else 1)
            if len(actions) >= seq_len:
                break
        going_right = not going_right
        # Drop down 3 rows (one 3×3-block height) before reversing
        for _ in range(3):
            actions.append(3)        # DOWN
            if len(actions) >= seq_len:
                break
    ind = creator.Individual(actions[:seq_len])
    return ind


def _setup_deap(seq_len):
    """Register DEAP primitives.  Called once; safe to call multiple times."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("attr_action", _biased_action)
    tb.register("individual", tools.initRepeat,
                creator.Individual, tb.attr_action, n=seq_len)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("evaluate",   evaluate)
    tb.register("mate",       tools.cxTwoPoint)
    tb.register("mutate",     tools.mutUniformInt,
                low=0, up=3, indpb=max(0.02, 1.0 / seq_len))
    tb.register("select",     tools.selTournament, tournsize=3)
    return tb


# ── Main GA loop ─────────────────────────────────────────────────────────────

def run_ga(seq_len=34, pop_size=50, n_gen=200,
           cxpb=0.8, mutpb=0.2, hof_size=5, stall_limit=25,
           seed=42, verbose=True):
    """
    Run GA with stall-based early termination.

    Parameters
    ----------
    seq_len    : chromosome length (number of directional moves)
    pop_size   : population size
    n_gen      : maximum generations
    cxpb       : crossover probability (per pair)
    mutpb      : mutation probability (per individual)
    hof_size   : Hall-of-Fame size (elites preserved across generations)
    stall_limit: stop after this many consecutive generations without improvement
    seed       : random seed for reproducibility

    Returns
    -------
    pop, logbook, hof
    """
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
    pop.insert(0, _make_sweep_individual(seq_len))   # always seed one good sweep

    # Diagnostic: show iron-count distribution before FEMM calls
    iron_counts = [int(np.sum(build_topology(ind))) for ind in pop]
    feasible    = sum(K1 <= c <= K2 for c in iron_counts)
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

    best_fitness  = hof[0].fitness.values[0]
    stall_count   = 0

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

        # Elitism: inject HallOfFame back into population
        elite = [tb.clone(ind) for ind in hof]
        offspring.extend(elite)

        # Evaluate only individuals with invalid (new/mutated) fitness
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

    # Logbook
    lb_path = os.path.join(DESIGNS_DIR, 'ga_logbook.json')
    with open(lb_path, 'w') as f:
        json.dump([dict(r) for r in logbook], f, indent=2)
    print(f"\nLogbook saved → {lb_path}")

    # Best topology
    best_mask = build_topology(hof[0])
    np.save(os.path.join(DESIGNS_DIR, 'ga_best_mask.npy'), best_mask)

    iron_count = int(np.sum(best_mask))
    print(f"\nBest individual:")
    print(f"  Action sequence : {list(hof[0])}")
    print(f"  Iron cells      : {iron_count}  (K1={K1}, K2={K2})")
    print(f"  Fitness         : {hof[0].fitness.values[0]:.4f}")


def plot_results(logbook, hof, save=True):
    """Two-panel figure: convergence + best topology."""
    gens   = logbook.select('gen')
    maxfit = logbook.select('max')
    meanfit = logbook.select('mean')
    minfit  = logbook.select('min')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: convergence ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(gens, maxfit,  'crimson', lw=2, label='Best')
    ax.plot(gens, meanfit, 'steelblue', lw=1.5, label='Mean')
    ax.fill_between(gens, minfit, maxfit, alpha=0.15, color='steelblue')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness  (mean |B| × 1000)')
    ax.set_title('GA Convergence\n(C-core SeqTO-v1, 3×3 controller)')
    ax.legend()
    ax.grid(alpha=0.3)

    # ── Right: best topology (grid view) ──────────────────────────────────
    ax = axes[1]
    best_mask = build_topology(hof[0])

    # Build region grid
    # 0=other  1=left-coil  2=right-coil  3=armature  4=iron  5=air-DW  6=excl
    region = np.zeros((ROWS, COLS), dtype=int)
    for i in range(ROWS):
        for j in range(COLS):
            if i < COIL_LEN and COIL1_COLS[0] <= j < COIL1_COLS[1]:
                region[i, j] = 1
            elif i < COIL_LEN and COIL2_COLS[0] <= j < COIL2_COLS[1]:
                region[i, j] = 2
            elif ARM_ROWS[0] <= i < ARM_ROWS[1] and ARM_COLS[0] <= j < ARM_COLS[1]:
                region[i, j] = 3
            elif is_design_cell(i, j):
                region[i, j] = 4 if best_mask[i, j] else 5
            elif DW_ROWS[0] <= i < DW_ROWS[1] and DW_COLS[0] <= j < DW_COLS[1]:
                region[i, j] = 6
            else:
                region[i, j] = 0

    CLRS = ['#D5D8DC',  # 0 buffer
            '#2980B9',  # 1 left coil
            '#E67E22',  # 2 right coil
            '#27AE60',  # 3 armature
            '#2C3E50',  # 4 iron (GA-placed)
            '#FEF9E7',  # 5 air in DW
            '#FAD7A0']  # 6 right-coil excl.

    ax.imshow(region, cmap=ListedColormap(CLRS), vmin=0, vmax=6,
              origin='upper', aspect='equal')
    ax.set_title(f'Best topology  (fitness={hof[0].fitness.values[0]:.4f})\n'
                 f'iron cells={int(np.sum(best_mask))}')
    ax.set_xlabel('column (mm)')
    ax.set_ylabel('row (mm)')
    ax.legend(handles=[
        mpatches.Patch(fc='#2C3E50', label='Iron (GA-placed)'),
        mpatches.Patch(fc='#FEF9E7', ec='grey', lw=0.5, label='Air (design window)'),
        mpatches.Patch(fc='#2980B9', label='Left coil'),
        mpatches.Patch(fc='#E67E22', label='Right coil'),
        mpatches.Patch(fc='#27AE60', label='Armature'),
    ], fontsize=8, loc='lower right')

    plt.suptitle('C-core SeqTO-v1 — Genetic Algorithm (DEAP)', fontsize=12)
    plt.tight_layout()

    if save:
        path = os.path.join(DESIGNS_DIR, 'ga_results.png')
        plt.savefig(path, dpi=150)
        print(f"Plot saved → {path}")
    plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description='GA for C-core SeqTO-v1 (DEAP)')
    p.add_argument('--seq-len',    type=int,   default=34,   help='chromosome length (steps, matches Q-learning max_steps)')
    p.add_argument('--pop',        type=int,   default=50,   help='population size')
    p.add_argument('--n-gen',      type=int,   default=200,  help='max generations')
    p.add_argument('--cxpb',       type=float, default=0.8,  help='crossover probability')
    p.add_argument('--mutpb',      type=float, default=0.2,  help='per-individual mutation prob')
    p.add_argument('--stall',      type=int,   default=25,   help='stall-termination patience')
    p.add_argument('--hof',        type=int,   default=5,    help='Hall-of-Fame size')
    p.add_argument('--seed',       type=int,   default=42,   help='random seed')
    p.add_argument('--no-plot',    action='store_true',      help='skip matplotlib output')
    return p.parse_args()


def main():
    args = _parse_args()

    print("═" * 60)
    print("C-core SeqTO-v1  ·  Genetic Algorithm (DEAP)")
    print("═" * 60)
    print(f"  Chromosome length : {args.seq_len} directional actions")
    print(f"  Population size   : {args.pop}")
    print(f"  Max generations   : {args.n_gen}  (stall limit: {args.stall})")
    print(f"  Crossover / mut   : {args.cxpb} / {args.mutpb}")
    print(f"  Random seed       : {args.seed}")
    print(f"  Design window     : rows {DW_ROWS[0]}–{DW_ROWS[1]-1}, "
          f"cols {DW_COLS[0]}–{DW_COLS[1]-1}  ({sum(is_design_cell(i,j) for i in range(ROWS) for j in range(COLS))} designable cells)")
    print(f"  Budget            : K1={K1} ≤ iron ≤ K2={K2}")
    print("─" * 60)

    pop, logbook, hof = run_ga(
        seq_len    = args.seq_len,
        pop_size   = args.pop,
        n_gen      = args.n_gen,
        cxpb       = args.cxpb,
        mutpb      = args.mutpb,
        hof_size   = args.hof,
        stall_limit= args.stall,
        seed       = args.seed,
    )

    save_results(logbook, hof, args.seq_len)

    if not args.no_plot:
        plot_results(logbook, hof, save=True)


if __name__ == '__main__':
    main()
