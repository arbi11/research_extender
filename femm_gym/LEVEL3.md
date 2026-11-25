# Level 3: generated papers with gaps

The player gets a paper-like write-up of a generated machine, rebuilds it in FEMM, and is
scored like a `daro_bench` replication: the scorer re-solves the saved model and compares its
own measurements with frozen targets. The write-up leaves some inputs out on purpose, so part
of the reward is for saying honestly which inputs were not stated.

Written before any Level 3 task was generated or solved.

## What the player gets and submits

The workspace has the `daro_bench` layout, so a model trained here meets the same contract in
the held-out benchmark:

- `paper/paper.md`: the generated paper (dimensions, winding, materials, published results);
- `task.md`: what is scored, the position and current conventions, and the input names to record;
- `task.json` (the benchmark's case format), the FEMM manual, an empty `submission/`.

The player submits, as in the benchmark, `model.fem`, `entities.json` (how the scorer drives the
model), `inputs.json` (every input with its provenance) and `notes.md`. FEMM runs are refused
until `inputs.json` exists, and each version of it is kept.

## The machine family `l3_spm`

An inner-rotor surface-magnet machine with a double-layer tooth-coil winding, built from
`femm_brain`'s tested components (`slotted_stator`, `tooth_coil_winding`,
`surface_magnet_rotor`, `air_gap_element`):

- slots and poles: 6/4, 9/8, 9/10, 12/8, 12/10, 12/14, 18/12, 18/16, 24/20;
- dimensions drawn from stated ranges and rounded to what the paper prints;
- magnets from FEMM's library (N30 to N45) or given by remanence and permeability;
- stator laminations from the library (M-15 to M-47); the rotor core a library lamination or solid steel;
- radial or parallel magnetization, 1 or 2 parallel paths, a rated current set from a current density.

## What is scored

The benchmark's `score.py` drives the model and measures; three acceptance targets, 2% each:

| Target | Quantity | In the paper? |
|---|---|---|
| E01 | Fundamental (peak) of the no-load phase back-EMF at the paper's speed, from 24 positions over one electrical period | yes |
| T01 | Mean torque at rated current, current in phase with the back-EMF, over 6 positions spanning 60 electrical degrees | yes |
| T02 | The same at three times the rated current | **no** |

T02 is held out. It depends on saturation, so a model that matches the two published numbers
for the wrong reasons fails it. Conventions are the benchmark's: at p = 0 phase A's no-load flux
linkage peaks, and phase B peaks 120 electrical degrees after A in the positive direction.

Conformance checks, also from the benchmark: stack length, magnet remanence, turns per coil side.

## Gaps

Every paper hides several inputs, in one of two ways. No gap is meant to be fitted to a
published result: the replication skill forbids tuning inputs, and Level 3 trains for that skill.

1. **Derivable.** The value follows from stated ones: a radius from a diameter, the slot-bottom
   radius from the yoke thickness, the magnet arc from a pole-arc ratio, the rotor-core radius
   from bore, air gap and magnet, turns per coil from turns per phase, coils and parallel paths,
   the coil sequence from the slot and pole numbers (star of slots), a magnet's remanence from
   its library grade. The truth is exact.
2. **Absent and insensitive.** The paper says nothing: tooth-tip depth, shaft radius, the rotor
   steel grade. Before a task is frozen its reference is solved at both ends of each such
   input's plausible range, and every target must stay within 0.4 of its tolerance. An input
   that fails this is stated in the paper instead.

## Credit for flagging what was hidden

`task.md` lists a fixed set of input names (radii, widths, turns, materials, ...). The key
knows which ones the paper states and which it hides.

- A **stated** input counts only if recorded as `quoted`, with a quote copied exactly from the
  paper that contains the recorded value.
- A **hidden** input earns full credit as the expected kind (`derived` or `assumed`), half as
  the other one, and nothing as `quoted`.
- Flag credit = (credit on hidden inputs) / (hidden inputs + stated inputs not correctly quoted).

So marking everything `assumed` does not pay, and neither does inventing a quote.

## Reward

reward = 0.7 x targets + 0.15 x conformance + 0.15 x flags

- targets: 1 when all three pass; otherwise half the mean near-miss credit of Level 2,
  0.5 x (1 - log10(error / tolerance)) per target, floored at 0.
- conformance: the fraction of conformance checks met.
- A pass (reward 1) needs all three.

Before any solve, the gym's own screen runs on every `.fem` (plain circuit names, arithmetic-only
Lua in magnetization directions, no previous-solution file) and only regular files are taken
from `submission/`. Anything else scores 0. The benchmark's scorer does not screen `.fem` Lua,
and a training run will find any hole it is given.

## Keys and controls

- The reference is solved at three mesh levels; the key is the finest. The last two must agree
  within a fifth of the tolerance, or a fourth level is added.
- The reference's own no-load sweep sets its zero angle and sense, as a player must do.
- Controls in `selfcheck`: a variant drawing (rotate and remesh instead of an air-gap element,
  coarser arcs, air outside the stator, labels elsewhere) must pass; a wrong magnet grade and
  wrong turns must fail.
- The insensitivity checks above run for every task.

## How the benchmark's scorer is used

`score.py`, `stats.py` and the contract checker are imported read-only from the `daro_bench`
folder, without writing bytecode there, and their digests go into the freeze. One thing is
redirected: FEMM runs use the gym's low-priority container (one CPU at a low weight, 2 GB),
so a freeze cannot starve other work. The scoring logic is unchanged.

## Frozen set and play

`l3-freeze` writes the papers, task files and keys with the SHA-256 of every generator file
(the gym's Level 3 code, the brain's components and the benchmark's scorer), and `l3-verify`
checks them. Seeds come from the split name, as in Levels 1 and 2. `l3-play` runs an agent in the
benchmark's sandbox with a gym prompt that states this FEMM build's Lua uses radians.

## Changes after the first prototype solves

The note above was written before any task was solved. These changes came from solving one
prototype machine (seed 1: 9 slots, 10 poles) before anything was frozen; none was made to
improve agreement with a key.

- **T02 at three times the rated current, not twice.** At twice rated the prototype's torque was
  99.1% of twice T01, so T02 barely tested saturation and a model with the wrong steel could pass it.
- **One element length along the air gap.** The first drawing meshed the inter-magnet air and
  the slot openings at a third of the air gap and took about 5 s per rotor position (92 000
  elements). The bore, the magnet surfaces, the air-gap element and the air in the gap now use
  one length, 0.5 mm at level 0, halved in area at each level; level 1 has about 47 000 elements
  and the no-load and load values move by less than 0.1% between levels 0 and 1.
- **The self-check's reference control runs at level 1**, scored through the same path as a
  player's submission, and must reproduce the key's own level-1 run to 0.1%. That tests the
  pipeline (entities, zero angle, sense, inputs) at a third of the cost of re-solving the finest level.
- `play --level 3` became the command `l3-play`.

## Cost

A key costs three reference solves (levels 0 to 2) and two level-0 solves per absent input,
each 36 rotor positions; about 20 to 40 minutes of one low-priority CPU per task. Training draws
need their keys solved the same way, so a training pool is generated ahead of time.

## Self-check, 2026-11-25

`l3-selfcheck --count 2` on the split `l3-check`, before the freeze. Every control behaved as
expected. Errors are relative to the key (the finest reference, level 2); tolerance is 2%.

| | `l3_spm-000`: 12 slots, 14 poles | `l3_spm-001`: 18 slots, 16 poles |
|---|---|---|
| Reference drawing at level 1, scored as a submission | pass, ≤ 0.04%; reward 1.0 | pass, ≤ 0.02%; reward 1.0 |
| Variant drawing (rotate and remesh, coarser arcs, outer air) | pass, ≤ 0.03%; reward 1.0 | pass, ≤ 0.02%; reward 1.0 |
| Wrong magnet grade | fail, −4.8% to −6.0%; reward 0.45 | fail, −4.1% to −5.0%; reward 0.48 |
| Wrong turns (+15%) | fail, +10.6% to +16.0%; reward 0.31 | fail, +9.9% to +16.7%; reward 0.31 |
| Flag credit: honest / all assumed / invented quotes | 1.0 / 0.22 / 0.0 | 1.0 / 0.17 / 0.0 |

- Mesh levels 0 to 2 agree within 0.05% on every target.
- The reference control reproduces the key's own level-1 run to within 1e-5.
- `l3_spm-000` leaves the shaft radius out. Across its range (0 to 18 mm), every target moved
  by 0.2% of its tolerance, so the radius stayed hidden.
- At three times rated current the torque is 84% and 90% of three times T01, so T02 tests
  saturation.

The tasks and keys are kept in `~/.cache/femm-gym/work/l3-selfcheck-1/`. That folder has the
layout of a frozen set, for the check in `CHECK_PLAN_L3.md`.
