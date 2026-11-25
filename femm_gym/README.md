# femm_gym

Generated FEMM tasks with exact answer keys, and a reward that checks a submission by
solving it again. It is the training and test ground for teaching a model to be a hands-on
FEMM expert, and it can measure any agent (Claude, an open model, an agent with or without
the second brain) on the same tasks.

## The ladder

| Level | The player is asked to | Reward | Status |
|---|---|---|---|
| 1. FEMM basics | Draw a described part (stator, rotor, pot core) | Which material lies where, compared point by point | **here** |
| 2. One result | Build a small device; the scorer measures one quantity | Inductance, flux linkage or force against the key | **here** |
| 3. Synthetic papers | Rebuild a generated machine from a paper-like write-up with gaps | Frozen targets, plus credit for flagging what was hidden | **here**; `l3-test-v1` frozen (24 tasks) |
| 4. Real papers | The second brain's paper cases (training only) | As in `daro_bench` | later |
| 5. Design | Improve a published design within limits | A gain that holds on a finer mesh | later |

The `daro_bench` papers are never used for training; they stay the held-out exam.

## Families (Levels 1 and 2)

| Family | Device | Scored | Pass when |
|---|---|---|---|
| `l1_stator` | Planar stator lamination, 12 to 48 trapezoidal slots | Material map | every material's IoU >= 0.95 |
| `l1_spm_rotor` | Planar rotor with 4 to 12 arc magnets | Material map | every material's IoU >= 0.95 |
| `l1_pot_core` | Axisymmetric pot core and winding | Material map | every material's IoU >= 0.95 |
| `l2_coil_inductance` | Axisymmetric air-core coil in open space | Self-inductance | within 2% |
| `l2_core_inductance` | Planar gapped C-core with a two-sided coil | Self-inductance | within 2% |
| `l2_magnet_force` | Axisymmetric ring magnet and coil | Axial force on the magnet, measured as the coil's reaction | within 2% |
| `l2_magnet_linkage` | The same, coil unpowered | Coil flux linkage from the magnet | within 2% |

Each device is built from `femm_brain`'s tested components, with random dimensions inside
stated ranges. Every input is in the task text: Levels 1 and 2 have no gaps.

## Level 3: generated papers with gaps

The player gets a paper about a generated surface-magnet machine with a tooth-coil winding
(`l3_spm`: 6/4 to 24/20 slots and poles, library or specified magnets, laminated or solid rotor
steel) and works in the benchmark's layout: it submits `model.fem`, `entities.json` and
`inputs.json`, and is scored by the benchmark's own `score.py`, imported read-only. Three targets
at 2%: the no-load back-EMF fundamental and the rated torque, both printed in the paper, and the
torque at three times rated current, held out. Each paper hides at least four inputs: some can be
derived from stated values, others are absent and were checked not to matter. A quarter of the
reward is credit for recording honestly which inputs were quoted, derived or assumed: a quote
must be copied from the paper and contain the value. Design, rules and changes: `LEVEL3.md`.

## Rules it keeps

- **The key is hidden and solved, not guessed.** A Level 2 reference is solved at three
  mesh densities (four if needed) and kept only when the last two agree within a fifth of
  the tolerance.
- **The reward comes from the scorer's own solve.** It opens the saved `model.fem`, sets
  the currents itself, solves with the player's mesh within 3 minutes, and measures with its
  own Lua. A number the player reports is recorded but never scored. The magnet's force is
  minus the Lorentz force over all blocks at once: FEMM's `mo_selectblock` toggles, so
  selecting a coil through each of its labels can leave it unselected.
- **Submissions cannot fake a result.** Before any solve, circuit names must be plain and
  the only Lua a `.fem` can carry (magnetization-direction expressions) may do arithmetic
  only. Output lines count only when they carry a random token made for that run.
- **Controls check the reward before anything is frozen.** For sampled tasks, the same
  device drawn differently (coarser arcs, labels elsewhere, FEMM's default mesh, a larger
  open boundary) must pass, and a deliberately wrong device must fail:
  `python3 -m femm_gym selfcheck`.
- **The test set is frozen.** `freeze` never overwrites. It records the SHA-256 of every
  generator file and of every task and key it writes, and `verify` checks both. Seeds come
  from the split name, so training draws never repeat a test task.
- **FEMM runs at low priority** (one CPU, low weight, 768 MB, no network), so a benchmark
  running alongside is not starved.

Reward: 1 for a pass. A miss earns at most 0.5: Level 2 gets 0.5 x (1 - log10(error /
tolerance)) floored at 0, Level 1 gets 0.5 x the mean IoU. A missing, invalid or unsolvable
submission gets 0, with its stage named.

## Commands

Run from the repository root:

```bash
python3 -m femm_gym list
python3 -m femm_gym sample l2_magnet_force --index 3
python3 -m femm_gym selfcheck --per-family 2
python3 -m femm_gym freeze --out femm_gym/testsets/test-v1 --per-family 30
python3 -m femm_gym verify --testset femm_gym/testsets/test-v1
python3 -m femm_gym score --testset femm_gym/testsets/test-v1 --task l2_coil_inductance-004 --submission path/to/submission
python3 -m femm_gym play --runner path/to/daro_bench/sandbox_run.py --out ~/.cache/femm-gym/plays/coil-0 \
  --model claude-opus-5-5 --family l2_coil_inductance --index 0 --split check
python3 -m unittest discover -s femm_gym/tests

# Level 3 (the daro_bench folder from --bench or $DARO_BENCH)
python3 -m femm_gym l3-sample --index 0 --key
python3 -m femm_gym l3-selfcheck --count 2 --jobs 2
python3 -m femm_gym l3-freeze --out femm_gym/testsets/l3-test-v1 --count 24 --jobs 3
python3 -m femm_gym l3-verify --testset femm_gym/testsets/l3-test-v1
python3 -m femm_gym l3-score --testset femm_gym/testsets/l3-test-v1 --task l3_spm-000 --submission path/to/submission
python3 -m femm_gym l3-play --runner path/to/daro_bench/sandbox_run.py --out ~/.cache/femm-gym/plays/l3-0 \
  --model claude-opus-5-5 --index 0 --split l3-check
```

Work files (solutions, scorer scripts) go to `~/.cache/femm-gym/work/` unless `--work` is given.

## A player's episode

The player gets `tasks/<id>.md` and a way to run FEMM (`femm-lua`), and writes
`submission/model.fem` (plus, for Level 2, an optional `submission/answer.json`). It never
sees `keys/`. Then `score` gives the reward record.

`play` runs such an episode with a Claude agent in the benchmark's sandbox (its
`sandbox_run.py`, used unchanged): the workspace holds only `task.md`, `task.json`, the FEMM
manual and an empty `submission/`; the key stays outside, where the sandbox cannot see it.
Checks of the gym itself use fresh tasks from their own split (`--split check`), so the
frozen test tasks stay unseen until a model is evaluated on them. The first check, with Opus 5.5
on one task per family, passed 7 of 7 in about a minute each (`CHECK_PLAN.md`, `CHECK_RESULTS.md`):
Levels 1 and 2 are a warm-up and a floor check, not a test that separates frontier agents.

## Known gaps

- A Level 3 key costs 20 to 40 minutes of one low-priority CPU (three mesh levels and the
  insensitivity checks, 36 rotor positions each), so Level 3 training needs a pool of keyed
  tasks made ahead of time.

- `play` borrows the benchmark's FEMM relay, which gives each run 2 CPUs and 2 GB; the scorer
  gives a solve 1 CPU and 768 MB at low priority. A model near those limits could solve for
  the player and fail for the scorer. The gym's own player tool should use the scorer's
  settings.
- This FEMM build's Lua takes radians in `sin`, `cos` and `atan2`. The benchmark's task
  template says degrees; the gym's prompt says radians.
- `reward.score` reads `model.fem` wherever it points. `play` therefore copies only regular
  files from inside the workspace and refuses links, which could name any of the user's files.
  The check belongs in `reward.score` itself; it moves there at the next freeze, since
  `reward.py` is one of the files `test-v1` is pinned to.
