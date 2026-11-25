# Stand-in check of femm_gym: results

Run on 2026-11-25 from 16:13 to 16:24 IST, following `CHECK_PLAN.md`, which was written before
any trial. The harness was pinned before the first trial in
`~/.cache/femm-gym/plays/check-v1/HARNESS.sha256` (Claude CLI 2.1.310, the benchmark's sandbox
unchanged). All evidence is in `~/.cache/femm-gym/plays/check-v1/`: transcripts, solver logs,
submissions and score records.

## Result: the gym passes

All 7 trials passed, so there is no failure in any class (player, task text or scorer).

| Task (split `check`, index 0) | Scored | Agent time | FEMM runs | Notional cost |
|---|---|---|---|---|
| `l1_pot_core` | least IoU 1.0 | 46 s | 1 | $0.10 |
| `l1_spm_rotor` | least IoU 1.0 | 47 s | 2 | $0.10 |
| `l1_stator` | least IoU 1.0 | 75 s | 2 | $0.19 |
| `l2_coil_inductance` | +0.024% (2% allowed) | 58 s | 3 | $0.09 |
| `l2_core_inductance` | −0.34% | 83 s | 4 | $0.16 |
| `l2_magnet_force` | −0.05% | 52 s | 2 | $0.12 |
| `l2_magnet_linkage` | −0.03% | 67 s | 3 | $0.12 |

Player: `claude-opus-5-5` at effort high, with 20 minutes, 30 FEMM runs and 900 s of FEMM time
per task.

## Observations

- In every Level 2 trial, the number the agent reported matched the scorer's own re-solve to
  four or more significant figures. The tasks describe the measurement clearly enough that
  player and scorer measure the same thing.
- No agent flagged anything as unclear or assumed, and none ran the benchmark's `femm-check`.
- One tool call failed and the agent recovered. In the C-core trial, the agent first ran FEMM
  from a folder outside the task folder; the relay refused it, and the agent moved. The prompt
  had not said that runs must start inside the task folder. It says so now (see below).
- The Level 2 agents chose meshes of 67,000 to 446,000 elements, up to 9 times the finest
  reference mesh. The scorer solved each in at most 4 s within its 768 MB, so the resource
  gap between the relay and the scorer (README, "Known gaps") did not matter at this size.
- The 5-hour usage window went from 64% to 66% across the check, this session included.

## What it means

- Levels 1 and 2 are easy for a frontier agent: about a minute and 1 to 4 FEMM runs per task,
  with near-exact results. They are a warm-up for the smaller open models that RL would train,
  and a floor check for any agent. They do not separate frontier agents.
- The level that can separate strong agents is Level 3, generated papers with gaps.

## Changes after the check

- The `play` prompt now says that a FEMM run must start in the task folder or a folder inside
  it. This changes only the prompt, not the frozen test set's tasks or keys.
