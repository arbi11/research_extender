# Stand-in check of femm_gym: plan

Written before any agent has played a gym task. It checks the gym, not the agent: can a capable
agent read each family's task, build the model and be scored fairly?

## Trials

- One fresh task per family from the split `check` (index 0), so the frozen test set stays
  unseen: 7 trials.
- Player: `claude-opus-5-5` at effort high, through `python3 -m femm_gym play` and the
  benchmark's `sandbox_run.py` (unchanged).
- Budget per trial: 20 minutes, 30 FEMM runs, 900 s of FEMM time.
- One trial at a time, and only when the benchmark is not running, since both draw on the same
  plan usage.

## What is recorded

For each trial: the stage reached, pass or fail, the reward, the error ratio or least IoU, the
FEMM runs and seconds used, wall time and notional cost. Everything stays under
`~/.cache/femm-gym/plays/`.

## How it is read

Each failure is put in one of three classes:

- **player**: the task was clear and the scorer right; the agent made a mistake;
- **task text**: the task was ambiguous or wrong;
- **scorer**: the scorer mis-measured, crashed or misread a valid model.

The gym passes the check if no failure is in the scorer class. A task-text or scorer fault is
fixed in the generator and the test set is frozen again as `test-v2`; `test-v1` is kept and
marked as superseded, with the reason. The agent's pass rate is a first baseline, not a
criterion: if it passes everything, Levels 1 and 2 are a warm-up for smaller models, not a test
for frontier ones.
