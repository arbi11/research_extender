# Level 3 check: plan

Written before any agent has played a Level 3 task. Like the check of Levels 1 and 2
(`CHECK_PLAN.md`), it checks the level, not the agent: can a capable agent read a generated
paper, rebuild the machine in FEMM, record which inputs the paper hides, and be scored fairly
by the benchmark's own scorer?

## Trials

- The two tasks whose controls the self-check verified: `l3_spm-000` and `l3_spm-001` of the
  split `l3-check`, played from the self-check's folder, so the frozen `l3-test-v1` stays unseen.
- Player: `claude-opus-5-5` at effort high, through `python3 -m femm_gym l3-play` and the
  benchmark's `sandbox_run.py`, unchanged.
- Budget per trial: the benchmark's, 75 minutes, 60 FEMM runs and 3600 s of FEMM time.
- One trial at a time, and only while the benchmark is not running: both use the same sign-in
  and plan usage.

## What is recorded

For each trial: the stage reached, pass or fail, the reward and its three parts, each target's
error, the conformance checks, the flag credit of every listed input, the inputs changed after
the first FEMM run, the FEMM runs and seconds used, wall time and notional cost. Everything
stays under `~/.cache/femm-gym/plays/l3-check-v1/`.

## How it is read

Each point lost is put in one of three classes:

- **player**: the paper was clear and the scorer right; the agent made a mistake (a wrong
  input, a modelling error, a gap it did not see, a wrong provenance);
- **paper or task text**: the paper or `task.md` was ambiguous, wrong or contradicted itself;
- **scorer**: the scorer mis-measured, crashed or misread a valid model, or the flag credit
  refused an honest record (a verbatim quote, a correct derivation).

The level passes the check if no point lost is in the scorer class. A fault in the paper, the
task text or the scorer is fixed in the generator before anything is trained on, and if
`l3-test-v1` is already frozen it is frozen again as `l3-test-v2`, with v1 kept and marked as
superseded with the reason. The agent's score is a first baseline, not a criterion.

## Added before the first trial (2026-11-25, 20:50 IST)

When the check was approved, the benchmark's repeats were running their last pair. So:

- The first trial waits for the benchmark to finish.
- A trial starts only when the sign-in token outlasts its time limit by 20 minutes. This is the
  benchmark runner's own rule, and its `ensure_token` is used unchanged. Otherwise the check waits
  for the token to expire and be refreshed.

A trial that outlived the token would refresh it inside the sandbox while other sessions refresh
it outside, and one of them could lose its sign-in.
