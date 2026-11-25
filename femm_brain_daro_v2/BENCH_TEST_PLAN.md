# Block 1 on daro_bench: test plan

**Status: approved by the user on 2026-11-26 ("Yes, build it").** Before the first trial, this
plan is committed on `red`. Its SHA-256 is then recorded in two places: in
`daro_bench/analyze_block1.py`, and in a note on the harness block in
`daro_bench/cases/HARNESS.sha256` that the pilot and the main run use.

## Question

Does block 1 help an agent replicate a machine on `daro_bench`? Block 1 is this folder's checked
catalogue of FEMM 5's magnetics commands for Python. The rebuild keeps a block only if a run with
it and a run without it show a gain.

## What earlier trials say

These counts were taken on 2026-11-26 from each trial's `run/solver.jsonl`. They cover the full
run and the repeats: 26 scored trials, all in Lua, run before `femm-python` existed.

- **Plain agents** ran FEMM 104 times. 13 of those runs failed (12%): 11 ended with an error
  (exit code 1) and 2 hit the per-run time limit (exit code 124).
- **Brain agents** ran FEMM 134 times. 21 runs failed (16%), all with exit code 1.
- That is about one failed run per plain trial, out of about eight runs. Saving those runs is
  the most direct gain block 1 can make.
- **Pass fraction was decided elsewhere.** Every full-run trial submitted a valid model. The gaps
  in pass fraction came from inputs the papers leave unstated, such as Aguilar's magnet direction
  and Ren's steel curve, not from scripting.

So block 1 is unlikely to change the pass fraction, and the rule below is mainly about failed
runs. Seven pairs can show only a large effect.

## Arms

Both arms use the harness block frozen for this test. Each trial gets:
- the model `claude-opus-5-5` at high effort;
- 75 minutes, 60 FEMM runs, 3600 FEMM seconds and a $200 cap;
- the same prompt;
- both `femm-lua` and `femm-python`.

The arms differ only in this:
- **plain**: nothing else.
- **block1**: in addition, `femm_brain_daro_v2/` with `catalogue.json`, `CHEATSHEET.md`,
  `lookup.py` and `README.md`, exactly as committed at `bd29771`. Its `TASK.md` also gets this
  paragraph:

  > - `femm_brain_daro_v2/`: a checked reference for driving FEMM 5 from Python (pyFEMM).
  >   `CHEATSHEET.md` lists every magnetics command with its arguments, and the engine's known
  >   pitfalls. Every entry was tested in this FEMM build. `python3 femm_brain_daro_v2/lookup.py
  >   get NAME` prints one command in full (`--test` adds the script that checked it), `search
  >   WORDS` finds commands, and `facts` lists the pitfalls. Lua uses the same command names. The
  >   README's `verify/` and `tests/` folders are not included.

The v1 brain arm does not run, because the question is block 1 against no help at all.

## Changes before the freeze

1. **The block1 arm.** `prepare.py` gets the `block1` arm, with `BLOCK1_COMMIT = "bd29771"`, its
   four paths and the paragraph above. `run_bench.py` records the commit in `config.json`.
2. **The two fixes left open after the repeats,** made for both arms:
   - **Keep generator scripts.** In the repeats, agents wrote them beside the task folder (for
     example in `~/work/scripts/`) or in `/tmp`, and both were lost with the sandbox. Both are
     now kept with the trial's evidence (`sandbox_run.py --keep-scratch`), and the task asks
     agents to keep their scripts in the task folder.
   - **A neutral `inputs.json` example.** "M19, 26 points" becomes "the named grade's datasheet
     curve, 26 points".
3. **`analyze_block1.py`.** It computes the measures and the rule below. It also checks the
   harness digests against this test's freeze block.
4. **Tests.**
   - A block1 workspace holds exactly the four files at `bd29771`.
   - Only the block1 task names them.
   - The existing tests still pass.
5. **A new harness block** from `freeze.py`, including this plan's digest and
   `analyze_block1.py`'s.

## Run order

1. **Pilot:** one pair on Kazup, outside every statistic. It checks the new harness end to end:
   - both trials are scored;
   - `femm-python` and `lookup.py` work in the sandbox;
   - `/tmp` is kept;
   - no sign-in token is left in the evidence.

   A harness fault is fixed under a new harness block, and then the pilot runs again.
2. **Main run:** all seven cases, one pair each, in the full run's order: Aguilar, Chung, Huang,
   Kaimori, Kazup, Mahariq, Ren. Both arms of a pair run at the same time:

   ```
   python3 daro_bench/run_bench.py --run block1 --cases <case> --arms plain block1
   ```

3. **Between pairs,** check usage:
   - if the weekly limit is above 60%, pause and tell the user;
   - if the 5-hour limit is above 60%, wait for it to reset.

   Infrastructure failures are rerun under the rules in `DESIGN.md`. Nothing is stopped or
   changed because of a result.

**Cost.**
- **Time per pair:** in the full run, a pair took 6 to 103 minutes, scoring included. Kazup was
  the shortest and Kaimori the longest; Chung's time is unclear because of a pause.
- **Plan use per pair:** in the repeats, a pair used 9 to 22% of a 5-hour plan window.
- **Main run:** about 4 to 7 hours, spread over one or two plan windows.

## Measures

Per trial:
- **Pass fraction:** acceptance targets passed ÷ acceptance targets, computed as in `analyze.py`.
  An unscored trial counts as 0.
- **Failed FEMM runs:** the `solve` events in `run/solver.jsonl` with a nonzero exit code.
- FEMM runs, FEMM seconds, agent minutes and notional cost.

These are also reported, but are not part of the rule:
- the language of each run (`.lua` or `.py`);
- the parts of block 1 each agent opened or ran, from its transcript;
- the first error line of each failed run;
- the median error ratio;
- whether each submission was valid;
- the inputs changed after the first solve.

## Rule

Block 1 stays only if both of these hold:

1. **Fewer failures.** block1's share of failed runs, pooled over its seven trials, is at most
   half of plain's. If plain has no failed run, block 1 cannot show this gain.
2. **No harm.** block1's mean pass fraction is at most 0.15 below plain's. That allows for about
   one case flipping completely (−1/7 ≈ −0.14).

Otherwise block 1 leaves the brain. It stays in git history, together with this result.

**How likely rule 1 is to hold.** This was simulated with plain's past failure rate, 40 to 60
runs per arm, and runs treated as independent:

| Block 1's true effect | Rule 1 holds |
|---|---|
| none | 11 to 17% of the time |
| failures cut by half | about 56% |
| failures cut by two thirds | about 76% |
| failures cut by 90% | about 97% |

Runs inside one trial are not truly independent, so the first figure is a lower bound.
