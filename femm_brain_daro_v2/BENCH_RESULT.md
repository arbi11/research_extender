# Block 1 on daro_bench: the result

**Block 1 leaves the brain.** It did not reduce failed FEMM runs. The rule in
[BENCH_TEST_PLAN.md](BENCH_TEST_PLAN.md), committed as cba95f9 before any trial, needed both of
its checks to hold, and the first did not:

| Check | Block 1 | Plain | Holds |
|---|---|---|---|
| 1. Failed FEMM runs, pooled: at most half of plain's share | 9 of 57 (15.8%) | 9 of 55 (16.4%) | no |
| 2. Mean pass fraction: at most 0.15 below plain's | 0.553 | 0.630 | yes (−0.077) |

Block 1 stays in git history. Commit bd29771 holds it in full, and the commit that adds this file
still holds it next to this result.

## The run

- **Run `block1`**, 2026-11-26 11:59 to 2026-11-28 05:56 IST.
  - One pair per case: a plain trial and a block1 trial, run side by side.
  - Seven cases, in the plan's order.
  - Each trial: claude-opus-5-5 at high effort, 75 minutes, at most 60 FEMM runs.
- **The Kazup pilot** (run `block1-pilot`) came first. It is not counted in any number here.
- **The harness** stayed as frozen at 2026-11-26T00:50:59+0530. Nothing in it changed during
  the run, and the plan matched its commit.
- **One infrastructure failure.** In Chung's plain trial the API sent no reply for 14 minutes.
  The runner kept that trial aside, and it was run again on 2026-11-27. Block1's Chung trial
  stood.
- **One change from the plan.** The plan said to pause above 60% weekly plan use. The user waived
  that pause before the first pair. No input or criterion changed.

## Targets passed

| Case | Plain | Block 1 |
|---|---|---|
| aguilar-zamorate-2023-12-10-spm-ipm-femm | 0/2 | 0/2 |
| chung-2021-pmlsm-10p12s-detent | 1/10 | 2/10 |
| huang-2018-vfrm-rotor-shaping | 8/8 | 8/8 |
| kaimori-2025-ieej-virtual-ipm-8p12s-10p12s | 10/11 | 3/11 |
| kazup-2021-axisym-ferrite-magnet-circuit | 3/3 | 3/3 |
| mahariq-2009-srm-8-6-static | 6/12 | 6/12 |
| ren-2018-cp-vs-inset-12-10-12-14 | 9/10 | 9/10 |

The measures outside the rule:

- Every submission in both arms was valid: no contract errors.
- Block 1's median error ratio was lower in 4 cases and higher in 3.
- Agent time: block 1 used 126 minutes, plain 159.
- FEMM time: block 1 used 4,083 s, plain 5,672 s.
- Inputs changed after the first solve: 4 times for block 1, 6 for plain.

## Why failures did not fall

Each failed run's error, sorted by cause:

| Cause | Plain | Block 1 |
|---|---|---|
| A file outside the folder FEMM can see (the trap below) | 1 | 6 |
| Wrong command or material name | 3 | 0 |
| Solver or mesher error | 4 | 1 |
| A bug in the agent's own script (Lua syntax, Python) | 1 | 1 |
| FEMM connection closed | 0 | 1 |

- **The folder trap.**
  - The relay runs each script in a container that sees only the script's own folder, mounted
    as `/work`. The task says so.
  - But rule 4, added for this test, suggests keeping scripts in `work/`.
  - Agents that did this, then saved to `../submission/` or built a path from the script's
    location, got "Could not save". In the container `/work/..` is `/`, so the path became
    `/submission`, which does not exist there.
  - Block 1's agents wrote Python in 6 of 7 cases, and Python scripts build such paths more
    often. Plain's agents wrote Lua in all 7.
  - The trap was found after the verdict, so it does not change it. The trap belongs to
    daro_bench, not to block 1. Fix it and freeze daro_bench again before testing another
    block.
- **What block 1 is for.** It is meant to prevent wrong command and material names. Those fell
  from 3 to 0, but they were too rare to move the share.
- **Kaimori.** Block 1 passed 3 of 11 targets; plain passed 10. Block 1's first eight targets all
  came out low, by 10% to 154%, while the rest matched plain's closely. One trial cannot say
  whether block 1 caused that.

## Records

- **Run data:** `daro_runs/bench/block1/` in the main checkout.
  - It is not in git, because the kept scratch holds the agents' unpacked copies of the papers.
  - `NOTES.md` there records the waived pause.
  - The analysis was made by daro_bench's `analyze_block1.py`:
    - `analysis_block1.json`: sha256 `d4b86e764481ba32888046238a18ce59098e55e158ac63e504eeb86a2fb3886a`
    - `analysis_block1.md`: sha256 `3ab198d515d0544fe18af364e43704172529d761daa8a346ab647a4aa75a9db9`
- **daro_bench** is not in git either, because its cases hold the benchmark's targets.
