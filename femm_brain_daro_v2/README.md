# femm_brain_daro_v2

A from-scratch second brain for replicating electric machines in FEMM, built one block at a
time. Each block is a file or a small tool, and stays only if it helps on `daro_bench`.

## Blocks tried so far

| Block | What it was | Result |
|---|---|---|
| 1 | A checked catalogue of FEMM 5's magnetics commands for Python (pyFEMM). It had 192 commands and 18 engine facts, each with a test that ran in FEMM, plus a lookup tool. | Left the brain on 2026-11-28: it did not reduce failed FEMM runs ([BENCH_RESULT.md](BENCH_RESULT.md)) |

Block 1 is kept in git history, at commit bd29771. To read it again:

```bash
git show bd29771:femm_brain_daro_v2/CHEATSHEET.md
git archive bd29771 femm_brain_daro_v2 | tar -x -C <an empty folder>
```

- [BENCH_TEST_PLAN.md](BENCH_TEST_PLAN.md): how block 1 was tested, fixed before any trial.
- [BENCH_RESULT.md](BENCH_RESULT.md): what the test found. It includes a daro_bench trap to fix
  before the next block is tested.
