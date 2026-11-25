# Level 3 check: results

Run on 2026-11-25 from 22:23 to 22:46 IST, following `CHECK_PLAN_L3.md`, which was written
before any trial. The harness was pinned before the first trial in
`~/.cache/femm-gym/plays/l3-check-v1/HARNESS.sha256`:
- Claude CLI 2.1.310;
- gym commit `bb5da1d`;
- the benchmark's sandbox and scorer, unchanged.

The first trial waited for the benchmark's last repeat pair to finish and for the token
refresh. All evidence is in `~/.cache/femm-gym/plays/l3-check-v1/`: transcripts, solver logs,
submissions and score records.

## Result: one scorer fault, so Level 3 does not pass the check yet

| Task (split `l3-check`) | Scored | Errors (2% allowed) | Flags | Agent time | FEMM runs | Notional cost |
|---|---|---|---|---|---|---|
| `l3_spm-000`, 12 slots / 14 poles | **pass**, reward 1.0 | E01 −0.20%, T01 −0.20%, T02 −0.46% | 1.0 | 608 s | 7 (279 s) | $1.22 |
| `l3_spm-001`, 18 slots / 16 poles | fail, reward 0.53 | E01 no result; T01 −0.03%, T02 −0.01% | 1.0 | 448 s | 9 (159 s) | $0.93 |

- **Player:** `claude-opus-5-5` at effort high, with the benchmark's budget (75 minutes, 60 FEMM
  runs, 3600 s of FEMM time).
- **Sign-in:** unchanged after both trials.
- **Usage:** the 5-hour usage window went from 38% to 71% over this period. That figure also
  covers the benchmark's last pair and this session.

## The scorer fault (trial 2)

The model is right. The agent's own 24-position no-load sweep gave E01 = 61.18 V; the key is
61.18 V. The scorer solved 22 of its 24 positions. At 11.25° and 33.75° the solver stopped with
"error code 2560": exit code 10, "Material properties have not been defined for all regions".

**Cause.** The scorer's `rotate` motion moves each moving group in turn, from the saved
position, by the absolute angle. Its comment explains this: "FEMM selects one group at a
time". The agent's rotor was group 1 and the air on the rotor side was group 2, with its one
label in an inter-magnet gap at 11.25°. At p = 11.25° and 33.75° this happens:

1. Turning group 1 puts a magnet label exactly on the air label, which has not moved yet.
2. After every move, FEMM re-adds all labels (`EnforcePSLG`). `AddBlockLabel` silently refuses
   a label that lies on another one. The saved state has 56 labels instead of 57.
3. Rotating group 2 then has no label to move, and the rotor-side air has no material.

**What was checked:**
- The failure repeats exactly when re-scored.
- Nudging the angle by 1e-7° does not avoid it.
- Rotating in the agent's 1.875° steps does avoid it.
- In this FEMM build `mi_selectgroup` adds to the selection, unlike original FEMM 4.2.
  Selecting groups 1 and 2 and rotating once keeps all 57 labels.
- Re-scored that way, trial 2 passes: E01 −0.005%, T01 −0.03%, T02 −0.01%, reward 1.0.

**Class.** This is a scorer fault: a valid model was misread. The benchmark's scorer has the
same latent flaw, for `rotate` and for `translate` motion. No benchmark submission so far
(0 of 74 that move a part) has used more than one moving group, so no benchmark result was
affected.

## Other observations

- **Inputs and provenance.** Both agents recorded every input honestly, with full flag credit:
  - the hidden dimensions were derived, each with its expression;
  - the absent shaft radius in trial 1 was assumed, with a reason;
  - the magnet values were taken from the named library grade.

  Trial 1 changed three inputs after its first FEMM run, all corrections to FEMM's library: a
  typical N40 remanence replaced by the library's, and two steel names made exact.
- **The paper's slot area (minor, no points lost).** The paper's fill factor and current density
  imply a slot about 11% larger than the drawn one. The generator estimates the slot as an
  annular sector minus the tooth body and ignores the tooth tips. The agent in trial 1 noticed
  this and correctly kept the stated geometry. It should take the area from the drawn slot.
- **Difficulty.** Both models were right in under 11 minutes, with the published values matched
  to 0.2%. Like Levels 1 and 2, Level 3 as generated is not hard for Opus 5.5. The hard part
  of a real paper (missing, conflicting or figure-only data) is not yet in these papers.

## What it means

By the plan, the scorer fault must be fixed before anything is trained on, and the set frozen
again as `l3-test-v2`, with v1 kept and marked as superseded. The fix is to move all of a
model's moving groups in one selection. Whether it goes into the gym only, or also into the
benchmark's next version, is the user's decision.
