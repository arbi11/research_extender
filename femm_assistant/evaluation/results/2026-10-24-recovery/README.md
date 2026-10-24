# Native controlled-recovery reference, 2026-10-24

A deliberately incorrect current passed native mesh and solver checks, failed a
frozen analytical field target, and was restored to the ancestral reference model.
Fresh native mesh, solver and target checks then passed. The original incorrect
model and its failed target check remain in the immutable journal.

Two separate runs are retained: a deterministic restoration of the known injected
cause, and one actual OpenRouter/Jev-assisted proposal followed by native execution.
These are real FEMM runs, not mocked solver outputs.

## Fixed reference and observed results

The planar conductor has radius 1 mm, depth 20 mm, one positive turn and a circular
air domain of radius 10 mm. The global mesh size is 0.2 mm. At probe `(2, 0)` mm,
Ampère's law gives `By = mu0 I / (2 pi r) = 0.0001 T` for the prescribed 1 A.
The target and its relative tolerance of 3% were frozen before execution and before
the 2 A fault injection. The same target definitions were rebound to each revision;
their values and tolerances were never changed.

| Attempt | Current | Mesh | Solver | Native By, T | Relative error against the fixed target | Target |
|---|---:|---|---|---:|---:|---|
| Reference | 1 A | Pass | Pass | 0.000099649474517809 | 0.350525% | Pass |
| Injected error | 2 A | Pass | Pass | 0.00019929894903562 | 99.298949% | Fail |
| Recovered | 1 A | Pass | Pass | 0.000099649474517809 | 0.350525% | Pass |

Both runs produced these quantities. The reference mesh has 9,445 nodes and
18,575 triangular elements, with a minimum angle of 30.0069 degrees. Its structural
checks passed; this benchmark does not establish refinement or boundary convergence.
The analytical reference is based on [Ampère's law](https://openstax.org/books/university-physics-volume-2/pages/12-6-amperes-law).

## Actual live proposal

The live run supplied the observed target failure, its operation history, two
ancestral model snapshots, and explicit permission to change only
`/circuits/wire/current_a`. The instruction identified the independently prescribed
1 A operating condition. Jev returned an argument mismatch as advisory evidence.
The draft model selected the ancestral 1 A checkpoint and proposed no extra edits.
The evaluator then restored a separate branch and ran fresh native checks.

The compact [proposal](recovery-proposal.json) was captured before execution; its
`executed: false` field describes that proposal record. The successful execution
is established separately by [summary.json](summary.json), its native artifact
hashes and its preservation checks. Jev's probability did not approve geometry
or replace any of the three execution checks.

## Resource accounting and limits

| Run | Explicit mesh jobs | Full solve jobs | Native FEMM invocations | Worker runtime | Wall time |
|---|---:|---:|---:|---:|---:|
| Deterministic known injection | 3 | 3 | 6 | 4.535 s | 5.466 s |
| Live OpenRouter-assisted proposal | 3 | 3 | 6 | 5.528 s | 9.924 s |

A full solve also meshes internally; the six invocations are not three native
calls. The deterministic run made no provider calls. The live proposal was bounded
to one Jev diagnostic and one draft response. Its provider usage, token counts and
cost were not captured by the current transport and remain **unknown**, not zero.

No restart-from-scratch comparator was run. **Comparative resource savings are not
proven.** This single known-cause injection does not estimate general diagnosis
accuracy, demonstrate arbitrary geometry recovery, or constitute a journal-paper
extraction/reproduction result.

## Evidence and reproduction

The public files contain allowlisted quantitative records, hashes and a scoped
proposal. Native artifacts, per-stage execution manifests and complete journals
remain in the local evaluation directories:

- `work/replication-development/recovery/recovery-loop`
- `work/replication-development/recovery-live-v1`

The source summary hashes are retained in `summary.json`. No credentials,
environment files, private documents or private document text are included.

From the `femm_assistant` directory in the patched native FEMM Docker image:

```bash
python -m evaluation.recovery_loop --output /results/new-recovery-reference
```

To request one bounded real provider proposal instead of deterministic restoration,
configure the existing server provider environment and use `--live`. Use a fresh
output directory; the runner refuses to overwrite earlier evidence.
