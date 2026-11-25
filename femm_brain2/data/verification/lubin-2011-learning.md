# First hierarchical learning case: Lubin 2011

The L1–L5 graph now reconstructs Example 1's geometry and no-load condition.
This is a learning paper, not a held-out test, and only the stated scope passed.
Loaded torque, Example 2 and the slot-opening sweeps remain outside this case.

Inputs and acceptance criteria were committed in `ff0482f` before the first
native solve. The final run used the same physical parameters and acceptance
criteria. The figure reference was extracted directly from 20 FEM markers in
Fig. 10, with the duplicated electrical-period endpoint excluded.

| Quantity | Frozen reference or bound | Real FEMM / observed |
| --- | --- | --- |
| Per-turn phase-A EMF fundamental, V peak | 0.805892075 | 0.814360163 |
| Absolute EMF error, V | <= 0.016058921 | 0.008468088 |
| Finer-mesh EMF fundamental, V peak | Diagnostic against baseline | 0.814387627 |
| Relative change with finer mesh | <= 1% | 0.003372% |
| Flux-linkage change with gauge offset, Wb | <= 1e-7 | 4.47129e-14 |
| Zero-magnet circuit flux linkage, Wb | <= 1e-10 | 0 |
| Zero-magnet probe field, T | Reported diagnostic | 0 |

The EMF allowance is additive: 1% numerical agreement plus 0.008 V for twice
the 0.004 V figure-reading uncertainty. The relative error is
1.0508%. No material value or effective length was fitted.

The final run completed **75 real solves** across five passing checks: pilot,
36-position baseline sweep, 36-position finer-mesh sweep, zero-magnet control
and gauge-offset control. The baseline has 39,975 nodes and 76,104 elements;
the finer mesh has 74,724 nodes and 144,171 elements. These are repeated solves
of one learning case, not 75 independent paper test cases.

Geometry checking reads saved native coordinates, segment/arc connectivity,
angles, groups, materials, circuit/turn assignments and each radial-magnet
expression. An independent solver-file reader checks mesh counts, finite
potentials/fields, total meshed area and individual coil/magnet region areas.
Every non-iron label must own a meshed region. The expected 1 mm air gap,
183.259571 mm² slots and 76.5-degree magnets follow from Table I's dimensions.

## Preserved findings and corrections

- **run1:** one native pilot solved, but the numeric-only label reader rejected
  FEMM's quoted radial-direction expressions. The original failed report,
  model and solver files remain unchanged. The reader now separates the nine
  numeric label fields from the direction expression and accepts ordinary
  header whitespace. FEMM's documented implementation normalizes zero turns
  to one; that serialization rule is checked without changing the input.
- Six air-air arcs were added to partition slot necks from the gap. They impose
  no boundary condition or material change and give each neck label its own
  region. This construction repair, with updated geometry counts, was
  committed before run2. The physical dimensions and EMF target did not change.
- **run2:** all 75 native solves completed. The initial and zero/gauge cases
  passed, while the two sweeps failed because the saved rotor-node groups
  were zero after movement/geometry cleanup. The assessment also referenced
  an incorrect tolerance key; it was corrected to the already-frozen field.
  A nonexistent B-field threshold was removed from the checker; the frozen
  zero-flux bound remains enforced and zero-field B remains a diagnostic.
- **run3:** the generated procedure restores rotor group membership after
  movement, using a selection circle strictly inside the air gap. All five
  checks and the independent evidence audit pass. Parameters, bindings,
  targets, tolerances and controls are identical to run2; a regression test
  checks that identity. The group-preservation finding is stored on the L5
  rotor procedure for later retrieval.

The FEMM program is restarted after twelve positions within the private
container to bound memory use. The pinned image, exact Lua binding, network
isolation and read-only filesystem configuration are recorded in the report.

## Inspect and reproduce

- [Current native report](knowledge-lubin-2011-20261125-run3/report.json)
- [Generated Lua](knowledge-lubin-2011-20261125-run3/baseline/native.lua)
- [API-to-procedure call trace](knowledge-lubin-2011-20261125-run3/baseline/api-trace.json)
- [Frozen knowledge snapshot](knowledge-lubin-2011-20261125-run3/knowledge.json)
- [Baseline measurements](knowledge-lubin-2011-20261125-run3/baseline/observations.obs)
- [Earlier failed pilot](knowledge-lubin-2011-20261125-run1/report.json)
- [Earlier complete sweep with failed group checks](knowledge-lubin-2011-20261125-run2/report.json)

Each run retains its first and final native model/solver files, all observation
rows, Lua, API traces, logs and fingerprints. Intermediate solver snapshots are
not retained. The audit verifies hashes and regenerates the script from the
current graph before recomputing the checks. Historical evidence is deliberately
not accepted as current after construction code or graph records change.

The implementation passes 42 package tests, including evidence tampering and
source/target preservation checks. The original package's 324 tests completed
successfully with two skips; graph validation and paper checks also pass.
