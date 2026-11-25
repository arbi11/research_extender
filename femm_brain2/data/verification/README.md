# Native FEMM evidence, November 25, 2026

The first L1–L5 learning case is now recorded separately from the API checks:
[Lubin 2011, Example 1 geometry and no-load EMF](lubin-2011-learning.md).
Its five checks pass across 75 real solves, with earlier failures retained.
This is one learning paper, not a held-out benchmark or complete-paper replication.

The current 68-command catalogue has three reports. Geometry and property
checks each passed 16/16 scenarios. The three analytical field solves and the
separate mesh-return capability check also passed: 36/36 native checks total.

| Current report | Outcome |
| --- | --- |
| `geometry-20261125-contours/report.json` | 16/16 geometry scenarios passed, covering 33 commands |
| `properties-20261125-contours/report.json` | 16/16 property scenarios passed, covering 15 commands |
| `solve-20261125-contours/report.json` | 4/4 field and mesh-return checks passed, covering 20 commands |

To attach all three reports, use the build command in the package README.
The database records all 68 commands passed for the tested cases. Each report is
bound to its exact reviewed command batch, binding source and container digest.
The importer rechecks artifact hashes and recomputes the observations.

## Analytical field results

A 0.04 m by 0.02 m linear region has depth 0.01 m, relative permeability 1,
and zero source current. Prescribed outer potential `A=A0+Bx*y` produces a
uniform field. Two equal-area pickup legs carry +10/-10 turns with zero current.
Their flux linkage is `10*depth*Bx*0.01`, independent of the offset A0.

| Quantity, positive case | Analytical target | Native FEMM |
| --- | --- | --- |
| Bx, T | 0.2 | 0.200000000028984 |
| Hx, A/m | 159154.943091895 | 159154.943114960 |
| Total energy, J | 0.127323954473516 | 0.127323954472177 |
| Pickup flux linkage, Wb-turn | 0.0002 | 0.000200000000000045 |

The reversed-field and zero-field controls also passed. Checks cover four
interior query points, potential/flux density/field intensity/energy density,
block area/volume/energy, zero circuit current and voltage, flux linkage, and
reloading the solution. Field values and tolerances are frozen in the protocol.
This is an analytical reference problem, not a journal-paper replication.

The independent `.ans` reader checks all solver nodal potentials against
`A0+Bx*y`, derives each triangle's B from nodal A, and integrates magnetic
energy. Each model has 3,847 nodes and 7,440 elements. The postprocessor counts
match the saved solver mesh.

## Final contour exercise

Targets, measurement geometry and tolerances were frozen in commit `b681d61`
before the native runs. The original three field problems are unchanged. Each
adds 62 checked contour quantities (186 measurements across three independent
solves); the mesh-return probe reuses the positive solution and is not a fourth
independent solve.

All six `mo_lineintegral` selectors passed in planar DC. Straight and reversed
contours check sign; a horizontal line supplies nonzero force and torque;
a closed clockwise rectangle checks cancellation. Both 180-degree bends check
six-chord length and endpoint integrals. Clearing the contour yields zero length
and area. These cases verify four new commands, bringing native coverage to 68.

| Quantity, positive field | Analytical target | Real FEMM |
| --- | --- | --- |
| Forward flux, Wb | -0.00002 | -0.0000200000000037946 |
| Reversed flux, Wb | +0.00002 | +0.0000200000000037946 |
| Horizontal-line Fy, N | -3.18309886183791 | -3.18309886064612 |
| Horizontal-line torque, N*m | -0.0636619772367582 | -0.0636619772055221 |
| Closed-loop Fx, N | 0 | -1.22639e-10 |
| Closed-loop Fy, N | 0 | 1.33846e-9 |
| Closed-loop torque, N*m | 0 | 1.45953e-12 |
| Bent six-chord length, m | 0.024846628329842 | 0.024846628329842 |

Negative and zero-field controls also passed against their frozen expectations.
Open-line stress values are surface contributions, not net force on a body.
AC, axisymmetric cases and the side of the bent contour remain untested.
All 36 native scenarios passed against the exact 68-command batch. Earlier
`*-mesh-count-fixed` runs are preserved for the preceding 64-command batch.

## Final contour exercise

Frozen commit `b681d61` adds four command contracts and 62 contour measurements
per field solve, with targets and tolerances committed before native execution.
The unchanged positive, negative and zero-field models all passed. The separate
mesh-return case reuses the positive model and is not an additional field solve.
Geometry and property suites were refreshed because evidence is bound to the
complete 68-command import batch. The runtime image is unchanged.

| Contour quantity, positive field | Analytical target | Native FEMM |
| --- | --- | --- |
| Upward-line flux, Wb | -0.00002 | -0.0000200000000037946 |
| Reversed-line flux, Wb | 0.00002 | 0.0000200000000037946 |
| Horizontal-line y force contribution, N | -3.18309886183791 | -3.18309886064612 |
| Horizontal-line torque contribution, N*m | -0.0636619772367582 | -0.0636619772055221 |
| Closed-contour x force, N | 0 | -1.22639e-10 |
| Closed-contour y force, N | 0 | 1.33846e-9 |
| Closed-contour torque, N*m | 0 | 1.45953e-12 |
| Bent-contour polygon length, m | 0.0248466283298420 | 0.0248466283298420 |

The tests exercise all six `mo_lineintegral` selectors, straight-contour
reversal, both +180/-180 degree bends with six straight subdivisions, and
clearing the contour. A bent-contour check establishes polygon length and
endpoint integrals; it does not independently establish which side the arc
occupies. Force/torque from an open line are surface contributions, while the
closed rectangle checks net cancellation in a uniform field. No body-force,
AC, axisymmetric or paper-replication validation is claimed.

The previous `*-20261125-mesh-count-fixed` reports remain preserved as passing
results for the earlier 64-command batch. All original physical inputs and
analytical targets remain identical; only new measurements have been added.

## Preserved solve runs and mesh limitation

Physical inputs, analytical targets and tolerances were committed before
native solving. They were not changed to obtain agreement.

- `solve-20261125-run1`: stopped before the first solve because `mi_createmesh`
  returned nil and the observation writer expected a number. The incomplete
  models, original protocol, script, log and failed report remain preserved.
- `solve-20261125-run2`: the driver records nil without aborting, and a separate
  strict capability case still requires the implementation count. All three
  field cases passed; the count case failed. Mesh export was enabled using
  the container's existing `FEMM_MESH_EXPORT=1` option.
- `solve-20261125-run3`: repeats the same physical cases after the catalogue
  explicitly documents the native return mismatch. Results remain 3/3 field
  cases passed plus the failed count case.

Those historical runs used a FEMM build whose mesh-creation binding omitted
the native return. Their 3/4 results and failed capability remain unchanged.

## Mesh-count fix and rerun

The separate image `femm-brain2:mesh-count-20261125` backports the actual mesh
vertex count through the shared C++ helper and the API binding. The original
runtime and source archive are retained. The solver executable is byte-identical
to the original; the headless API executable is rebuilt from the retained source
and the checked patch. Build details are in `runtime/` and each new run's
`runtime-build.json`.

The new profile pins image
`sha256:e927378c714cca415c6471122a73b34bae0cebf0ab00381b01a4a5f1199aa0c3`.
Every field control requires a real returned count and comparison with the
postprocessor and solver file: 3,847 nodes and 7,440 elements. All four solve
checks passed. All physical inputs, analytical values and tolerances were kept;
every non-count observation exactly equals the earlier run. The geometry and
property checks were rerun in the same new image and all passed.

## Geometry and property history

All older reports and superseded refreshes remain preserved. The historical
38-command geometry run passed 14/16 initially, then 16/16 after two checker
corrections committed before rerunning: locate the assigned rectangle edge by
endpoints rather than row number, and expect mirroring to retain the original
geometry as well as its copies. Original failures and expectations remain.

Property scenarios verify all material arguments and modification selectors,
sorted/deduplicated B-H points and clearing, all eight boundary formats,
complex coefficients/currents, point properties, series/parallel circuits,
deletion, assignments and assignment persistence after reopening models.

Reports named `*-property-refresh`, `properties-20261125-run1` and the earlier
`*-solve-refresh` belong to prior command batches. They remain readable but
cannot be attached to the current batch.

Every run retains its script, protocol, actual saved models, captured log and
artifact fingerprints. Downloaded binding source is not committed. All native
runs use a networkless, read-only container with a writable output directory.
