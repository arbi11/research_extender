# Native FEMM evidence, November 25, 2026

The current 64-command catalogue has three reports. Geometry and property
checks each passed 16/16 scenarios. All three analytical field solves passed;
one separate mesh-return capability check failed and remains visible.

| Current report | Outcome |
| --- | --- |
| `geometry-20261125-field-refresh/report.json` | 16/16 geometry scenarios passed, covering 33 commands |
| `properties-20261125-field-refresh/report.json` | 16/16 property scenarios passed, covering 15 commands |
| `solve-20261125-run3/report.json` | 3/3 analytical field cases passed; separate mesh return-count check failed |

To attach all three reports, use the build command in the package README.
The database records 63 commands passed and `mi_createmesh` as `check_failed`
with its source interpretation flagged `needs_clarification`. Each report is
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

The newer pinned C++ source returns a mesh node count, but this older container
returns no value for `mi_createmesh`. `mi_analyze` still constructs the mesh and
solves successfully. Solved counts are available through `mo_numnodes` and
`mo_numelements`. `check-solve` reports 3/4 passed with exit code 2; the failed
capability is not relabeled a pass.

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
