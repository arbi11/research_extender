# FEMM API lookup catalogue

The catalogue contains 68 source-checked FEMM 5.0 API entries for the Lua 4 binding:
33 geometry, selection, labeling and editing commands; all 15 material, boundary
and circuit property commands in section 3.6.9; and 20 file, mesh, solve and
postprocessing commands. All 68 have passed the recorded native checks for
their tested cases.

The first L1–L5 learning graph is now implemented for Lubin 2011, Example 1.
See [SECOND_BRAIN.md](SECOND_BRAIN.md) for its records, code, retrieval commands
and real FEMM evidence. Start with `python3 -m femm_brain2 brain tree`.

SQLite is the working lookup database. Commands return JSON. The versioned
`data/commands.json` file is the reviewed import batch used to reproduce this
milestone; updates go through validation and a rebuild, rather than independent
edits to the database and import batch. Raw documentation stays in a local cache.

## Try it

Run these from `/home/akhan/code/research_extender`:

```bash
python3 -m femm_brain2 fetch-sources
python3 -m femm_brain2 validate
python3 -m femm_brain2 build
python3 -m femm_brain2 search "draw arc"
python3 -m femm_brain2 get mi_addarc
python3 -m femm_brain2 search "magnetic problem frequency" --physics magnetics
python3 -m femm_brain2 stats
```

The default database is `~/.local/share/femm-brain2/catalogue.sqlite3` (or the
equivalent under `XDG_DATA_HOME`). Every command accepts `--db /path/to/file`.
Source downloads use `~/.cache/femm-brain2/sources/`, respecting `XDG_CACHE_HOME`.
`fetch-sources`, `validate`, `build` and all three `check-*` commands accept
`--cache`.

`get` and `search` support `--version 5.0 --binding lua4`. Search also accepts
`--physics` and `--limit`. Exact lookup returns the complete contract. Search
returns concise matches with signatures, review status and source references.
`export` writes the complete catalogue as JSON to stdout. `stats --all-pending`
includes the full list of detected manual commands awaiting review.

Lookup is read-only and works offline after the database is built. A missing
database produces an error with the build command. An inventory name without a
reviewed entry returns `not_reviewed`, rather than an invented API contract.

## What an entry contains

- Actual command name, FEMM version, binding, physics and a flat category.
- Purpose, normalized signature, ordered arguments, units, optional arguments,
  documented defaults and allowed values where supported by the source.
- Return values in order, preconditions and effects on the current model.
- Source URLs, document pages/sections, source hashes and definition hashes.
- Review status, unresolved questions and explicitly labelled example fragments.
- An independent execution status, with optional verified native evidence.

The importer detects 558 distinct manual definitions; 490 remain an inventory for later
work. This count describes this pinned manual's detected definitions, not a
claim that every alias, convenience wrapper or undocumented function is covered.
Repeated definitions retain their additional page locations in the inventory.

## Sources and who authored the entries

The official source set is pinned in [sources.json](data/sources.json):

1. [FEMM 5.0 User's Manual](https://femm.info/doku/lib/exe/fetch.php?media=femm-qt%3Adocs%3Amanual.pdf),
   dated October 1, 2026, SHA-256 beginning `e1273290fd7856a0`.
2. The FEMM project's Lua launcher guide and `femm.lua` binding, pinned to
   upstream revision `31ee4bf6a8cae1a1860c108e7934ab3403bfd6e3`.
3. The matching common and magnetic C++ API implementations, with line locators
   and definition hashes for the reviewed excerpts.
4. The geometry implementation that confirms mirroring creates copies.

The coding agent read the manual definitions and binding source and authored
the semantic records in [commands.json](data/commands.json). PDF text
extraction and command-name inventory detection are mechanical. The Python
validator checks record consistency and source evidence; it does not independently
prove that every agent-written interpretation is correct. Relevant manual pages
were also rendered and reviewed, including the return table that spans pages
108-109.

Full source text is cached locally rather than committed. Definition fingerprints
bind the reviewed records to the cited source passages without copying those
passages into the repository. Fresh downloads are accepted only if their hashes
match the pinned snapshots. Build validates the sources before atomically
replacing an existing catalogue belonging to this project.

Known limitations stay visible in lookups:

- `mi_addarc.maxseg`: the manual's segment-count wording conflicts with the
  implementation. The adder writes the same angular-discretization field used
  by `mi_setarcsegmentprop`; its unit is degrees per mesh element.
- The six transform wrappers do not forward the manual's optional `editaction`.
  The catalogue uses their callable Lua signatures. The pinned C++ implementation
  defaults omission to all selected entity types (4), rather than reading edit mode.
- `mi_probdef`: the manual signature has a missing closing parenthesis. The
  normalized signature repairs that typography and separately marks optional
  arguments. Unstated optional defaults remain null.
- Dimensional units absent from the point-value return table remain null.
- Example snippets are agent-authored fragments with listed preconditions, not
  claimed runnable standalone programs or copied official examples.
- Documentation review does not establish behaviour in every installed solver build.

## Native geometry checks

Sixteen frozen toy scenarios check the 33 geometry commands by reading the models
saved by real FEMM: object counts, coordinates, group assignments, arc angles and
property values. These are API checks; they do not solve fields or replicate papers.

```bash
python3 -m femm_brain2 check-geometry --output /tmp/femm-brain2-geometry-run
python3 -m femm_brain2 build --evidence /tmp/femm-brain2-geometry-run/report.json
python3 -m femm_brain2 get mi_addarc
```

Each run requires an empty output folder and preserves its models, log, protocol
and hash manifest. It uses the frozen image and exact reviewed Lua binding inside
a networkless, read-only container. The runner explicitly loads the reviewed Lua binding from the verified source
cache. The image digest identifies the FEMM program used; observations establish
only the tested behaviour in that container.

`build --evidence` rechecks artifact hashes and compares the saved models with the
frozen expectations before attaching results to command lookup. Source JSON cannot
assert an execution pass. A plain `build` imports documentation alone, so execution
status remains `not_tested`. Example fragments remain separately untested.

The geometry checks passed **16/16 native scenarios covering 33 commands**,
including a fresh run against the final 68-command catalogue.
[Saved models and reports](data/verification/README.md) also retain the original
14/16 run and explain its two checker corrections.

## Materials, boundaries and circuits

The 15 property entries cover adding, modifying and deleting definitions,
library material loading, and adding/clearing B-H curve points. Modification
commands include a `property_selectors` table with each numeric selector,
its meaning, type and units where established by the source.

Sixteen additional native scenarios passed. They read back all material
arguments, nonlinear curve samples, all eight boundary formats, complex point
and circuit values, property edits and deletions, geometry assignments, and
the same assignments after closing and reopening the saved model.
These checks establish property persistence and assignment in the pinned
container; they do not solve a field problem.

```bash
python3 -m femm_brain2 get mi_addmaterial
python3 -m femm_brain2 search "series circuit"
python3 -m femm_brain2 check-properties --output /tmp/femm-brain2-properties-new
```

Attach all three current reports to retain the measured outcomes for 68 commands:

```bash
python3 -m femm_brain2 build \
  --evidence femm_brain2/data/verification/geometry-20261125-contours/report.json \
  --evidence femm_brain2/data/verification/properties-20261125-contours/report.json \
  --evidence femm_brain2/data/verification/solve-20261125-contours/report.json
```

Each report is tied to the exact command batch. Historical reports remain
readable, but a changed catalogue requires a new native run before execution
claims can be attached. Repeat `--evidence` for distinct suites; overlapping
reports for the same command are rejected.

## Analytical field solution checks

Three planar magnetostatic cases use +0.2 T, -0.2 T with a potential offset,
and zero field with the same offset. An explicit linear material has relative
permeability 1 and zero source current. Prescribed outer potential `A=A0+Bx*y`
gives the analytical uniform field. Two equal-area pickup legs have +10/-10
turns and zero current, making flux linkage independent of the potential offset.

All three field cases passed. Checks cover point A/B/H, energy density, block
area/volume/energy, circuit current/voltage/flux linkage, and reloading the saved
solution. An independent reader checks nodal potentials in the real `.ans`
files and derives element B and total energy from those nodes. Inputs,
analytical formulas and tolerances were committed before solving.

```bash
python3 -m femm_brain2 check-solve --output /tmp/femm-brain2-solve-new
```

The separate `explicit_mesh_return` check now passes. The original FEMM build
created a mesh but omitted the native return. The isolated runtime
`femm-brain2:mesh-count-20261125` returns the actual mesh vertex count through
the C++ helper and binding. Its source patch, image IDs and binary fingerprints
are recorded in the build manifest copied into each run.

All three field controls now require the returned count to match both the
postprocessor count and the `.ans` mesh. Each returns 3,847 nodes, distinct from
the 7,440 elements. `check-solve` reports 4/4 passed with exit code 0; the complete
native check set is 36/36 passed. Physical inputs, analytical values and
tolerances are unchanged, and all non-count observations equal the earlier run.
Historical nil-return failures remain preserved. To rebuild the isolated image,
see [runtime/README.md](runtime/README.md).

These are analytical reference problems, not journal-paper replications.

## Contours and line integrals — final catalogue exercise

Four additional entries cover `mo_addcontour`, `mo_clearcontour`,
`mo_bendcontour` and `mo_lineintegral`. The last includes a searchable
`result_selectors` table: selectors 0–5, ordered return values, units and the
DC/AC return differences found in the pinned source.

The same three field solutions now each check 62 additional contour quantities:
forward/reverse flux, H.t, length/swept area, squared normal field, force and
torque on straight lines, cancellation around a closed rectangle, and the
subdivision length and endpoint integrals of two bent contours. Empty-contour
length and area must return zero. Every expected value and tolerance was
committed before running FEMM; the existing physical inputs and targets remain
unchanged. All new measurements passed in real FEMM.

For the positive field, FEMM returned flux -2.00000000038e-5 Wb (target -2e-5),
horizontal-line force -3.18309886065 N (target -3.18309886184), and torque
-0.0636619772055 N*m (target -0.0636619772368). Reversing the contour reverses
signed flux and stress contributions. The closed contour gives near-zero net
force and torque, as expected in a uniform field.

An open-line stress result is a surface contribution. Net force on a body
requires a suitable closed contour around that body. Native evidence here covers
planar DC only; AC, axisymmetric behaviour and bent-contour side placement are
not execution-verified.

```bash
python3 -m femm_brain2 search "contour flux torque"
python3 -m femm_brain2 get mo_lineintegral
```

This completed the catalogue foundation. The first hierarchical learning case
now links motor features to parameterized construction code and this API catalogue.


## Code map

| File | Responsibility |
|---|---|
| `source.py` | Fetch pinned sources, verify their hashes, and locate manual and Lua definitions |
| `data/commands.json` | Agent-authored, source-reviewed import records |
| `schema.sql` | SQLite catalogue and FTS5 search tables |
| `catalogue.py` | Validate/import records and perform read-only searches/lookups |
| `__main__.py` | JSON command-line interface |
| `geometry_check.py` | Isolated native runner and saved-model evidence checks |
| `property_check.py` | Read and compare saved material, boundary, point and circuit records |
| `solve_check.py` | Compare native observations and derive fields/energy directly from solver nodes |
| `tests/native_geometry/` | Frozen Lua fixtures and expected observations |
| `tests/native_properties/` | Frozen property fixtures and expected observations |
| `tests/native_solve/` | Frozen analytical model inputs, Lua driver and expected results |
| `runtime/` | Pinned container build and native mesh-count return backport |

There are no imports from the original `femm_brain` package. The new hierarchy,
construction procedures and first paper case are described in [SECOND_BRAIN.md](SECOND_BRAIN.md).
No background service is required.

## Check the implementation

Python 3.12+, SQLite with FTS5, and Poppler's `pdftotext` for source validation
are sufficient; no third-party Python packages are required.

```bash
python3 -m unittest discover -s femm_brain2/tests -v
python3 -m femm_brain2 validate
```

Catalogue software tests use synthetic documentation for search and import
checks. Evidence tests re-read saved real FEMM artifacts, including solver
nodes; they do not launch a fresh solver. `check-solve` launches native FEMM,
and `validate` checks the real pinned sources.
