# First paper in the hierarchical second brain

The first learning source is Lubin, Mezani and Rezzoug (2011),
*2-D Exact Analytical Model for Surface-Mounted Permanent-Magnet Motors With
Semi-Closed Slots*, DOI 10.1109/TMAG.2010.2095874.

This implementation covers Example 1's geometry and no-load back-EMF. It does
not claim the entire paper has been reproduced, and it is a learning example,
not a held-out benchmark. The second and subsequent papers have not been added.

## See the hierarchy

From `/home/akhan/code/research_extender`:

```bash
python3 -m femm_brain2 brain tree
python3 -m femm_brain2 brain search "semi closed slot"
python3 -m femm_brain2 brain show semi_closed_radial_slot
python3 -m femm_brain2 brain plan lubin-2011-spm-semi-closed-slots
```

The first path is:

```text
L1 Electric motor
  L2 Inner-rotor surface permanent-magnet motor
    L3 Stator
      L4 Semi-closed radial-sided slot
        L5 Construct radial slots and ideal stator boundaries
           -> recipes.py: semi_closed_radial_stator(...)
```

There are 17 records: one L1, one L2, three L3 components, six L4 features and
six L5 procedures. The stator, rotor and air-gap branches share applicable
procedures rather than copying their code. A procedure may therefore appear
under two features in the printed tree while remaining one stored record.

## Follow one extracted dimension

1. Table I, printed page 7 (PDF page 8), gives the stator bore radius as 2.8 cm.
   The coding agent identifies the row and records its page, short quote,
   unit and value as `r3_cm` in `data/knowledge.json`.
2. `r3_mm` records the expression `r3_cm*10`; the importer recomputes 28 mm
   rather than trusting a second handwritten number.
3. The paper's binding passes `$r3_mm` to the L5 procedure's `bore_radius`.
4. `semi_closed_radial_stator` uses that radius with the opening and slot
   dimensions to construct nodes, arcs, segments and labels.
5. Its L5 record links to the existing catalogue entries for `mi_addnode`,
   `mi_addarc`, `mi_addsegment`, selection and property assignment. The Lua
   generator checks every call's linked contract and argument count.
6. The native checker reads the actual FEMM file back and compares its
   coordinates, topology, groups, assignments and solver mesh with the frozen
   construction and dimension-derived areas.

The hierarchy captures the physical structure. `requires` links between L5
procedures capture construction order: slots must exist before winding labels;
rotor and bore surfaces must exist before the air-gap region is labelled.

## Who creates the records

The coding agent reads the paper, interprets its figures and writes the
structured candidate records and parameterized code. Python validates those
records, extracts explicitly located numerical data and builds the searchable
SQLite index. FEMM supplies execution evidence.

`paper_source.py` is not a universal paper-understanding agent. For this paper
the agent specified the figure, marker colour, plot bounds and axis calibration.
The extractor mechanically locates 20 red FEM markers in Fig. 10's vector data,
checks their angular spacing and excludes the duplicated period endpoint.
Its derived fundamental is approximately 0.805892 V per turn. The stored
extraction includes every reading and its locator.

The acceptance allowance was frozen before the native run: 1% numerical
agreement plus 0.008 V for twice the 0.004 V figure-reading uncertainty.
The model uses the stated 1500 r/min, two pole pairs, one-turn windings and
zero current. No fitted magnet strength, steel curve or effective length is used.

The original `femm_brain` records were inspected as a cross-check. The new code
imports none of its components or graph. The paper's ideal-iron assumption is
explicit: excluded iron regions give natural H_t=0 interfaces; this construction
must not be reused as a model of finite-permeability steel without another
procedure and validation.

## Records, SQLite and code

| File | Purpose |
| --- | --- |
| `data/knowledge.json` | Reviewed hierarchy, source references, paper parameters, bindings, frozen targets and controls |
| `knowledge.py` | Validate levels/links, evaluate restricted parameter expressions, retrieve and order the construction plan |
| `recipes.py` | Six reusable L5 construction functions, with no paper IDs or fixed paper dimensions |
| `paper_source.py` | Verify cached PDF fingerprints, parameter locators and the explicit figure extraction |
| `knowledge_compile.py` | Translate the retrieved construction into Lua, checking links to the API catalogue |
| `knowledge_run.py` | Execute real FEMM in isolated Docker and preserve/audit the run |
| `knowledge_check.py` | Independently read saved geometry and solver meshes, then score the frozen no-load scope |
| `schema.sql` | API tables plus hierarchy nodes, relationships, API links, paper cases and full-text search |

The versioned records and Python source rebuild the working SQLite database.
Edit these reviewable sources and rebuild; do not independently edit SQLite.
The PDF and full extracted paper text stay in the local source cache.

To rebuild the hierarchy while retaining the catalogue's existing API evidence:

```bash
python3 -m femm_brain2 build --knowledge \
  --evidence femm_brain2/data/verification/geometry-20261125-contours/report.json \
  --evidence femm_brain2/data/verification/properties-20261125-contours/report.json \
  --evidence femm_brain2/data/verification/solve-20261125-contours/report.json \
  --paper-evidence femm_brain2/data/verification/knowledge-lubin-2011-20261125-run3/report.json
```

The last argument attaches the first paper's audited `report.json`. Omit it
before a new paper run, then attach the new audited result afterward. A plain
catalogue build remains available without the hierarchy. Source records cannot
assert an execution pass; runtime evidence is stored separately. `brain search`
returns source records; `brain show` also displays the attached native evidence.

## Execute and inspect

```bash
python3 -m femm_brain2 brain validate
python3 -m femm_brain2 brain run lubin-2011-spm-semi-closed-slots \
  --output /tmp/lubin-first-paper-new-run
python3 -m femm_brain2 brain audit /tmp/lubin-first-paper-new-run/report.json
```

A run requires an empty directory. It first solves one position, then 36
positions across an electrical period, repeats at a finer mesh, and checks
zero magnet strength and an arbitrary vector-potential offset. Each sweep
restarts the FEMM program periodically inside its private container to bound
memory use. Network access is disabled, the container is read-only, and only
its run directory is mounted writable. No external LLM provider is called.

The evidence retains the exact knowledge snapshot, generated Lua, API-call
trace, source/runtime/code fingerprints, logs and all observation rows.
First and final FEMM models and solver files are retained for each sweep;
intermediate solver files are overwritten by FEMM, while their observations
remain recorded. Auditing regenerates the Lua from the graph, checks artifacts,
re-reads the native results and recomputes the assessment.

## Scope for the next paper

Retrieval currently constructs this recorded model deterministically. It does
not autonomously read an arbitrary new paper or choose unsupported geometry.
The next intake adds its source-backed features and parameter bindings, reuses
applicable L5 functions, and creates new parameterized procedures only where
the existing ones do not apply. Native verification remains specific to the
tested geometry and operating conditions.
