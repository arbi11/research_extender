# FEMM second brain

For another agent continuing development, read the repository
[handover](https://github.com/arbi11/research_extender/blob/codex/femm-brain-solver-20261025/detailedREADME.md) for branch relationships, current evidence,
known failures, protected services and the next milestones.

A local, source-backed engineering knowledge graph with exactly three semantic
layers:

1. **Engineering workflow:** geometry, materials, excitation, boundaries,
   meshing/solving and result extraction.
2. **Components and methods:** stators, rotors, windings, air gaps, analysis
   choices and conditions for reusing them.
3. **Exact operations:** parameterized coordinates, nodes, segments, arcs,
   transformations and FEMM commands.

Official examples, source attachments and validation reports are evidence
attached to these layers. They are not another abstraction layer. The graph
preserves alternative methods and their applicability conditions; two induction
motor examples can share stator/winding procedures without sharing their rotor
current or analysis assumptions.

## Bundled source snapshot

The current bundle contains **46 curated study profiles**, **1,159 knowledge
nodes**, and **1,985 relationships**, including **654 documented API functions**
and **eight parameterized construction recipes**. These profiles describe
source-backed methods; they do not mean 46 numerical studies were reproduced.

The inventory retains **73 source records: 70 fetched and three unavailable**.
There are **315 attachment records: 311 downloaded and four failed**. Collection
failures remain visible in the manifest rather than disappearing from coverage
counts. The offline viewer includes **67 native model previews**: 59 magnetics,
six electrostatics and two heat-flow models. Use `python3 -m femm_brain stats`
and the source-coverage view to inspect the actual bundled snapshot.

## Run

From the repository root, with Python 3.12 or later:

```bash
python3 -m femm_brain build
python3 -m femm_brain validate
python3 -m femm_brain serve --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010>. The bundled corpus, graph, source references,
construction previews and downloads work locally without an AI credential or a
FEMM installation. The graph server uses the Python standard library; its graph
renderer is bundled locally. Opening an external source link requires network
access.

To keep a separate local index:

```bash
python3 -m femm_brain build --db /tmp/femm-brain.sqlite3
python3 -m femm_brain serve --port 8010 --db /tmp/femm-brain.sqlite3
```

This application is independent of the existing FEMM assistant. It does not
change that assistant's settings, jobs, development instance or deployment.

## Generate and solve from the knowledge base

Nine registered studies now generate fresh geometry, assign physics, mesh and
solve with native FEMM. Selection, parameter conversion, construction and
assessment use local code and cited knowledge records. **No AI credential or
external model call is used.** The eight teaching construction recipes and these
nine executable studies have separate catalogs and verification evidence.

| Case ID | Physical problem | Frozen comparison |
|---|---|---|
| `aircore-short` | Axisymmetric winding: 100 turns, radii 1–2 in, length 1 in | Published FEMM inductance 0.64707 mH |
| `aircore-long` | Axisymmetric winding: 100 turns, radii 1–1.25 in, length 4 in | Published FEMM inductance 0.23081 mH |
| `aircore-flat` | Axisymmetric winding: 100 turns, radii 0.25–2 in, length 0.25 in | Published FEMM inductance 0.40955 mH |
| `uniform-planar` | Magnetized IABC shells around a circular planar region | Analytic interior field magnitude 0.5 T |
| `uniform-axisymmetric` | Magnetized IABC shells around a spherical region | Analytic interior field magnitude 2/3 T |
| `parallel-wires` | Opposite 1 A currents in two circular wires, radius 0.2 m, centers 1 m apart | Analytic repulsive force 2×10⁻⁷ N/m |
| `gapped-ei-core` | Gapped EI iron core, signed 66-turn winding, linear µr=2500 | Published 1.73 mH, 0.000865042 J and 1.11347 N; force agrees, L/energy mismatch retained |
| `radial-bearing` | Eight-pole nonlinear M-19 magnetic bearing, four circuits and 16 winding sections | Published rounded upward rotor force ≈400 N |
| `ansys-vm270` | Axisymmetric SmCo magnet and 280-turn coil at 50 mA, with measured displacement | Measured ±0.0112815 N magnet/coil force, independent estimators and action-reaction check |

The five FEMM cases follow [AirCore](https://www.femm.info/wiki/AirCore) and
[UniformField](https://www.femm.info/wiki/UniformField). Uniform-field shell
permeabilities come from the cited source-model data; the circular construction
is recentered and scaled. Signed native field components are retained separately
from the source magnitude target.

The sixth case transfers the physical problem from [COMSOL's public parallel-wire
tutorial](https://doc.comsol.com/6.4/doc/com.comsol.help.models.acdc.parallel_wires/parallel_wires.html).
It replaces COMSOL's infinite-element exterior with a finite circular FEMM A=0
boundary. The baseline domain radius is 10 m; the recorded comparison uses 20 m
and two mesh scales. Changing domain radius also changes the global air mesh, so
this is **combined domain/mesh sensitivity**, not proof of infinite-domain or
monotone convergence. FEMM weighted stress and volumetric Lorentz force are
reported separately. The planar depth is exactly 1 m, so native force in N equals
force per length in N/m numerically. This is an independently implemented method
transfer, not execution or certification of a COMSOL model.

The [gapped EI source](https://www.femm.info/wiki/InductanceExample) and its
[force companion](https://www.femm.info/wiki/ACForceExample) supply three
independent published outputs. The 1% comparison remains **failed** for
inductance and energy: the supplied source model in the current engine gives
1.756068 mH and 0.000878034 J. A control loading the original FEM file as passive
data gives exactly the same values as our fresh reconstruction. Force is
1.108277 N, within 0.47% of 1.11347 N; linkage and all-domain energy agree
internally. This rules out reconstruction drift for the retained run, while the
historical engine/model/article cause remains unresolved. Targets and tolerances
were not changed to make the result pass.

The [nonlinear bearing](https://www.femm.info/wiki/RadialMagneticBearing) retains
its 13-point M-19 B-H curve and signed 80-turn winding sections. The native file
uses 1.6 inches of depth and unit lamination fill, while the prose gives nominal
4 cm and 0.98; the inherited difference is shown explicitly. Current edits do
not inherit the source's ≈400 N target or an assumed quadratic force law.

The [ANSYS VM270 / TEAM23 verification case](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM270.html)
is reconstructed from its public dimension/material table and
[input listing](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/ans_vm/Hlp_V_VM270TXT.html),
without executing APDL. Its SmCo magnet uses the input expression
`mu_r=Br/(mu0*Hc)` with Br=1.02 T and Hc=720000 A/m. The exact measured input
target is 0.0112815 N, distinct from the rounded table value 0.0113 N and ANSYS's
numerical result 0.0112 N. Both signed body forces and their balance are assessed.
Source infinite elements are replaced by a finite semicircle; default radius is
20 mm and a 40 mm control uses the same physical mesh lengths. Coarse and local-
only refinements failed the predeclared 2% criterion. A graded air mesh with
0.1 mm exterior spacing passed, preserving the physical inputs and original
criterion. The default solve has approximately 937,000 triangles and may take
20–30 seconds on this development laptop. No general convergence certificate is
claimed. Editing current or displacement makes the measured target unassessed.

`verify-solves` keeps a nonzero exit when a published target fails. CI then runs
`python3 -m femm_brain.benchmark_audit --summary RESULTS/summary.json --output RESULTS/audit.json`.
Only the reviewed EI discrepancy may receive `passed_with_reference_mismatch`,
requiring agreement with the independent source control, a passing force check,
physical cross-checks and every construction/mesh/solver check. Individual failed
targets remain failed. CI success therefore means the audit passed; it does not
mean every published output was reproduced. The discrepancy registry is
`data/verification/reference_discrepancies.json`.

Mesh previews show at most 100,000 actual native triangles to keep the browser
responsive. Full native connectivity remains in the downloadable `.node` and
`.ele` files. No model geometry or solver mesh is decimated.

List studies and prepare a reviewable plan without installing FEMM or Docker:

```bash
python3 -m femm_brain studies
python3 -m femm_brain plan 'aircore short with current 2 A'
python3 -m femm_brain plan --case parallel-wires --parameters '{"domain_radius":20}'
```

The local language parser selects a **known catalog profile**, handles named
parameters with explicit units, and reports ambiguity or unsupported details. It
does not generate arbitrary CAD, infer missing motor dimensions, interpret a
journal PDF or silently discard unfamiliar instructions. Use explicit case IDs
and the typed parameter editor when a request needs clarification. Numeric JSON
parameters use the units shown in `studies`.

For Docker solving, build the native engine from a full repository checkout:

```bash
docker build --target femm-build -t femm-brain-native:local -f femm_assistant/Dockerfile femm_assistant
python3 -m femm_brain solve aircore-short --output work/brain-runs/aircore-short
python3 -m femm_brain solve parallel-wires --parameters '{"mesh_scale":0.5}' --output work/brain-runs/parallel-wires-fine
python3 -m femm_brain verify-solves --output work/brain-runs/all-baselines
```

Docker image creation downloads/builds the pinned native engine. Subsequent
solves run with networking disabled, a read-only container filesystem, a
read-only source mount and a separate writable results directory. No provider
credentials are passed. Supply `--image <tag>` or `FEMM_BRAIN_IMAGE` to use another
prepared engine image. Each result directory must be new and empty; previous
evidence is never overwritten. The portable offline package can browse and plan
with Python alone; building the Docker image requires the full checkout or an
already prepared image.

Enable the web solver explicitly after preparing the image:

```bash
python3 -m femm_brain serve --port 8010 --enable-solver --jobs work/brain-runs/web
```

Open the executable-study panel to select a profile, edit parameters, inspect
the generated geometry and frozen target, then run it. Ordinary `serve` keeps
native execution disabled. The results include actual geometry, triangular mesh,
field samples, signed metrics, native FEMM files, a report and content hashes.
Geometry, mesh, solver completion and target assessment have separate statuses.

Published AirCore numbers apply to the original dimensions and turn count.
Changing those values produces `solved_unassessed` rather than inheriting a
baseline pass. Changing current retains the inductance comparison because this
model is linear. Uniform-field analytic references scale with excitation and
remain valid under geometric similarity; parallel-wire force is recomputed from
the declared current and spacing. The 1% FEMM and 5% transfer acceptance bands
are declared test tolerances, not accuracies certified by either source. A passing
comparison does not establish convergence or validate all documented studies.

## Reproduce the official examples

`reproduce` checks the brain against FEMM's own examples at up to four
independent levels, all sharing one typed measurement plan:

| Level | What is built | What it shows |
|---|---|---|
| `original` | The unmodified official model file, opened as data | The published number is reproducible on this engine at all |
| `rebuild` | A fresh model compiled from the brain's stored model record, then compared field by field with the saved native file | The brain's stored knowledge is complete for this model |
| `article` | A model built from parameters quoted from the article text, with official library entries and FEMM defaults where the article relies on them | The brain can construct the problem from its description, without the model file |
| `components` | A machine assembled from reusable machine components (slotted stator, star-of-slots winding, magnet rotor, library materials), with stated dimensions quoted and the rest identified from named model entities | The components rebuild a real machine: identical to the official model region by region, with native results that agree with the original's |

Published targets, quotes and the 1% tolerance policy are frozen in
`data/reproductions.json`; every quote is verbatim source text, cited by source
page or by artifact digest, and a test checks it against the restored cache.
Comparators (mesh-dependent values, other models or measurements) are reported
but never decide a pass. Diagnostic controls scale explicit mesh constraints,
override article or component parameters, toggle smart meshing, reorder the
model's entities or move its nodes by a sub-nanometre jitter, and report each
metric's change without affecting acceptance. Steps may modify properties or
rotate a group (for example a rotor) between solves.

```bash
python3 -m femm_brain restore-cache --for-reproductions   # official model files, digest-checked
python3 -m femm_brain reproduce --all --controls --output work/reproductions \
  --image research-extender-femm:dev-hierarchy-resumed-20261025
python3 -m femm_brain reproduction-report --results work/reproductions --html work/reproductions.html
```

At the recorded snapshot, 27 cases from 19 official examples ran: 22 pass every
attempted level, 3 fail their frozen targets and 2 have comparators only. All 26
rebuilds reproduce their original's native values exactly (1,064 raw values); 6
article builds pass, and the LRK motor passes from machine components (below).
The failures are kept with findings rather than relabeled:

- **EI inductance/energy** remains about 1.5% above the article. Mesh refinement
  moves it further away, so discretization does not explain it.
- **Coilgun ball force** changes from -2.5% to +8% of the published value as the
  mesh is refined; this small weighted-stress force is not mesh-converged.
- **Rotor-motion cogging torque** varies by tens of percent between adjacent
  6-degree samples and with smart meshing at the source's coarse mesh.

### LRK outrunner rebuilt from machine components

`lrkanalysis-flux-sweep` reproduces the LRK hobby motor's analysis script: with
the phase currents set to zero, the rotor group turns through one pole pitch in
61 positions and each position is meshed, solved and measured. The published
flux linkage of phases A, B and C at every position is an acceptance target
(183 targets, 1% relative or 1% of the amplitude, since the waveforms cross
zero), and the published cogging torque is a comparator (0.2 mN*m).

The `components` level rebuilds the motor from machine components alone: a
12-tooth outward slotted stator, a single-layer tooth-coil winding laid out by
the star of slots (12 slots, 7 pole pairs, reverse sequence) and 14 inward and
outward block magnets. Dimensions the design worksheet states are quoted from
it; the rest are identified from named entities of the official model. Before
solving, the component model must equal the official model region by region:
the same contours, the same used materials, circuits and boundaries, and the
same material, circuit, turns, magnetization, group and mesh size in every
region, wherever each model's labels sit (`compare_regions`). It does, with every
node within 2e-14 mm of its official coordinate.

- The original file, the stored record and the component build pass all 183
  flux targets; the worst error is 5% of the tolerance.
- The `phase_linkage` assessor's prediction from the construction places each
  phase's fundamental within 0.002 electrical degrees of the solved one, peaking
  A, C, B as the rotor turns counterclockwise; the published waveforms agree
  with the same prediction within 0.05 degrees.
- Native agreement with the original is bounded on the measurements behind
  acceptance targets and assessors: flux linkage agrees within 0.025% of the
  amplitude against a declared 1% bound.
- Cogging torque differs by up to 28% of its 1.4 mN*m peak and is reported, not
  bounded. The diagnosis: the mesher branches on last-bit coordinate differences
  in this rotationally symmetric geometry. Moving the official model's own nodes
  by at most 1e-12 mm changes its torque by 18 to 32% of the peak, reordering
  its entities by up to 13.5%, and shrinking its mesh by a quarter by up to 40%,
  while flux linkage moves at most 0.085%. Agreement was first declared over
  every raw value; after this first run showed the torque difference, the bound
  was narrowed to acceptance quantities, consistent with comparators never
  deciding a pass, and the narrowing is recorded here and in the case findings.
- With FEMM 5's library N40 magnet in place of the model's NdFeB 40 MGOe, flux
  linkage falls 0.86% and every target still passes: the published table does
  not separate the two grades.

Each case declares its coverage: **core** when it checks the example's main
published results, **partial** otherwise, with what was and was not reproduced.
An example's workflow node becomes `numerically_checked` only through a passing
core case and `partially_checked` through partial ones (14 core and 13 partial
cases; the transient-heat example, for instance, checks only its initial
state). Labels use only fresh evidence: each record carries a digest of the case
definition, so editing a case's inputs, measurements or targets makes its
evidence stale until it is run again. Article builds exist for six examples; the
others need dimensions that appear only in figures.

`reproduction-report` audits the record before accepting it. Every catalog case
and every applicable level must be present and fully built, solved and assessed;
the record must come from the current catalog; every rebuild must reproduce its
original; every component rebuild must match its official model region by region
and agree with it natively within its declared bound. A failed target is
accepted only if `reviewed_reproduction_failures.json` lists it with the same
case definition and frozen target, and the observed value stays within a 1e-6
relative regression bound of the reviewed value (the CI and development images
agree bit for bit). A reviewed failure that starts passing also fails the audit,
so the registry cannot go stale.

## Reusable components and new designs

Article builds contain no per-example code. Each is a data-only **assembly** of
23 reusable, executable **components** (`components.py`, knowledge records in
`data/components.json`): geometry and property operations (segment, polyline,
arc, circle, isolated node, block label, unmeshed region, material, official
library material, circuit, conductor, boundaries), layer-2 methods (open
boundary builder, mixed open boundary, convection, axisymmetric winding section,
bar magnet, air-gap element, Kelvin exterior with an applied field) and electric
machine components:

- **`slotted_stator`**: parallel-sided teeth with shoes repeated around the axis,
  pointing outward (outer rotor) or inward (inner rotor), with optional slot
  dividers for double layers and chords that close the slot openings. Teeth,
  tips and shoes that would overlap are refused.
- **`tooth_coil_winding`** on **`winding_layout`**: a star-of-slots layout for
  any odd number of phases, as single- or double-layer tooth coils or a
  distributed single layer, in either phase sequence. Each coil's phasor joins
  the phase whose signed axis owns its half-open belt; windings whose phases get
  unequal coil sides are refused. A positive coil carries +turns in the slot
  clockwise of its tooth, driving flux radially inward through it.
- **`surface_magnet_rotor`**: block or arc magnets with alternating polarity,
  radial (position-dependent `theta` direction) or parallel magnetization, an
  optional mesh size on the air-gap face, and seat arcs that close the rotor
  core between arc magnets.
- **`library_material`** draws on the whole FEMM 5 library (513 entries: 282
  magnetic with 132 B-H curves, 95 electrostatic, 136 thermal), with a folder to
  disambiguate repeated names and a model name of the assembly's choosing.
  Curves are validated: B-H curves need at least three nonnegative points with B
  and H strictly increasing; k(T) curves need increasing temperature and
  positive conductivity (two points suffice, as in FEMM's own heat library).

Every assembled model must also give each block label a closed region of its
own (`check_regions`), which catches, for instance, a double-layer winding drawn
into undivided slots. Methods bind to the existing documented method nodes; a
component is marked numerically checked only when a passing native assembly used
it (an article build, a new design, or a machine-component rebuild that matched
its official model region by region), and the node cites the cases.

Assemblies use parameter references (`$name`) and a whitelisted arithmetic
language (`=sqrt(a**2+b**2)/2`, with `pi`, `mu0`, `eps0`) that is parsed, never
evaluated as code. Before building, every parameter is checked against its
declared type (integers, booleans, finite numbers, points), bounds (lengths and
mesh sizes strictly positive) and applicability (physics, problem type). The
builder rejects nonfinite coordinates, zero-length or duplicate contours,
duplicate labels, turns without a nonzero integer winding, unphysical material
constants, and library entries from another physics or with properties the model
cannot represent. The six article assemblies reproduce the earlier per-example
builders exactly; frozen digests in the tests keep further changes deliberate.
Component evidence is bound to an implementation digest (the function, its
record, the shared builder and the components it uses), so a component's checked
status lapses when any of them changes.

`data/designs.json` holds new designs assembled only from components, in
combinations no official example uses, each checked against a closed-form result
with the 1% policy fixed before running:

| Design | Components exercised | Result |
|---|---|---|
| Coaxial capacitor | conductors on circles, unmeshed core | C within 0.11% |
| Twin-lead line in open space | open boundary builder in planar electrostatics | C' within 0.23% |
| Cylindrical heat shell | fixed-temperature conductors, library brick | Q within 0.017% |
| Dipole torque variant | Kelvin exterior, air-gap element, bar magnet at new size, field and direction | m x B within 0.1% with the predicted field |
| Kelvin field at 70 degrees | Kelvin exterior alone, prediction declared before running | uniform 0.800 T at -70.0 degrees, as predicted |
| Nonlinear M-19 ring | library B-H curve (33 points) around a 1000 A conductor; Ampere's law fixes H wherever the probes sit | B within 0.063% of the curve and H within 0.54% at four curve points from 1.56 to 1.84 T |
| Inner-rotor 12-slot 10-pole SPM | inward teeth with divided slots, double-layer star-of-slots winding, radial arc magnets with seat arcs, laminated and solid nonlinear M-19, 24 rotor positions | each phase's flux-linkage fundamental within 0.005 electrical degrees of the construction's prediction, peaking A, B, C; amplitudes equal within 4e-6 |

```bash
python3 -m femm_brain design --all --output work/designs --image research-extender-femm:dev-hierarchy-resumed-20261025
python3 -m femm_brain reproduction-report --catalog designs --results work/designs
```

The Kelvin exterior's applied field is predicted, not measured: FEMM's mesher
orients each periodic chain with its region on the left and reverses the second
(`tangle.cpp`, `buildPbcTwinFromCDT`), so the exterior rim angle maps to 180
degrees minus the domain angle, and the disk's magnetization at angle alpha
appears in the domain as a uniform field at -alpha with magnitude mu0*Hc/2.
Torque checks compare against m x B with this predicted field and separately
require the measured field to match it within 0.5 degrees and 1%.

Winding sign and phase linkage are checked the same way, against a prediction
from the construction (`phase_linkage` in `reference_models.py`). Each coil side
contributes its signed turns at the angle of its region's centroid and each
magnet its radial magnetization; magnetization continues radially across the air
gap, so a circuit whose conductors sum to S = sum(turns e^(i p theta)) links
Im(S e^(-i arg R)), where R is the rotor's fundamental, and turning the rotor
counterclockwise by alpha advances arg R by p alpha. Over one electrical period
(or half of one, extended by a declared half-wave symmetry) the assessor fits
each phase's fundamental and requires the predicted and measured phases to agree
within 1 electrical degree, the phases to peak in the declared counterclockwise
order, the winding to be balanced and the amplitudes to agree within 1%. Tests
show that swapping two phases or reversing one coil fails it.

The ring's mesh was refined after its first native run, with its targets left
as frozen: at 0.5 mm the 8 mm probe, in deep saturation (relative permeability
about 74, where a 0.27% error in B is a 2.3% error in H), read H 2.3% low. Ring
meshes of 0.5, 0.25 and 0.125 mm gave errors of 2.3%, 0.54% and 0.28% there, so
the error was discretization; the design records this history beside its mesh
parameter.

## Replicate machines from papers

Coding agents (Codex, Claude Code) replicate machines from papers, reports or
PDFs with the replicate-paper skill (`skills/replicate-paper/SKILL.md`), under
the repository's `AGENTS.md`. Each paper becomes one case,
`data/papers/<id>.json`, which a later harness can check stage by stage:

1. Intake: the PDF in its own cache directory (`~/.cache/femm-brain/papers/<sha256>/`),
   with extracted text, page images and OCR of every page, since tables are
   often images. Papers are never committed.
2. Extraction: every parameter is `quoted` (verbatim, with its location),
   `derived` (an expression, recomputed when a control varies an input),
   `figure` (scaled, with uncertainty), `assumed` (with a control that varies
   it) or `numerical` (a mesh choice, with a mesh control).
3. Gaps and contradictions are decided before solving and recorded.
4. Targets are frozen and committed before the first native solve, with
   tolerances set by how each number was obtained (analytic, 2D or 3D FEA,
   measured, digitized from a plot).
5. The machine is a data-only assembly of components; a missing capability
   becomes a generic component with tests and a synthetic design.
6. `papers --check` validates the case: derived values follow their
   expressions, assumptions have controls, quotes are found verbatim in the
   cached text or OCR, weak targets cannot decide acceptance, regions close.
   `papers --render` draws the model for comparison with the paper's figures.
7. `replicate` solves natively with controls; `reproduction-report --catalog
   papers` audits the evidence like the official catalog.
8. Findings, lessons (`skills/replicate-paper/lessons.md`) and a retrospective
   feed back into the skill and the brain.

`data/papers.json` lists ten selected papers: six for developing the skill and
four held out to measure it with the skill frozen, plus alternates.

```bash
python3 -m femm_brain papers                                  # index and case checks
python3 -m femm_brain papers --check elrefaie-2006-fscw-spm
python3 -m femm_brain papers --render elrefaie-2006-fscw-spm --output model.svg
python3 -m femm_brain replicate elrefaie-2006-fscw-spm --controls --output work/papers \
  --image research-extender-femm:dev-hierarchy-resumed-20261025
python3 -m femm_brain reproduction-report --catalog papers --results work/papers
```

The first development paper, El-Refaie et al. (IEEE TIA 2006), a 36-slot
30-pole concentrated-winding surface-PM machine, is a partial replication:

- Its RMS PM flux linkage is 34.30 mWb against the paper's analytic 34.8 mWb
  (−1.4%, tolerance 3%, set before solving from the paper's own FEA-versus-
  analysis spread), and is mesh-converged. Winding signs and phase order match
  the construction's prediction; the mutual inductance is about zero, as stated.
- Its 2D phase inductance is 1.20 mH against the paper's FEA 1.16 mH (+3.5%,
  tolerance 1%): a reviewed failure. A control with unit magnet permeability
  gives 1.1607 mH (+0.06%), and the paper's quoted 14 mm effective air gap also
  treats the magnets as air, so the paper's FEA most likely used μr = 1 where
  its table states 1.22. The case keeps the stated value.
- Table I's tooth width contradicts its slot widths. The slot widths are used,
  because they match the cross-section figure and reproduce the stated iron
  mass within 0.9%; controls show what each other reading would give.

## Explore

- Search for a machine, component or method and select a source example.
- Expand a workflow into its components and then its underlying operations.
- Switch between hierarchy, execution dependencies and reuse relationships.
- Inspect inputs, outputs, assumptions, source citations and verification status.
- Choose a construction recipe, change its parameters and regenerate the SVG.
- Select a line or arc to inspect the operation that constructs it.
- Inspect an available source-model preview to compare a recorded FEMM geometry.
- Download Python, Lua, FEMM or SVG construction files.

The parameterized recipes are reconstructed teaching procedures with explicit
parameter ranges and illustrative defaults. Their default dimensions must not
be mistaken for dimensions taken from an official motor example. A source-model
preview and a generalized construction recipe have separate identities and
evidence.

## Inspect and export

```bash
python3 -m femm_brain stats
python3 -m femm_brain search induction
python3 -m femm_brain export --output femm-brain.json
```

Versioned JSON is the canonical knowledge representation. SQLite is a rebuildable
index for local queries. Nodes record their layer, parameters and units, required
inputs, produced outputs, constraints, capabilities, source references and
verification evidence. Typed edges distinguish decomposition, implementation,
dependencies, alternatives, reuse and execution order.

The HTTP API includes:

| Endpoint | Purpose |
|---|---|
| `/api/graph` | Graph, examples, sources and coverage statistics |
| `/api/node/<id>` | Node details and its relationships |
| `/api/examples/<id>` | One source-backed example and its graph |
| `/api/sources` | Source inventory, attachments and collection coverage |
| `/api/search?q=...` | Search the local corpus |
| `/api/recipes` | Parameterized construction recipes |
| `/api/geometry?recipe=...` | Regenerated recipe SVG and ordered operations |
| `/api/geometry?artifact=...` | An available source-model geometry preview |
| `/api/download?recipe=...&format=python` | Portable construction export |
| `/api/simulations` | Executable studies and solver-enabled state |
| `/api/simulation-request?q=...` | Local catalog selection and explicit parameter parsing |
| `/api/simulation-plan?case=...` | Frozen generated study, source bindings and reference |
| `/api/simulation-download?case=...&format=lua` | Study plan, Lua or SVG export |
| `POST /api/simulations` | Queue a registered native study when solving is enabled |
| `/api/simulation-jobs/<id>` | Per-stage native results and target status |
| `/api/simulation-artifacts/<id>/<filename>` | Retained native artifacts and reports |

The geometry, study-plan and download endpoints accept a URL-encoded JSON
`parameters` object. Teaching recipe downloads support `python`, `lua`, `fem`
and `svg`; executable-study plan exports support `json`, `lua` and `svg`.

## Refresh source evidence

The bundled manifest records discovered official FEMM pages and attachments,
retrieval times, content hashes and collection failures. It distinguishes the
number of discovered sources from curated examples and verified procedures.
Refreshing sources requires network access:

```bash
python3 -m femm_brain ingest
python3 -m femm_brain build
python3 -m femm_brain validate
```

The collection cache is not bundled. To restore the recorded snapshot without
changing the manifest, `restore-cache` re-fetches each attachment and keeps it
only when its bytes match the recorded digest. Wiki pages embed a per-request
timestamp, so crawls also record each page's extracted-text digest and revision
time; change detection compares text whenever the curation baseline holds it.

```bash
python3 -m femm_brain restore-cache --report work/restore-report.json
```

Source collection is bounded and follows official FEMM links. Downloaded HTML,
PDFs and archives stay in the local collection cache rather than being bundled
as a copy of the website. Scripts are inspected as source data; collection never
executes downloaded scripts. Extracted attachment metadata and available model
previews retain their original source identity.

Refreshing an inventory does not automatically create a verified generalized
procedure. New examples and methods must be represented in the canonical JSON,
linked to evidence and checked before their status changes. Existing variants
remain separate when their physics, excitation or symmetry assumptions differ.
The curation manifest records the source hashes supporting the reviewed graph.
A changed source discovered during refresh is flagged for review; a new download
does not silently reapprove the old curation against changed evidence.

## Evidence and verification

- **Documented:** supported by a cited source; not a claim that it was run here.
- **Reconstructed:** an independently assembled procedure derived from source
  concepts or recorded geometry.
- **Geometry checked:** specified geometry/operation checks have run.
- **Numerically checked:** a particular native calculation and its stated
  comparison have run. This status must name the supporting report.

The graph may describe physics supported by FEMM but not yet implemented in the
assistant's current execution harness. Required capabilities remain explicit.
FEMM 4.2 examples and FEMM 5 operation contracts retain their version context.
A valid graph, rendered shape or completed mesh does not establish field accuracy,
mesh convergence or a universally applicable design rule.

The bundled [native geometry report](data/verification/native_geometry.json)
records actual FEMM 5 construction and save/readback checks for all eight
reconstructed recipe exports. Three geometry-only cases were also meshed:
circle (7,293 elements), rectangle (7,384 elements), and circular air gap
(8,769 elements). **No field solves were performed in this report.** It records
the pinned upstream source identity and implementation hashes. These construction
and mesh checks are separate from the documented study profiles and do not
establish their numerical reproduction or convergence.

The registered solve layer records additional field-solve evidence per run;
the historical construction-only report is not retroactively relabeled as a
solve report. Run `verify-solves` to generate current native evidence for all
nine executable studies. External vendor documentation retains its own publisher,
version, source hash and redistribution status in `data/external_sources.json`;
it does not become an official FEMM source or an open-licensed document merely
because a method transfer is indexed here.

## Tests

```bash
python3 -m unittest discover -s femm_brain/tests -v
node --check femm_brain/static/app.js
node --check femm_brain/static/simulation.js
```

The standalone browser test additionally requires Playwright and Chromium:

```bash
python3 -m playwright install chromium
python3 femm_brain/tests/browser_smoke.py
```

Graph/API/browser tests verify navigation, provenance, consolidation,
construction and exports. They do not imply a successful native FEMM solve for
every documented source example. Use the actual validation report and its
coverage counts when reporting what has been checked. The native CI job performs
the construction checks and all nine actual field solves in Docker with networking
disabled, then applies the explicit reference-discrepancy audit described above.
It uploads the separate reports and native artifacts; the EI inductance and
energy comparisons remain failed.
