# FEMM second brain

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
six executable studies. External vendor documentation retains its own publisher,
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
the construction checks and all six actual field solves in Docker with networking
disabled, and uploads their separate reports and native artifacts.
