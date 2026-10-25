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

The current bundle contains **44 curated study profiles**, **1,142 knowledge
nodes**, and **1,893 relationships**, including **654 documented API functions**
and **eight parameterized construction recipes**. These profiles describe
source-backed methods; they do not mean 44 numerical studies were reproduced.

The inventory retains **70 source records: 67 fetched and three unavailable**.
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

The geometry and download endpoints accept a URL-encoded JSON `parameters`
object. Download formats are `python`, `lua`, `fem` and `svg`.

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
and mesh checks are separate from the 44 documented study profiles and do not
establish their numerical reproduction or convergence.

## Tests

```bash
python3 -m unittest discover -s femm_brain/tests -v
node --check femm_brain/static/app.js
```

The standalone browser test additionally requires Playwright and Chromium:

```bash
python3 -m playwright install chromium
python3 femm_brain/tests/browser_smoke.py
```

Graph/API/browser tests verify navigation, provenance, consolidation,
construction and exports. They do not imply a successful native FEMM solve for
every documented source example. Use the actual validation report and its
coverage counts when reporting what has been checked.
