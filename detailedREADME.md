# FEMM assistant: detailed development README

Updated 25 October 2026. Read this before continuing the work. This is a
development handover, not a claim that general paper replication is solved.

## Start here

The current reliable path is **second brain → registered construction method →
native FEMM geometry and physical assignments → mesh → solve → frozen numerical
comparison**. It runs locally with Docker and makes no external LLM calls.

Continue on `claude/femm-brain-official-examples`, the branch of
[PR #7](https://github.com/arbi11/research_extender/pull/7), stacked on
[PR #6](https://github.com/arbi11/research_extender/pull/6). It adds the
official-example reproduction ladder described below on top of PR #6's solver
work. Start by checking the current branch, worktree status and remote; preserve
any subsequent work by other agents.

For a new checkout only:

```bash
git clone --branch claude/femm-brain-official-examples \
  https://github.com/arbi11/research_extender.git research_extender
cd research_extender
```

For an existing checkout, inspect it before switching branches or pulling.

```bash
git status --short --branch
git log -5 --oneline
python3 -m femm_brain validate
python3 -m femm_brain stats
python3 -m femm_brain studies
```

Run commands from the repository root. Python 3.12 is the tested CI runtime.
Browsing, planning and host unit tests need only the standard library. Do not
install the repository's large thesis/LightRAG dependency set just to run
`femm_brain`. Browser verification additionally uses Node and Playwright; native
solving uses Docker.

## User's goal and working rules

- Develop on this laptop, use Docker for FEMM, preserve existing settings and
  services, and leave the old laptop unchanged until an explicitly authorized
  production deployment.
- Test useful milestones, commit them and push the development branch. A pushed
  PR is not a production deployment. Do not merge or deploy just because the
  code is ready.
- Rely heavily on the second brain. Keep external LLM use limited to tasks that
  need it, through OpenRouter. Preserve the user's provider/model settings.
  Never commit API keys, assistant access tokens, `.env` files or editor swaps.
- Jev is for function routing and diagnosing an actual failed call. Do not
  restore an LLM confidence threshold that blocks otherwise valid geometry.
  Optional visual comparisons are advisory and must identify the actual image
  supplied to the model.
- Keep geometry/physical assignments, mesh, solver and target checks separate.
  A solved mesh is not proof that the requested physical problem was reproduced.
- The eventual inputs are natural language, a paper, or both. Preserve both
  input paths; the current second-brain parser only selects known study profiles.
- Freeze physical inputs, targets and tolerances before a benchmark. Preserve
  failed attempts. Do not change physical inputs or acceptance criteria to fit
  an expected result. For a blind evaluation, freeze the code before exposure
  and finish the locked cohort before making product changes.

The latest user-approved direction is summarized below. The current request was
to write and push this handover; no new physics feature was implemented as part
of that request.

## Repository and branch relationships

Repository: <https://github.com/arbi11/research_extender>.

| Work | Branch | Last implementation milestone | PR |
| --- | --- | --- | --- |
| Assistant, hierarchical paper intake and recovery foundation | `codex/femm5-linux-assistant-20261022` | `cbd58722161578b4f44d7feabca1f6db60a3247a` | [#3](https://github.com/arbi11/research_extender/pull/3) |
| Original three-layer second brain | `codex/femm-second-brain-20261025` | `e94244902d85781e86cf185cbc643397e965a26c` | [#5](https://github.com/arbi11/research_extender/pull/5) |
| Second-brain native solving and nine executable studies | `codex/femm-brain-solver-20261025` | `c46abba83b95882b4eff5df20a1c15d712265583` | [#6](https://github.com/arbi11/research_extender/pull/6) |
| Official-example reproduction ladder, source-cache restore and maintenance fixes | `claude/femm-brain-official-examples` | see the PR head | [#7](https://github.com/arbi11/research_extender/pull/7) |

These are stacked: PR #5's commit descends from the recorded PR #3 head,
PR #6 is based on PR #5, and PR #7 on PR #6. The current solver branch contains that earlier source
history. PR #3 and PR #6 remain draft development PRs at handover. Do not
independently cherry-pick the entire stacks or reset an existing checkout.

## What works, and what does not

The bundled graph has three semantic layers: **workflow → components/methods →
exact operations**. At the tested snapshot it contains 46 documented studies,
1,159 nodes, 1,985 relationships, 73 source records, 67 native model previews and
eight editable teaching construction recipes. These counts do not mean all
documented studies were numerically reproduced. The teaching recipe catalog is
separate from the nine executable studies.

The browser can select a registered study, edit typed parameters, display its
geometry, run Docker FEMM, display actual native mesh and sampled field vectors,
and download plans, scripts, native files, solutions and audit reports. Unknown
or ambiguous language remains unresolved rather than silently dropping details.
Changed physical parameters cannot inherit an inapplicable published target;
such a solve can be `solved_unassessed`.

The second brain does **not** yet provide arbitrary CAD synthesis, automatic
paper-to-geometry replication, motor assembly beyond slotted stators with
tooth-coil or single-layer distributed windings and surface-magnet rotors,
AC-loss execution or automatic controlled recovery. The older assistant has experimental hierarchical
paper intake and recovery code, but that is a separate pipeline and its paper
replication has not succeeded. Do not present those capabilities as integrated
into the second brain merely because their source is in the same repository.

### Executable numerical baseline

The following are retained baseline results, not promises for every parameter
variation or future engine build. Values and reference applicability live in the
versioned data and generated plans.

| Case ID | Retained comparison | State |
| --- | --- | --- |
| `aircore-short` | Inductance versus FEMM 0.64707 mH; 0.0188% error | Passed |
| `aircore-long` | Inductance versus FEMM 0.23081 mH; 0.0828% error | Passed |
| `aircore-flat` | Inductance versus FEMM 0.40955 mH; 0.0466% error | Passed |
| `uniform-planar` | Analytic field magnitude 0.5 T; 0.0041% error | Passed |
| `uniform-axisymmetric` | Analytic field magnitude 2/3 T; 0.0057% error | Passed |
| `parallel-wires` | Repulsive force per length 2e-7 N/m; 3.0317% error | Passed within declared 5% |
| `gapped-ei-core` | Force 1.108276659 N versus 1.11347 N; 0.4664% error. Inductance 1.756068 mH versus 1.73 mH and energy 0.000878034 J versus 0.000865042 J | Force passed; L/energy FAILED original 1% |
| `radial-bearing` | Nonlinear eight-pole M-19 bearing: 398.319641 N versus rounded 400 N; 0.4201% error | Passed within declared 5% |
| `ansys-vm270` | Magnet stress force 0.011324689 N versus measured 0.0112815 N; 0.3828% error. Independent coil Lorentz force magnitude error 0.6282% | Passed original 2%, with signed force/balance checks |

Important qualifications:

- **EI mismatch remains unresolved.** Independently opening and solving the
  original passive source FEM file gives exactly the same L, energy and force as
  our reconstruction in the retained engine. This rules out reconstruction
  drift for that run; it does not identify a historical engine/model/article
  cause. Do not loosen the 1% comparison. See
  [the discrepancy registry](femm_brain/data/verification/reference_discrepancies.json).
- The bearing retains its source B-H data and signed winding circuits. Native
  source depth/lamination fill differ from nominal prose values; the discrepancy
  is disclosed. Do not replace it silently. Half-current, reversal, zero-current
  and mesh controls are retained; do not assume nonlinear force scales as I².
- VM270 uses the public ANSYS input expression `mu_r = Br/(mu0*Hc)` rather than
  the slightly inconsistent rounded permeability in the table. Infinite exterior
  elements are replaced with a finite A=0 semicircle. A fixed-physical-mesh
  20-to-40 mm domain check and independent force estimators were run. Twenty
  development attempts, including coarse/local mesh failures, were retained.
  The successful baseline has about 937,000 triangles and took 20–30 seconds
  on this laptop. This is not a general convergence certificate.
- The COMSOL wire transfer also substitutes a finite A=0 exterior. Its earlier
  domain sweep changed global air mesh too; call it combined sensitivity
  evidence, not proof of convergence.
- Mesh previews show at most 100,000 actual triangles. Full `.node`, `.ele` and
  solution files remain available; the solver mesh itself is not decimated.

### Official-example reproduction (PR #7)

`python3 -m femm_brain reproduce` checks the brain against FEMM's official
examples at four levels sharing one measurement plan: **original** (the
unmodified official model file, opened as data), **rebuild** (compiled from the
brain's stored model record and compared field by field with the saved native
file), **article** (built only from parameters quoted from the article text)
and **components** (assembled from reusable machine components, then required
to equal the official model region by region).
Targets, verbatim quotes and the 1% policy are frozen in
`femm_brain/data/reproductions.json`; the evidence record is
`femm_brain/data/verification/official_reproductions.json`. See the
[second-brain README](femm_brain/README.md#reproduce-the-official-examples).

- 27 cases from 19 examples: 22 pass every attempted level, 3 fail, 2 have
  comparators only. All 26 rebuilds reproduce their original exactly. Cases
  declare core (14) or partial (13) coverage; labels use only fresh evidence.
- The audit requires every case and level, the current catalog, exact rebuild
  agreement, component rebuilds identical to their official model region by
  region and in native agreement, and reviewed failures bound to their case
  definition and target within a 1e-6 regression bound.
- Article-only builds pass for the magnetics tutorial, square capacitor, two
  spheres, tube eddy currents, torque benchmark and heat-flow chimney.
- The torque benchmark (milestone 1 below) passes the whole 0–90° curve against
  Table 1 and m×B at all three levels. Domain-radius, gap-radius and mesh
  controls change torque by at most 0.5%.
- AC references (milestone 5): the tube matches its analytic eddy-current
  density; TEAM 28 matches published losses within 0.35% and force within 0.5%
  under both open-boundary methods, converging toward the finest-mesh value.
- Kept failures with findings: EI inductance (mesh refinement moves it away from
  1.73 mH), coilgun ball force (not mesh-converged), rotor-motion cogging torque
  (very mesh- and angle-sensitive at the source's ~1,400-node mesh).
- Article builds are data-only assemblies of reusable components (milestone 2
  below); seven new designs built only from components pass closed-form or
  construction-predicted checks, including a field-direction prediction declared
  before running, a nonlinear library B-H curve read back natively, and the
  phase linkage of an inner-rotor machine no example contains.
- The LRK outrunner is rebuilt from machine components (slotted stator,
  star-of-slots tooth-coil winding, block-magnet rotor) identically to its
  official model region by region and passes all 183 published flux-linkage
  targets over 61 rotor positions, as do the original and rebuild. Its cogging
  torque is a comparator: sub-nanometre coordinate changes alone move it by up
  to 32% of its peak at this mesh, so component agreement is bounded on flux
  linkage (0.025% observed against 1%). See the
  [LRK section](femm_brain/README.md#lrk-outrunner-rebuilt-from-machine-components).

Maintenance in the same PR: `restore-cache` restores the source snapshot by
digest (the original cache was not on disk); scalar-physics model records keep
conductors and k(T) curves; source-change detection compares page text because
wiki markup carries a per-request timestamp; reproduced examples mark only their
own workflow node as numerically checked. CI restores the referenced model files
by digest and runs the full catalog in the engine container.

## Resume locally without disturbing existing services

On the original development laptop, the active checkout is:

```text
/home/akhan/Documents/Codex/2026-10-24/develop-the-femm-assistant-on-this/work/femm-brain-dev
```

The older assistant checkout is the sibling `work/femm-dev`. The protected
user clone is `/home/akhan/femm-dev`; its assistant `.env`, editor swap and
`deploy/compose.yaml` were checked unchanged. Read credential files only when
needed for an explicitly intended provider operation; never print their values.

| Port | Existing service at handover | Action |
| --- | --- | --- |
| 8000 | Production Docker container `deploy-assistant-1` | Leave unchanged |
| 8001 | Earlier development Docker container `femm-assistant-dev-20261024` | Leave unchanged |
| 8010 | Port used by the second-brain server in the previous session; not listening when this handover was written | Start it only after checking the port, or use an isolated free port |

No old-laptop access or production deployment was performed. Preserve settings,
container identities/images/start times and existing job evidence. Do not stop a
service just to free a familiar port. For a separate agent session, use a free
port and separate index/jobs, for example:

```bash
python3 -m femm_brain serve --host 127.0.0.1 --port 8012 \
  --db work/brain-agent/index.sqlite3
```

This browsing-only server requires neither Docker nor a key. For solving, the
installed image used by the retained tests was:

```text
Tag: research-extender-femm:dev-hierarchy-resumed-20261025
ID:  sha256:85f07dce379bff58dc539f8ed49ffba53a94eba332fbf81d4c9fd6c3df94725d
Native engine: FEMM headless engine v5.0 23Oct2026
```

The tag is laptop-local; do not assume it exists on another machine. Check it
with `docker image inspect`. The harness pins the resolved image ID in receipts.
If a new image is needed, build under a new development tag, without overwriting
the production/deployed image tags:

```bash
docker build --target femm-build -t femm-brain-native:handover \
  -f femm_assistant/Dockerfile femm_assistant
```

Example using the already installed laptop image:

```bash
export FEMM_BRAIN_IMAGE=research-extender-femm:dev-hierarchy-resumed-20261025
python3 -m femm_brain serve --host 127.0.0.1 --port 8012 \
  --db work/brain-agent/index.sqlite3 --enable-solver \
  --image "$FEMM_BRAIN_IMAGE" --jobs work/brain-agent/jobs
```

`serve` disables solving unless `--enable-solver` is supplied. The default image
name is `femm-brain-native:local`, not the installed snapshot tag above; explicitly
pass `--image` or set `FEMM_BRAIN_IMAGE`. Solves run network-disabled with a
read-only container/source mount and a fresh writable result mount. Provider
credentials are not passed. The web solver is loopback-only.

Web job files persist on disk, but the current `SimulationJobs` implementation
does not reload historical jobs into its in-memory index after a server restart.
The native evidence remains available on disk; an empty jobs UI after restart
does not mean earlier results were deleted.

## Useful commands and validation

Prepare plans without Docker or provider calls:

```bash
python3 -m femm_brain plan 'nonlinear radial bearing current_scale=0.5'
python3 -m femm_brain plan 'ANSYS VM270 force at 50 mA'
python3 -m femm_brain plan --case gapped-ei-core --parameters '{"mesh_scale":0.5}'
```

For a fresh native run, choose an unused output directory. The harness refuses
to overwrite prior output or its sibling execution receipt:

```bash
mkdir -p work
RUN_ROOT="$(mktemp -d "$PWD/work/brain-handover.XXXXXX")"
python3 -m femm_brain solve radial-bearing \
  --image "$FEMM_BRAIN_IMAGE" --output "$RUN_ROOT/bearing"
python3 -m femm_brain verify-solves \
  --image "$FEMM_BRAIN_IMAGE" --output "$RUN_ROOT/baselines"
```

**The strict nine-case `verify-solves` command is expected to exit 1** while EI
L/energy fail. Do not hide that exit or relabel the individual targets. Run the
separate mandatory audit on its summary:

```bash
python3 -m femm_brain.benchmark_audit \
  --summary "$RUN_ROOT/baselines/summary.json" \
  --output "$RUN_ROOT/baselines/audit.json"
```

Expected audit state for the retained baseline: `passed_with_reference_mismatch`,
with `all_published_targets_passed: false`. The narrow exception requires current
plan/source/parameter bindings, the reviewed original-source regression, passing
EI force/internal consistency and all native checks. Any new failure must fail
the audit. A green CI job does not mean all published targets passed.

Routine host checks from the repository root:

```bash
python3 -m femm_brain validate
python3 -m unittest discover -s femm_brain/tests -v
node --check femm_brain/static/app.js
node --check femm_brain/static/simulation.js
python3 femm_brain/tests/browser_smoke.py
```

The browser smoke test needs Playwright and Chromium installed in the selected
Python environment. At `c46abba`, 145 host tests ran: 143 passed and two opt-in
native integration tests skipped. Separate fresh Docker runs covered all nine
studies; three real browser Docker jobs covered EI, bearing and VM270, including
downloaded artifacts, loaded images and displayed/submitted/executed plan-hash
agreement. Do not treat host skips as proof that native checks ran.

The implementation passed both jobs in
[CI run 40169159790](https://github.com/arbi11/research_extender/actions/runs/40169159790).
Its downloaded native artifact digest and all nine plan hashes were independently
checked against local evidence. See
[the CI workflow](.github/workflows/femm-brain.yml) for the full native recipe
checks and mandatory discrepancy audit. Documentation-only changes may not
trigger this path-filtered workflow; the link above identifies the tested code
snapshot, not a promise of a run for every subsequent documentation commit.

## Code and evidence map

| Location | Responsibility |
| --- | --- |
| [`femm_brain/core.py`](femm_brain/core.py), `store.py` | Canonical graph loading/validation and rebuildable SQLite index |
| [`femm_brain/data/`](femm_brain/data/) | Versioned source, workflow, method, operation and study records; JSON is canonical |
| `data/simulations.json` | Executable study schemas, defaults, references and applicability |
| `data/benchmark_methods.json`, `transfer_cases.json`, `external_sources.json` | Source-backed benchmark methods and vendor transfers |
| `data/models/`, `model_manifest.json` | Extracted passive source-model data and provenance |
| [`geometry.py`](femm_brain/geometry.py), `catalog.py` | Typed construction/export, native-file parsing, teaching recipes and previews |
| [`planner.py`](femm_brain/planner.py) | Local known-study selection, explicit parameter/unit parsing and unsupported-request guards |
| [`simulation.py`](femm_brain/simulation.py) | Frozen plans/scripts, native execution/physical checks, mesh/field rendering and target assessments |
| [`source_recipe.py`](femm_brain/source_recipe.py) | Strict reconstruction from pinned passive native data and independent assignment verification |
| [`vm270.py`](femm_brain/vm270.py) | VM270 geometry/materials/mesh, independent saved-model verifier and signed force checks |
| [`execution.py`](femm_brain/execution.py) | Isolated Docker jobs, image pinning, queue, durable receipts, cleanup and plan-hash checks |
| [`benchmark_audit.py`](femm_brain/benchmark_audit.py) | Strict baseline audit and the narrowly scoped EI discrepancy |
| [`native_check.py`](femm_brain/native_check.py) | Actual native construction/mesh checks for the teaching recipes |
| [`server.py`](femm_brain/server.py), [`static/`](femm_brain/static/) | Local API and browser UI |
| [`tests/`](femm_brain/tests/) | Graph, planning, physics verification, audit, API, execution and browser checks |
| [`femm_assistant/`](femm_assistant/) | Earlier natural-language/paper assistant; read its own README and retained experiment reports before modifying |

Committed portable evidence is under
[`femm_brain/data/verification/`](femm_brain/data/verification/):

- `vendor_benchmarks.json`: latest nine-case reports, controls and test
  metadata at the tested implementation snapshot.
- `reference_discrepancies.json`: EI original-source control and exact allowed
  reference failures; this does not change the failed published targets.
- `native_simulations.json` and `engineering_benchmarks.json`: historical
  six-case and eight-case receipts. Do not call their old plan hashes current.

Additional original-laptop evidence is outside Git, relative to
`/home/akhan/Documents/Codex/2026-10-24/develop-the-femm-assistant-on-this`:

- `work/next-benchmark-validation/final-native/`: final nine fresh native runs,
  complete baseline files, `summary.json` and `audit.json`.
- `work/next-benchmark-validation/browser-final/`: three final browser jobs.
- `work/next-benchmark-validation/source-control/`: original EI native control.
- `work/next-vendor-source-research/`: vendor source receipts, VM270 protocol,
  twenty-attempt audit, mesh/domain controls and source shortlist.
- `outputs/femm-engineering-benchmark-audit.html`: user-facing report, linked
  native evidence ZIP and offline application package.
- `outputs/femm-engineering-ci.json`, `femm-engineering-preservation.json` and
  `femm-engineering-deliverables.json`: CI, preservation and deliverable hashes.

These laptop files are optional supplemental evidence, not dependencies of a
fresh clone. The CI run also publishes native evidence and an offline package;
artifacts can expire. Do not commit credential-bearing job data, bulk caches,
downloaded vendor programs, or third-party PDFs just to make a handover complete.
Inspect downloaded scripts as source; do not execute them.

## Earlier LLM/paper experiment: retain the negative result

PR #3 has hierarchical strict contracts for machine/case inventory, source facts,
component intent and typed features. It also has source/geometry/mesh image
comparison and a bounded recovery foundation with revisions and frozen targets.
Neither model confidence nor image review is a geometry approval gate.

The last frozen three-model experiment used Haiku, DeepSeek and GLM through
OpenRouter. Only one executable natural-language wire case completed faithfully;
no paper case reached an accepted geometry and numerical reproduction. The common
paper stop was a `baseline` parameter/case handoff mismatch, alongside other
provenance, scope, reference and provider-output failures. Do not reduce the
diagnosis to a weak extraction model or quietly patch a held-out case mid-test.
Read the committed
[resumed experiment report](femm_assistant/evaluation/results/2026-10-25-resumed-hierarchy/README.md)
and [assistant documentation](femm_assistant/README.md) before attempting to
reuse that pipeline. Controlled recovery has not demonstrated comparative
resource savings, and general dependency-aware backtracking is not complete.

## Next milestones agreed with the user

1. **Torque over rotor position.** Done in PR #7 for the documented
   [FEMM torque benchmark](https://www.femm.info/wiki/TorqueBenchmark), with
   domain, gap and mesh controls. The rotor-motion example's cogging torque
   remains a recorded failure; a converged-mesh study of that machine is the open
   follow-up.
2. **Reusable construction methods.** 23 executable components with knowledge
   records in `femm_brain/data/components.json`, bound to the documented method
   nodes and marked checked only by passing assemblies. The machine set added
   in PR #7: a slotted stator (outward or inward parallel-sided teeth with
   shoes, optional slot dividers), a star-of-slots winding for any odd phase
   count (single- or double-layer tooth coils, distributed single layer;
   unbalanced windings refused), a surface-magnet rotor (block or arc magnets,
   radial or parallel, seat arcs), isolated nodes, and the full FEMM 5 material
   library (513 entries, 132 B-H curves). The LRK outrunner is rebuilt from
   them identically to its official model, region by region. Still missing:
   interior (buried) magnets, distributed double-layer and fractional-slot
   distributed windings, rotor bars, and result-extraction methods as
   graph-bound components (measurements are typed in the catalog).
3. **Natural language through the second brain.** Retrieve methods and fill typed
   parameters. Use OpenRouter only when interpretation or missing details need
   it. Unsupported combinations should state what is missing. Preserve the
   distinction between a documented source, illustrative recipe and executed
   validated study.
4. **Controlled recovery in this pipeline.** Track each feature/operation, inputs,
   dependencies and checkpoints. Diagnose an actual failed geometry/mesh/solve
   operation, roll back and retain an evidence-supported repair branch. Prefer
   mesh-only repair initially. A target mismatch requires investigation, not
   optimization of physical inputs to match the answer. Keep every failed run
   and the original target. Measure savings rather than assuming them.
5. **AC and eddy-current losses.** Documented references now pass (tube eddy
   currents, TEAM 28 losses, induction-motor complex inductance at one frequency).
   Frequency sweeps, previous-solution chains and moving geometry are not yet
   expressible as reproduction steps; they block the remaining AC examples.
6. **Blind documentation transfer, then journal papers.** Select an unseen,
   sufficiently specified vendor/tutorial problem, freeze implementation and
   target criteria, and audit extraction → construction → mesh → solve → target.
   Complete the frozen experiment before changing the implementation. Return
   to journal replication after this transfer is reliable; papers often omit
   essential geometry and operating conditions.

Keep the EI mismatch open in parallel. Additional candidates already researched
include [QuickField's nonlinear solenoid](https://quickfield.info/advanced/magn2.htm),
[COMSOL's axial magnetic bearing](https://doc.comsol.com/6.3/doc/com.comsol.help.models.acdc.axial_magnetic_bearing/axial_magnetic_bearing.html),
and [TEAM induction-motor examples](https://quickfield.com/advanced/induction_motor_analysis.htm).
These are candidates, not completed reproductions; some still need missing
clearances, figure digitization or AC/motion capabilities. Verify complete source
inputs and access before selecting a case.

For each milestone, record the source and assumptions, declare checks before
solving, preserve failures, review native assignments independently, run relevant
tests, commit and push. Update this handover when the verified state changes.
