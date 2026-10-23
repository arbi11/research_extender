# FEMM assistant implementation pipeline

This is the durable implementation plan. The general-geometry staged workflow is
implemented; release verification and remaining extensions are tracked separately.
Current CI results must be checked before deployment.

## Current sequence

`prompt → semantic preflight → typed plan → geometry/material/BC checks`

`→ semantic fidelity → preview → mesh only → measured mesh report → solve → export`

Every transition uses an explicit stage result. Semantic probabilities cannot
override a failed geometry check, failed mesh, or failed solve.

| Stage | Responsibility | Status |
|---|---|---|
| Request | Capture intent and required physical inputs; clarify omissions | Implemented |
| Semantic preflight | Jev checks supported intent, operation and input sufficiency | Implemented; mandatory for general AI planning |
| Typed plan | Draft or modify constrained model data, never executable AI code | Implemented |
| Plan checks | Validate contours, labels, references, physics and mesh budget | Implemented |
| Semantic fidelity | Jev checks whether the valid draft follows the request | Implemented; mandatory for general AI planning |
| Preview | Show proposed geometry and assumptions for reader review | Implemented |
| Mesh only | Run native meshing without solving the magnetic field | Implemented |
| Mesh report | Inspect actual triangles, counts, shape, domain and label coverage | Implemented; not an accuracy certificate |
| Solve | Require a passed mesh for the exact current spec; run native FEMM | Implemented |
| Extract | Read computed quantities, probes, field samples and eligible forces | Implemented |
| Export | Save editable Python CAD, native Lua, spec, artifacts and reports | Implemented |

## Supported baseline

General plans use circles, rectangles and simple polygons with separate or nested
regions, linear isotropic materials, DC circuits, signed turns, labeled regions,
an enclosing A=0 domain and bounded meshing. Physics is 2D planar DC magnetostatics
in millimeters with explicit out-of-plane depth. Contours cannot touch or cross.
The separate C-core numeric example retains its original solve workflow.

The mesh report uses the native mesh, not a geometry-preview estimate. It checks
triangle validity, resource limits, domain coverage and compiler block labels,
and reports minimum angle and aspect ratio. Slender-element warnings and a passed
structural gate do not establish mesh-refinement or outer-boundary convergence.
Editing the spec invalidates the previous mesh approval in both API and UI.
The native analysis remeshes the identical validated spec; it does not consume the
diagnostic `.node`/`.ele` files from the earlier mesh job.

## Jev's bounded role

OpenRouter's [Jev interface](https://openrouter.ai/labs/jev), model
`typesafe/jev-1.14`, uses `/api/alpha/decisions` for bounded semantic questions.
The draft model defaults to `openai/gpt-4.1-mini`. Only `OPENROUTER_API_KEY` is
needed for these defaults; blank model overrides fall back to them. Custom
decision settings need exported variables or explicit Compose forwarding.

Preflight and fidelity probabilities are recorded with the proposal. They concern
intent, not physical confidence, mesh quality or numerical accuracy. The default
`0.90` threshold is provisional until calibrated against labeled requests and
FEMM-backed cases. Low confidence or contradictory intent leads to clarification.
At most two Jev calls and two draft calls are allowed within 90 seconds overall;
each draft is limited to 6000 generated tokens, with one validation-guided repair.

Arithmetic, unit checks, topology, label placement, material/circuit references,
resource limits and stage transitions remain deterministic. Jev never generates
executable code or changes a failed numerical stage into a success.

## Portable model and native mesh export

Downloads include `model.py` with editable `MODEL` data and named Python CAD
stages, `model.lua` as the native bridge, and `run.py` as a compatibility launcher.
`python3 model.py` solves; `--mesh-only` meshes; `--write-only` regenerates source.
Rerunning needs FEMM 5, without an AI key, repository or Python dependencies.

`deploy/patch_femm_mesh.py` adds an opt-in `FEMM_MESH_EXPORT=1` native callback
that writes the actual in-memory mesh. It does not change numerical algorithms.
The image retains the original upstream archive and patcher. Detailed mesh
diagnostics require this patch; portable model scripts can run with stock FEMM 5.
Existing deployments need one FEMM recompilation. An old laptop can instead load
the workflow's tested Linux `amd64` image once a successful run publishes its
three-day artifact, or an image built on a powerful machine. Preserve `.env` and
local Compose edits when updating.

## Verification and remaining milestones

1. **Release verification.** Run unit/API checks, browser stage-gate checks and
   native live suites before recording a passing milestone. Live checks cover
   mesh-only output, Python CAD reruns, wire/Ampere comparison, refinement,
   current/energy scaling and C-core force/gap regressions. Browser fixtures test
   UI behavior, not real provider decisions or field accuracy.
2. **Calibration and independent references.** Evaluate semantic thresholds on
   labeled supported, ambiguous and unsupported requests; expand reference cases
   and boundary-distance/convergence studies before claiming quantitative accuracy.
   The [evaluation guide](evaluation/README.md) now provides 28 labeled public
   development cases, live/replay scoring and a real OpenRouter-to-native-CAD
   acceptance test. A 2026-10-24 live run matched 27/28 gate labels at the unchanged
   0.90 threshold: zero false accepts, one conservative rejection for an explicit
   assumptions request, and zero provider errors. The four CAD acceptance checks
   passed, including create/modify, actual mesh/solve/export, incomplete inputs and
   unsupported AC. The wire probe error was 0.74% and double-current energy scaled
   by 4.0. This seed set is not an independent holdout or a calibrated confidence
   guarantee; runtime settings remain unchanged.
   Native references now also check the thick-wire interior and exterior field
   vectors, mesh refinement with fixed domain, domain-radius changes with fixed
   mesh, and a nested coaxial return-current shell including exterior cancellation.
   Both new reference tests passed in the development Docker image. Measurements
   are saved in `evaluation/results/2026-10-24/numerical-references.json`; these
   circular symmetry cases do not establish convergence for arbitrary boundaries.
3. **Broader physics and CAD.** Axisymmetry, AC, nonlinear B-H curves, permanent
   magnets, shared-edge Boolean geometry and 3D remain future work. Add each with
   explicit schemas, deterministic validation and native reference tests.

Complete, test and push milestones to the existing draft PR #3. Do not present
pending CI results or planned physics as verified capabilities.
