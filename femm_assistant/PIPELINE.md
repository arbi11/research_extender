# FEMM assistant implementation pipeline

This records the implemented workflow and remaining verification work. A feature
being implemented does not establish that a particular release, provider run or
numerical reference passed. Check the reports and CI for the commit being used.

## Current sequence

`prompt and/or cited paper case → Jev function route → strict inventory → source facts/settings → component/feature filling → deterministic plan checks → preview`

`→ immutable revision → native mesh check → exact-spec prerequisite → solve → frozen target comparison → export`

After an observed failure, one requested recovery uses checkpoint ancestry,
feature edits and native evidence to propose a scoped branch repair. Its fresh
mesh and solve are checked against unchanged frozen targets. Paid recovery
requests are claimed/cached per failed job; duplicate clicks do not initiate a
second provider attempt. Queue pressure can resume the already prepared repair.

An actual wire/local stage failure can receive one argument diagnostic and one
repair of that stage, retaining earlier checkpoints. Provider/configuration failures
do not justify changing physics. No default model output cap or AI response deadline
is imposed. Optional visual
review and manual native failure diagnosis produce advice beside this sequence;
they do not approve geometry or control whether it can be meshed or solved.

| Stage | Responsibility | Status |
|---|---|---|
| Request | Capture a new geometry or changes to a current plan | Implemented |
| Paper intake | Preserve PDF bytes, page quotes, explicit/derived parameters and missing information | Implemented; text PDFs only |
| Source binding | Bind selected paper case to exact generated model and immutable revision | Implemented; provenance only |
| Source arithmetic | Check cited uniform current density against one winding's area, turns and circuit current | Opt-in; reports ambiguous/inapplicable cases |
| Function route | Jev selects `create_plan`, `modify_plan` or `clarify` | Implemented; fixed catalog |
| Typed stages | Separate strict inventory, facts, domain, component and feature contracts; custom/template components compose locally | Implemented; no executable AI code or ordinary-JSON fallback |
| Source images | Preserve PDF page/crop provenance and compare exact source/geometry/mesh pixels | Optional; advisory, separate image model |
| Plan checks | Validate contours, labels, references and executable physics | Implemented; deterministic |
| Failed-call diagnostic | Compare actual failed function arguments with its contract and errors | Optional; advisory |
| Preview | Show valid typed geometry and assumptions for user inspection | Implemented; no confidence gate |
| Visual review | Compare request, rendered geometry and typed plan, optionally actual mesh | Optional; advisory |
| Mesh only | Run native meshing without solving the magnetic field | Implemented |
| Mesh report | Inspect actual triangles, shape, domain and label coverage | Implemented; not an accuracy certificate |
| Solve | Require a passed mesh for the exact current spec; run native FEMM | Implemented |
| Extract | Read computed quantities, probes, field samples and eligible forces | Implemented |
| Target | Freeze numerical references before execution and compare actual native results | Implemented; separate from solver success |
| Recovery | Diagnose an observed failure and run one scoped checkpoint/repair branch | Implemented; bounded and user initiated |
| Export | Save editable Python CAD, native Lua, spec, artifacts and reports | Implemented |

## Jev's bounded role

The draft model defaults to `google/gemini-3.1-flash-lite`. Jev defaults to
`typesafe/jev-1.14` through `/api/alpha/decisions`. Only `OPENROUTER_API_KEY` is
needed for these defaults. Compose forwards `DECISION_MODEL` and
`DECISION_BASE_URL`; plain Uvicorn uses exported environment variables. Existing
explicit `AI_MODEL` and `VISION_MODEL` choices override the defaults; preserve
those settings when updating an installation.

Jev chooses a function from a fixed catalog. A valid route is used regardless of
its reported probability or confidence. It does not run support, sufficiency or
fidelity approval checks. `DECISION_THRESHOLD` remains accepted for existing
settings and is ignored. The draft model can ask for missing dimensions,
materials or excitation, or explain unsupported physics, through its typed
clarification response.

When `create_plan` or `modify_plan` actually throws a validation error, one Jev
question considers whether that function and its actual arguments match the
function contract and requested geometry, using the schema and reported errors.
The result is diagnostic information for the bounded repair. A yes answer does
not turn the failed call into a success. A diagnostic provider failure does not
prevent the repair from being validated normally.

Each legacy planning request has at most one logical route call, one logical
failed-call diagnostic and two drafts. Hierarchical creation applies the same
one-diagnostic/one-repair rule independently to an actually failed typed stage,
keeping completed stages fixed. A Jev logical call can make at most two transport
attempts after connection-establishment failures. It never retries an ambiguous
read/write failure or creates a whole-case retry. These paths impose no output
token or reasoning cap and no default AI read/wall deadline. Arithmetic, units,
topology, labels, material/circuit references and stage transitions remain
deterministic. Native execution can still report actual resource failures.

Users may manually diagnose failed native `mesh_plan` or `solve_plan` jobs using
the actual spec, original error and bounded log. The result is cached as an
advisory on that failed job. It preserves the original error and failed status;
it never creates successful mesh evidence or enables a failed solve.

## Optional visual observations

Manual visual review sends the original request, typed plan and geometry PNG to
an image-input model. `VISION_MODEL` defaults to
`google/gemini-3.1-flash-lite`; an explicitly empty value disables the action.
CairoSVG renders local passive SVG to PNG. A second image is included only when
an actual completed, passed mesh belongs to the exact current spec.

The reviewer reports matches, issues or uncertainty with concrete locations and
suggestions. Findings and provider errors are advisory and read-only: they do
not modify the spec or jobs, certify numerical accuracy, or disable mesh/solve.
No visual request runs automatically during drafting or numerical execution.

## Supported geometry and native prerequisites

General plans use circles, rectangles and simple polygons with separate or nested
regions, linear isotropic materials, DC circuits, signed turns, labeled regions,
an enclosing A=0 domain and native meshing. Physics is planar or axisymmetric DC magnetostatics in millimeters. Planar
models have explicit out-of-plane depth; axisymmetric models use `depth_mm=null`
and r/z coordinates with full revolved volumes. Contours cannot touch or cross,
except for explicitly supported shared axis edges at r=0 in axisymmetric models.
The separate C-core numeric example retains its original solve workflow.

The mesh report uses the actual native mesh. It checks triangle validity,
resource limits, domain coverage and compiler block labels, and reports minimum
angle and aspect ratio. A passed structural check does not establish mesh or
outer-boundary convergence. Editing any part of the spec invalidates the previous
mesh prerequisite. Native analysis remeshes the identical validated spec; it does
not consume diagnostic `.node`/`.ele` files from the earlier mesh job.

## Portable model and deployment isolation

Downloads include `model.py` with editable `MODEL` data and named Python CAD
stages, `model.lua` as the native bridge, and `run.py` as a compatibility launcher.
`python3 model.py` solves; `--mesh-only` meshes; `--write-only` regenerates source.
Rerunning needs FEMM 5, without an AI key, repository or Python dependencies.

`deploy/patch_femm_mesh.py` adds opt-in `FEMM_MESH_EXPORT=1` output from the actual
in-memory mesh without changing numerical algorithms. The image retains the
upstream archive and patcher. Detailed mesh diagnostics require the patch;
portable scripts can also run with stock FEMM 5. CairoSVG renders passive previews
and pypdf extracts bounded page text for paper intake.

Development runs in Docker on this laptop at port 8001. Preserve `.env`, tokens,
keys, local Compose edits and jobs. Leave the old port-8000 installation and old
laptop unchanged until production deployment. A tested Linux `amd64` workflow
image can later be loaded on the older laptop without recompilation. Its
availability depends on a successful workflow for that exact commit.

## Verification and remaining milestones

1. **Release verification.** Run unit/API checks, browser workflows and native
   live suites before recording a tested milestone. Browser fixtures verify UI
   state and native prerequisites, not real provider behavior or field accuracy.
2. **Repeated language-to-CAD execution.** `evaluation.batch` runs public create,
   modify and clarification requests with independent typed/physical comparisons,
   previews and actual native mesh/solve/export where appropriate. Two rounds of
   28 cases plan 56 attempts; retain every failure and error. Actual reports,
   rather than planned counts, establish the results.
3. **Optional vision comparison.** `evaluation.vision_benchmark` defaults to six
   fixtures across two image-input models, planning 12 calls. Record errors,
   findings, latency and reported usage without treating observations as gates.
4. **Independent numerical references.** Retain wire interior/exterior vector,
   current/energy scaling, refinement, domain-radius and coaxial cancellation
   checks. Circular symmetry cases do not establish convergence for arbitrary
   boundaries. See the [evaluation guide](evaluation/README.md) and saved reports.
5. **Broader physics and CAD.** AC, nonlinear B-H curves, permanent magnets, shared-edge Boolean geometry and 3D need explicit schemas,
   deterministic validation and native reference tests before being offered.

Earlier Jev semantic-gate reports, including the original 28-case run, describe a
retired policy. Keep them for historical offline replay; their match rates and
thresholds do not describe the current routing architecture.

Complete, test and push milestones to the existing PR #3. Do not present pending
CI or live runs as successful. No production deployment is implied by development
image builds or benchmark runs.

The Lubin 2012 axisymmetric coil paper supplies three independently transcribed
native reference cases. Published target tolerance (2%) and mesh/domain criteria
(0.5%) are fixed separately from an auxiliary energy identity diagnostic. Initial
coarse-mesh failures remain recorded. See the evaluation guide for the refined
protocol, source audit and distinction between gold-fixture execution and live
PDF-to-model extraction.
