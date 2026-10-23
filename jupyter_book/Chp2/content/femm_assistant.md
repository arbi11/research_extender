# An interactive FEMM companion

The repository includes a companion application in
[`femm_assistant/`](https://github.com/arbi11/research_extender/tree/main/femm_assistant).
Readers can describe a new geometry, modify an existing model plan, inspect its
preview, check the mesh, run native FEMM 5, and download complete scripts and
results: **AI → preview → Check mesh → Solve → Download**.

General geometry mode supports circles, rectangles and simple polygons, including
separate or nested material regions. It models two-dimensional planar DC
magnetostatics with specified linear isotropic permeability, DC circuits, an
out-of-plane depth, and an outer zero-potential boundary. Contours cannot touch,
cross, or share edges. Axisymmetric models, AC, nonlinear B-H curves, permanent
magnets and full 3D CAD are outside this version. A separate C-core example keeps
its parameter form and established pole-gap refinement.

The AI proposes a structured plan rather than executable code. Jev checks request
intent and draft fidelity; its probabilities describe semantic decisions, not
physical accuracy. Its default `0.90` threshold remains provisional. Geometry,
labels, references, physics and mesh budgets are checked deterministically. The
reader reviews the plan and an actual triangular mesh report before solving;
changing the model requires a new mesh check. Missing physical inputs produce
questions. FEMM supplies numerical fields and quantities,
which are distinct from the surrogate predictions discussed in this chapter.

A source download contains `model.py`, an editable Python CAD script with named
construction, material, boundary, mesh, solve and extraction stages; `model.lua`,
its native bridge; `run.py`, a compatibility launcher; and validated `spec.json`.
Run `python3 model.py`, or use `--mesh-only` and `--write-only` for those stages.
After a successful solve, the full download also includes the FEMM model and
solution, field CSV, quantities and logs. Rerunning the scripts requires FEMM 5,
without the assistant, its AI key, third-party Python packages or a repository
checkout. Detailed pre-solve mesh diagnostics use the opt-in native export patch
included in the assistant image; portable scripts can also run with stock FEMM 5.
See the
[application instructions](https://github.com/arbi11/research_extender/blob/main/femm_assistant/README.md)
for setup and deployment.

The round-wire example supports comparison with Ampere's law,
$B=\mu_0 I/(2\pi r)$ outside the conductor. Further exercises include refining
the mesh, varying current and comparing energy scaling. The C-core example adds
air-gap sensitivity and force scaling: for linear materials, flux scales with
current and force with its square. Check mesh and outer-boundary convergence
before interpreting quantitative results. A passed mesh report checks structural
validity and domain/label coverage; it does not establish numerical convergence.
