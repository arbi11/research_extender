# An interactive FEMM companion

The repository includes a companion application in
[`femm_assistant/`](https://github.com/arbi11/research_extender/tree/main/femm_assistant).
Readers can describe a new geometry, modify an existing model plan, inspect its
preview, run native FEMM 5, and download the complete scripts and results.

General geometry mode supports circles, rectangles and simple polygons, including
separate or nested material regions. It models two-dimensional planar DC
magnetostatics with specified linear isotropic permeability, DC circuits, an
out-of-plane depth, and an outer zero-potential boundary. Contours cannot touch,
cross, or share edges. Axisymmetric models, AC, nonlinear B-H curves, permanent
magnets and full 3D CAD are outside this version. A separate C-core example keeps
its parameter form and established pole-gap refinement.

The AI proposes a structured plan rather than executable code. Geometry, labels,
references, physics and mesh budgets are validated before the application
compiles the plan to Lua. Missing physical inputs produce questions; the reader
reviews the plan before solving. FEMM supplies numerical fields and quantities,
which are distinct from the surrogate predictions discussed in this chapter.

A source download contains `model.lua`, the complete editable model and solve
script, a standalone Python launcher `run.py`, and the validated `spec.json`.
After a successful solve, the full download also includes the FEMM model and
solution, field CSV, quantities and logs. Rerunning the scripts requires FEMM 5,
without the assistant, its AI key, or a repository checkout. See the
[application instructions](https://github.com/arbi11/research_extender/blob/main/femm_assistant/README.md)
for setup and deployment.

The round-wire example supports comparison with Ampere's law,
$B=\mu_0 I/(2\pi r)$ outside the conductor. Further exercises include refining
the mesh, varying current and comparing energy scaling. The C-core example adds
air-gap sensitivity and force scaling: for linear materials, flux scales with
current and force with its square. Check mesh and outer-boundary convergence
before interpreting quantitative results.
