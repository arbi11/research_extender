# An interactive FEMM companion

The repository now includes a companion application in
[`femm_assistant/`](https://github.com/arbi11/research_extender/tree/main/femm_assistant).
It lets a reader configure a C-core actuator, preview its geometry, run a native
FEMM 5 solve, and download the complete geometry/solve script and results.

The initial model uses two-dimensional planar DC magnetostatics with a linear
core permeability. Its paired winding cross-sections, pole gaps, stack depth,
mesh size, and excitation are recorded explicitly. These assumptions make the
model suitable for a reproducible teaching workflow; saturation and losses are
outside its scope.

Natural-language configuration proposes a model specification for the reader to
inspect. FEMM computes the fields and forces. These results are distinct from
the neural-network surrogate predictions discussed elsewhere in this chapter.

The download contains an editable Lua script and a standalone Python launcher.
It can be rerun with FEMM 5 without the assistant, its API key, or the repository.
Use the [application instructions](https://github.com/arbi11/research_extender/blob/main/femm_assistant/README.md)
to run it locally or deploy it on a Linux server.

Useful exercises are to vary current and examine flux/force scaling, increase
the air gap, and repeat a solve with a finer mesh. For a linear material, flux
linkage scales with current and force scales with its square. Numerical results
should be checked for mesh and outer-boundary convergence before interpretation.
