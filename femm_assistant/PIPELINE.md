# FEMM assistant implementation pipeline

This plan separates the working general-geometry harness from the next
implementation milestones. It records intended behavior, not a claim that every
stage below is already available.

## Current working baseline

- General geometry plans support circles, rectangles and simple polygons,
  including separate or nested regions, for 2D planar DC magnetostatics.
- Materials are linear and isotropic. Plans define circuits, signed turns,
  region labels, an enclosing A=0 domain, bounded meshing and optional probes/force.
- The AI proposes typed data. Deterministic validation checks the plan before
  compiling its geometry/material/boundary/mesh/solve/export sequence to Lua.
- Native FEMM produces the solution and numerical outputs. Current downloads
  contain `model.lua`, `spec.json` and `run.py`; `run.py` is a launcher, not a
  self-contained Python implementation of the model.
- Current meshing and solving run together. There is no independent mesh-only
  stage or actual mesh-quality report yet. Jev is not integrated yet.
- Verification includes application tests, a round-wire comparison with Ampere's
  law, mesh/current/energy scaling checks, and the C-core regression suite.

## Target sequence

`prompt → semantic classification → typed plan → geometry/material/BC checks`

`→ mesh only → measured mesh report → solve → extract and export`

Every transition uses an explicit stage result. A semantic probability cannot
override a failed geometry check, a failed mesh, or a failed solve.

| Stage | Responsibility | Status |
|---|---|---|
| Request | Capture intent, units and required physical inputs; ask about omissions | Implemented |
| Semantic classification | Use bounded Jev decisions for supported intent and ambiguity | Planned |
| Typed plan | Propose or modify a constrained model specification | Implemented |
| Plan checks | Validate contours, labels, references, supported physics and mesh budget | Implemented |
| Mesh only | Generate a native mesh without running the field solver | Next milestone |
| Mesh report | Report measured mesh counts/quality and deterministic gate outcomes | Next milestone |
| Solve | Run native FEMM with resource/time limits after the mesh gate passes | Implemented together with meshing; separation planned |
| Extract | Read computed quantities, probes, field samples and eligible forces | Implemented |
| Export | Save the specification, complete model script, artifacts and reports | Lua/launcher implemented; full Python model planned |

## Jev's bounded role

Use OpenRouter's [Jev interface](https://openrouter.ai/labs/jev), model
`typesafe/jev-1.14`, through `/api/alpha/decisions` for narrowly defined semantic
choices. Examples include whether a request describes supported planar DC
physics, requests an unsupported analysis, or needs clarification.

Persist the decision and returned probabilities with the model request. Treat
them as semantic evidence, not physical confidence, mesh quality or simulation
accuracy. Decision thresholds are provisional until evaluated against labeled
requests and FEMM-backed cases. Low confidence or contradictory intent should
lead to clarification rather than silent substitution of another physical model.

Arithmetic, unit conversion checks, topology, label placement, material/circuit
references, resource limits and stage transitions remain deterministic. Jev must
not generate executable code or decide that a failed numerical stage succeeded.

## Delivery milestones

1. **Independent mesh stage.** Split model construction/meshing from solving.
   Save mesh artifacts and an explicit stage status. Check actual mesh output,
   degenerate elements, counts and resource bounds. Keep analytical field results
   unavailable until a successful solve.
2. **Measured mesh gate.** Produce a report from the generated mesh rather than
   the current area-over-size planning estimate. Record quality criteria and
   reasons for failure. Verify refinement and boundary-distance studies on known
   models before adopting numerical acceptance thresholds.
3. **Complete Python export.** Add a standalone `model.py` containing the actual
   construction, materials, boundaries, mesh, solve and extraction workflow.
   Check it against the Lua export using the same specification and FEMM version.
4. **Jev semantic decisions.** Add the bounded classifier without replacing the
   deterministic validators. Test supported, ambiguous and unsupported requests,
   provider failures, and probability-threshold behavior on labeled cases.
5. **Integrated regression.** Exercise the staged workflow through the API and
   browser, including restart/failure recovery and reproducible downloads.

Complete, test and push each milestone to the existing draft PR #3 before moving
to the next. Use native FEMM tests for changes affecting geometry, meshing,
solving or exports. Preserve the existing installation's `.env` and local Compose
settings. Update the README only when a capability becomes available; identify
remaining work here rather than presenting planned behavior as ready.
