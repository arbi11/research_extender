# Lessons for replicating machines in FEMM

Generic knowledge from earlier work. Each lesson names the case that taught it;
add new ones the same way, and correct or retire a lesson when a case
contradicts it.

## Numerics

- **Cogging torque and torque ripple are mesh noise on symmetric machines.**
  Rotationally symmetric geometry puts many points on common circles, and the
  mesher branches on the last bits of their coordinates. In LRK, moving every
  node by at most 1e-12 mm changed cogging torque by 18–32% of its peak while
  flux linkage moved 0.02%. Make such quantities comparators, unless a mesh
  study shows convergence. (`lrkanalysis-flux-sweep`)
- **Weighted-stress-tensor forces on small parts converge slowly.** Coilgun
  force moved from −2.5% to +8% of the published value under refinement. Prefer
  an air-gap element for rotating machines, where one is available, and record
  a mesh study. (`coilgun-z1p5`, `torque-benchmark`)
- **Deep saturation amplifies field errors.** Where the relative permeability is
  low (about 74 for M-19 near 1.84 T), a 0.27% error in B is a 2.3% error in H.
  Probe-based targets in saturated iron need a finer mesh than flux or torque.
  (`design-nonlinear-ring-bh`)
- **The engine container's 2 GB limit** kills FEMM at roughly 470,000 elements.
  Scale mesh controls to fit (0.75 rather than 0.5 for a full machine).
  (`lrkanalysis-flux-sweep`)
- **FEMM solves one frequency at a time.** A slip-frequency induction model
  captures the fundamental only; space harmonics (5th, 7th) need their own solves
  at the frequencies they see. (research for TEAM 30)

## Construction

- **Rotor positions** come from `rotate_group` steps on the rotor's group. With
  alternating magnets and a symmetric stator, one pole pitch with a declared
  half-wave symmetry covers a full electrical period.
  (`lrkanalysis-flux-sweep`)
- **Radial magnetization of arc magnets** uses FEMM's direction function
  (`theta` or `theta+180`); FEMM then stores the numeric angle as zero.
  (`design-inner-rotor-spm-phase-linkage`)
- **Winding signs:** a positive coil carries +turns in the slot clockwise of its
  tooth and drives flux radially inward. The `phase_linkage` assessor checks
  balance and the counterclockwise phase order from the construction alone.
  (`lrkanalysis-flux-sweep`)
- **Double-layer windings need divided slots**, or two labels share one region;
  the region check catches it. (`design-inner-rotor-spm-phase-linkage`)
- **Exact geometry reproduces to the last bit only when coordinates come from
  the same arithmetic.** A faithful rebuild can still mesh differently, so
  compare region by region rather than entity by entity.
  (`lrkanalysis-flux-sweep`)

## Materials

- FEMM needs a coercivity, not a remanence: `H_c = B_r / (mu0 * mu_r)`. Sintered
  NdFeB and SmCo have mu_r of about 1.05; bonded NdFeB about 1.1–1.25; ferrite
  about 1.05–1.1. Record mu_r as assumed when the paper omits it.
- The bundled FEMM 5 library holds 513 materials (132 B-H curves). Look up a
  named grade first; if the grade is absent, ask before downloading a datasheet,
  or take the nearest library grade as an assumption with a sensitivity control.
  Library names repeat across folders; give the folder.
- A 33-point library curve read back natively within 0.06% in B
  (`design-nonlinear-ring-bh`), so library B-H transfer is trustworthy.

## Sources and extraction

- Text extraction scrambles tables, Greek letters and superscripts; read tables
  from page images and quote the text verbatim.
- Papers often leave rms or peak unstated (TEAM 30's 310 A/cm2). Resolve it
  from a second source or the paper's own numbers, and record which you chose.
- Many publishers block automated fetchers; the built-in browser can read
  landing pages. Author copies under IEEE copyright are for reading only: keep
  them in the cache, never in the repository.
- Measured inductances usually exceed 2D values because of end-winding leakage
  (one surface-PM paper: 2D 1.16 mH, measured about 1.3 mH). Treat measured
  values as comparators in a 2D replication unless an allowance is argued.
