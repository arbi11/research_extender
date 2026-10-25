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
  the region check catches it. In narrow slots, set `layer_offset` so each
  label falls inside its half slot. (`design-inner-rotor-spm-phase-linkage`,
  `design-outer-rotor-profiled-slots`)
- **Slots given by tabulated widths** (opening width and height, wedge, top and
  bottom widths, slot height) are built by `trapezoidal_slot_stator`; teeth
  given by width and shoe use `slotted_stator`. (`elrefaie-2006-fscw-spm`)
- **Components compute shared corner points once.** Two computations of the
  same corner by different angle arithmetic can differ in the last bits, enough
  to split a node and open a contour. (`elrefaie-2006-fscw-spm`)
- **Exact geometry reproduces to the last bit only when coordinates come from
  the same arithmetic.** A faithful rebuild can still mesh differently, so
  compare region by region rather than entity by entity.
  (`lrkanalysis-flux-sweep`)

- **Tubular and linear machines are one period with periodic ends.** Build the
  armature with `magnet_array`, pair the two end edges of each region with a
  periodic boundary (`segment_boundary` for edges another component drew), and
  model ideal iron as natural boundaries when the paper does. Periodic and
  natural boundaries alone leave the potential undetermined: fix it at one node
  with `fixed_point_potential`. (`wang-2005-tubular-halbach`)
- **Halbach directions decide which side the field is on.** Read each block's
  arrow from the figure, then confirm with an independent sign in the paper's
  results before trusting the reading. (`wang-2005-tubular-halbach`)

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

- **Tables are often images.** OCR every page at intake (`ocr.txt`) so table
  quotes can be checked mechanically; read the page image as well, because OCR
  garbles Greek letters and subscripts. (`elrefaie-2006-fscw-spm`)
- **A table can contradict itself.** Table I gave an 11.4 mm tooth width that no
  slot of the tabulated widths admits. Settle such conflicts with independent
  evidence (a figure, a mass or area the paper states) before solving, record
  the choice, and add a control for the rejected reading.
  (`elrefaie-2006-fscw-spm`)
- **"Top" and "bottom" of a slot are ambiguous.** Decide which end each width
  belongs to from a figure and from whether the teeth come out parallel.
  (`elrefaie-2006-fscw-spm`)
- **Rounded tables need an absolute allowance.** Harmonics given to three
  decimals carry up to 0.0005 T of rounding, over 1% of a 0.047 T harmonic;
  use 1% or 0.001 T, whichever is larger. (`wang-2005-tubular-halbach`)
- **Unquantified agreement still bounds a tolerance.** When a paper says its
  analysis was "verified by FEA" without numbers, read the overlay figure at
  600 dpi before solving and set the analytic target's tolerance to the spread
  it shows (3% in that paper's back-EMF overlay). (`elrefaie-2006-fscw-spm`)
- **Cogging from a figure needs dense samples.** Use at least eight rotor
  positions per cogging period, and still treat it as a comparator.
  (`elrefaie-2006-fscw-spm`)

- Text extraction scrambles tables, Greek letters and superscripts; read tables
  from page images and quote the text verbatim.
- Papers often leave rms or peak unstated (TEAM 30's 310 A/cm2). Resolve it
  from a second source or the paper's own numbers, and record which you chose.
- Many publishers block automated fetchers; the built-in browser can read
  landing pages. Author copies under IEEE copyright are for reading only: keep
  them in the cache, never in the repository.
- Measured inductances usually exceed 2D values because of end-winding leakage
  (2D 1.20 mH against a measured 1.3 mH). Treat measured values as comparators
  in a 2D replication unless an allowance is argued. (`elrefaie-2006-fscw-spm`)
- **A paper's FEA may not use its own stated material data.** El-Refaie's 2D
  FEA inductance (1.16 mH) matches unit magnet permeability within 0.06%, not
  the 1.22 its table states (+3.5%), and its quoted 14 mm effective air gap
  (magnet depth plus gap) implies the same. Keep the stated value, and use a
  control on the property to show what the FEA most likely used.
  (`elrefaie-2006-fscw-spm`)
