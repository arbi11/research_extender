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
- **FEMM's memory grows with every solve in one process**, by about 360 bytes
  per mesh element per solve: TEAM 30's 79k-element model filled the 2 GB
  container after 125 solves. For long sweeps set `engine.solves_per_process`
  (elements times solves times 360 B well under 2 GB); the engine restarts FEMM
  and replays earlier modifications, and results reproduce bit for bit across
  the restarts. (`team-30-induction`)
- **Read a discrepancy's signature before blaming an input.** A gap or a
  material changes where a flux-linkage curve bends: a 5% longer gap lowered
  an SRM's aligned flux linkage 4% at low current but 0.24% in saturation, and
  a more permeable steel raised it 8.8% and 0.4%. A ratio that stays constant
  across current at each rotor position (0.961 aligned, within 0.25% from 4 to
  18 A) is not either; look at how the result was evaluated, and record the
  failure when no input explains it. (`hamouda-2019-srm-8-6`)
- **FEMM solves one frequency at a time and has no velocity term.** For a
  rotationally uniform moving conductor in a linear problem, solve each spatial
  order m of the excitation at its own slip frequency, omega - m Omega
  (conjugating the excitation when that is negative), take order m out of
  circle samples, and superpose (`harmonic_superposition`). Torque comes from
  two gap circles, rotor loss from each order's slip power. In a single-phase
  machine the net torque is a near-cancellation of counter-rotating waves, so
  higher orders carry 40-230% of it: never stop at the fundamental.
  (`team-30-induction`, `design-rotating-cylinder-belt-winding`)

- **FEMM stores what it is given by its own rules.** `addbhpoint` starts every
  B-H curve at the origin, and `addarc` keeps at most 10 degrees as an arc's
  maximum segment, so a 0.2 mm fillet asked for in 29 degree pieces is meshed in
  10 degree ones. The build check applies both rules; count elements with them
  in mind. (`gierczynski-2024-prius-gen4-ipm`)
- **Estimate the element count before freezing.** Fillets and short arcs in a
  supplied drawing multiply elements: the Prius cross-section first meshed to
  3.35 million elements. A minimum arc piece (`min_segment_length`) and a model
  of one pole (antiperiodic ends, `sector_air_gap`) brought it to 109k; check
  such a model against the full machine at one operating point first (psi_d
  0.072890 against 0.072896 Wb). (`gierczynski-2024-prius-gen4-ipm`)

- **FEMM's air-gap-element torque of a sector is the whole machine's.** For
  an (anti)periodic sliding band FEMM extends the field round the full circle,
  so scaling the gap torque by the pole count read four times too high; a full
  cross-section solve gave the same torque at every position (flux linkages,
  by contrast, are the sector's and need the scale). (`ieej-2012-d-model-ipmsm`)
- **Average torque needs dense positions when the ripple is sharp.** Six
  positions over 60 electrical degrees put two samples in the D model's torque
  dips and read the mean 2.7% low; 30 positions at 1 degree matched the
  flux-linkage torque, 3/2 p (psi_d i_q - psi_q i_d), to 0.0001 N m. Compare
  the two before trusting a stress-tensor mean. (`ieej-2012-d-model-ipmsm`)
- **An air-gap element and unmeshed stator iron can give a wrong field.** With
  the stator's ideal iron left unmeshed and reaching the model's outer
  boundary, a sliding band in the gap gave half the paper's no-load flux
  linkage and a no-load torque with a nonzero mean. Meshing the gap and turning
  the rotor by remeshing (stress-tensor torque), or meshing the iron at a high
  permeability, reproduced the paper; unmeshed pole pieces between two bands
  were fine. Check an idealized-iron model against meshed iron at one position
  first. (`lubin-2011-spm-semi-closed-slots`, `lubin-2010-magnetic-gear`)

- **A linear model's torque for any current phase follows from a few solves.**
  With ideal or linear iron and linear magnets, torque is quadratic in the
  sources: the no-current torque plus terms in cos phi, sin phi, cos^2 phi,
  sin^2 phi and sin phi cos phi for a balanced set at phasor angle phi. Six
  solve sets (no current, phi = 0, 45, 90, 180, 270) give every term exactly,
  and two suffice when the rotor has no saliency. Use it to test whether any
  current reference reproduces a figure, as a diagnostic outside the frozen
  case. (`lubin-2011-spm-semi-closed-slots`, `lubin-2012-surface-inset-pm`)

- **Measure a target the way the paper defines it.** The gear's stall torque is
  "the maximum point" of a curve with a 3% third harmonic; read as the
  fundamental it came out 3% low. Do not substitute a statistic on an untested
  assumption (a sinusoidal curve); if the definition needs the peak, make sure
  the sweep puts a sample on it. (`zhang-2014-coaxial-magnetic-gear`)

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

- **Salient-pole machines** (switched reluctance, wound-field rotors) are
  `salient_poles` for the stator and the rotor (coil blocks share the pole
  sides, so no sliver of air separates them) and `salient_pole_winding`. A
  switched-reluctance phase's opposite poles carry opposite polarity: one
  series circuit with ±turns. Build the rotor unaligned and turn it toward
  alignment counterclockwise, so motoring torque is positive. (`hamouda-2019-srm-8-6`)

- **Tubular and linear machines are one period with periodic ends.** Build the
  armature with `magnet_array`, pair the two end edges of each region with a
  periodic boundary (`segment_boundary` for edges another component drew), and
  model ideal iron as natural boundaries when the paper does. Periodic and
  natural boundaries alone leave the potential undetermined: fix it at one node
  with `fixed_point_potential`. (`wang-2005-tubular-halbach`)
- **Halbach directions decide which side the field is on.** Read each block's
  arrow from the figure, then confirm with an independent sign in the paper's
  results before trusting the reading. (`wang-2005-tubular-halbach`)

- **Unmeshed labels turn with their region.** A rotor turned by `rotate_group`
  must carry every label in it, `<No Mesh>` ones included, or a label left
  behind lands on an edge once the rotor has turned half a sector (the mesher
  then stops). `ring_sectors` gives its unmeshed sectors the ring's group; a
  `hole` inside a moving region takes `group`. (`lubin-2012-surface-inset-pm`)
- **Ideal iron is an unmeshed region** when the paper idealizes it (subdomain
  models do): `ring_sectors` leaves a sector with no material unmeshed, so its
  surfaces are natural boundaries; fix the potential at one node. Mind the
  air-gap-element pitfall above. (`lubin-2010-magnetic-gear`)
- **Slots shared by two phases, with uniform current density,** are one-turn
  circuits each: a slot's current is the connecting matrix's sum of the phase
  currents, and a phase's flux linkage per turn the matrix's sum of the slots'
  (metric `terms`). FEMM gives a block one circuit only. (`lubin-2012-surface-inset-pm`)
- **Benchmark drawings can be R12 DXF with old POLYLINEs and doubled
  contours.** The reader now takes POLYLINE and VERTEX (it used to skip them),
  and refuses contour types it cannot convert; look for a contour drawn twice
  (one slot opening closed by both arcs and a chord left a 0.018 mm sliver).
  (`ieej-2012-d-model-ipmsm`)

- **Drawings supplied as DXF need joining.** Endpoints that should meet can
  miss by 1-8 micrometres, and a magnet corner can rest on a pocket edge without
  a node there. Snap endpoints within a stated distance and split contours at
  T-junctions (`entity_pattern`), then confirm every region has one label.
  (`gierczynski-2024-prius-gen4-ipm`)

## Materials

- FEMM needs a coercivity, not a remanence: `H_c = B_r / (mu0 * mu_r)`. Sintered
  NdFeB and SmCo have mu_r of about 1.05; bonded NdFeB about 1.1–1.25; ferrite
  about 1.05–1.1. Record mu_r as assumed when the paper omits it.
- The bundled FEMM 5 library holds 513 materials (132 B-H curves). Look up a
  named grade first; if the grade is absent, ask before downloading a datasheet,
  or take the nearest library grade as an assumption with a sensitivity control.
  Library names repeat across folders; give the folder.
- **A lamination stacking factor can decide a saturated map.** Building the
  Prius steel solid instead of at Motor-CAD's 0.97 fill raised psi_q by 3-4%
  and failed half the targets; read whether the reference applies the factor
  and give it a control. (`gierczynski-2024-prius-gen4-ipm`)
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
- **Calibrate plot axes on every gridline, never on the frame.** Hamouda's
  Fig. 8 frame sits above its 0.9 Wb gridline; a frame calibration read every
  flux linkage 2.2% low. Fit all gridlines, report the misfit, and check one
  reading against another figure of the same quantity (an inductance plot
  against flux linkage over current). (`hamouda-2019-srm-8-6`)
- **Track curves rather than threshold columns.** Follow each curve from a
  guess with local line and quadratic fits over several windows (their spread
  is part of the uncertainty); count stacked curves from the bottom in columns
  clear of marker symbols; expect a curve hidden under another where they
  coincide. State the uncertainty in the target's unit when it does not scale
  with the value. (`hamouda-2019-srm-8-6`)
- **A rendering of the authors' FE model shows what they solved.** Scale it by
  a tabulated diameter, check the scale with a second one, and fit circles and
  lines to its outlines. Hamouda's Fig. 13 showed Table 1's 96.7 mm "bore
  diameter" as the rotor's diameter, rotor pole roots 0.6 mm from the tabulated
  pole height, and tapered rotor poles; with that geometry the unaligned flux
  linkage matched the paper's FEM within 0.2%, against 1.25% for the
  conventional reading. Build FE targets from the rendering and give the
  table's reading a control. (`hamouda-2019-srm-8-6`)
- **Stated masses and areas decide ambiguous dimensions.** Where a paper leaves
  open at which radius a ratio is taken, compare its stated masses across its
  designs: the right reading gives one density for all of them (the gear's PM
  masses: 0.36% spread at one radius, 1.3-2.6% at the others), whatever the
  density is. Areas do the same for drawing details (four rounded corners in
  the double-cage bar). (`zhang-2014-coaxial-magnetic-gear`, `nemeth-2001-double-cage-bar`)
- **Know what a quoted number is before freezing it.** Nemeth's text gives
  skin factors "in this case" for an approximate circuit; the FE values are in
  its Fig. 8 (k_ri 4.8, not 2.36). Find the same quantity in the paper's tables
  or plots and freeze the one that is the FE (or analytic) result.
  (`nemeth-2001-double-cage-bar`)
- **Compare shapes as well as values.** A uniform factor across operating
  points, with the curve's shape reproduced, points at a scale (an effective
  length, a current or voltage convention) rather than at geometry or materials:
  the SynRM's torque was 1.28-1.36 times the paper's at two currents while its
  torque-angle curve matched within 1.1%, and no geometry or steel control moved
  it more than 3.2%. Check the paper's operating point against its own rating
  too (15.63 N m at 6000 rpm is the stated 10 kW). (`ibrahim-2016-synrm`)
- **A table row can be mislabelled.** The D model's EMF table labels its last
  row 9; digitizing the figure it summarizes gave orders 1-7 within 4% of the
  table, a 9th of 1.4 V and an 11th equal to that row. When a value looks
  impossible, recompute it from the figure before blaming the model.
  (`ieej-2012-d-model-ipmsm`)
- **Benchmarks may state dq values on the power-invariant scale.** The IEEJ
  page's psi_m is sqrt(3/2) times the phase flux-linkage fundamental, and its
  dq currents are sqrt(3) times rms; check a convention by recomputing one
  published number from another (psi_m from the EMF fundamental) before
  freezing. (`ieej-2012-d-model-ipmsm`)
- **Text values are rounded readings of a figure.** "A maximum torque of 75 Nm"
  was 73.8 N m on the paper's own FE points. Freeze the figure's points and keep
  the text as a comparator; check pole counts against field plots too (one
  example's text swapped the two rotors' pole pairs). (`lubin-2010-magnetic-gear`)
- **A static-torque figure can disagree with the paper's EMF figure.** Lubin
  2011 states Ia = I, Ib = Ic = -I/2, and its phase-A EMF is largest at 0
  degrees, so the torque must be largest there; its static-torque figure is
  zero there. The same model reproduced that figure within 0.07 N m with the
  current phasor turned 90 electrical degrees. Its cogging figure also has the
  opposite sign to the ripple inside its own torque figure. Check a figure's
  reference and sign against another result before freezing (see the skill's
  reference angles). (`lubin-2011-spm-semi-closed-slots`)
- **Fit a digitized curve's harmonics as well as its points.** A gear curve's
  fundamental and 3rd harmonic agreed within 0.2% while its points missed by
  -11% to +27%, alternating: the slot ripple was in antiphase. Point errors
  alone would have suggested a wrong amplitude. (`lubin-2010-magnetic-gear`)
- **Published FE points sit at k times range/N, rarely on round angles** (19 or
  39 intervals here). Find N from the digitized positions and put the sweep on
  them; find markers by eroding the image (filled diamonds) or by closing the
  gap a plotted line cuts through a dot. (`lubin-2010-magnetic-gear`, `lubin-2011-spm-semi-closed-slots`)

- **Tabulated optimum designs are rounded.** Before blaming the rounding, solve
  the rounded variables' neighbours once each: in the gear it moved the torque
  by at most 0.3%, too little to explain 1.3%. (`zhang-2014-coaxial-magnetic-gear`)
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

- **A commercial tool's model file is data for what the paper leaves out.** The
  Prius Motor-CAD file gave region materials, magnet directions and remanence,
  B-H tables, the stacking factor and the shaft material; read it as text and
  run none of its scripts. Its maps are interpolations between a few FE points
  (`ModelBuildPoints_*`), and dq maps from such tools are consistent with
  averages over rotor position: compare averages, not one position.
  (`gierczynski-2024-prius-gen4-ipm`)
- Text extraction scrambles tables, Greek letters and superscripts; read tables
  from page images and quote the text verbatim.
- **Conventions can differ between columns of one table.** TEAM 30 states
  310 A/cm^2 without saying rms or peak; its torques and losses (and the
  analysis's Fourier strengths) fix it as rms, and its voltages and fields are
  rms as well, so a model that reports phasor peaks is exactly sqrt 2 high in
  those columns only. Decide each column's convention from evidence, record it,
  and freeze a comparator in the other convention where it is inferred.
  (`team-30-induction`)
- **A drawing and a table can be mirror images.** Reflecting a machine in the
  x-axis reverses B_r on that axis and keeps H_theta, while torque, loss and
  voltage do not change. TEAM 30's field table follows Fig. 1's drawing, its
  analysis the mirror image; signed field values need the drawing they belong
  to. (`team-30-induction`)
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
