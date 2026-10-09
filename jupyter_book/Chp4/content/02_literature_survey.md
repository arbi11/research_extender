# Literature Review: Topology Optimization Methodologies and Machine Learning Applications

This chapter reviews topology optimization methods for electromagnetic device design, identifying the research gaps that motivate the sequential decision-making framework developed in subsequent sections. Two recent surveys organize the field: Lucchini et al. {cite}`lucchini2022topology` survey electromagnetics TO across gradient-based and gradient-free methods, noting the growing role of neural-network ideas; Nishanth and Wang {cite}`nishanth2022review` provide an electric-machine-specific review covering SynRM and adjacent motor classes. Both conclude that discrete methods remain dominant in practice but carry substantial computational and manufacturability costs — the precise gap SeqTO-v1 targets. We build on their taxonomy to evaluate each method class and to position the MDP-based framework within the broader optimization landscape.

## Traditional Topology Optimization Methodologies

Distributing materials within a design domain to optimize performance represents a fundamental challenge across multiple engineering disciplines: electromagnetics, solid mechanics, heat transfer, acoustics, fluid dynamics, and multiphysics problems. Several distinct TO methodologies have emerged, each with characteristic advantages and limitations: homogenization methods, density-based approaches (SIMP), boundary-based methods (level-set), and discrete ON/OFF techniques. These approaches have been applied to electromagnetic device design with varying degrees of success and practical adoption.

### Homogenization Methods

Bendsøe and Kikuchi {cite}`bendsoe1988generating` introduced the homogenization-based approach in the late 1980s, establishing a methodology that received substantial attention in structural engineering through subsequent work by Bendsøe and Sigmund {cite}`bendsoe1993topology` and Allaire and colleagues {cite}`allaire1997homogenization`. The method's first application to electromagnetic problems appeared in Yoo and colleagues' work on structural and electromagnetic topology optimization {cite}`yoo2000structural`.

The homogenization methodology transforms optimal shape determination into a material distribution optimization problem through microstructure parameterization, as illustrated in Figure {numref}`fig-homogenization-microstructure`. In two-dimensional design spaces, three design variables describe each microstructure unit cell, resulting in a large total variable count across the complete design domain. The approach rests on the mathematical concept of relaxation: making ill-posed problems well-posed by enlarging the space of admissible shapes through composite materials with prescribed microstructures {cite}`allaire2019homogenization`.

```{figure} ../_static/figures/homo_microStructure.png
---
name: fig-homogenization-microstructure
width: 95%
---
Homogenization method for topology optimization showing domain discretization and 2D unit microstructure with cavity. This method uses three design variables to describe microstructure, transforming optimal shape problems into material distribution optimization. Renewed interest due to additive manufacturing capabilities for microstructure fabrication.
```

While mathematically elegant, the homogenization method faces practical limitations. The large number of variables renders the approach computationally expensive for large systems. Additionally, the method produces non-smooth estimates of topology boundaries, complicating post-processing and manufacturing interpretation {cite}`midha_2018`.

Recent years have witnessed renewed interest in homogenization methods driven by advances in additive manufacturing. The practical feasibility of fabricating complex microstructures through AM creates synergy with homogenization's strength in microstructure optimization {cite}`guo2013additive`. Figure {numref}`fig-3d-printed-lattice` illustrates a 3D-printed lattice structure for a jet engine mounting bracket, demonstrating how modern manufacturing capabilities enable realization of homogenization-optimized designs {cite}`morgan2014ge,allaire2019homogenization`.

```{figure} ../_static/figures/Crystallon.jpg
---
name: fig-3d-printed-lattice
width: 50%
---
Example of 3D printed lattice structure for jet engine mounting bracket. Additive manufacturing enables fabrication of complex microstructures designed through homogenization-based topology optimization, combining optimal material distribution with manufacturing feasibility.
```

### Density-Based Methods (SIMP)

The Solid Isotropic Material with Penalization (SIMP) method, proposed by Bendsøe {cite}`bendsoe1989optimal` and further developed by Rozvany and colleagues {cite}`rozvany1991coc`, has become the dominant TO approach due to its versatility, effectiveness, and relative ease of implementation compared to homogenization. Following success in structural mechanics, SIMP applications proliferated in electromagnetics through work by Yoo {cite}`yoo2005optimal`, Byun {cite}`byun2002topology`, Wang {cite}`wang2004multi`, and Okamoto {cite}`okamoto20063`.

Density-based methods optimize objective functions by determining whether each finite element should contain solid material or remain void. The discrete nature of this decision creates an extremely large-scale combinatorial problem. SIMP addresses this through continuous relaxation: material properties are explicitly interpreted as continuous design variables (typically density, ρ ∈ [0,1]), transferring the discrete problem to a continuous formulation amenable to gradient-based optimizers. Penalty methods force solutions toward crisp "0/1" or "solid/void" topologies. Regularization and filtering techniques mitigate checkerboard patterns and mesh-dependency issues.

Figure {numref}`fig-density-method-ccore` demonstrates SIMP application to a C-core actuator, showing (a) initial topology, (b) optimal design from density method, and (c) results after averaging filtering to eliminate checkerboard patterns {cite}`midha2019selection`.

```{figure} ../_static/figures/density_c_core_final.png
---
name: fig-density-method-ccore
width: 100%
---
Density-based topology optimization results for C-core actuator. (a) Initial topology, (b) Optimal design from SIMP density method, (c) Result with averaging filtering to eliminate checkerboard patterns. Density methods are popular but require filtering techniques to handle grayscale materials and mesh-dependent solutions.
```

Despite SIMP's popularity, several limitations constrain its application. Sigmund and Petersson {cite}`Sigmund1998` documented difficulties including convergence to local optima, mesh-dependent structures, and checkerboard patterns. Bourdin {cite}`bourdin2001filters` established the mathematical foundations of density filtering, showing that it regularizes the ill-posed discrete problem and suppresses checkerboard instabilities — but introduces filter parameters that require problem-specific tuning with no universally effective setting. Byun {cite}`byun2004node` and Okamoto {cite}`okamoto2006investigation` observed that optimized results depend critically on initial conditions.

The persistent challenge of intermediate density elements—regions where ρ assumes values between 0 and 1, representing "grayscale" materials—remains problematic. While penalization schemes and filtering techniques eliminate grayscales in some problems, these approaches depend crucially on artificial parameters lacking rational guidelines for *a priori* determination. The sensitivity of solutions to filter parameters and the absence of universally effective parameter sets across different TO problems constitute ongoing research challenges {cite}`midha2019selection`.

### Boundary-Based Methods (Level Set)

Boundary-based methods, particularly the level-set approach developed by Osher and Sethian {cite}`osher1988fronts`, represent a fundamentally different TO paradigm. Rather than parameterizing the design domain through explicit functions (as in density methods), boundary-based approaches employ implicit functions Φ(x,t) = c that define structural boundaries {cite}`deaton2014survey`.

The level-set method introduces an additional dimension, representing 2D curves as intersections between planes and surfaces, as illustrated in Figure {numref}`fig-level-set-3d` and {numref}`fig-level-set-2d`. The circular curve Φ=0 represents the structural boundary, specified as the zero level set. Material distribution throughout the design domain follows from the level-set function values:

$$
\text{Material} = \begin{cases}
0 & \text{if } \Phi < 0 \\
1 & \text{if } \Phi \geq 0
\end{cases}
$$

::::{grid} 2
:gutter: 3

:::{grid-item}
```{figure} ../_static/figures/level1a.png
---
name: fig-level-set-3d
width: 100%
---
**(a)** 3D level-set surface. The circular curve (Φ=0) is the structural boundary; material fills the region where Φ≥0.
```
:::

:::{grid-item}
```{figure} ../_static/figures/level1b.png
---
name: fig-level-set-2d
width: 100%
---
**(b)** Top view. The zero level set separates material (Φ>0, green ring) from void (Φ<0, orange interior and exterior).
```
:::
::::

Applications of level-set TO span electromagnetic antenna design {cite}`zhou2010level`, metamaterial optimization {cite}`zhou2011topology`, and dielectric material distribution {cite}`otomori2012topology`. In the electromagnetic actuator domain — directly relevant to this chapter's C-core benchmark — Park et al. {cite}`park2009` maximized actuator force under material constraints using a level-set formulation, and Lim et al. {cite}`lim2011topology` extended this with a phase-field approach to control geometric complexity and boundary smoothness. These works define the closest prior formulations against which SeqTO-v1's connectivity and manufacturability claims should be judged.

Boundary-based methods differ fundamentally from density approaches in their optimization products. Density methods typically generate optimized topologies containing substantial intermediate-density elements, requiring post-processing for interpretation. Level-set methods produce topologies with crisp, smooth edges directly. However, this advantage comes with trade-offs: optimized boundaries remain represented by discretized, non-smooth meshes unless alternative techniques are applied. Additionally, level-set solutions exhibit heavy dependence on initial conditions, and the method can generate irregular, toothed topologies in some applications {cite}`deaton2014survey`.

### Discrete Methods (ON/OFF Approach)

Discrete or ON/OFF methods constitute the binary approach to TO, where each design space element assumes discrete states: material present (1) or void (0), eliminating the grayscale elements that complicate density-based methods {cite}`midha_2018`. This discrete formulation transforms the material distribution problem into a combinatorial optimization naturally suited to stochastic search algorithms.

Density-based methods can be understood as continuous relaxations of the underlying discrete problem, enabling gradient-based optimization. In contrast, discrete methods embrace the combinatorial nature directly, typically employing Evolutionary Algorithms (EA) for solution. While gradient-based approaches generally converge faster, stochastic methods offer superior global search capability, stronger robustness against local optima, and parallel search characteristics {cite}`li2019numerical`. The ability to find global solutions represents a decisive advantage, particularly for highly non-convex electromagnetic TO landscapes.

However, discrete methods face substantial computational demands. Comprehensive TO for an electromagnetic device easily requires 10,000+ finite element analyses. For complex 3D FEA where individual simulations demand 30+ minutes, optimization timelines extend to weeks or months—prohibitive for many industrial applications {cite}`lei2017review`.

Additionally, discrete methods produce topologies with undesirable artifacts: floating material pieces disconnected from the main structure, checkerboard patterns, and thin "stepping stone" connections. Practitioners employ various remediation strategies: filtering or smoothing post-processing {cite}`campelo2008topology`, incorporating smoothness directly into the optimization objective {cite}`dupre2014ant`, or applying specialized clustering techniques {cite}`campelo2010survey,campelo2008,sato2014`. Unfortunately, filtering proves unreliable, occasionally producing infeasible solutions that drastically degrade performance. Furthermore, filter parameters (dimensions, thresholds, operating characteristics) require problem-specific tuning, and no universal parameter set exists across different TO problems.

Despite these limitations, ON/OFF methods maintain popularity due to convenient implementation across diverse TO problems and straightforward integration with commercial FEA software. The fundamental compatibility between evolutionary algorithms and binary optimization, combined with EA's inherent global search capabilities, ensures continued application to electromagnetic design problems where global optima are critical {cite}`im2003hybrid`.

For synchronous reluctance motor (SynRM) design specifically, several alternatives to standard ON/OFF EA have been proposed. Sato et al. {cite}`sato2015synrm` introduced normalized Gaussian network (NGnet) parameterization to jointly optimize average torque and iron loss. Otomo and Igarashi {cite}`otomo2021gabor` demonstrated that Gabor-filter-based topology parameterization produces thin layer-shaped flux barriers with better torque than both conventional designs and the NGnet baseline. Lee et al. {cite}`lee2021gta` combined GA with deterministic ON/OFF refinement in a genetic topology algorithm (GTA), achieving effective SynRM designs without continuous relaxation. These methods constitute the direct motor-class baselines against which SeqTO-v1's SynRM results are evaluated in Section 7.

## Machine Learning in Topology Optimization

### Learning to Reduce Expensive Evaluations

The dominant cost in TO is repeated field solves: each candidate topology requires a FEMM or FEA evaluation, and a typical run demands 10,000–40,000 such evaluations {cite}`lei2017review`. The primary ML response to this has been surrogate modeling — training neural networks to predict device performance without running the full solver.

Sasaki and Igarashi {cite}`sasaki2019topology` demonstrated CNN-based screening for IPM motor TO, filtering unpromising candidates before FEA and substantially reducing total computation. Asanuma et al. {cite}`asanuma2020transfer` extended this with transfer learning across related motor geometries, reporting total computation below 15% of the conventional method. Barmada et al. {cite}`barmada2021deep` applied deep network surrogates to the electromagnetic TEAM 25 benchmark, learning the FEA relationship and using it as an optimization proxy.

These approaches share a common structure: surrogates reduce the cost of evaluating a design, but the optimization algorithm (GA, random search, gradient descent) that decides which designs to generate remains unchanged. They accelerate evaluation without altering exploration. This is the key distinction from reinforcement learning.

### Reinforcement Learning for Sequential Design Synthesis

Reinforcement learning addresses a different problem: rather than reducing the cost of evaluating a given design, an RL agent learns which designs to generate through direct interaction with the environment. This aligns naturally with sequential TO formulations where construction proceeds through iterative decisions.

RL has been applied to structural TO with increasing sophistication. Brown et al. {cite}`brown2022deep` showed that RL agents can learn generalized sequential design strategies on elementally discretized domains, with reward signals derived from physical simulations. These works also clarify that state representation — flat binary vector, 2D spatial observation, or feature-mapped embedding — is a consequential design decision. A flat vector encoding is appropriate for tabular or low-dimensional value-based learning but does not scale to large domains or preserve the spatial structure that CNN-based policies exploit.

Applications targeting electromagnetic TO have been more limited. The controller-based sequential framework (SeqTO-v1) recasts electromagnetic TO as a movement-sequence problem, creating an MDP formulation amenable to both Q-learning and GA {cite}`khan2020sequence`. A subsequent application to SynRM demonstrated that a trained policy generalizes to related unseen design scenarios while using fewer FEA evaluations than the GA baseline {cite}`khan2022reinforcement`. Crucially, the SeqTO representation is the contribution that makes the MDP natural — the controller-motion idea preceded and motivated the RL application, not the reverse.

This chapter's tabular Q-learning implementation operates in the same SeqTO-v1 environment and extends the comparison to the C-core actuator benchmark, enabling a controlled RL-vs-GA evaluation where both algorithms face identical state, action, and reward structures.

## Comparative Analysis of TO Methodologies

Table 4.1 provides a category-level analytical view of the TO landscape, organized around the dimensions that matter most for the SeqTO argument: how designs are represented, how artifacts are handled, and whether the approach has been validated on electromagnetic problems.

**Table 4.1:** Analytical taxonomy of topology optimization approaches by method category

| Category | Representative Methods | Design Representation | Artifact Control | EM Validated | Key Limitation |
|---|---|---|---|---|---|
| **Homogenization** | Bendsøe & Kikuchi 1988; Allaire et al. | Microstructure density | Smooth via relaxation; non-smooth boundary | Limited | High variable count; impractical for most EM geometries |
| **Density-based (SIMP)** | SIMP, RAMP; EM extensions | Continuous density ρ ∈ [0,1] | Filter post-processing required | Yes — actuators, motors, antennas | Filter parameter sensitivity; grayscale elements; local optima |
| **Level-set / boundary** | Osher & Sethian; phase-field; Park 2009; Lim 2011 | Implicit function Φ | Crisp boundaries by construction | Yes — magnetic actuators, antennas | Initial condition dependency; irregular topologies on coarse meshes |
| **Discrete ON/OFF + EA** | GA, SA, PSO with binary grid | Binary grid | Filtering / clustering required | Yes — motors (NGnet, GTA, Gabor), actuators | 10,000–40,000 FEA calls; floating elements persist without filtering |
| **ML surrogate / screening** | Sasaki 2019 CNN; Asanuma 2020 transfer; Barmada 2021 DNN | Binary grid / image | Not addressed | Yes — IPM, SynRM | Accelerates evaluation only; does not change which designs are generated |
| **RL for sequential TO** | Brown 2022; SOgym 2025 | Sequential element decisions | Action masking; depends on formulation | No — structural only | No EM benchmark; modern variants require PPO / DreamerV3, not tabular Q |
| **SeqTO-v1 + MDP (this work)** | Controller trail + Q-learning or GA | Action sequence over discrete grid | Structural — by construction within controller constraints | Yes — C-core (40 N → 70 N), SynRM (2.5 → 3.40 Nm) | Sequential constraints; tabular Q-learning requires compact state encoding |

Table 4.2 follows with a method-by-method reference summary.

**Table 4.2:** Method-by-method reference summary for electromagnetic device design

| Method | Era | Primary Strength | Critical Limitation | Typical Applications | Key References |
|--------|-----|------------------|---------------------|---------------------|----------------|
| **Homogenization** | 1988-present | Microstructure optimization with theoretical rigor | High variable count; non-smooth boundaries | AM-enabled lattice structures; metamaterials | {cite}`bendsoe1988generating`, {cite}`allaire2019homogenization` |
| **SIMP (Density)** | 1989-present | Versatile; gradient-based efficiency; wide adoption | Grayscale materials; checkerboard patterns; filter parameter sensitivity | General structural TO; beam/truss optimization | {cite}`bendsoe1989optimal`, {cite}`rozvany1991coc`, {cite}`Sigmund1998` |
| **Level Set** | 1988-present | Crisp boundaries without intermediate densities | Initial condition dependency; irregular topologies | Antenna design; metamaterials; smooth boundary requirements | {cite}`osher1988fronts`, {cite}`zhou2010level`, {cite}`deaton2014survey` |
| **ON/OFF + EA** | 2000-present | Global search; binary clarity; FEA integration | High computational cost (10,000+ FEA); checkerboard/floating elements; filter dependency | Electromagnetic devices where global optima critical | {cite}`im2003hybrid`, {cite}`lei2017review` |
| **SeqTO-v1 + MDP** | This work | Reduces disconnected-fragment artifacts by construction; no filtering for addressed artifact class; RL/GA compatible | Sequential construction constraints; minimum feature size and smooth boundaries remain open; tabular Q-learning limited to compact state spaces | EM motors; actuators; C-core and SynRM benchmarks | This chapter |

The comparison reveals a fundamental trade-off across existing methods: approaches offering mathematical elegance and gradient-based efficiency (homogenization, SIMP, level-set) require post-processing to address artifacts (checkerboards, grayscales, irregular boundaries), while discrete methods embracing the combinatorial nature (ON/OFF) demand prohibitive computational expense and filtering to achieve manufacturability.

Sequential TO formulations—particularly MDP-based approaches enabling RL application—offer a conceptually distinct alternative: constructing topologies through iterative local decisions inherently produces connected configurations, eliminating the checkerboard problem *by construction* rather than through post-processing remediation. This paradigm shift motivates systematic investigation of whether the sequential framework, despite constraining the search space to constructible topologies, can discover competitive designs more efficiently than traditional approaches exploring the full combinatorial space.

## Research Gaps and Positioning

Analysis of the literature reveals several interconnected gaps motivating the MDP-based framework developed in this chapter:

**Gap 1: Artifact Elimination Without Filtering**

All established discrete TO methods—both density-based and ON/OFF approaches—produce undesirable artifacts: checkerboard patterns, isolated material islands, thin connections, and floating elements {cite}`Sigmund1998,campelo2008topology`. Current remediation strategies rely on filtering and smoothing techniques requiring careful parameter tuning, with no universal parameter set effective across different problems {cite}`midha2019selection`. The absence of principled methods guaranteeing manufacturable topologies without filtering represents a significant practical limitation.

**Gap 2: Principled Sequential Decision Framework for Electromagnetic TO**

While recent work has applied RL to structural TO {cite}`brown2022deep`, no prior research had formulated electromagnetic TO as a Markov Decision Process with well-defined state, action, transition, and reward specifications coupled to a FEMM-based field solver. Electromagnetic TO presents distinct challenges — nonlinear material behavior, coupled field physics, complex objective landscapes — that require domain-specific environment design rather than direct transfer from structural benchmarks.

**Gap 3: Controlled RL vs. GA Comparison and Computational Efficiency**

Existing comparisons between RL and evolutionary methods typically evaluate across different problem formulations — different state spaces, different reward structures, different device geometries — making it impossible to isolate whether observed differences come from the learning algorithm or the problem encoding. Surrogate-based ML reduces evaluation cost but does not change which designs are generated; RL changes the exploration strategy but existing EM applications lack a GA baseline in the same environment. A controlled comparison — where both algorithms operate within identical sequential state, action, and reward structures — and explicit accounting of FEA cost (optimization-stage calls only, not amortized training) are both absent from the electromagnetic TO literature.

**Contributions Addressing These Gaps**

This chapter addresses these gaps through four interconnected contributions:

1. **Sequential TO Framework (SeqTO-v1)**: Controller-based construction that structurally reduces disconnected-fragment and checkerboard artifacts for reachable configurations, without requiring filtering or penalty terms for that artifact class
2. **MDP Formulation for Electromagnetic TO**: First explicit coupling of a controller-trail MDP to FEMM-based magnetostatic reward evaluation, enabling direct application of tabular Q-learning to C-core and SynRM benchmarks
3. **Q-Learning Implementation and Validation**: Demonstrates that tabular Q-learning within SeqTO-v1 discovers competitive electromagnetic TO solutions using approximately 15–20% fewer FEA evaluations than the GA baseline
4. **Controlled RL vs. GA Comparison**: Both algorithms evaluated on identical SeqTO-v1 environment instances, isolating the effect of the learning strategy from the problem encoding

Together, these contributions establish a controller-based sequential representation for electromagnetic TO that embeds a connectivity bias into the feasible design process, admits a natural MDP formulation, and is benchmarked against GA on C-core and SynRM problems.

**Positioning of SeqTO-v1 in the landscape.** Existing methods address electromagnetic TO through three broad strategies: continuous relaxation (density-based, level-set), which produces artifacts requiring post-processing filters with no universal parameter set; discrete evolutionary search (ON/OFF + GA), which avoids relaxation but incurs prohibitive FEA cost and still requires filtering for manufacturability; and ML surrogates, which reduce evaluation cost but leave the exploration strategy unchanged. SeqTO-v1 occupies a distinct position: a discrete constructive representation that avoids the artifact problem structurally rather than through post-processing, operates at FEA cost comparable to density-based methods, and admits both RL and GA as solvers within the same environment. It does not claim RL is universally superior to evolutionary search — the controlled comparison in Section 7 tests precisely that question. Section 3 formalizes this setup as a Markov Decision Process and derives the Q-learning algorithm that operates within it.