# Research proposal: quantum-assisted topology optimization

The book's SeqTO method constructs connected material layouts through controller
movements. The classical FEMM environment supplies performance feedback. A useful
research extension is to use a quantum optimizer to propose **small local design
updates**, with the electromagnetic response still evaluated classically.

This section is a proposal. No QAOA implementation, quantum annealing run, or
topology-optimization result is included in this preliminary appendix.

## A bounded first experiment

Begin with a fixed, connected C-core layout. Choose 8–12 optional material blocks
near its boundary. Let $z_i\in\{0,1\}$ indicate whether block $i$ is present.
Keep coils, armature, excitation, material model, and gap fixed. Prefer optional
blocks attached to a fixed connected backbone, and check every candidate for
connectivity, coil clearance, feature size, and material budget.

This is a new local-update representation, not an assertion that the original
SeqTO movement sequence is already a quadratic binary objective. Restricting the
representation changes the feasible design space and must be reported.

The existing
[`calculate_reward`](https://github.com/arbi11/research_extender/blob/main/jupyter_book/Chp4/content/seqTO_env.py)
returns both a magnetic-field reward and a force-related output. Establish the
force extraction's sign, depth scaling, and units against an independently checked
baseline before using it as a physical objective. A field proxy and a measured
force objective should be reported separately.

## From FEM evaluations to a QUBO

Fit a quadratic response model on sampled layouts:

$$
\widehat F(z)=c+\sum_i a_i z_i+\sum_{i<j}b_{ij}z_iz_j.
$$

Nonlinear magnetic saturation and long-range interactions are not generally
quadratic in material indicators. The fitted model is an approximation, not an
exact transformation of Maxwell's equations. Check held-out errors and candidate
rankings; use a local trust region if the surrogate is unreliable far from the
current layout.

For an illustrative fixed block count $K$, minimize

$$
C(z)=-\widehat F(z)+\lambda\left(\sum_i z_i-K\right)^2.
$$

This is a quadratic unconstrained binary optimization (QUBO) objective because
$z_i^2=z_i$. A fixed count is simpler than the book's interval budget. For unequal
block volumes, use volume weights. An inequality budget generally needs slack
variables or a constraint-preserving method; additional binary variables increase
the qubit requirement. Penalty weights require validation against objective scales.

Pairwise neighbor preferences do not guarantee global connectivity. Keep the
backbone construction and explicit feasibility check unless a complete
connectivity encoding has been derived and tested.

## QAOA and annealing

Substitute $z_i=(1-Z_i)/2$ to obtain an Ising cost Hamiltonian. The quantum
approximate optimization algorithm (QAOA) alternates cost evolution and mixing
operations, with a classical optimizer tuning their angles {cite}`farhi2014`.
Measured bitstrings propose layouts; FEMM validates their actual performance.
QAOA at small depth has no general guarantee of outperforming classical methods.

Quantum annealing is another possible solver for the same binary objective.
Classical simulated annealing and digital annealers also solve such objectives;
using a QUBO alone does not mean quantum hardware was used. Maruo, Soeda, and
Igarashi demonstrated electromagnetic topology optimization with a Digital
Annealer alternating with FEM calculations {cite}`maruo2022`. That is relevant
formulation precedent, not a demonstrated quantum-hardware speedup.

## Fair experiment and decision criteria

For 8–12 variables, enumerate all $2^n$ bitstrings to obtain the exact QUBO
minimum. Compare QAOA with that reference, simulated annealing, and a classical
local optimizer. Then validate a predeclared number of candidates with FEMM.
Compare the resulting FEMM-validated performance against matched FEM-evaluation
budgets for classical search, including surrogate-training evaluations.

Report actual force, field proxy, material use, feasibility rate, QUBO optimality
gap, FEM call count, circuit depth, two-qubit gate count, shots, and total elapsed
time. Keep simulation, hardware, and queue times separate. Record all preprocessing
and retraining costs. Do not reuse expensive test evaluations as free training data.

Proceed to a larger problem only if the surrogate reliably ranks candidates and
the quantum proposal mechanism adds value against strong classical baselines.
An informative negative result still belongs in this appendix.
