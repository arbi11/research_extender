# Learning milestones and research criteria

This appendix is intended to grow through completed, reviewable experiments.
The main chapters do not depend on any result here.

## Milestone 1: understand and reproduce the supplied example

Run the C-core notebook and explain the role of every register and reflection.
Verify state normalization, the probability encoded in the objective qubit,
unitarity, and agreement between the circuit simulation and the independent
eigenphase formula. Reproduce the classical/quantum sampling comparison with its
fixed random seed. The notebook includes these checks.

Distinguish finite-grid truth from the probability under a continuous input
distribution. Explain why changing the grid or force model changes the problem.
Identify why an exact sum is the strongest classical baseline for the supplied
small table.

## Milestone 2: implement a quantum SDK version

Translate the same small example into Qiskit or another maintained gate-model
SDK. Verify objective-qubit ordering and phase-register endianness against this
reference implementation. Start on a noiseless simulator; then inspect the
compiled circuit on a chosen hardware connectivity map.

Implement iterative amplitude estimation following {cite}`grinko2021`, or use a
maintained implementation after checking its API and statistical guarantees.
Record the package version and distinguish its interval guarantee from the
modal estimator in the preliminary notebook. Compare query counts, depth, and
confidence at matched accuracy. Neither the SDK port nor iterative QAE is
implemented in the initial appendix.

## Milestone 3: connect to the book's FEMM model

Use the existing C-core model to generate a small independent force dataset over
current and gap. Record geometry, gap definition, excitation convention,
nonlinear B–H curve, mesh convergence, and force-extraction convention. Choose
tolerances and a minimum-force requirement from an engineering specification;
the values in the notebook are synthetic teaching assumptions.

Calibrate a small analytical or polynomial response model and reserve held-out
FEMM evaluations for validation, especially near the failure boundary. An existing
neural network is a possible classical reference, but its PyTorch forward pass
cannot simply be called coherently inside a quantum algorithm. Investigate a
reversible evaluator and include its arithmetic precision and uncomputation cost.

Report physical-input uncertainty separately from surrogate epistemic uncertainty.
Ensemble spread or Monte Carlo dropout does not, by itself, define a manufacturing
tolerance distribution. More accurate integration of a surrogate does not improve
its calibration or fix a biased response near a failure threshold.

Compare ordinary Monte Carlo, scrambled quasi-Monte Carlo, and suitable quadrature.
For small failure probabilities, also consider importance sampling or other
classical reliability methods. Add hardware execution only after validating the
noiseless model, with circuit size, shots, device, noise treatment, and uncertainty
reported. No hardware account is needed for the present appendix.

## Milestone 4: test a small topology-update problem

Follow the [topology proposal](04_topology_optimization.md). Establish feasibility
and force validation first, then fit the quadratic model and benchmark the QUBO
solver. Keep its classical training and FEMM costs in the final comparison.

The most interesting question is whether a quantum proposal mechanism finds
better feasible designs per total computational budget. Replacing a classical
policy network with a quantum circuit is a separate hypothesis and has no assumed
advantage.

## A reusable experiment record

Each future result should include:

- **Question and target:** physical quantity, units, threshold, input distribution.
- **Model provenance:** source revision, geometry, material law, validation data.
- **Encoding:** registers, distribution loader, response evaluator, precision.
- **References:** exact sum where feasible and strong classical alternatives.
- **Costs:** data generation, training, gate synthesis, queries, shots, execution.
- **Error:** model, discretization, estimation, and hardware contributions.
- **Outcome:** what was observed, uncertainty, and what remains untested.

For the initial notebook, the completed outcome is a verified classical simulation
of a small quantum algorithm on a synthetic magnetic-circuit problem. FEMM
calibration, efficient response encoding, iterative QAE, hardware runs, and QAOA
remain future work. No practical quantum advantage has been established.

## References

```{bibliography}
```
