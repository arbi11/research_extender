# Appendix B: Quantum Computing for Electromagnetic Design

This appendix explores quantum computing through the electromagnetic models and
statistical questions developed in the book. It is a place for reproducible
preliminary experiments and clearly identified research proposals. It can evolve
independently of the main chapters.

The immediate goal is to learn enough quantum computing to formulate and test a
useful electromagnetic example. A practical speedup is a research question to
evaluate, rather than an assumption behind the appendix.

## Connections to the book

| Existing material | Quantum extension | Status in this appendix |
| --- | --- | --- |
| Chapter 1: computational motivation and research framework | Compare algorithms using explicit input, output, and cost assumptions | Discussion |
| Chapter 2: magnetic fields, force surrogates, and uncertainty | Estimate expected force and probability of insufficient force | Implemented on a simplified magnetic circuit |
| Chapter 3: motor performance maps and uncertainty | Estimate expected loss over uncertain operating conditions | Proposed follow-on |
| Chapter 4: sequential topology optimization | Optimize small binary design updates with QAOA or annealing | Research proposal |
| Chapter 5: learned design agents | Use a quantum optimizer inside a classical design workflow | Longer-term question |

Relevant repository entry points are
[`train_c_core_ann.py`](https://github.com/arbi11/research_extender/blob/main/jupyter_book/Chp2/content/train_c_core_ann.py),
which predicts force from current, gap, core width, and turns;
the [Chapter 2 uncertainty notebook](https://github.com/arbi11/research_extender/blob/main/jupyter_book/Chp2/content/05_uncertainty_quantification.ipynb);
the [Chapter 3 uncertainty notebook](https://github.com/arbi11/research_extender/blob/main/jupyter_book/Chp3/content/06_uncertainty_quantification.ipynb);
and [`seqTO_env.py`](https://github.com/arbi11/research_extender/blob/main/jupyter_book/Chp4/content/seqTO_env.py).

The foundational thesis is Arbaaz Khan's *Statistical Methods for Design and
Analysis of Electrical Machines*, McGill University, 2021. The simplified circuit
and uncertainty ranges used here are new teaching assumptions, not parameters
or validation results copied from the thesis.

## Two complementary directions

**Statistical estimation.** Given uncertain current and gap, estimate expected
force or the probability of failing a minimum-force requirement. Quantum amplitude
estimation (QAE) connects directly to expectations and Monte Carlo error analysis
{cite}`brassard2002,montanaro2015`. This is the implemented learning example.

**Design optimization.** Approximate a small binary material-update problem by a
quadratic objective, optimize it with a quantum algorithm, and validate the
candidate using the classical electromagnetic solver. This connects to the
book's topology-optimization methods and has precedent in electromagnetic QUBO
formulations {cite}`maruo2022`.

## What counts as progress

The first milestone is a working, independently checked numerical example whose
physics, probability model, and quantum encoding can all be explained. The next
milestone is a comparison with strong classical alternatives. A claim of practical
advantage would additionally require the costs of input preparation, reversible
model evaluation, measurement, noise handling, and any classical training or
precomputation.

The supplied simulation establishes the first milestone. It does not execute a
quantum circuit on hardware, accelerate the existing neural networks, or replace
FEMM. Its finite-grid problem is small enough for exact enumeration.

## Reading order

1. [Foundations](02_foundations.md): learn the quantum concepts through the force problem.
2. [Worked notebook](03_ccore_reliability.ipynb): inspect the simulation and its limitations.
3. [Topology optimization](04_topology_optimization.md): consider a deeper research extension.
4. [Research plan](05_research_plan.md): follow the milestones and reporting criteria.
