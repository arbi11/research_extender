"""Chapter 1 prose: replace the thesis-era numbers about this book's own methods, hardware and gains with
what the repo measured (user: "do the same for chapter 1", 2026-10-11). Markdown only.
Left alone: cited literature values, the global-energy / EV / efficiency-class statistics of C1/01 and the
design-era narrative of C1/02 (background, several carry [CITATION NEEDED] comments in the source).
Measured sources: C2 notebook outputs and logs (see _runs/REPORT.md), C3 _runs/REPORT.md + c3_logs
(19–22 s per design, 2,462 design folders, 2,234 valid; training 19:47–00:44), C5 queue logs (DQN 38,768 s,
A2C 22,305 s), hardware from lscpu / free / nvidia-smi on the laptop that ran everything.
"""
import nbformat
B = '/home/akhan/code/compem_book/C1/'
NBS = {}
def nb(name):
    if name not in NBS: NBS[name] = nbformat.read(B + name, as_version=4)
    return NBS[name]
def sub(name, i, old, new):
    c = nb(name).cells[i]
    assert c.cell_type == 'markdown' and c.source.count(old) == 1, (name, i, old[:70], c.source.count(old))
    c.source = c.source.replace(old, new)

def sub_span(name, i, start, end, new):
    """Replace from the line containing `start` through the line containing `end` (inclusive)."""
    c = nb(name).cells[i]; ls = c.source.split('\n')
    a = [k for k, l in enumerate(ls) if start in l]; b = [k for k, l in enumerate(ls) if end in l]
    assert c.cell_type == 'markdown' and len(a) == 1 and len(b) == 1 and a[0] <= b[0], (name, i, start[:40], a, b)
    ls[a[0]:b[0] + 1] = new.split('\n'); c.source = '\n'.join(ls)

# ───────────────────────── 01_electric_machine ─────────────────────────
N = '01_electric_machine.ipynb'
sub(N, 3, "Machine learning methods that provide 10<sup>3</sup>-10<sup>6</sup>× speedup over traditional simulation enable the design space exploration necessary to identify optimal solutions within available timeframes.",
"In Chapter 2 of this book a trained network returns a 256×256 field map in 16 ms where the finite element solve that produced the training data took 2.1 s on the same laptop, a factor of about 130; published studies on larger finite element models, where one solve takes hours, report far larger factors {cite}`khan2020efficiency`. Acceleration of that order is what makes design space exploration tractable within available timeframes.")
sub(N, 3, "This dissertation addresses these challenges through systematic investigation of machine learning methods for electromagnetic design acceleration.",
"This book addresses these challenges through systematic investigation of machine learning methods for electromagnetic design acceleration. Every number it reports about its own experiments was produced by the notebooks it contains.")

# ───────────────────────── 02_computation_sim ─────────────────────────
N = '02_computation_sim.ipynb'
sub_span(N, 3, "[VERIFY: FEA computation time claims", "Days to weeks",
"""**Computational cost**: it depends on mesh size, material nonlinearity and physics coupling. The two-dimensional magnetostatic FEMM models used in this book are small and solve in seconds on a laptop: 2.1 s for the coil of Chapter 2 (including extraction of the field onto a grid), 5.9 s for the interior permanent magnet motor of Chapter 2's pipeline, and 19–22 s per design for the full performance-map characterisation of Chapter 3, which solves tens of operating points per design. Industrial models with finer meshes, transient time-stepping and coupled thermal or structural physics take far longer; no such model was timed for this book.""")
sub(N, 3, "- **Sequential computation time**: 10,000 designs × 2 hours = 20,000 hours ≈ 2.3 years of continuous computation",
"- **Sequential computation time**: 10,000 designs at the 20 s per design of Chapter 3's pipeline take 56 h on one solver; Chapter 3's 2,462-design sweep took about 20 h of wall time with two FEMM workers on a laptop, restarts included. At the hours per design of an industrial model, the same sweep takes years.")
sub(N, 4, "**Data generation**: Execute FEA simulations on a representative sample of design space (typically 1,000-10,000 designs selected through Latin Hypercube sampling or similar space-filling methods).",
"**Data generation**: Execute FEA simulations on a representative sample of design space (1,000 designs in Chapter 2 and 2,234 in Chapter 3 of this book, selected through Latin Hypercube sampling).")
sub(N, 4, "**Rapid inference**: Once trained, the neural network predicts performance for novel designs in milliseconds, enabling exploration of millions of candidates within hours.",
"**Rapid inference**: Once trained, the neural network predicts performance for novel designs in milliseconds (16 ms per field map in Chapter 2), so a hundred thousand candidates take under half an hour on a laptop GPU.")
sub_span(N, 4, "**Computational speedup**:", "This orders-of-magnitude acceleration transforms",
"""**Computational speedup, measured in this book** (laptop with an NVIDIA GeForce MX450):
- FEMM solve of the Chapter 2 coil, with extraction of the field onto a grid: 2.1 s
- Trained U-Net, one 256×256 field map: 16 ms on the GPU, 159 ms on two CPU threads
- **Speedup factor: about 130× on the GPU, 13× on the CPU**

The network's cost does not grow with the finite element mesh, so the factor scales with the solver time it replaces; published surrogate studies on models that take hours per solve report far larger factors {cite}`khan2020efficiency`.

**Design space exploration capacity** (Chapter 2 numbers):
- 1,000 coil designs with FEMM: 34 min
- 1,000 coil designs with the trained network: 16 s
- 100,000 network predictions: 27 min

This acceleration transforms previously infeasible computational methods (genetic algorithms requiring 100,000+ function evaluations, multi-objective Pareto frontier exploration with millions of trade-off evaluations) into practical engineering tools accessible within conventional development schedules.""")
sub(N, 4, "necessitating FEA simulation to generate training datasets comprising 1,000-10,000 geometry-performance pairs.",
"necessitating FEA simulation to generate training datasets (1,000 field maps in Chapter 2 and 2,234 performance maps in Chapter 3 of this book).")
sub(N, 4, "**Gradient-based optimization**: Algorithms such as L-BFGS or Adam can perform 10,000+ function evaluations to converge to local optima. With FEA, this requires months; with ML surrogates, hours.",
"**Gradient-based optimization**: Algorithms such as L-BFGS or Adam can perform 10,000+ function evaluations to converge to local optima. At the 20 s per design of Chapter 3's FEMM pipeline, 10,000 evaluations take 56 h; at 16 ms per network prediction, under 3 min.")
sub(N, 4, "**Economic impact**: Organizations are achieving positive return on investment within 2-3 product development cycles, as initial ML infrastructure costs (computing resources, software development, training data generation) amortize across multiple product applications.",
"**Economic impact**: The initial costs (computing resources, software development, training data generation) amortize across the predictions that follow. In Chapter 2, data generation plus training (71 min on a laptop) cost as much as about 2,000 FEMM solves; every prediction after that saves 2 s.")
sub(N, 4, "**Democratization of optimization**: Consumer-grade GPUs (\\$500-\\$2,000) provide sufficient computational capacity for training and deploying surrogate models {cite}`khan2020efficiency`.",
"**Democratization of optimization**: Consumer-grade GPUs provide sufficient computational capacity for training and deploying surrogate models {cite}`khan2020efficiency`; everything in this book ran on a laptop with a 2 GB GPU and, for one training run, a free cloud GPU.")

# ───────────────────────── 03_research_framework ─────────────────────────
N = '03_research_framework.ipynb'
sub(N, 0, "# Research Framework and Dissertation Structure", "# Research Framework and Book Structure")
sub(N, 1, "The investigation demonstrates that orders-of-magnitude computational acceleration can be achieved while maintaining accuracy sufficient for practical engineering applications.",
"In the experiments of this book the trained surrogates evaluate a design two orders of magnitude faster than the finite element solve on the same laptop (16 ms against 2.1 s in Chapter 2), at 0.17% normalised mean squared error for field maps (Chapter 2) and a cross-validated R² of 0.91–0.97 for efficiency maps (Chapter 3).")

sub(N, 2, "Chapter 2 establishes the foundation for machine learning-based electromagnetic modeling through two complementary approaches: data-driven convolutional neural networks and physics-informed neural networks. The investigation focuses on magnetic field distribution prediction for three problems of increasing complexity: a coil in air, a transformer with nonlinear magnetic core, and an interior permanent magnet motor. Data-driven CNNs employ encoder-decoder architectures with dilated convolutions to capture long-range spatial dependencies dictated by electromagnetic field theory, achieving 0.1-1% normalized mean squared error with 180,000× computational speedup compared to finite element analysis. Bayesian uncertainty quantification via Monte Carlo dropout provides prediction confidence estimates. Physics-informed neural networks (PINNs) encode Maxwell's equations directly into the loss function, enabling learning from minimal training data through physics constraints while guaranteeing field solutions satisfy governing equations. This dual methodology demonstrates that both data-driven learning (high accuracy with sufficient data) and physics-constrained learning (data efficiency and extrapolation) effectively accelerate electromagnetic field prediction when appropriately designed with domain-aware architectural choices.",
"Chapter 2 establishes the foundation for machine learning-based electromagnetic modeling through two complementary approaches: data-driven convolutional neural networks and physics-informed neural networks. Its data pipeline simulates three problems of increasing complexity — a coil in air, a transformer with a nonlinear magnetic core, and an interior permanent magnet motor — and the learning experiments use the coil: 1,000 FEMM solutions, a U-Net whose dilated encoder convolutions capture the long-range spatial dependencies dictated by electromagnetic field theory, 0.17% normalized mean squared error on held-out field maps, and a prediction cost of 16 ms against 2.1 s per finite element solve. Monte Carlo dropout is implemented for uncertainty estimates and its calibration is measured rather than assumed (the nominal 90% interval covered 52% of the test points on a synthetic task). Physics-informed neural networks (PINNs) add the magnetostatic Poisson equation to the loss and reach 0.14% error from 100 field observations of one coil, against 15.8% for the same network trained on the data alone, with an ablation of each ingredient. This dual methodology shows that both data-driven learning (high accuracy with sufficient data) and physics-constrained learning (data efficiency) accelerate electromagnetic field prediction when designed with domain-aware choices — in the PINN's case, giving the network the radial coordinates the solution is built from.")

sub(N, 2, "Building on the spatial field prediction capabilities developed in Chapter 2, Chapter 3 extends the deep learning framework to temporal sequences and multi-objective performance prediction. The focus shifts from instantaneous field distributions to complete performance maps spanning the entire torque-speed operating envelope—a requirement for evaluating motor performance across diverse drive cycles rather than at isolated operating points. Recurrent neural networks with attention mechanisms model the complex dependencies between operating conditions (torque, speed) and multiple performance metrics (efficiency, power factor, losses) simultaneously, capturing both spatial patterns within individual maps and correlations between different performance quantities. Transfer learning techniques reduce data requirements when extending trained models to new motor topologies, demonstrating that models trained on one motor family can accelerate learning for related designs. This chapter shows that deep learning addresses not only the question of \"what are the fields\" (Chapter 2's domain) but \"how well does the machine perform\" across its operational space, providing the rapid performance evaluation capability essential for design optimization algorithms that require thousands to millions of function evaluations.",
"Building on the spatial field prediction of Chapter 2, Chapter 3 moves from instantaneous field distributions to complete performance maps spanning the torque–speed operating envelope — the quantity needed to evaluate a motor over drive cycles rather than at isolated operating points. A FEMM pipeline characterises each interior permanent magnet design over tens of operating points (19–22 s per design) and produced 2,234 valid designs of 2,800 sampled, plus 60 designs of a fractional-slot concentrated-winding variant. The chapter reviews recurrent and attention-based sequence models and trains a map decoder — a small multilayer perceptron on the design parameters feeding a transposed-convolution decoder of the 60×80 efficiency map — reaching a cross-validated R² of 0.912 for the baseline and 0.971 for the widest variant. Transfer-learning experiments measure when a model trained on one task accelerates learning on another: fine-tuning a pretrained model with its output layer reset beat training from scratch at the smallest data sizes (+0.05 R² at 10 designs for the power-factor maps) and, for the widest model, at every size tested. The chapter thus answers \"how well does the machine perform\" across its operating space, the evaluation that design optimization algorithms call thousands of times.")

sub(N, 2, "Chapter 4 marks a methodological transition from supervised prediction tasks to optimization-driven design generation. Rather than applying machine learning methods directly to the topology optimization problem, this chapter reformulates the problem itself—specifically, synchronous reluctance motor flux barrier design—as a sequential decision-making process.",
"Chapter 4 marks a methodological transition from supervised prediction tasks to optimization-driven design generation. Rather than applying machine learning methods directly to the topology optimization problem, this chapter reformulates the problem itself — a C-core actuator and synchronous reluctance motor flux barrier design — as a sequential decision-making process.")
sub(N, 2, "Validation through tree-search algorithms (Monte Carlo Tree Search) demonstrates that the sequential formulation inherently produces manufacturable topologies satisfying connectivity constraints without requiring penalty terms or filtering operations.",
"Tabular Q-learning and a genetic algorithm run on this environment with a FEMM solve at every step, and the chapter examines whether the sequential formulation produces connected, manufacturable topologies without penalty terms or filtering operations.")

sub(N, 2, "Policy gradient methods enable the agent to discover design strategies autonomously through trial-and-error experimentation, guided by reward functions that encode engineering objectives such as torque production or efficiency. The chapter demonstrates that reinforcement learning agents, despite being sample-inefficient during their training phase, can master the sequential design task and produce high-performing motor topologies without requiring human-curated training datasets.",
"Deep Q-networks, advantage actor-critic and proximal policy optimisation agents, implemented in PyTorch, discover design strategies through trial-and-error experimentation, guided by reward functions that encode engineering objectives such as force or torque production. The chapter trains these agents for a fixed budget of episodes with a FEMM solve at every step (the deep Q-network and actor-critic runs took 10.8 h and 6.2 h on a laptop) and compares what each reached against the Q-learning and genetic-algorithm baselines of Chapter 4, without any human-curated training dataset.")
sub(N, 2, "The supervised learning chapters demonstrate that convolutional and recurrent architectures can approximate complex physics with sufficient accuracy for engineering decision-making.",
"The supervised learning chapters show that convolutional and decoder architectures can approximate the field and performance maps of the problems studied with accuracies of 0.17% NMSE and R² up to 0.97.")
sub(N, 2, "Chapter 5 then demonstrates that reinforcement learning agents can master this sequential design process through interaction with the environment, learning which actions lead to high-performing designs without requiring labeled examples.",
"Chapter 5 then tests whether reinforcement learning agents can learn this sequential design process through interaction with the environment, which actions lead to high-performing designs, and at what cost in solver calls, without labeled examples.")

sub(N, 3, "Computational implementations throughout employ Python as the primary programming language, with PyTorch serving as the deep learning framework for neural network development and training. Electromagnetic simulations utilize FEMM (Finite Element Method Magnetics), an open-source 2D field solver. All neural network architectures and training algorithms are implemented from fundamental components to ensure transparency and educational value. The computational requirements remain modest—all models presented were trained on a single consumer-grade NVIDIA 1080 Ti GPU, demonstrating that sophisticated machine learning for electromagnetic design does not necessitate high-performance computing cluster access. Training durations range from hours (RNN performance predictors) to days (reinforcement learning agents with iterative environment interaction), all feasible on standard research-grade workstations.",
"Computational implementations throughout employ Python 3.12, with PyTorch 2.14 as the deep learning framework. Electromagnetic simulations use FEMM 5 (the open-source femm-qt build of Finite Element Method Magnetics), driven from Python on Linux. Network architectures and training loops are written in PyTorch from basic modules to ensure transparency and educational value. The computational requirements are modest: every result in this book was produced on one laptop — an Intel Core i7-11370H with 14 GB of memory and an NVIDIA GeForce MX450 with 2 GB — except one Chapter 2 training run on a free cloud T4 GPU. Training durations range from seconds (the force network of Chapter 2; 110 s for the PINN) through tens of minutes (36 min per U-Net) and hours (Chapter 3's twelve training runs, about 5 h in total) to the reinforcement-learning agents of Chapter 5, which spend most of their 6–11 h per run inside the finite element solver.")

for name, n in NBS.items():
    nbformat.write(n, B + name); print('written', name)

# Applied afterwards, inline (2026-10-11 23:45): the two sentences with "2-8 hours" + {cite}`FEA_market_report`
# in 01 cell 2 and 02 cell 2 were rewritten to quote the measured FEMM times and keep the industrial
# statement qualitative with its citation; the [VERIFY] HTML comments were removed with them.
