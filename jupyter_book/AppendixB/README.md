# Appendix B: Quantum Computing for Electromagnetic Design

Preliminary, standalone supplementary work for *Statistical Methods for Design
and Analysis of Electrical Machines*. This appendix follows the repository's
separate Jupyter Book structure. Chapters 1–5 and Appendix A are not required to
build it, and their tables of contents are unchanged.

Start with [scope and connections to the book](content/01_scope.md), then run
the [C-core reliability notebook](content/03_ccore_reliability.ipynb).
The [topology-optimization section](content/04_topology_optimization.md) is a
research proposal, not an implemented quantum optimizer.

## Run locally

From the repository root, using Python 3.10 or newer:

```bash
python -m venv .venv-quantum
source .venv-quantum/bin/activate
python -m pip install -r jupyter_book/AppendixB/requirements.txt
python jupyter_book/AppendixB/content/quantum_ccore.py
```

On Windows, activate with `.venv-quantum\Scripts\activate` instead.
Open the notebook in an editor with a Python notebook kernel. Its imports work
when launched from the repository root, the appendix directory, or `content/`.
The helper module uses only NumPy; plotting uses Matplotlib. No FEMM installation,
trained weights, quantum account, hardware subscription, or API key is needed.

## Build the appendix

```bash
jupyter-book build jupyter_book/AppendixB --warningiserror --keep-going
```

Open `jupyter_book/AppendixB/_build/html/index.html`. The saved notebook includes
executed results. Notebook execution is disabled during ordinary builds, matching
the existing appendix convention. To recompute the example and build it:

```bash
jupyter-book build jupyter_book/AppendixB --execute --all --warningiserror --keep-going
```

## What is implemented

- A synthetic, constant-permeability magnetic-circuit model of C-core force.
- Discrete input uncertainty and exact enumeration of the resulting probability.
- Explicit state preparation, Grover reflections, controlled powers, and inverse
  quantum Fourier transform in a small NumPy statevector simulator.
- Verification against an independent eigenphase formula and repeated-trial
  comparisons with classical sampling at matched logical oracle budgets.
- A separate estimate of input-grid discretization error.

This is **canonical QAE on a classical simulator**, not iterative QAE or a quantum
execution of FEMM. The dense response loader is deliberately educational. The
example makes no hardware-speedup or engineering-validation claim.
