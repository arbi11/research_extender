"""Rewrite the PINN claims in C2/06a and 06b after the 06b rebuild (radial inputs).
Usage: rewrite_06_prose.py <06b_executed.ipynb> <nmse_06b e.g. 0.26%> [nmse_06a=15.8%] [nmse_G=0.24%]
Edits markdown cells only (outputs untouched); writes the 06b copy in place of its path and 06a in place.
"""
import sys, nbformat
NB06B, NMSE06B = sys.argv[1], sys.argv[2]
NMSE06A = sys.argv[3] if len(sys.argv) > 3 else '15.8%'
NMSEG   = sys.argv[4] if len(sys.argv) > 4 else '0.24%'
P06A = '/home/akhan/code/compem_book/C2/06a_pinn_theory_formulation.ipynb'

def fill(s):
    return s.replace('@@NMSE06B@@', NMSE06B).replace('@@NMSE06A@@', NMSE06A).replace('@@NMSEG@@', NMSEG)

def set_md(nb, i, must, new):
    c = nb.cells[i]
    assert c.cell_type == 'markdown' and must in c.source, (i, c.source[:80])
    c.source = fill(new)

def sub_md(nb, i, old, new):
    c = nb.cells[i]
    assert c.cell_type == 'markdown' and c.source.count(old) == 1, (i, old[:60])
    c.source = c.source.replace(old, fill(new))

# ───────────────────────────── 06b ─────────────────────────────
nb = nbformat.read(NB06B, as_version=4)

set_md(nb, 0, '# Physics-Informed Neural Networks: Proper Implementation', '''# Physics-Informed Neural Networks: Proper Implementation

## From @@NMSE06A@@ to @@NMSE06B@@ NMSE on the FEMM Coil Dataset

Section 6a established that a vanilla, data-only PINN on the FEMM coil dataset plateaus at **@@NMSE06A@@ NMSE** due to five diagnosable failures.  This notebook implements a targeted fix for each one and reaches **@@NMSE06B@@ NMSE** on the same sample with the same 100 observations — below the 0.9% of the infinite-space analytic wire field, which ignores the walls of the FEMM box.

**Learning objectives**:
1. Understand why wire-scale nondimensionalisation balances loss terms
2. Implement a hard Dirichlet BC via a distance-function multiplier
3. Give the network the radial coordinates the solution is built from — and see why random Fourier features overfit sparse data instead
4. Add targeted collocation near the wire to enforce the Poisson source term
5. Compare the full training curve and field reconstruction against the vanilla baseline

:::{seealso}
**Prerequisites**:
- Section 6a: Theory, Maxwell's equations, vector potential formulation, vanilla PINN baseline
:::
''')

set_md(nb, 1, '## The Five Fixes at a Glance', '''## The Five Fixes at a Glance

Before diving into code, here is a map from failure to fix:

| # | Failure (Section 6a) | Fix (this notebook) |
|---|---------------------|---------------------|
| 1 | Loss scale mismatch ($10^{5}$–$10^{10}$ gap for a naive residual) | Wire-scale nondim → data loss $O(0.01)$, physics loss $O(1)$ at init, $\\lambda = 0.02$ |
| 2 | Wrong physics equation (Laplace, not Poisson) | Correct residual: $\\nabla^2_{\\tilde{x}}\\tilde{u}+2\\cdot\\mathbf{1}_{\\text{in}}=0$ |
| 3 | Gauge underdetermination (no BC) | Hard Dirichlet BC via distance-function multiplier |
| 4 | Representation limit: a tanh MLP on $(x/500, y/500)$ cannot resolve $\\ln r$ at a 14 mm wire; wire-scale Fourier features overfit the 100 points | Radial inputs $r/500$ and $\\ln((r+1\\,\\text{mm})/500)$ — the 2-D Green's-function coordinate |
| 5 | Sparse wire sampling (0 of the 1000 uniform collocation points inside this wire) | +200 inside-wire +300 near-wire collocation pts/epoch |

We implement them cumulatively in a single `ProperPINN` class and a `ProperPINNTrainer`.
''')

set_md(nb, 3, '## Fix 2: Hard Dirichlet BC + Fourier Features', '''## Fixes 3 and 4: Hard Dirichlet BC + Radial Input Features

### Hard Boundary Condition

Instead of a soft BC loss, we multiply the network output by a distance function that is **exactly zero at the domain walls**:

$$A_z(x,y) = D(x,y) \\times f_\\theta(x,y)$$

$$D(x,y) = \\text{relu}\\!\\left(1 - \\left(\\tfrac{x}{500}\\right)^2\\right) \\cdot \\text{relu}\\!\\left(1 - \\left(\\tfrac{y}{500}\\right)^2\\right)$$

This enforces $A_z = 0$ at $|x|=|y|=500$ mm **exactly**, with zero training cost — no BC loss term, no $\\lambda_{\\text{BC}}$ to tune.  It also makes the prediction exactly zero outside the box, which is what the FEMM solution is on the ±600 mm validation grid.

### Radial Input Features

Outside the wire the exact solution is $A_z \\propto -\\ln r$ plus a smooth correction from the wall images; inside it is quadratic in $r$.  A tanh MLP that only sees $(x/500,\\,y/500)$ has to build this $1/r$ structure at 14 mm out of smooth ridge functions spanning a 1000 mm box, and it cannot: with the correct, well-scaled physics loss it stalls at 5.9% NMSE with a physics residual of about 0.02 whatever $\\lambda$ is (Section 6a, cause 4).  Random Fourier features at wire scale do not help either — on 100 sparse observations they overfit the points and the field in between is wrong (159% NMSE data-only, 55% with physics).

The network therefore receives four coordinate inputs:

| Input | Role |
|-------|------|
| $x/500$, $y/500$ | Position in the box; the hard-BC multiplier is built from them |
| $r/500$ | The wire field depends on $r$ only (inside the wire $A_z$ is quadratic in $r$) |
| $\\ln\\!\\big((r + 1\\,\\text{mm})/500\\big)$ | The 2-D Green's-function coordinate: $A_z \\propto -\\ln r$ outside the wire; the +1 mm keeps it finite on the axis |

With these the network only has to learn a smooth correction, and the physics weight stops mattering (0.24% at $\\lambda = 0.02$, 0.28% at $\\lambda = 1$ in the ablation).  This is plain feature engineering: give the network the coordinates its target is built from.
''')

set_md(nb, 9, '## Quantitative Results', '''## Quantitative Results

### Observed Performance (this notebook, sample 0, 100 observations)

| Implementation | Epochs | NMSE (±600 mm grid) | Key missing piece |
|---------------|--------|------|------------------|
| Vanilla, data-only (Section 6a) | 3000 | @@NMSE06A@@ | Physics, BC, radial inputs, near-wire collocation |
| **ProperPINN (this notebook)** | **3000** | **@@NMSE06B@@** | — all fixes applied |
| Analytic wire $\\mu_0 I / 2\\pi r$, zero outside the box | — | 0.9% | Image currents of the $A_z = 0$ walls |

The hard BC makes the prediction exactly zero outside the ±500 mm box, like the FEMM padding, so the NMSE on the ±600 mm grid equals the NMSE inside the box.

### What the Improvement Comes From — Measured

Each ingredient was switched on and off separately with the same losses, schedule and seed (sample 0, CPU, 3000 epochs unless noted; the ablation network takes the four coordinate inputs only, this notebook adds the two parameter inputs, constant for a single sample):

| Network inputs | Hard BC | $\\lambda_{\\text{physics}}$ | NMSE |
|---|---|---|---|
| $x/500, y/500$ (vanilla, Section 6a) | no | 0 | @@NMSE06A@@ |
| $x/500, y/500$ | yes | 0 | 11.9% |
| $x/500, y/500$ | yes | 0.02 | 5.9% |
| $x/500, y/500$ | yes | 1 | 7.7% |
| $x/500, y/500$ | yes | 10 | 22.1% |
| 32 random Fourier features (domain + wire scale) | yes | 0 (1500 epochs) | 159% |
| 32 random Fourier features (domain + wire scale) | yes | 0.02 | 55.3% |
| **$x/500, y/500, r/500, \\ln r$** | **yes** | **0.02** | **@@NMSEG@@** |
| $x/500, y/500, r/500, \\ln r$ | yes | 1 | 0.28% |

Reading the table:

1. **Hard BC** alone takes the data-only error from @@NMSE06A@@ to 11.9%: the gauge and the far field are fixed.
2. **Physics** helps the plain network only to 5.9%, and more weight makes it worse: its physics residual stays at 0.015–0.03 whatever $\\lambda$, so the network cannot represent the Poisson solution at the wire — a representation limit, not a weighting problem.
3. **Random Fourier features** overfit the 100 points: data loss $10^{-6}$, field wrong elsewhere.  The physics term only partly repairs this (159% → 55%).
4. **Radial inputs** remove the representation limit: 0.24% at $\\lambda = 0.02$ and 0.28% at $\\lambda = 1$ — the result no longer depends on the weight.
5. **Near-wire collocation** is what lets the physics term act at all: the 1000 uniform collocation points contain no point inside this wire.

### Comparison with Published Literature {cite}`raissi2019physics,karniadakis2021physics`

Published PINN results for magnetostatics typically report 1–5% NMSE with carefully normalised losses, hard or analytically enforced BCs and 3000+ epochs — usually with denser observations than the 100 used here.  The sub-1% figure above comes from the radial inputs: the network is handed the $\\ln r$ structure of the 2-D Green's function instead of having to discover it.

### Practical Guidance

For production use:
1. **$\\lambda_{\\text{physics}}$**: with the wire-scale residual, 0.02 and 1 give the same result — verify at init that $\\mathcal{L}_{\\text{data}} \\approx \\mathcal{L}_{\\text{physics}} \\times \\lambda$ and stop tuning it.
2. **Input features**: before adding Fourier features, ask which coordinates the solution depends on ($r$, $\\ln r$, distance to an interface) and feed those; with sparse data, high-frequency features overfit.
3. **Multi-sample training** requires conditioning on $(I, r_w)$ — the PINN base class already supports this (the two parameters are inputs alongside the four coordinate features).
''')

set_md(nb, 10, '## Summary: From Vanilla to Proper', '''## Summary: From Vanilla to Proper

This notebook demonstrated that the gap between **@@NMSE06A@@ and @@NMSE06B@@ NMSE** on the FEMM coil dataset is entirely explained by implementation choices, not by fundamental limitations of PINNs.

### Key Achievements

**1. Wire-scale nondimensionalisation** {cite}`raissi2019physics,karniadakis2021physics`:
- Residual $\\nabla^2_{\\tilde{x}}\\tilde{u}+2\\cdot\\mathbf{1}_{\\text{inside}}=0$ is $O(1)$ for any $(I, r_w)$
- Data and physics losses are $O(10^{-2})$ and $O(1)$ at init → $\\lambda=0.02$ without hyperparameter search ($\\lambda = 1$ works as well)

**2. Hard Dirichlet BC**:
- $A_z = D(x,y) \\times f_\\theta(x,y)$ eliminates gauge ambiguity at zero cost: @@NMSE06A@@ → 11.9% on its own
- The prediction is exactly zero outside the box, like the FEMM solution

**3. Radial input features + near-wire collocation**:
- $r/500$ and $\\ln r$ give the network the Green's-function structure it cannot build from $(x, y)$ alone: 5.9% → 0.24%
- 500 extra collocation points near/inside the wire guarantee a physics-residual signal where $J_z \\neq 0$
- Random Fourier features at wire scale were tried and rejected: they overfit 100 sparse observations (55–159% NMSE)

**4. Stable training**:
- Cosine LR schedule 1e-3 → 1e-5 prevents oscillation
- Gradient clipping (max_norm=1) prevents explosion near wire

### PINN Design Checklist

For any new electromagnetic PINN problem:
- [ ] Nondimensionalise: make physics residual $O(1)$ at init
- [ ] Check $\\mathcal{L}_{\\text{data}} / \\mathcal{L}_{\\text{physics}}$ at epoch 0 — set $\\lambda$ accordingly
- [ ] Enforce BCs hard if possible (distance function or level-set)
- [ ] Feed the network the coordinates the solution depends on ($r$, $\\ln r$, interface distances) before reaching for Fourier features; with sparse data, high-frequency features overfit
- [ ] Oversample regions where the PDE source term is non-zero
- [ ] Use LR decay + gradient clipping for deep networks
- [ ] Compare against the cheapest physical reference available (here the analytic wire field: 0.9%)

:::{seealso}
**Chapter 2 completion**:
- Section 6a: Theory, governing equations, vanilla PINN baseline
- Section 6b (this notebook): Proper implementation at @@NMSE06B@@ NMSE
- Chapter 3: Apply to performance map prediction and motor design optimisation
:::''')

nbformat.write(nb, NB06B)
print('06b markdown cells 0,1,3,9,10 rewritten ->', NB06B)

# ───────────────────────────── 06a ─────────────────────────────
nb = nbformat.read(P06A, as_version=4)

sub_md(nb, 2, '''- Wire-scale nondimensionalization, hard Dirichlet BC, Fourier features
- Proper training schedule and near-wire collocation
- Field prediction achieving **~7% NMSE** on the FEMM coil dataset''',
'''- Wire-scale nondimensionalization, hard Dirichlet BC, radial input features
- Proper training schedule and near-wire collocation
- Field prediction at **@@NMSE06B@@ NMSE** on the FEMM coil dataset (vanilla baseline of this notebook: @@NMSE06A@@)''')

sub_md(nb, 5, '''### Why This Problem Challenges a Naïve PINN

1. All 100 observations lie **far from the wire** ($r > 98$ mm, $r_w \\approx 32$ mm) — the near-field must be inferred from physics, not data.
2. $J_z$ is stored in **A/mm²** (not SI A/m²) — a factor-of-$10^6$ unit trap.
3. The field varies on **two scales**: slow decay at domain scale (±500 mm) and rapid variation at wire scale ($r_w \\approx 32$ mm).
4. $A_z$ is determined only up to a **gauge constant** without explicit boundary conditions.''',
'''### Why This Problem Challenges a Naïve PINN

Sample 0, used throughout Sections 6a and 6b: $I = 66.8$ A, $r_w = 14.4$ mm.

1. The observations are **sparse where the field is**: only 8 of the 100 points lie within 100 mm of the wire (1 inside it, 4 within $3r_w$), yet that disc holds 56% of the field energy on 3% of the box area — the near field must come from physics, not data.
2. $J_z$ is stored in **A/mm²** (not SI A/m²) — a factor-of-$10^6$ unit trap.
3. The field varies on **two scales**: the $1/r$ decay across the ±500 mm box and the linear rise inside the 14 mm wire — a 35:1 ratio of scales.
4. $A_z$ is determined only up to a **gauge constant** without explicit boundary conditions.
5. The validation grid is **±600 mm** while the FEMM box ends at ±500 mm (walls at $A_z = 0$); the 31% of grid points outside the box are zero, so a predictor must return zero there too (Section 6b does this exactly through its hard boundary condition).''')

sub_md(nb, 9, '''Without physics constraints, the network must extrapolate the near-wire field
($r < 98$ mm) from 100 observations all located at $r > 98$ mm.
This is an **ill-posed extrapolation problem** — physics is what makes it tractable.''',
'''Without physics constraints, the network must reconstruct the near-wire field — 56% of the
field energy lies inside $r < 100$ mm — from the 8 observations that fall there.
This is an **ill-posed interpolation problem** — physics is what makes it tractable.''')

set_md(nb, 13, '## Why the Vanilla PINN Fails: 5 Root Causes', '''## Why the Vanilla PINN Fails: 5 Root Causes

The @@NMSE06A@@ NMSE is not noise.  The baseline is data-only, and its error map above shows where the data run out: the near-wire field.  Section 6b removes the error in five steps; where a step was measured on its own (sample 0, 3000 epochs, the losses and schedule of Section 6b), the number is given so that nothing has to be taken on faith.

---

### 1 · Loss Scale Mismatch

The vanilla loss has no PDE term, and a naive one cannot simply be added.  With the network output in T·mm, the residual $\\nabla^2 A_z + \\mu_0 J_z$ in mm/SI units is $O(10^{-7})$ at initialisation against a data loss of $O(10^{-2})$:

| Term | Magnitude at init | Effect |
|------|------------------|--------|
| $\\mathcal{L}_{\\text{data}}$ (normalised by $B_{\\text{ref}}$) | $\\sim 10^{-2}$ | Gradient drives weight updates |
| $\\mathcal{L}_{\\text{physics}}$ naive ($\\nabla^2 A_z$ in mm, SI source) | $\\sim 10^{-7}$ | Invisible to optimiser — needs $\\lambda \\approx 10^{5}$–$10^{10}$ |

**Fix**: Wire-scale nondimensionalisation makes the data loss $O(10^{-2})$ and the physics loss $O(1)$ at init, so $\\lambda = 0.02$ balances them (and $\\lambda = 1$ works too).

---

### 2 · Wrong Physics Equation

Rescaling the raw Laplacian by a large constant does NOT yield the Poisson source term.  The source $\\mu_0 J_z^{\\text{SI}}$ is $O(10^{-6})$, swamped by the rescaled Laplacian, and the optimiser satisfies $\\nabla^2 A_z \\approx 0$ (Laplace) instead of $\\nabla^2 A_z = -\\mu_0 J_z$.

**Fix**: Wire-scale residual $\\nabla^2_{\\tilde{x}}\\,\\tilde{u} + 2\\cdot\\mathbf{1}_{\\text{inside}} = 0$ — both terms are $O(1)$ for any $(I, r_w)$.

---

### 3 · Gauge Underdetermination

$A_z + C$ gives the same B-field for any constant $C$, and the plain network has no boundary condition to fix the gauge or the far field.

**Fix**: Hard Dirichlet BC — multiply the network output by $D(x,y) = \\text{relu}(1-(x/500)^2)\\cdot\\text{relu}(1-(y/500)^2)$, forcing $A_z = 0$ at the domain walls **exactly**, at zero training cost.  **Measured**: data-only @@NMSE06A@@ → 11.9% with the hard BC alone.

---

### 4 · Representation: a tanh MLP on $(x/500, y/500)$ Cannot Resolve the Wire

The solution is $\\ln r$ outside a 14 mm wire in a 1000 mm box.  A tanh network fed only the Cartesian box coordinates must build the $1/r$ singularity from smooth ridge functions.  With the correct, well-scaled physics loss added it reaches 5.9% ($\\lambda = 0.02$), 7.7% ($\\lambda = 1$) and 22% ($\\lambda = 10$), and its physics residual stays at 0.015–0.03 whatever the weight: the limit is what the network can represent, not how the losses are weighted.

The usual remedy, random Fourier features at wire scale, makes it worse here: with 100 sparse observations the high-frequency features fit the points (data loss $10^{-6}$) and the field everywhere else is wrong — 159% NMSE data-only (1500 epochs), 55% with physics.  This is the overfitting regime of Fourier-feature networks on sparse data (Tancik et al. 2020; Wang, Wang & Perdikaris 2021), the opposite of the spectral-bias problem the features are meant to cure.

**Fix**: Give the network the coordinates the solution is built from — $r/500$ and $\\ln((r+1\\,\\text{mm})/500)$, the 2-D Green's-function coordinate — alongside $x/500$ and $y/500$.  **Measured**: 5.9% → @@NMSEG@@, and the physics weight stops mattering (0.28% at $\\lambda = 1$).

---

### 5 · No Collocation Inside or Near the Wire

The 1000 uniform collocation points put **none** inside this wire (expected 0.65 for $r_w = 14.4$ mm) and 3 within $3r_w$.  The Poisson source $J_z \\neq 0$ only inside — so the inhomogeneous term receives no gradient signal.

**Fix**: Add 200 inside-wire + 300 near-wire ($r_w < r < 3r_w$) extra collocation points every epoch.

---

### Training Schedule

A constant learning rate oscillates around the minimum once the data loss is $10^{-4}$, and the second-derivative loss near the wire produces occasional large gradients.  Section 6b uses cosine annealing $10^{-3} \\to 10^{-5}$ and gradient clipping at norm 1.

---

### The Fix in One Line

$$\\text{Section 6b} \\;:\\; @@NMSE06A@@ \\;\\xrightarrow{\\text{5 fixes}}\\; @@NMSE06B@@ \\text{ NMSE}$$

For reference, the infinite-space analytic wire field $B = \\mu_0 I/(2\\pi r)$ — exact physics, no walls — scores 0.9% inside the box because it ignores the image currents of the $A_z = 0$ walls.  The trained PINN does better than that.

:::{seealso}
Section 6b implements all five fixes on the same FEMM coil dataset.
:::''')

nbformat.write(nb, P06A)
print('06a markdown cells 2,5,9,13 rewritten ->', P06A)
