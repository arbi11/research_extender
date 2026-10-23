# Quantum concepts through an electromagnetic problem

Consider a fixed actuator with uncertain supply current $I$ and total effective
air gap $g$. Let $F(I,g)$ be its force. We want either an expectation
$\mathbb{E}[F]$ or the probability

$$
a=\Pr\{F(I,g)<F_{\min}\}.
$$

This question gives each quantum concept a concrete role.

## Qubits, amplitudes, and measurement

A qubit has state $\alpha|0\rangle+\beta|1\rangle$, where
$|\alpha|^2+|\beta|^2=1$. A measurement yields one outcome, with probabilities
$|\alpha|^2$ and $|\beta|^2$. Multiple qubits can index the input grid: two
qubits for current and two for gap represent $4\times4=16$ input combinations.
They do not provide all sixteen field or force values in a single measurement.

For a finite input distribution, prepare $\sum_x\sqrt{p_x}|x\rangle$.
Here $x$ indexes a current-gap pair. The example uses independent uniforms and
equal-probability midpoint bins. More general input distributions require a
different loader, whose implementation cost must be included.

## Encoding the engineering quantity

For a response $f(x)$ in $[0,1]$, define a preparation operator $A$ by

$$
A|0\rangle=|\psi\rangle
=\sum_x\sqrt{p_x}|x\rangle
\left[\sqrt{1-f(x)}|0\rangle+\sqrt{f(x)}|1\rangle\right].
$$

The last qubit is the objective qubit. Its probability of being one is

$$
a=\sum_xp_xf(x)=\mathbb{E}[f(X)].
$$

For failure probability, choose $f(x)=\mathbf{1}[F(x)<F_{\min}]$. For expected
force with known bounds $F_{\mathrm{lo}}\le F(x)\le F_{\mathrm{hi}}$, choose
$f(x)=(F(x)-F_{\mathrm{lo}})/(F_{\mathrm{hi}}-F_{\mathrm{lo}})$ and recover
$\mathbb{E}[F]=F_{\mathrm{lo}}+(F_{\mathrm{hi}}-F_{\mathrm{lo}})a$.
Bounds must cover the modeled input domain; clipping otherwise changes the target.

The worked example constructs this vector directly and uses a dense Householder
matrix for $A$. This makes the algebra inspectable, but is not an efficient
general-purpose state loader. Arbitrary state preparation can erase a proposed
advantage {cite}`vazquez2021`.

## Interference and amplitude amplification

Let $\Pi_{\mathrm{good}}$ project onto objective-qubit outcome one. Define

$$
S_{\mathrm{good}}=I-2\Pi_{\mathrm{good}},\qquad
D=2|\psi\rangle\langle\psi|-I,\qquad Q=DS_{\mathrm{good}}.
$$

The two reflections produce a rotation in the subspace associated with good and
bad outcomes. Write $a=\sin^2\theta$. The relevant eigenvalues of $Q$ are
$e^{\pm2i\theta}$. Interference lets a circuit extract this angle through
controlled applications of $Q$, rather than simply repeat independent samples.
The reflection $D$ can be written $A(2|0\rangle\langle0|-I)A^\dagger$;
this is why the cost includes both $A$ and its inverse.

## Phase estimation and the inverse quantum Fourier transform

With $m$ phase qubits and $M=2^m$, canonical QAE starts the phase register in
$M^{-1/2}\sum_{k=0}^{M-1}|k\rangle$. Controlled powers create

$$
\frac{1}{\sqrt M}\sum_{k=0}^{M-1}|k\rangle Q^k|\psi\rangle.
$$

Apply an inverse quantum Fourier transform to the phase register and measure an
integer $y$. Convert it to $\hat a=\sin^2(\pi y/M)$. Both $y$ and $M-y$
give the same estimate. The notebook simulates the controlled powers using NumPy
and the inverse transform using an FFT. The FFT is a classical simulation
technique; it is not a claim that a classical FFT produces quantum acceleration.

Canonical QAE is used here because its operations are explicit. Iterative QAE,
which avoids the phase register and full phase-estimation circuit, is a later
learning milestone {cite}`grinko2021`. The current code does not implement it.

## Error and cost accounting

For bounded responses, ordinary Monte Carlo has root-mean-square error
$O(N^{-1/2})$. Ideal QAE can achieve additive error $\epsilon$ with
$O(\epsilon^{-1})$ coherent queries at fixed success probability, compared with
$O(\epsilon^{-2})$ classical samples in the generic Monte Carlo setting.
Confidence amplification adds overhead. These are oracle-model comparisons;
quadrature or quasi-Monte Carlo can be much better than ordinary sampling on
structured problems {cite}`brassard2002,montanaro2015`.

For one execution with $m$ phase qubits, the controlled powers use $M-1$ logical
applications of $Q$. Counting $A$ and $A^\dagger$ separately gives
$1+2(M-1)$ preparation-or-inverse calls. Multiply by the number of measurement
shots. Counting only the number of shots would hide the amplification work.
The notebook uses nine shots and a folded modal estimate, with empirical error
over repeated trials; it supplies no formal confidence interval for that
aggregation rule.

Track four different errors: the simplified physics model, input discretization,
statistical/phase estimation, and hardware noise. Reducing estimation error does
not correct a biased force model, uncertain material data, an inaccurate surrogate,
or a coarse uncertainty grid.
