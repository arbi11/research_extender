"""Small, explicit statevector model for the quantum-computing appendix.

This is a classical simulation of canonical quantum amplitude estimation (QAE).
Dense state preparation and the response table are educational constructions,
not scalable quantum implementations of FEMM or the book's neural networks.
Only NumPy is required. Run this file to print the reproducible example.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MagneticCircuit:
    """Illustrative single-effective-gap, constant-permeability magnetic circuit.

    gap_m is the TOTAL effective air-gap length. area_m2 is the effective pole
    area. This is not a calibrated replica of the thesis's two-coil geometry.
    """

    turns: int = 500
    area_m2: float = 1.0e-4
    core_length_m: float = 0.10
    relative_permeability: float = 1500.0

    def __post_init__(self):
        parameters = (self.turns, self.area_m2, self.core_length_m,
                      self.relative_permeability)
        if not all(np.isfinite(x) and x > 0 for x in parameters):
            raise ValueError("Magnetic-circuit parameters must be finite and positive.")

    def field_and_force(self, current_a, gap_m):
        current_a, gap_m = np.broadcast_arrays(
            np.asarray(current_a, dtype=float), np.asarray(gap_m, dtype=float)
        )
        if (not np.all(np.isfinite(current_a)) or
                not np.all(np.isfinite(gap_m)) or np.any(gap_m <= 0)):
            raise ValueError("Current must be finite; total gap must be finite and positive.")
        mu0 = 4.0e-7 * np.pi
        effective_length = gap_m + self.core_length_m / self.relative_permeability
        field_t = mu0 * self.turns * current_a / effective_length
        force_n = self.area_m2 * field_t**2 / (2.0 * mu0)
        return field_t, force_n


def midpoint_grid(size, lower, upper):
    """Equal-probability midpoint discretization of a uniform random variable."""
    if not isinstance(size, int) or size < 1 or not lower < upper:
        raise ValueError("Use a positive integer size and increasing bounds.")
    return lower + (np.arange(size) + 0.5) * (upper - lower) / size


def reliability_problem(points_per_input=4, minimum_force_n=14.0):
    """Synthetic independent current/gap uncertainty; not measured tolerances.

    I ~ Uniform(0.8, 1.2) A, g_total ~ Uniform(0.8, 1.2) mm.
    Grid size is a power of two so the input states map to whole qubits.
    """
    n = points_per_input
    if not isinstance(n, int) or n < 2 or n & (n - 1):
        raise ValueError("points_per_input must be a power of two, at least two.")
    if not np.isfinite(minimum_force_n):
        raise ValueError("minimum_force_n must be finite.")
    currents = midpoint_grid(n, 0.8, 1.2)
    gaps = midpoint_grid(n, 0.8e-3, 1.2e-3)
    current_mesh, gap_mesh = np.meshgrid(currents, gaps, indexing="ij")
    fields, forces = MagneticCircuit().field_and_force(current_mesh, gap_mesh)
    return {
        "currents_a": currents,
        "gaps_m": gaps,
        "fields_t": fields,
        "forces_n": forces,
        "probabilities": np.full(n * n, 1.0 / (n * n)),
        "failure_values": (forces.ravel() < minimum_force_n).astype(float),
        "minimum_force_n": minimum_force_n,
    }


def encoded_state(probabilities, response):
    """Return sum_x sqrt(p_x)|x>(sqrt(1-f_x)|0>+sqrt(f_x)|1>).

    The objective qubit is the least significant bit. f must lie in [0,1].
    An indicator f estimates a failure probability; a normalized force f
    estimates an expectation. All response entries are computed classically.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    response = np.asarray(response, dtype=float)
    if (probabilities.ndim != 1 or response.shape != probabilities.shape or
            len(probabilities) == 0 or len(probabilities) & (len(probabilities) - 1)):
        raise ValueError("Use matching one-dimensional arrays of power-of-two length.")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or
            not np.isclose(probabilities.sum(), 1.0)):
        raise ValueError("Probabilities must be finite, nonnegative, and sum to one.")
    if (not np.all(np.isfinite(response)) or
            np.any(response < 0) or np.any(response > 1)):
        raise ValueError("The encoded response must be finite and lie in [0,1].")
    state = np.empty(2 * len(probabilities))
    state[0::2] = np.sqrt(probabilities * (1.0 - response))
    state[1::2] = np.sqrt(probabilities * response)
    return state


def preparation_unitary(state):
    """Dense Householder A with A|0>=state, for inspecting the circuit algebra.

    This arbitrary-state loader is intentionally not presented as efficient.
    """
    state = np.asarray(state, dtype=float)
    if state.ndim != 1 or not np.isclose(np.linalg.norm(state), 1.0):
        raise ValueError("Expected a normalized real statevector.")
    zero = np.zeros_like(state)
    zero[0] = 1.0
    difference = zero - state
    norm = np.linalg.norm(difference)
    identity = np.eye(len(state))
    if norm < 1e-14:
        return identity
    direction = difference / norm
    return identity - 2.0 * np.outer(direction, direction)


def grover_operator(state):
    """Q=(2|psi><psi|-I)(I-2 Pi_good), good = objective qubit one.

    Equivalently Q=A(2|0><0|-I)A^dagger S_good. Its relevant eigenphases
    are +/-2 theta, where sin(theta)^2 is the encoded expectation.
    """
    state = np.asarray(state, dtype=float)
    # Reuse the state-preparation validation, keeping the small example explicit.
    preparation_unitary(state)
    good_reflection = np.ones(len(state))
    good_reflection[1::2] = -1.0
    reflection_about_state = 2.0 * np.outer(state, state) - np.eye(len(state))
    return reflection_about_state * good_reflection[np.newaxis, :]


def qae_distribution(state, phase_qubits):
    """Simulate H gates, controlled Q powers, and the inverse QFT explicitly.

    Rows index the phase register, columns the input/objective register. Dense
    matrix squaring and an FFT simulate the circuit; their classical runtime
    is NOT a quantum-resource estimate. Limited to small educational circuits.
    """
    if not isinstance(phase_qubits, int) or not 1 <= phase_qubits <= 10:
        raise ValueError("Use 1 to 10 phase qubits for this small simulator.")
    state = np.asarray(state, dtype=float)
    power = grover_operator(state).astype(complex)
    phase_states = 2**phase_qubits
    joint = np.tile(state, (phase_states, 1)).astype(complex) / np.sqrt(phase_states)
    rows = np.arange(phase_states)
    for bit in range(phase_qubits):
        controlled_rows = (rows & (1 << bit)) != 0
        joint[controlled_rows] = joint[controlled_rows] @ power.T
        power = power @ power
    joint = np.fft.fft(joint, axis=0) / np.sqrt(phase_states)
    probabilities = np.sum(np.abs(joint)**2, axis=1)
    # Remove tiny floating-point drift before using a random-number generator.
    return probabilities / probabilities.sum()


def analytic_qae_distribution(amplitude, phase_qubits):
    """Independent eigenphase formula used to verify the circuit simulation."""
    if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("Amplitude probability must lie in [0,1].")
    size = 2**phase_qubits
    theta = np.arcsin(np.sqrt(amplitude))
    phases = (theta / np.pi, (-theta / np.pi) % 1.0)
    outcomes = np.arange(size)
    distribution = np.zeros(size)
    for phase in phases:
        delta = phase - outcomes / size
        # sinc ratio is the finite geometric sum, stable at delta=0.
        kernel = np.sinc(size * delta) / np.sinc(delta)
        distribution += 0.5 * kernel**2
    return distribution / distribution.sum()


def estimate_from_counts(counts):
    """Fold conjugate phase outcomes y and M-y, then choose their modal bin.

    Fixed-shot modal aggregation is a teaching choice, not IQAE or a supplied
    confidence interval. Repeated trials below expose its finite-sample error.
    """
    counts = np.asarray(counts)
    size = len(counts)
    if (counts.ndim != 1 or size < 2 or size & (size - 1) or
            np.any(counts < 0) or not np.all(np.isfinite(counts)) or counts.sum() <= 0):
        raise ValueError("Expected nonnegative nonempty power-of-two phase counts.")
    folded = counts[:size // 2 + 1].astype(float).copy()
    for outcome in range(1, size // 2):
        folded[outcome] += counts[size - outcome]
    best = np.argmax(folded)
    return float(np.sin(np.pi * best / size)**2)


def resource_counts(phase_qubits, shots):
    """Logical oracle accounting only; excludes the table and gate synthesis."""
    grover_calls = shots * (2**phase_qubits - 1)
    return {
        "shots": shots,
        "grover_calls": grover_calls,
        "preparation_calls_A_or_Adagger": 2 * grover_calls + shots,
    }


def benchmark(state, seed=2026, trials=200, shots=9, phase_sizes=(2, 3, 4, 5, 6)):
    """Compare repeated-trial RMSE at matched A/A-dagger query budgets.

    Classical Bernoulli sampling counts one response query per sample. QAE
    counts initial A and both A/A-dagger in every Q. This does not equate gate
    costs or demonstrate wall-clock advantage; both methods run classically.
    """
    rng = np.random.default_rng(seed)
    truth = float(np.sum(np.asarray(state)[1::2]**2))
    results = []
    for phase_qubits in phase_sizes:
        resources = resource_counts(phase_qubits, shots)
        budget = resources["preparation_calls_A_or_Adagger"]
        distribution = qae_distribution(state, phase_qubits)
        histograms = rng.multinomial(shots, distribution, size=trials)
        quantum_estimates = np.array([estimate_from_counts(c) for c in histograms])
        classical_estimates = rng.binomial(budget, truth, size=trials) / budget
        results.append({
            "phase_qubits": phase_qubits,
            **resources,
            "qae_rmse": float(np.sqrt(np.mean((quantum_estimates - truth)**2))),
            "mc_rmse": float(np.sqrt(np.mean((classical_estimates - truth)**2))),
        })
    return results


if __name__ == "__main__":
    problem = reliability_problem()
    state = encoded_state(problem["probabilities"], problem["failure_values"])
    print(f"Exact probability on the 4 x 4 grid: {np.sum(state[1::2]**2):.6f}")
    print("phase bits | A/A-dagger calls | QAE RMSE | MC RMSE")
    for row in benchmark(state):
        print(f"{row['phase_qubits']:10d} | "
              f"{row['preparation_calls_A_or_Adagger']:16d} | "
              f"{row['qae_rmse']:.6f} | {row['mc_rmse']:.6f}")
