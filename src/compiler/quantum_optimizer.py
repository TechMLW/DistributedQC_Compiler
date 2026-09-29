import numpy as np
from scipy.optimize import minimize

from qiskit.circuit.library import qaoa_ansatz
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit.quantum_info import SparsePauliOp

from compiler.route_optimization import RouteSolution


def qubo_to_ising(problem):

    size = problem.num_variables

    linear = np.array(problem.linear, dtype=float)
    quadratic = np.zeros((size, size))

    for (i, j), coefficient in problem.quadratic.items():
        quadratic[i, j] += coefficient

    offset = 0.0

    for indices in problem.variable_groups().values():

        penalty = problem.onehot_penalty
        offset += penalty

        for index in indices:
            linear[index] -= penalty

        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                quadratic[indices[a], indices[b]] += 2.0 * penalty

    h = np.zeros(size)
    couplings = np.zeros((size, size))

    for index in range(size):
        offset += linear[index] / 2.0
        h[index] -= linear[index] / 2.0

    for i in range(size):
        for j in range(i + 1, size):

            coefficient = quadratic[i, j]

            if coefficient == 0.0:
                continue

            offset += coefficient / 4.0
            h[i] -= coefficient / 4.0
            h[j] -= coefficient / 4.0
            couplings[i, j] += coefficient / 4.0

    return h, couplings, offset


def ising_to_operator(h, couplings):

    size = len(h)
    terms = []

    for index in range(size):

        if abs(h[index]) < 1e-12:
            continue

        label = ["I"] * size
        label[size - 1 - index] = "Z"
        terms.append(("".join(label), h[index]))

    for i in range(size):
        for j in range(i + 1, size):

            if abs(couplings[i, j]) < 1e-12:
                continue

            label = ["I"] * size
            label[size - 1 - i] = "Z"
            label[size - 1 - j] = "Z"
            terms.append(("".join(label), couplings[i, j]))

    if not terms:
        terms.append(("I" * size, 0.0))

    return SparsePauliOp.from_list(terms)


class QAOARouteSolver:

    def __init__(self, reps, max_qubits, maxiter, shots):

        self.reps = reps
        self.max_qubits = max_qubits
        self.maxiter = maxiter
        self.shots = shots

    def can_solve(self, problem):
        return 0 < problem.num_variables <= self.max_qubits

    def solve(self, problem, run_seed):

        if not self.can_solve(problem):
            return RouteSolution(
                {}, float("inf"), "qaoa_not_applicable", False,
                details={"num_variables": problem.num_variables},
            )

        h, couplings, offset = qubo_to_ising(problem)
        operator = ising_to_operator(h, couplings)

        ansatz = qaoa_ansatz(operator, reps=self.reps).decompose(reps=3)

        estimator = StatevectorEstimator()
        evaluations = {"count": 0}

        def expectation(parameters):

            evaluations["count"] += 1
            result = estimator.run([(ansatz, operator, [parameters])]).result()[0]

            return float(np.real(result.data.evs[0]))

        rng = np.random.default_rng(run_seed)
        initial = rng.uniform(0.0, np.pi, ansatz.num_parameters)

        optimization = minimize(
            expectation, initial, method="COBYLA", options={"maxiter": self.maxiter}
        )

        bound = ansatz.assign_parameters(optimization.x)
        bound.measure_all()

        sampler = StatevectorSampler(seed=run_seed)
        counts = sampler.run([bound], shots=self.shots).result()[0].data.meas.get_counts()

        scored = []

        for bitstring, occurrences in counts.items():
            bits = [int(character) for character in reversed(bitstring)]
            scored.append((problem.energy(bits), bitstring, bits, occurrences))

        scored.sort(key=lambda item: (item[0], item[1]))

        onehot_samples = 0
        feasible_samples = 0
        best_selection = None

        for _, _, bits, occurrences in scored:

            selection = problem.decode(bits)

            if selection is None:
                continue

            onehot_samples += occurrences

            if not problem.is_capacity_feasible(selection):
                continue

            feasible_samples += occurrences

            if best_selection is None:
                best_selection = selection

        details = {
            "num_qubits": problem.num_variables,
            "reps": self.reps,
            "run_seed": run_seed,
            "optimized_expectation": float(optimization.fun) + offset,
            "onehot_sample_fraction": onehot_samples / max(self.shots, 1),
            "feasible_sample_fraction": feasible_samples / max(self.shots, 1),
            "unique_bitstrings": len(counts),
            "lowest_sampled_energy": scored[0][0] if scored else None,
        }

        if best_selection is None:
            return RouteSolution(
                {}, float("inf"), "qaoa", False,
                evaluations=evaluations["count"], details=details,
            )

        return RouteSolution(
            selection=best_selection,
            energy=problem.selection_energy(best_selection),
            method="qaoa",
            feasible=True,
            evaluations=evaluations["count"],
            details=details,
        )
