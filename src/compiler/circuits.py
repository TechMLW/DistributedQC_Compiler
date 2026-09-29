import math
import os
from collections import Counter

from qiskit import QuantumCircuit, qasm2, transpile
from qiskit.circuit.library import efficient_su2
from qiskit.synthesis.qft import synth_qft_full


RANDOMIZED_BENCHMARKS = ("layered_random", "random")
LAYERED_BENCHMARKS = ("layered_random", "random", "hardware_efficient")
SUPPORTED_BENCHMARKS = ("layered_random", "random", "ghz", "qft", "hardware_efficient")

NON_INTERACTION_OPERATIONS = {"barrier", "measure", "reset", "delay", "snapshot"}

NORMALIZATION_BASIS = ["cx", "u", "measure", "reset", "barrier"]


def generate_benchmark(name, num_qubits, layers, rng):

    if num_qubits < 2:
        raise ValueError("A benchmark circuit needs at least two logical qubits")

    if name == "layered_random":
        return _layered_random(num_qubits, layers, rng)

    if name == "random":
        return _random_activity(num_qubits, layers, rng)

    if name == "ghz":
        return _ghz(num_qubits)

    if name == "qft":
        return synth_qft_full(num_qubits)

    if name == "hardware_efficient":
        return _hardware_efficient(num_qubits, layers, rng)

    raise ValueError(
        f"Unsupported benchmark '{name}'. Choose from {', '.join(SUPPORTED_BENCHMARKS)}."
    )


def _layered_random(num_qubits, layers, rng):

    circuit = QuantumCircuit(num_qubits, name="layered_random")

    for _ in range(layers):

        order = list(range(num_qubits))
        rng.shuffle(order)

        for index in range(0, num_qubits - 1, 2):
            circuit.cx(order[index], order[index + 1])

        circuit.barrier()

    return circuit


def _random_activity(num_qubits, layers, rng):

    circuit = QuantumCircuit(num_qubits, name="random")

    for _ in range(layers):

        active = [qubit for qubit in range(num_qubits) if rng.random() < 0.5]
        rng.shuffle(active)

        for index in range(0, len(active) - 1, 2):
            circuit.cx(active[index], active[index + 1])

        idle = sorted(set(range(num_qubits)) - set(active[: len(active) // 2 * 2]))

        for qubit in idle:
            circuit.rz(rng.uniform(0.0, 2.0 * math.pi), qubit)

        circuit.barrier()

    return circuit


def _ghz(num_qubits):

    circuit = QuantumCircuit(num_qubits, name="ghz")
    circuit.h(0)

    for qubit in range(num_qubits - 1):
        circuit.cx(qubit, qubit + 1)

    return circuit


def _hardware_efficient(num_qubits, layers, rng):

    ansatz = efficient_su2(num_qubits, reps=layers, entanglement="linear")

    values = [rng.uniform(0.0, 2.0 * math.pi) for _ in range(ansatz.num_parameters)]

    return ansatz.assign_parameters(values)


def load_circuit_file(path):

    if not os.path.exists(path):
        raise FileNotFoundError(f"Circuit file not found: {path}")

    try:
        return qasm2.load(
            path, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        )
    except Exception as qasm2_error:

        try:
            from qiskit import qasm3
            return qasm3.load(path)
        except Exception:
            raise ValueError(
                f"Could not parse {path} as OpenQASM 2 ({qasm2_error})"
            ) from qasm2_error


def requires_normalization(circuit):

    for instruction in circuit.data:

        operation = instruction.operation

        if operation.name in NON_INTERACTION_OPERATIONS:
            continue

        if len(instruction.qubits) > 2:
            return True

        if getattr(operation, "definition", None) is not None and operation.name not in {
            "cx", "cz", "cp", "swap", "rzz", "rxx", "ryy", "ecr", "cy", "ch",
            "crx", "cry", "crz", "cu", "h", "x", "y", "z", "s", "sdg", "t",
            "tdg", "rx", "ry", "rz", "u", "u1", "u2", "u3", "p", "sx", "sxdg", "id",
        }:
            return True

    return False


def normalize_circuit(circuit, seed):

    if not requires_normalization(circuit):
        return circuit

    return transpile(
        circuit,
        basis_gates=NORMALIZATION_BASIS,
        optimization_level=0,
        seed_transpiler=seed,
    )


def two_qubit_interactions(circuit):

    interactions = []

    for instruction in circuit.data:

        if instruction.operation.name in NON_INTERACTION_OPERATIONS:
            continue

        if len(instruction.qubits) != 2:
            continue

        first = circuit.find_bit(instruction.qubits[0]).index
        second = circuit.find_bit(instruction.qubits[1]).index

        interactions.append((first, second))

    return interactions


def circuit_summary(circuit, source):

    interactions = two_qubit_interactions(circuit)

    operations = Counter(
        instruction.operation.name
        for instruction in circuit.data
        if instruction.operation.name != "barrier"
    )

    distinct_pairs = {tuple(sorted(pair)) for pair in interactions}

    return {
        "source": source,
        "num_qubits": circuit.num_qubits,
        "depth": circuit.depth(filter_function=lambda item: item.operation.name != "barrier"),
        "gate_count": sum(operations.values()),
        "two_qubit_gate_count": len(interactions),
        "distinct_interaction_pairs": len(distinct_pairs),
        "operations": dict(sorted(operations.items())),
    }


def circuit_to_qasm(circuit):

    try:
        return qasm2.dumps(circuit)
    except Exception:
        from qiskit import qasm3
        return qasm3.dumps(circuit)
