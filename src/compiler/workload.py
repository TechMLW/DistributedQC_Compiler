from dataclasses import dataclass, field

from qiskit.converters import circuit_to_dag

from compiler.circuits import NON_INTERACTION_OPERATIONS


@dataclass
class Communication:
    id: str
    qubit_u: int
    qubit_v: int
    load: float
    created_step: int
    window_index: int
    sequence: int
    age: int = 0

    @property
    def qubits(self):
        return (self.qubit_u, self.qubit_v)


@dataclass
class ExecutionWindow:
    index: int
    gate_pairs: list = field(default_factory=list)

    @property
    def size(self):
        return len(self.gate_pairs)


def qubit_to_qpu_map(partition):
    return {qubit: qpu for qpu, qubits in partition.items() for qubit in qubits}


def build_execution_windows(circuit):

    dag = circuit_to_dag(circuit)
    windows = []

    for layer in dag.layers():

        pairs = []

        for node in layer["graph"].op_nodes():

            if node.op.name in NON_INTERACTION_OPERATIONS or len(node.qargs) != 2:
                continue

            u = circuit.find_bit(node.qargs[0]).index
            v = circuit.find_bit(node.qargs[1]).index

            pairs.append((min(u, v), max(u, v)))

        if pairs:
            windows.append(ExecutionWindow(index=len(windows), gate_pairs=sorted(pairs)))

    return windows


class DemandStream:

    def __init__(self, windows, load_per_interaction):

        self.windows = windows
        self.load_per_interaction = load_per_interaction

    def window_for_step(self, step):

        if not self.windows:
            return None

        return self.windows[(step - 1) % len(self.windows)]

    def lookahead(self, next_step, count):

        return [
            self.window_for_step(step)
            for step in range(next_step, next_step + count)
            if self.window_for_step(step) is not None
        ]

    def communications_for_step(self, step):

        window = self.window_for_step(step)

        if window is None:
            return None, []

        communications = [
            Communication(
                id=f"s{step:04d}-w{window.index:03d}-g{position:03d}-q{u}_{v}",
                qubit_u=u,
                qubit_v=v,
                load=self.load_per_interaction,
                created_step=step,
                window_index=window.index,
                sequence=position,
            )
            for position, (u, v) in enumerate(window.gate_pairs)
        ]

        return window, communications
