from qiskit.converters import circuit_to_dag

class CircuitAnalyzer:

    def analyze(self, circuit):

        dag = circuit_to_dag(circuit)

        communication = []

        for node in dag.op_nodes():

            if len(node.qargs) > 1:

                communication.append({
                    "gate": node.name,
                    "qubits": [q._index for q in node.qargs]
                })

        return {

            "qubits": circuit.num_qubits,

            "depth": circuit.depth(),

            "gate_count": circuit.size(),

            "operations": dict(circuit.count_ops()),

            "communication_gates": communication

        }