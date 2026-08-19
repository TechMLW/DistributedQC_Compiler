import networkx as nx

class CommunicationGraph:

    def build(self, circuit):

        graph = nx.Graph()

        for instruction in circuit.data:

            operation = instruction.operation
            qubits = instruction.qubits

            # Only consider two-qubit gates
            if len(qubits) == 2:

                q1 = circuit.find_bit(qubits[0]).index
                q2 = circuit.find_bit(qubits[1]).index

                if graph.has_edge(q1, q2):
                    graph[q1][q2]["weight"] += 1
                else:
                    graph.add_edge(q1, q2, weight=1)

        return graph