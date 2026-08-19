class CommunicationCost:

    def calculate(self, graph, partitions):

        # Create a lookup table: qubit -> QPU
        qubit_to_qpu = {}

        for qpu, qubits in partitions.items():
            for qubit in qubits:
                qubit_to_qpu[qubit] = qpu

        communication_cost = 0

        for u, v, data in graph.edges(data=True):

            if qubit_to_qpu[u] != qubit_to_qpu[v]:
                communication_cost += data["weight"]

        return communication_cost