from network import link


class CommunicationCost:

    def __init__(self, monitor=None):

        self.monitor = monitor

    def calculate(self, graph, partitions):

        # Qubit -> QPU lookup
        qubit_to_qpu = {}

        for qpu, qubits in partitions.items():
            for qubit in qubits:
                qubit_to_qpu[qubit] = qpu

        communication_cost = 0

        for u, v, data in graph.edges(data=True):

            if qubit_to_qpu[u] == qubit_to_qpu[v]:
                continue

            weight = data["weight"]

            if self.monitor is None:
                communication_cost += weight
                continue

            qpu1 = qubit_to_qpu[u]
            qpu2 = qubit_to_qpu[v]

            link = self.monitor.links.get((qpu1, qpu2))

            link = self.monitor.get_state(qpu1, qpu2)

            if link is None:
                communication_cost += weight
                continue

            latency = link["latency"]
            fidelity = link["fidelity"]
            congestion = link["congestion"]
            bell_pairs = max(link["bell_pairs"], 1)

            network_penalty = (
                latency
                * (1 + congestion / 100)
                * (1 / fidelity)
                * (20 / bell_pairs)
            )

            communication_cost += weight * network_penalty

        return communication_cost