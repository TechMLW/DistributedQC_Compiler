import networkx as nx

from compiler.circuits import two_qubit_interactions


class CommunicationGraph:

    def build(self, circuit):

        graph = nx.Graph()
        graph.add_nodes_from(range(circuit.num_qubits))

        for first, second in two_qubit_interactions(circuit):

            if first == second:
                continue

            if graph.has_edge(first, second):
                graph[first][second]["weight"] += 1
            else:
                graph.add_edge(first, second, weight=1)

        return graph


def communication_graph_to_dict(graph):

    return {
        "num_logical_qubits": graph.number_of_nodes(),
        "num_interaction_edges": graph.number_of_edges(),
        "total_interaction_weight": sum(
            data["weight"] for _, _, data in graph.edges(data=True)
        ),
        "edges": [
            {"u": min(u, v), "v": max(u, v), "weight": data["weight"]}
            for u, v, data in sorted(
                graph.edges(data=True), key=lambda item: (min(item[:2]), max(item[:2]))
            )
        ],
    }
