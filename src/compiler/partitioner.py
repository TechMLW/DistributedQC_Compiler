# import networkx as nx

# class Partitioner:

#     def partition(self, graph, num_qpus):

#         qubits = sorted(graph.nodes())

#         partitions = {i: [] for i in range(num_qpus)}

#         for i, qubit in enumerate(qubits):

#             partitions[i % num_qpus].append(qubit)

#         return partitions

import networkx as nx


class Partitioner:

    def partition(self, graph, num_qpus):

        # Sort edges by communication weight (highest first)
        edges = sorted(
            graph.edges(data=True),
            key=lambda x: x[2]["weight"],
            reverse=True
        )

        partitions = {i: [] for i in range(num_qpus)}
        assigned = {}
        max_size = (len(graph.nodes()) + num_qpus - 1) // num_qpus

        current_qpu = 0

        for u, v, data in edges:

            if u not in assigned and v not in assigned:

                partitions[current_qpu].extend([u, v])

                assigned[u] = current_qpu
                assigned[v] = current_qpu

                current_qpu = (current_qpu + 1) % num_qpus

            elif u in assigned and v not in assigned:

                qpu = assigned[u]
                if len(partitions[qpu]) < max_size:
                    partitions[qpu].append(v)
                    assigned[v] = qpu

            elif v in assigned and u not in assigned:

                qpu = assigned[v]
                if len(partitions[qpu]) < max_size:
                    partitions[qpu].append(u)
                    assigned[u] = qpu

        # Assign isolated qubits
        for node in graph.nodes():

            if node not in assigned:

                smallest = min(partitions, key=lambda x: len(partitions[x]))

                partitions[smallest].append(node)
                assigned[node] = smallest

        # Remove duplicates
        for qpu in partitions:
            partitions[qpu] = sorted(list(set(partitions[qpu])))

        return partitions