import networkx as nx


class PartitionCapacityError(ValueError):
    pass


class CapacityAwareKLPartitioner:

    def __init__(self, rng, restarts, refinement_passes):

        self.rng = rng
        self.restarts = max(1, restarts)
        self.refinement_passes = max(0, refinement_passes)

    def partition(self, graph, qpu_capacities):

        nodes = sorted(graph.nodes())
        qpu_ids = sorted(qpu_capacities)

        total_capacity = sum(qpu_capacities[qpu] for qpu in qpu_ids)

        if len(nodes) > total_capacity:
            raise PartitionCapacityError(
                f"{len(nodes)} logical qubits cannot be placed on {len(qpu_ids)} QPUs "
                f"with total capacity {total_capacity}"
            )

        assignment = self._assign(graph, nodes, qpu_ids, qpu_capacities)

        partition = {qpu: sorted(assignment.get(qpu, [])) for qpu in qpu_ids}

        partition = self._refine(graph, partition, qpu_capacities)

        return {qpu: sorted(qubits) for qpu, qubits in partition.items()}

    def _assign(self, graph, nodes, qpu_ids, qpu_capacities):

        if len(qpu_ids) == 1:
            return {qpu_ids[0]: list(nodes)}

        middle = len(qpu_ids) // 2
        left_qpus, right_qpus = qpu_ids[:middle], qpu_ids[middle:]

        left_capacity = sum(qpu_capacities[q] for q in left_qpus)
        right_capacity = sum(qpu_capacities[q] for q in right_qpus)

        proportional = round(len(nodes) * left_capacity / (left_capacity + right_capacity))
        left_size = min(left_capacity, max(len(nodes) - right_capacity, proportional))

        left_nodes, right_nodes = self._bisect(graph, nodes, left_size)

        assignment = self._assign(graph, left_nodes, left_qpus, qpu_capacities)
        assignment.update(self._assign(graph, right_nodes, right_qpus, qpu_capacities))

        return assignment

    def _bisect(self, graph, nodes, left_size):

        if left_size <= 0:
            return [], list(nodes)

        if left_size >= len(nodes):
            return list(nodes), []

        subgraph = graph.subgraph(nodes)

        best_key = None
        best_split = None

        for _ in range(self.restarts):

            shuffled = list(nodes)
            self.rng.shuffle(shuffled)

            initial = (set(shuffled[:left_size]), set(shuffled[left_size:]))

            if subgraph.number_of_edges() == 0:
                left, right = initial
            else:
                first, second = nx.community.kernighan_lin_bisection(
                    subgraph, partition=initial, weight="weight"
                )
                left, right = (first, second) if len(first) == left_size else (second, first)

            if len(left) != left_size:
                raise RuntimeError("Kernighan-Lin changed the bisection sizes")

            cut = nx.cut_size(subgraph, left, right, weight="weight")
            key = (cut, sorted(left))

            if best_key is None or key < best_key:
                best_key = key
                best_split = (sorted(left), sorted(right))

        return best_split

    def _refine(self, graph, partition, qpu_capacities):

        qubit_to_qpu = {q: qpu for qpu, qubits in partition.items() for q in qubits}
        members = {qpu: set(qubits) for qpu, qubits in partition.items()}

        def affinity(qubit, qpu):
            return sum(
                graph[qubit][neighbour]["weight"]
                for neighbour in graph.neighbors(qubit)
                if qubit_to_qpu[neighbour] == qpu and neighbour != qubit
            )

        for _ in range(self.refinement_passes):

            improved = False

            for qubit in sorted(qubit_to_qpu):

                home = qubit_to_qpu[qubit]
                home_affinity = affinity(qubit, home)

                best_gain = 0
                best_move = None

                for target in sorted(members):

                    if target == home:
                        continue

                    if len(members[target]) < qpu_capacities[target]:

                        gain = affinity(qubit, target) - home_affinity

                        if gain > best_gain:
                            best_gain = gain
                            best_move = (target, None)

                    for other in sorted(members[target]):

                        link_weight = (
                            graph[qubit][other]["weight"]
                            if graph.has_edge(qubit, other) else 0
                        )

                        gain = (
                            affinity(qubit, target) - home_affinity
                            + affinity(other, home) - affinity(other, target)
                            - 2 * link_weight
                        )

                        if gain > best_gain:
                            best_gain = gain
                            best_move = (target, other)

                if best_move is None:
                    continue

                target, other = best_move

                members[home].discard(qubit)
                members[target].add(qubit)
                qubit_to_qpu[qubit] = target

                if other is not None:
                    members[target].discard(other)
                    members[home].add(other)
                    qubit_to_qpu[other] = home

                improved = True

            if not improved:
                break

        return {qpu: sorted(qubits) for qpu, qubits in members.items()}


def validate_partition(partition, num_logical_qubits, qpu_capacities):

    problems = []
    seen = {}

    for qpu, qubits in partition.items():

        if qpu not in qpu_capacities:
            problems.append(f"QPU {qpu} does not exist")
            continue

        if len(qubits) > qpu_capacities[qpu]:
            problems.append(
                f"QPU {qpu} holds {len(qubits)} qubits but capacity is {qpu_capacities[qpu]}"
            )

        for qubit in qubits:
            if qubit in seen:
                problems.append(f"qubit {qubit} assigned to QPU {seen[qubit]} and {qpu}")
            seen[qubit] = qpu

    missing = sorted(set(range(num_logical_qubits)) - set(seen))

    if missing:
        problems.append(f"qubits without a QPU: {missing}")

    return problems
