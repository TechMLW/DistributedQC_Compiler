from compiler.workload import qubit_to_qpu_map


def static_partition_metrics(graph, partition, topology, router, cost_model):

    qubit_map = qubit_to_qpu_map(partition)

    cut_weight = 0
    remote_edges = 0
    weighted_route_cost = 0.0
    weighted_hops = 0

    for u, v, data in graph.edges(data=True):

        source, destination = qubit_map[u], qubit_map[v]

        if source == destination:
            continue

        weight = data["weight"]
        route = min(
            router.candidate_routes(topology, source, destination),
            key=lambda candidate: (cost_model.base_cost(candidate, topology), candidate.path),
        )

        remote_edges += 1
        cut_weight += weight
        weighted_hops += weight * route.hop_count
        weighted_route_cost += weight * cost_model.base_cost(route, topology)

    total_weight = sum(data["weight"] for _, _, data in graph.edges(data=True))

    return {
        "remote_interaction_edges": remote_edges,
        "cut_weight": cut_weight,
        "local_weight": total_weight - cut_weight,
        "remote_weight_fraction": cut_weight / total_weight if total_weight else 0.0,
        "weighted_hops": weighted_hops,
        "static_communication_cost": weighted_route_cost,
    }
