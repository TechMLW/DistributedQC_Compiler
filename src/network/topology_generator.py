import math
from dataclasses import asdict, dataclass

from network.topology import Topology, link_key
from utils.seeding import derived_rng


SUPPORTED_TOPOLOGIES = ("line", "ring", "grid", "mesh", "star", "random_connected")
STOCHASTIC_TOPOLOGIES = ("random_connected",)


@dataclass(frozen=True)
class LinkPropertyRanges:
    latency: tuple
    fidelity: tuple
    capacity: tuple
    bell_pair_pool: tuple
    bell_pair_regeneration: tuple

    def to_dict(self):
        return {name: list(values) for name, values in asdict(self).items()}


def generate_physical_edges(model, num_qpus, rng, extra_link_probability):

    if num_qpus < 1:
        raise ValueError("At least one QPU is required")

    if num_qpus == 1:
        return []

    if model == "line":
        return [(i, i + 1) for i in range(num_qpus - 1)]

    if model == "ring":
        edges = [(i, i + 1) for i in range(num_qpus - 1)]
        if num_qpus > 2:
            edges.append((0, num_qpus - 1))
        return sorted(edges)

    if model == "mesh":
        return [(i, j) for i in range(num_qpus) for j in range(i + 1, num_qpus)]

    if model == "star":
        return [(0, i) for i in range(1, num_qpus)]

    if model == "grid":
        return _grid_edges(num_qpus)

    if model == "random_connected":
        return _random_connected_edges(num_qpus, rng, extra_link_probability)

    raise ValueError(
        f"Unsupported topology '{model}'. Choose from {', '.join(SUPPORTED_TOPOLOGIES)}."
    )


def _grid_edges(num_qpus):

    columns = math.ceil(math.sqrt(num_qpus))
    edges = []

    for node in range(num_qpus):

        right = node + 1
        below = node + columns

        if node % columns + 1 < columns and right < num_qpus:
            edges.append((node, right))

        if below < num_qpus:
            edges.append((node, below))

    return sorted(edges)


def _random_connected_edges(num_qpus, rng, extra_link_probability):

    order = list(range(num_qpus))
    rng.shuffle(order)

    selected = set()

    for position in range(1, num_qpus):
        attach_to = order[rng.randrange(position)]
        selected.add(link_key(order[position], attach_to))

    for a in range(num_qpus):
        for b in range(a + 1, num_qpus):

            draw = rng.random()

            if (a, b) not in selected and draw < extra_link_probability:
                selected.add((a, b))

    return sorted(selected)


def generate_link_properties(a, b, ranges, topology_seed):

    rng = derived_rng(topology_seed, f"link-properties-{a}-{b}")

    return {
        "latency": float(rng.randint(int(ranges.latency[0]), int(ranges.latency[1]))),
        "fidelity": round(rng.uniform(*ranges.fidelity), 6),
        "capacity": rng.randint(int(ranges.capacity[0]), int(ranges.capacity[1])),
        "bell_pair_pool": rng.randint(
            int(ranges.bell_pair_pool[0]), int(ranges.bell_pair_pool[1])
        ),
        "bell_pair_regeneration": round(rng.uniform(*ranges.bell_pair_regeneration), 3),
    }


def generate_topology(
    model,
    qpu_capacities,
    ranges,
    congestion_threshold,
    topology_seed,
    extra_link_probability,
):

    topology = Topology(model)

    for qpu in sorted(qpu_capacities):
        topology.add_qpu(qpu, qpu_capacities[qpu])

    edge_rng = derived_rng(topology_seed, "edges")

    edges = generate_physical_edges(
        model, len(qpu_capacities), edge_rng, extra_link_probability
    )

    for a, b in edges:
        topology.connect(
            a,
            b,
            congestion_threshold=congestion_threshold,
            **generate_link_properties(a, b, ranges, topology_seed),
        )

    if not topology.is_connected():
        raise RuntimeError(f"Generated '{model}' topology is disconnected")

    return topology
