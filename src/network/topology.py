import networkx as nx

from network.link import Link


class NonexistentPhysicalLinkError(ValueError):
    pass


def link_key(a, b):
    return (a, b) if a <= b else (b, a)


class Topology:

    def __init__(self, model):

        self.model = model
        self.qpu_capacities = {}
        self.links = {}
        self.graph = nx.Graph()
        self._hop_distances = None

    def add_qpu(self, qpu_id, capacity):

        if capacity < 0:
            raise ValueError("QPU capacity cannot be negative")

        self.qpu_capacities[qpu_id] = capacity
        self.graph.add_node(qpu_id)
        self._hop_distances = None

    def connect(self, a, b, **properties):

        if a == b:
            raise ValueError("A physical link cannot connect a QPU to itself")

        if a not in self.qpu_capacities or b not in self.qpu_capacities:
            raise ValueError(f"Link ({a},{b}) references an unknown QPU")

        key = link_key(a, b)

        if key in self.links:
            raise ValueError(f"Physical link {key} already exists")

        link = Link(key[0], key[1], **properties)

        self.links[key] = link
        self.graph.add_edge(key[0], key[1], latency=link.latency)
        self._hop_distances = None

        return link

    @property
    def qpu_ids(self):
        return sorted(self.qpu_capacities)

    def has_link(self, a, b):
        return link_key(a, b) in self.links

    def get_link(self, a, b):
        return self.links.get(link_key(a, b))

    def all_links(self):
        return [self.links[key] for key in sorted(self.links)]

    def links_on_path(self, path):

        found = []

        for a, b in zip(path[:-1], path[1:]):

            link = self.links.get(link_key(a, b))

            if link is None:
                raise NonexistentPhysicalLinkError(
                    f"Path {list(path)} uses nonexistent physical link ({a},{b})"
                )

            found.append(link)

        return found

    def is_connected(self):

        if self.graph.number_of_nodes() <= 1:
            return True

        return nx.is_connected(self.graph)

    def hop_distance(self, a, b):

        if self._hop_distances is None:
            self._hop_distances = dict(nx.all_pairs_shortest_path_length(self.graph))

        return self._hop_distances.get(a, {}).get(b)

    def sync_routing_weights(self):

        for (a, b), link in self.links.items():
            self.graph[a][b]["latency"] = link.latency

    def congested_links(self):
        return [link for link in self.all_links() if link.is_congested]

    def saturated_links(self):
        return [link for link in self.all_links() if link.is_saturated]

    def utilization_summary(self):

        values = [link.utilization for link in self.links.values()]

        if not values:
            return 0.0, 0.0

        return max(values), sum(values) / len(values)

    def capacity_invariant_holds(self):
        return all(
            link.active_load <= link.capacity + 1e-9 for link in self.links.values()
        )

    def observe_window(self):

        for link in self.all_links():
            link.observe_window()

    def summary(self):

        degrees = dict(self.graph.degree())

        return {
            "model": self.model,
            "num_qpus": len(self.qpu_capacities),
            "num_physical_links": len(self.links),
            "connected": self.is_connected(),
            "diameter_hops": (
                nx.diameter(self.graph)
                if self.graph.number_of_nodes() > 1 and self.is_connected() else 0
            ),
            "min_degree": min(degrees.values()) if degrees else 0,
            "max_degree": max(degrees.values()) if degrees else 0,
            "physical_links": [f"{a}-{b}" for a, b in sorted(self.links)],
            "total_link_capacity": sum(link.capacity for link in self.links.values()),
        }

    def to_dict(self):

        return {
            "summary": self.summary(),
            "qpus": [
                {"qpu": qpu, "capacity": self.qpu_capacities[qpu]}
                for qpu in self.qpu_ids
            ],
            "links": [link.to_dict() for link in self.all_links()],
        }
