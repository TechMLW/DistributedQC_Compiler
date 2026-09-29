from dataclasses import dataclass
from itertools import islice

import networkx as nx

from network.topology import link_key


@dataclass(frozen=True)
class RouteMetrics:
    hop_count: int
    latency: float
    fidelity: float
    bottleneck_utilization: float


class Route:

    def __init__(self, path):
        self.path = tuple(path)

    @property
    def hop_count(self):
        return len(self.path) - 1

    @property
    def is_local(self):
        return self.hop_count == 0

    def link_keys(self):
        return tuple(link_key(a, b) for a, b in zip(self.path[:-1], self.path[1:]))

    def uses_link(self, key):
        return tuple(key) in self.link_keys()

    def metrics(self, topology):

        links = topology.links_on_path(self.path)

        if not links:
            return RouteMetrics(0, 0.0, 1.0, 0.0)

        fidelity = 1.0

        for link in links:
            fidelity *= link.fidelity

        return RouteMetrics(
            hop_count=self.hop_count,
            latency=sum(link.latency for link in links),
            fidelity=fidelity,
            bottleneck_utilization=max(link.utilization for link in links),
        )

    def label(self):
        return "-".join(str(node) for node in self.path)

    def __eq__(self, other):
        return isinstance(other, Route) and self.path == other.path

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return f"Route({self.label()})"


class Router:

    def __init__(self, candidate_limit, max_extra_hops):

        self.candidate_limit = max(1, candidate_limit)
        self.max_extra_hops = max(0, max_extra_hops)
        self._cache = {}

    def begin_window(self):
        self._cache.clear()

    def candidate_routes(self, topology, source, destination):

        if source == destination:
            return [Route((source,))]

        key = (source, destination)

        if key in self._cache:
            return list(self._cache[key])

        shortest_hops = topology.hop_distance(source, destination)

        if shortest_hops is None:
            self._cache[key] = []
            return []

        hop_limit = shortest_hops + self.max_extra_hops

        routes = []

        try:
            generator = nx.shortest_simple_paths(
                topology.graph, source, destination, weight="latency"
            )

            for path in islice(generator, self.candidate_limit * 8):

                if len(path) - 1 > hop_limit:
                    continue

                route = Route(path)
                topology.links_on_path(route.path)
                routes.append(route)

                if len(routes) >= self.candidate_limit:
                    break

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            routes = []

        self._cache[key] = routes

        return list(routes)
