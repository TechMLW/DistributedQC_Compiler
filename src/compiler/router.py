import networkx as nx

class Router:

    def route(self, topology, source, destination):

        return nx.shortest_path(
            topology,
            source,
            destination,
            weight="latency"
        )