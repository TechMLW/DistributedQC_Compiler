from networkx import shortest_path

class Router:

    def route(self,
              topology_graph,
              source_qpu,
              destination_qpu):

        path = shortest_path(
            topology_graph,
            source_qpu,
            destination_qpu
        )

        return path