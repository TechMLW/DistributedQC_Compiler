import networkx as nx


class KLPartitioner:

    def partition(self, graph):

        left, right = nx.community.kernighan_lin_bisection(
            graph,
            weight="weight"
        )

        partitions = {
            0: sorted(list(left)),
            1: sorted(list(right))
        }

        return partitions