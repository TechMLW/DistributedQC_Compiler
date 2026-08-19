import matplotlib.pyplot as plt
import networkx as nx


class Visualizer:

    def draw(self, graph):

        plt.figure(figsize=(8, 6))

        pos = nx.kamada_kawai_layout(graph)

        edge_labels = nx.get_edge_attributes(graph, "weight")

        # nx.draw_networkx_nodes(graph, pos, node_size=800)

        # nx.draw_networkx_edges(graph, pos)

        # nx.draw_networkx_labels(graph, pos)

        # nx.draw_networkx_edge_labels(
        #     graph,
        #     pos,
        #     edge_labels=edge_labels
        # )

        nx.draw(

            graph,
        
            pos,
        
            with_labels=True,
        
            node_size=1200,
        
            node_color="skyblue",
        
            font_size=12,
        
            width=2
        )
        plt.title("Communication Graph")

        plt.axis("off")

        # plt.show()
        plt.savefig("communication_graph.png", dpi=300, bbox_inches="tight")
        print("Communication graph saved as communication_graph.png")
        plt.close()