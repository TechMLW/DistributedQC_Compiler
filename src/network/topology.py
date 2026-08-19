from network.qpu import QPU
from network.link import Link
import networkx as nx


class Topology:

    def __init__(self):

        self.qpus = {}
        self.links = []
        self.graph = nx.Graph()

    def add_qpu(self,id,capacity):
  
      self.qpus[id]=QPU(id,capacity)
  
      self.graph.add_node(id)

    def connect(
        self,
        a,
        b,
        latency,
        fidelity,
        bell_pairs):

        link = Link(
            a,
            b,
            latency,
            fidelity,
            bell_pairs
        )
    
        self.links.append(link)
    
        self.graph.add_edge(
            a,
            b,
            weight=latency
    )

    def print_topology(self):

        print("\n===== TOPOLOGY =====\n")

        for qpu in self.qpus.values():
            print(qpu)

        print()

        for link in self.links:

            print(
                f"{link.source} <--> {link.destination}"
                f"  latency={link.latency}"
                f" fidelity={link.fidelity}"
                f" bell_pairs={link.bell_pairs}"
            )