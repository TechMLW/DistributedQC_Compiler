class Link:

    def __init__(self,
                 source,
                 destination,
                 latency,
                 fidelity,
                 bell_pairs):

        self.source = source
        self.destination = destination

        self.latency = latency
        self.fidelity = fidelity
        self.bell_pairs = bell_pairs

    def cost(self):

        return self.latency + (1-self.fidelity)*100