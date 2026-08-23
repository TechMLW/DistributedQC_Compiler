class RepartitionTrigger:

    def __init__(
        self,
        latency_limit=20,
        fidelity_limit=0.95,
        congestion_limit=20,
        bellpair_limit=10
    ):

        self.latency_limit = latency_limit
        self.fidelity_limit = fidelity_limit
        self.congestion_limit = congestion_limit
        self.bellpair_limit = bellpair_limit

    def should_repartition(self, monitor):

        # for _, _, link in monitor.graph.edges(data=True):
        for state in monitor.links.values():

            if state["latency"] > self.latency_limit:
                return True

            if state["fidelity"] < self.fidelity_limit:
                return True

            if state["congestion"] > self.congestion_limit:
                return True

            if state["bell_pairs"] < self.bellpair_limit:
                return True

        return False