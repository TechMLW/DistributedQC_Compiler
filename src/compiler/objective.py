class ObjectiveFunction:

    def __init__(self,
                 monitor,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 delta=1.0,
                 epsilon=1.0,
                 zeta=1.0):

        self.monitor = monitor

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta


    def evaluate(self,
                 communication_cost,
                 partitions):

        # -------------------------------
        # Load imbalance
        # -------------------------------

        sizes = [len(v) for v in partitions.values()]
        imbalance = max(sizes) - min(sizes)

        latency_cost = 0
        fidelity_cost = 0
        bellpair_cost = 0
        congestion_cost = 0

        # -------------------------------
        # Network state
        # -------------------------------

        qpus = list(partitions.keys())

        for i in range(len(qpus)):
            for j in range(i + 1, len(qpus)):

                state = self.monitor.get_state(qpus[i], qpus[j])

                if state is None:
                    continue

                latency_cost += state["latency"]

                fidelity_cost += (1 - state["fidelity"])

                bellpair_cost += 1 / (state["bell_pairs"] + 1)

                congestion_cost += state["congestion"]

        total = (

            self.alpha * communication_cost +

            self.beta * imbalance +

            self.gamma * latency_cost +

            self.delta * fidelity_cost +

            self.epsilon * bellpair_cost +

            self.zeta * congestion_cost

        )

        return total