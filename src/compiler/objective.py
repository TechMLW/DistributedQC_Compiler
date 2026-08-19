class ObjectiveFunction:

    def __init__(self,
                 alpha=1,
                 beta=1):

        self.alpha = alpha
        self.beta = beta


    def evaluate(self,
                 communication_cost,
                 partitions):

        sizes = [len(x) for x in partitions.values()]

        imbalance = max(sizes) - min(sizes)

        return (

            self.alpha * communication_cost

            +

            self.beta * imbalance

        )