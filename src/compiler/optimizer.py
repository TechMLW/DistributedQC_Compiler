from copy import deepcopy
from compiler.communication_cost import CommunicationCost
from compiler.objective import ObjectiveFunction


class Optimizer:

    def __init__(self):
        self.cost = CommunicationCost()
        self.objective = ObjectiveFunction()

    def optimize(self, graph, initial_partition):

        best_partition = deepcopy(initial_partition)

        best_score = self.objective.evaluate(
            self.cost.calculate(graph, best_partition),
            best_partition
        )

        improved = True

        while improved:

            improved = False

            qpus = list(best_partition.keys())

            for q1 in qpus:
                for q2 in qpus:

                    if q1 == q2:
                        continue

                    for a in list(best_partition[q1]):

                        for b in list(best_partition[q2]):

                            candidate = deepcopy(best_partition)

                            if a not in candidate[q1] or b not in candidate[q2]:
                                continue
                            
                            candidate[q1].remove(a)
                            candidate[q2].remove(b)

                            candidate[q1].append(b)
                            candidate[q2].append(a)

                            score = self.objective.evaluate(

                                self.cost.calculate(graph, candidate),

                                candidate

                            )

                            if score < best_score:

                                best_score = score
                                best_partition = candidate
                                improved = True

        return best_partition, best_score