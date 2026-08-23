from copy import deepcopy
from compiler.communication_cost import CommunicationCost
from compiler.objective import ObjectiveFunction


class Optimizer:

    def __init__(self, monitor):
        self.monitor = monitor
        self.cost = CommunicationCost(monitor)
        self.objective = ObjectiveFunction(monitor)

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
    
    def adaptive_optimize(self, graph, current_partition, communication_cost, threshold=5):

        current_score = self.objective.evaluate(communication_cost, current_partition)
    
        candidate_partition, _ = self.optimize(graph, current_partition)
    
        candidate_comm_cost = self.cost.calculate(graph, candidate_partition)
    
        candidate_score = self.objective.evaluate(candidate_comm_cost, candidate_partition)
    
        print("\n===== Adaptive Decision =====")
        print(f"Current Score   : {current_score:.3f}")
        print(f"Candidate Score : {candidate_score:.3f}")
    
        if candidate_score < current_score - threshold:
            print("Decision : Repartition")
            return candidate_partition, candidate_score
    
        print("Decision : Keep Current Partition")
        return current_partition, current_score
    
    