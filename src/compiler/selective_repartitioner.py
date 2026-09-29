from dataclasses import dataclass, field

from compiler.workload import qubit_to_qpu_map


TOLERANCE = 1e-9


@dataclass
class PlacementEvaluation:
    objective: float
    communication_cost: float
    deferred_load: float
    migration_cost: float
    remote_interactions: int
    peak_link_utilization: dict = field(default_factory=dict)


@dataclass
class RepartitionOutcome:
    step: int
    reason: str
    reason_detail: str
    trigger_link_keys: list
    affected_qubits: list
    accepted: bool = False
    rejection_reason: str = ""
    moves: list = field(default_factory=list)
    candidates_evaluated: int = 0
    before: PlacementEvaluation = None
    after: PlacementEvaluation = None
    old_partition: dict = field(default_factory=dict)
    new_partition: dict = field(default_factory=dict)
    unaffected_qubits_moved: list = field(default_factory=list)
    utilization_at_trigger: dict = field(default_factory=dict)

    @property
    def moved_qubits(self):
        return sorted({qubit for move in self.moves for qubit in move["qubits"]})


def partition_from_map(qubit_map, qpu_ids):

    partition = {qpu: [] for qpu in qpu_ids}

    for qubit in sorted(qubit_map):
        partition[qubit_map[qubit]].append(qubit)

    return partition


class SelectiveRepartitioner:

    def __init__(
        self,
        router,
        cost_model,
        qpu_capacities,
        lookahead_windows,
        max_moves,
        improvement_margin,
        deferral_penalty,
        migration_weight,
        bell_pairs_per_unit_load,
    ):

        self.router = router
        self.cost_model = cost_model
        self.qpu_capacities = dict(qpu_capacities)
        self.lookahead_windows = lookahead_windows
        self.max_moves = max_moves
        self.improvement_margin = improvement_margin
        self.deferral_penalty = deferral_penalty
        self.migration_weight = migration_weight
        self.bell_pairs_per_unit_load = bell_pairs_per_unit_load

    def propose(self, step, partition, decision, topology, demand_stream, queued, admission):

        original = qubit_to_qpu_map(partition)
        affected = set(decision.affected_qubits)

        outcome = RepartitionOutcome(
            step=step,
            reason=decision.reason,
            reason_detail=decision.reason_detail,
            trigger_link_keys=list(decision.trigger_link_keys),
            affected_qubits=sorted(affected),
            old_partition={qpu: list(qubits) for qpu, qubits in partition.items()},
            new_partition={qpu: list(qubits) for qpu, qubits in partition.items()},
            utilization_at_trigger={
                key: topology.links[key].utilization for key in decision.trigger_link_keys
            },
        )

        horizon = self._horizon(step, demand_stream, queued)
        persisting = admission.persisting_load()
        load = demand_stream.load_per_interaction

        def evaluate(qubit_map):
            return self._evaluate(qubit_map, original, horizon, persisting, load, topology)

        best_map = dict(original)
        outcome.before = evaluate(best_map)
        best_eval = outcome.before
        applied = []

        for _ in range(self.max_moves):

            best_candidate = None

            for candidate_map, move in self._neighbours(best_map, affected):

                outcome.candidates_evaluated += 1
                evaluation = evaluate(candidate_map)
                key = (evaluation.objective, move["order"])

                if best_candidate is None or key < best_candidate[0]:
                    best_candidate = (key, candidate_map, move, evaluation)

            if best_candidate is None:
                break

            _, candidate_map, move, evaluation = best_candidate

            if evaluation.objective >= best_eval.objective - TOLERANCE:
                break

            best_map, best_eval = candidate_map, evaluation
            applied.append(move)

        outcome.after = best_eval
        outcome.moves = [{k: v for k, v in move.items() if k != "order"} for move in applied]
        outcome.unaffected_qubits_moved = sorted(
            qubit for qubit in best_map
            if best_map[qubit] != original[qubit] and qubit not in affected
        )

        required = outcome.before.objective * (1.0 - self.improvement_margin)

        if not applied:
            outcome.rejection_reason = "no_improving_selective_move"
        elif outcome.unaffected_qubits_moved:
            outcome.rejection_reason = "would_move_unaffected_qubits"
        elif not self._capacity_respected(best_map):
            outcome.rejection_reason = "qpu_capacity_violation"
        elif best_eval.objective > required + TOLERANCE:
            outcome.rejection_reason = "improvement_below_margin"
        else:
            outcome.accepted = True
            outcome.new_partition = partition_from_map(best_map, sorted(partition))

        return outcome

    def _horizon(self, step, demand_stream, queued):

        horizon_length = self.lookahead_windows or len(demand_stream.windows)

        windows = [list(window.gate_pairs)
                   for window in demand_stream.lookahead(step + 1, horizon_length)]

        queued_pairs = [
            (min(c.qubit_u, c.qubit_v), max(c.qubit_u, c.qubit_v)) for c in queued
        ]

        if windows:
            windows[0] = queued_pairs + windows[0]
        elif queued_pairs:
            windows = [queued_pairs]

        return windows

    def _neighbours(self, qubit_map, affected):

        occupancy = {qpu: 0 for qpu in self.qpu_capacities}

        for qpu in qubit_map.values():
            occupancy[qpu] += 1

        ordered = sorted(affected)

        for qubit in ordered:

            home = qubit_map[qubit]

            for target in sorted(self.qpu_capacities):

                if target == home:
                    continue

                if occupancy[target] < self.qpu_capacities[target]:
                    candidate = dict(qubit_map)
                    candidate[qubit] = target
                    yield candidate, {
                        "type": "move", "qubits": [qubit],
                        "from": [home], "to": [target],
                        "order": (0, qubit, target, -1),
                    }

                for other in ordered:

                    if other <= qubit or qubit_map[other] != target:
                        continue

                    candidate = dict(qubit_map)
                    candidate[qubit], candidate[other] = target, home
                    yield candidate, {
                        "type": "swap", "qubits": [qubit, other],
                        "from": [home, target], "to": [target, home],
                        "order": (1, qubit, target, other),
                    }

    def _capacity_respected(self, qubit_map):

        occupancy = {qpu: 0 for qpu in self.qpu_capacities}

        for qpu in qubit_map.values():
            occupancy[qpu] += 1

        return all(occupancy[qpu] <= self.qpu_capacities[qpu] for qpu in occupancy)

    def _migration_cost(self, qubit_map, original, topology):

        total = 0.0

        for qubit, qpu in qubit_map.items():

            if qpu == original[qubit]:
                continue

            routes = self.router.candidate_routes(topology, original[qubit], qpu)
            total += self.migration_weight * min(
                self.cost_model.base_cost(route, topology) for route in routes
            )

        return total

    def _evaluate(self, qubit_map, original, horizon, persisting, load, topology):

        bell_pairs_needed = load * self.bell_pairs_per_unit_load

        communication_cost = 0.0
        deferred = 0.0
        remote = 0
        peak = {key: 0.0 for key in topology.links}

        for pairs in horizon:

            loads = dict(persisting)
            bell_pairs = {key: link.available_bell_pairs for key, link in topology.links.items()}

            for u, v in pairs:

                source, destination = qubit_map[u], qubit_map[v]

                if source == destination:
                    continue

                remote += 1
                best = None

                for route in self.router.candidate_routes(topology, source, destination):

                    keys = route.link_keys()

                    if any(
                        loads[key] + load > topology.links[key].capacity + TOLERANCE
                        or bell_pairs[key] + TOLERANCE < bell_pairs_needed
                        for key in keys
                    ):
                        continue

                    cost = self.cost_model.congestion_aware_cost(
                        route, topology, added_load=load, background_load=loads
                    )

                    if best is None or (cost, route.path) < (best[0], best[1].path):
                        best = (cost, route)

                if best is None:
                    deferred += load
                    continue

                communication_cost += load * best[0]

                for key in best[1].link_keys():
                    loads[key] += load
                    bell_pairs[key] -= bell_pairs_needed

            for key, link in topology.links.items():
                if link.capacity > 0:
                    peak[key] = max(peak[key], loads[key] / link.capacity)

        migration = self._migration_cost(qubit_map, original, topology)

        return PlacementEvaluation(
            objective=communication_cost + self.deferral_penalty * deferred + migration,
            communication_cost=communication_cost,
            deferred_load=deferred,
            migration_cost=migration,
            remote_interactions=remote,
            peak_link_utilization=peak,
        )
