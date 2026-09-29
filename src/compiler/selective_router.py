from dataclasses import dataclass, field

from compiler.route_optimization import RouteChoiceEntry, build_route_choice_problem
from utils.seeding import derive_seed


POLICY_QAOA_FIRST = "qaoa_first"
POLICY_BEST_OF = "best_of"


@dataclass
class RerouteRecord:
    step: int
    block_index: int
    communication_id: str
    qubit_u: int
    qubit_v: int
    source_qpu: int
    destination_qpu: int
    old_path: str
    new_path: str
    old_hops: int
    new_hops: int
    old_latency: float
    new_latency: float
    old_fidelity: float
    new_fidelity: float
    old_cost: float
    new_cost: float
    method: str


@dataclass
class BlockDecision:
    step: int
    block_index: int
    communication_ids: list
    num_variables: int
    capacity_conflict_pairs: int
    classical_method: str
    classical_energy: float
    classical_evaluations: int
    qaoa_attempted: bool
    qaoa_feasible: bool
    qaoa_energy: float
    qaoa_evaluations: int
    qaoa_details: dict
    executed_method: str
    executed_energy: float
    executed_raw_cost: float
    keep_current_energy: float
    changed_routes: int


@dataclass
class SelectiveRerouteResult:
    affected_ids: list = field(default_factory=list)
    untouched_ids: list = field(default_factory=list)
    reroutes: list = field(default_factory=list)
    kept_ids: list = field(default_factory=list)
    blocks: list = field(default_factory=list)
    admission_failures: int = 0
    affected_link_utilization: list = field(default_factory=list)
    affected_cost_before: float = 0.0
    affected_cost_after: float = 0.0

    @property
    def reroute_count(self):
        return len(self.reroutes)


class SelectiveCongestionRouter:

    def __init__(
        self,
        router,
        admission,
        cost_model,
        classical_solver,
        quantum_solver,
        use_quantum,
        selection_policy,
        block_max_variables,
        qaoa_seed,
        qubo_parameters,
    ):

        self.router = router
        self.admission = admission
        self.cost_model = cost_model
        self.classical_solver = classical_solver
        self.quantum_solver = quantum_solver
        self.use_quantum = use_quantum and quantum_solver is not None
        self.selection_policy = selection_policy
        self.block_max_variables = max(2, block_max_variables)
        self.qaoa_seed = qaoa_seed
        self.qubo_parameters = qubo_parameters

    def reroute(self, step, topology, report):

        result = SelectiveRerouteResult(
            affected_ids=list(report.affected_communication_ids),
            untouched_ids=list(report.unaffected_communication_ids),
        )

        if not result.affected_ids:
            return result

        affected = [self.admission.allocations[cid] for cid in result.affected_ids]

        before = {
            allocation.id: self._route_snapshot(allocation.route, allocation.load, topology)
            for allocation in affected
        }
        result.affected_cost_before = sum(snapshot["cost"] for snapshot in before.values())

        for block_index, block in enumerate(self._blocks(topology, affected, report)):
            result.blocks.append(self._solve_block(step, block_index, topology, block, result))

        for allocation in affected:

            current = self.admission.allocations.get(allocation.id)

            if current is None:
                continue

            after = self._route_snapshot(current.route, current.load, topology)
            result.affected_cost_after += after["cost"]

            old = before[allocation.id]

            if current.route.path == old["route"].path:
                result.kept_ids.append(allocation.id)
                continue

            result.reroutes.append(
                RerouteRecord(
                    step=step,
                    block_index=next(
                        block.block_index for block in result.blocks
                        if allocation.id in block.communication_ids
                    ),
                    communication_id=allocation.id,
                    qubit_u=allocation.communication.qubit_u,
                    qubit_v=allocation.communication.qubit_v,
                    source_qpu=allocation.source_qpu,
                    destination_qpu=allocation.destination_qpu,
                    old_path=old["route"].label(),
                    new_path=current.route.label(),
                    old_hops=old["metrics"].hop_count,
                    new_hops=after["metrics"].hop_count,
                    old_latency=old["metrics"].latency,
                    new_latency=after["metrics"].latency,
                    old_fidelity=old["metrics"].fidelity,
                    new_fidelity=after["metrics"].fidelity,
                    old_cost=old["cost"],
                    new_cost=after["cost"],
                    method=next(
                        block.executed_method for block in result.blocks
                        if allocation.id in block.communication_ids
                    ),
                )
            )

        for key in report.congested_link_keys:
            result.affected_link_utilization.append({
                "step": step,
                "link": f"{key[0]}-{key[1]}",
                "utilization_before": report.link_utilization[key],
                "utilization_after": topology.links[key].utilization,
            })

        return result

    def _route_snapshot(self, route, load, topology):
        return {
            "route": route,
            "metrics": route.metrics(topology),
            "cost": self.cost_model.communication_cost(load, route, topology),
        }

    def _option_estimate(self, topology, allocation):

        routes = set(
            self.router.candidate_routes(
                topology, allocation.source_qpu, allocation.destination_qpu
            )
        )
        routes.add(allocation.route)

        return min(len(routes), self.block_max_variables)

    def _blocks(self, topology, affected, report):

        congested = set(report.congested_link_keys)

        def ordering(allocation):
            touched = sorted(congested.intersection(allocation.route.link_keys()))
            return (touched[0] if touched else (-1, -1), allocation.id)

        blocks = []
        current = []
        current_size = 0

        for allocation in sorted(affected, key=ordering):

            size = self._option_estimate(topology, allocation)

            if current and current_size + size > self.block_max_variables:
                blocks.append(current)
                current, current_size = [], 0

            current.append(allocation)
            current_size += size

        if current:
            blocks.append(current)

        return blocks

    def _solve_block(self, step, block_index, topology, block, result):

        released = [self.admission.release(allocation.id, consumed=False)
                    for allocation in block]

        background = self.admission.load_by_link()
        per_block_limit = max(1, self.block_max_variables // len(released))

        candidates = {}
        entries = []

        for allocation in released:

            options = [allocation.route]

            alternatives = sorted(
                (
                    route for route in self.router.candidate_routes(
                        topology, allocation.source_qpu, allocation.destination_qpu
                    )
                    if route != allocation.route
                ),
                key=lambda route: (
                    self.cost_model.congestion_aware_cost(
                        route, topology, added_load=allocation.load
                    ),
                    route.path,
                ),
            )

            for route in alternatives:

                if len(options) >= per_block_limit:
                    break

                if self.admission.check_route(route, allocation.load)[0]:
                    options.append(route)

            candidates[allocation.id] = options
            entries.append(
                RouteChoiceEntry(
                    communication_id=allocation.id,
                    load=allocation.load,
                    bell_pairs_per_link=allocation.bell_pairs_per_link,
                )
            )

        problem = build_route_choice_problem(
            entries, candidates, topology, self.cost_model, background,
            **self.qubo_parameters,
        )

        keep_current = {entry.communication_id: 0 for entry in entries}
        keep_current_energy = problem.selection_energy(keep_current)

        classical = self.classical_solver.solve(problem)

        if not classical.feasible:
            classical.selection = keep_current
            classical.energy = keep_current_energy
            classical.method = "keep_current"
            classical.feasible = True

        quantum = None

        if self.use_quantum and self.quantum_solver.can_solve(problem):
            quantum = self.quantum_solver.solve(
                problem, derive_seed(self.qaoa_seed, f"step-{step}-block-{block_index}")
            )

        executed = classical

        if quantum is not None and quantum.feasible:
            if self.selection_policy == POLICY_QAOA_FIRST:
                executed = quantum
            elif quantum.energy <= classical.energy + 1e-9:
                executed = quantum

        admitted = []
        failed = False

        for allocation in released:

            route = candidates[allocation.id][executed.selection[allocation.id]]

            ok, _ = self.admission.admit(
                allocation.communication, allocation.source_qpu,
                allocation.destination_qpu, route, step,
            )

            if not ok:
                failed = True
                break

            admitted.append(allocation)

        if failed:

            result.admission_failures += 1

            for allocation in admitted:
                self.admission.release(allocation.id, consumed=False)

            for allocation in released:
                self.admission.admit(
                    allocation.communication, allocation.source_qpu,
                    allocation.destination_qpu, allocation.route, step,
                )

            executed_selection = keep_current
            executed_method = "keep_current_after_admission_failure"
        else:
            executed_selection = executed.selection
            executed_method = executed.method

        return BlockDecision(
            step=step,
            block_index=block_index,
            communication_ids=[allocation.id for allocation in released],
            num_variables=problem.num_variables,
            capacity_conflict_pairs=problem.capacity_conflict_pairs,
            classical_method=classical.method,
            classical_energy=classical.energy,
            classical_evaluations=classical.evaluations,
            qaoa_attempted=quantum is not None,
            qaoa_feasible=bool(quantum and quantum.feasible),
            qaoa_energy=quantum.energy if quantum and quantum.feasible else None,
            qaoa_evaluations=quantum.evaluations if quantum else 0,
            qaoa_details=quantum.details if quantum else {},
            executed_method=executed_method,
            executed_energy=problem.selection_energy(executed_selection),
            executed_raw_cost=problem.raw_cost(executed_selection),
            keep_current_energy=keep_current_energy,
            changed_routes=sum(1 for index in executed_selection.values() if index != 0),
        )
