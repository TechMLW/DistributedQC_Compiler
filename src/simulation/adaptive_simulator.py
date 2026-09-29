from compiler.admission import REJECT_BELL_PAIRS, REJECT_CAPACITY
from compiler.congestion import CongestionDetector
from compiler.workload import qubit_to_qpu_map
from simulation.records import StepRecord


def _format_keys(keys):
    return "|".join(f"{a}-{b}" for a, b in keys)


def _format_partition(partition):
    return {qpu: list(qubits) for qpu, qubits in sorted(partition.items())}


class AdaptiveSimulator:

    def __init__(
        self,
        topology,
        initial_partition,
        demand_stream,
        evolution,
        router,
        cost_model,
        admission,
        selective_router,
        trigger_policy,
        repartitioner,
        max_queue_age,
        loggers,
    ):

        self.topology = topology
        self.partition = {qpu: list(qubits) for qpu, qubits in initial_partition.items()}
        self.demand_stream = demand_stream
        self.evolution = evolution
        self.router = router
        self.cost_model = cost_model
        self.admission = admission
        self.detector = CongestionDetector()
        self.selective_router = selective_router
        self.trigger_policy = trigger_policy
        self.repartitioner = repartitioner
        self.max_queue_age = max_queue_age
        self.log = loggers

        self.queue = []
        self.dropped_total = 0

        self.step_records = []
        self.reroute_records = []
        self.block_decisions = []
        self.affected_link_utilization = []
        self.repartition_outcomes = []
        self.link_state_history = []
        self.partition_history = [(0, _format_partition(self.partition))]

    def run(self, steps):

        for step in range(1, steps + 1):
            self.run_step(step)

    def run_step(self, step):

        log = self.log.simulation

        self.router.begin_window()
        self.admission.reset_window_counters()

        evolution_changes = self.evolution.evolve(self.topology)
        regenerated = sum(change["bell_pairs_regenerated"] for change in evolution_changes)

        window, fresh = self.demand_stream.communications_for_step(step)

        carried, dropped = self._age_queue(step)
        demand = carried + fresh

        qubit_map = qubit_to_qpu_map(self.partition)

        local = []
        remote = []

        for communication in demand:
            if qubit_map[communication.qubit_u] == qubit_map[communication.qubit_v]:
                local.append(communication)
            else:
                remote.append(communication)

        became_local = sum(1 for communication in local if communication.age > 0)

        in_flight = self.admission.in_flight()
        in_flight_load = sum(a.load * a.route.hop_count for a in in_flight)

        admitted, primary, diverted, deferred = self._initial_admission(step, remote, qubit_map)

        report = self.detector.analyse(self.topology, self.admission, step)
        max_before, _ = self.topology.utilization_summary()

        if report.has_congestion:
            log.info(
                "step %d congestion on links %s (utilization %s); affected=%d unaffected=%d "
                "in_flight_on_congested=%d",
                step, _format_keys(report.congested_link_keys),
                ", ".join(
                    f"{a}-{b}:{report.link_utilization[(a, b)]:.2f}"
                    for a, b in report.congested_link_keys
                ),
                report.affected_count, report.unaffected_count,
                report.in_flight_on_congested,
            )

        unaffected_routes_before = {
            cid: self.admission.route_of(cid) for cid in report.unaffected_communication_ids
        }

        reroute = self.selective_router.reroute(step, self.topology, report)

        unaffected_violations = sum(
            1 for cid, route in unaffected_routes_before.items()
            if self.admission.route_of(cid) != route
        )

        self._log_reroute(step, report, reroute)

        self.reroute_records.extend(reroute.reroutes)
        self.block_decisions.extend(reroute.blocks)
        self.affected_link_utilization.extend(reroute.affected_link_utilization)

        max_after, avg_after = self.topology.utilization_summary()
        congested_after = self.topology.congested_links()
        saturated_after = self.topology.saturated_links()

        capacity_ok = self.topology.capacity_invariant_holds()
        ledger_ok = self.admission.load_ledger_consistent()

        if not capacity_ok or not ledger_ok or unaffected_violations:
            log.error(
                "step %d invariant failure: capacity_ok=%s ledger_ok=%s unaffected_rerouted=%d",
                step, capacity_ok, ledger_ok, unaffected_violations,
            )

        self.topology.observe_window()

        resource_deferred = [
            communication for communication, reason in deferred
            if reason in (REJECT_CAPACITY, REJECT_BELL_PAIRS)
        ]

        decision = self.trigger_policy.evaluate(
            step, self.topology, self.admission, resource_deferred
        )

        outcome = None

        if decision.triggered:
            outcome = self._selective_repartition(step, decision, [c for c, _ in deferred])
        elif decision.suppressed_by_cooldown:
            self.log.repartition.info(
                "step %d trigger '%s' suppressed by cooldown (last repartition at step %s)",
                step, decision.reason_detail, self.trigger_policy.last_repartition_step,
            )

        window_cost = sum(
            self.cost_model.communication_cost(a.load, a.route, self.topology)
            for a in self.admission.allocations.values() if a.admitted_step == step
        )

        self._record_link_states(step)

        self.admission.start_admitted(step)
        completed = self.admission.complete_window()

        if not self.admission.load_ledger_consistent():
            ledger_ok = False
            log.error("step %d load ledger inconsistent after releasing completions", step)

        self.queue = [communication for communication, _ in deferred]

        if deferred:
            log.info(
                "step %d deferred %d communications to queue (%s); queue length %d",
                step, len(deferred),
                ", ".join(sorted({reason for _, reason in deferred})), len(self.queue),
            )

        if dropped:
            log.warning(
                "step %d dropped %d queued communications older than %d windows",
                step, dropped, self.max_queue_age,
            )

        record = StepRecord(
            step=step,
            window_index=window.index if window else -1,
            fresh_interactions=len(fresh),
            local_interactions=len(local),
            remote_demand=len(remote),
            carried_from_queue=len(carried),
            became_local_after_repartition=became_local,
            admitted=len(admitted),
            admitted_on_primary_route=primary,
            diverted_at_admission=diverted,
            deferred=len(deferred),
            deferred_for_capacity=sum(1 for _, r in deferred if r == REJECT_CAPACITY),
            deferred_for_bell_pairs=sum(1 for _, r in deferred if r == REJECT_BELL_PAIRS),
            dropped_from_queue=dropped,
            queue_length_after=len(self.queue),
            prevented_capacity_violations=self.admission.prevented_capacity_violations,
            rejected_for_capacity=self.admission.rejected_for_capacity,
            rejected_for_bell_pairs=self.admission.rejected_for_bell_pairs,
            rejected_for_no_route=self.admission.rejected_for_no_route,
            in_flight_communications=len(in_flight),
            in_flight_load=in_flight_load,
            congested_links_before_reroute=len(report.congested_link_keys),
            saturated_links_before_reroute=len(report.saturated_link_keys),
            congested_link_keys=_format_keys(report.congested_link_keys),
            affected_communications=report.affected_count,
            unaffected_communications=report.unaffected_count,
            in_flight_on_congested_links=report.in_flight_on_congested,
            selective_reroutes=reroute.reroute_count,
            affected_kept_route=len(reroute.kept_ids),
            reroute_blocks=len(reroute.blocks),
            qaoa_blocks=sum(1 for block in reroute.blocks if block.qaoa_attempted),
            qaoa_executed_blocks=sum(
                1 for block in reroute.blocks if block.executed_method == "qaoa"
            ),
            affected_cost_before_reroute=reroute.affected_cost_before,
            affected_cost_after_reroute=reroute.affected_cost_after,
            congested_links_after_reroute=len(congested_after),
            saturated_links_after_reroute=len(saturated_after),
            max_utilization_before_reroute=max_before,
            max_utilization_after_reroute=max_after,
            avg_link_utilization=avg_after,
            total_active_load=sum(link.active_load for link in self.topology.all_links()),
            window_communication_cost=window_cost,
            completed_communications=len(completed),
            bell_pairs_available=sum(
                link.available_bell_pairs for link in self.topology.all_links()
            ),
            bell_pairs_consumed_total=sum(
                link.consumed_bell_pairs for link in self.topology.all_links()
            ),
            bell_pairs_regenerated=regenerated,
            repartition_trigger_reason=decision.reason_detail,
            repartition_triggered=decision.triggered,
            repartition_suppressed_by_cooldown=decision.suppressed_by_cooldown,
            repartition_accepted=bool(outcome and outcome.accepted),
            logical_qubits_moved=len(outcome.moved_qubits) if outcome and outcome.accepted else 0,
            capacity_invariant_ok=capacity_ok,
            load_ledger_ok=ledger_ok,
            unaffected_route_violations=unaffected_violations,
            reroute_admission_failures=reroute.admission_failures,
        )

        self.step_records.append(record)

        return record

    def _age_queue(self, step):

        carried = []
        dropped = 0

        for communication in self.queue:

            communication.age += 1

            if self.max_queue_age is not None and communication.age > self.max_queue_age:
                dropped += 1
                self.log.simulation.info(
                    "step %d dropped queued communication %s (age %d)",
                    step, communication.id, communication.age,
                )
                continue

            carried.append(communication)

        self.dropped_total += dropped

        return carried, dropped

    def _initial_admission(self, step, remote, qubit_map):

        admitted = []
        deferred = []
        primary_count = 0
        diverted = 0

        ordered = sorted(remote, key=lambda c: (-c.age, c.created_step, c.sequence))

        for communication in ordered:

            source = qubit_map[communication.qubit_u]
            destination = qubit_map[communication.qubit_v]

            candidates = sorted(
                self.router.candidate_routes(self.topology, source, destination),
                key=lambda route: (self.cost_model.base_cost(route, self.topology), route.path),
            )

            if not candidates:
                self.admission.check_route(None, communication.load)
                deferred.append((communication, "no_route"))
                continue

            last_reason = None
            chosen_index = None

            for index, route in enumerate(candidates):

                ok, reason = self.admission.admit(
                    communication, source, destination, route, step
                )

                if ok:
                    chosen_index = index
                    break

                last_reason = reason

            if chosen_index is None:
                deferred.append((communication, last_reason))
                continue

            admitted.append(communication)

            if chosen_index == 0:
                primary_count += 1
            else:
                diverted += 1

        return admitted, primary_count, diverted, deferred

    def _log_reroute(self, step, report, reroute):

        log = self.log.rerouting

        if not report.affected_count:
            return

        log.info(
            "step %d selective rerouting: %d affected communications on congested links %s; "
            "%d unaffected communications left untouched",
            step, report.affected_count, _format_keys(report.congested_link_keys),
            report.unaffected_count,
        )

        for block in reroute.blocks:
            log.info(
                "step %d block %d: %d communications, %d QUBO variables, "
                "%d pairwise capacity conflicts; classical=%s E=%.4f; qaoa=%s%s; "
                "executed=%s E=%.4f (keep-current E=%.4f); routes changed=%d",
                step, block.block_index, len(block.communication_ids), block.num_variables,
                block.capacity_conflict_pairs, block.classical_method, block.classical_energy,
                "attempted" if block.qaoa_attempted else "not run",
                f" feasible E={block.qaoa_energy:.4f}" if block.qaoa_feasible
                else (" infeasible" if block.qaoa_attempted else ""),
                block.executed_method, block.executed_energy, block.keep_current_energy,
                block.changed_routes,
            )

        for record in reroute.reroutes:
            log.info(
                "step %d rerouted %s (q%d QPU%d -> q%d QPU%d): %s -> %s, cost %.2f -> %.2f",
                step, record.communication_id, record.qubit_u, record.source_qpu,
                record.qubit_v, record.destination_qpu, record.old_path, record.new_path,
                record.old_cost, record.new_cost,
            )

        for entry in reroute.affected_link_utilization:
            log.info(
                "step %d link %s utilization %.2f -> %.2f after selective rerouting",
                step, entry["link"], entry["utilization_before"], entry["utilization_after"],
            )

    def _selective_repartition(self, step, decision, deferred_communications):

        log = self.log.repartition

        queued = list(deferred_communications)

        outcome = self.repartitioner.propose(
            step, self.partition, decision, self.topology,
            self.demand_stream, queued, self.admission,
        )

        self.repartition_outcomes.append(outcome)

        log.info(
            "step %d repartition trigger reason=%s (%s) links=%s affected_qubits=%s",
            step, decision.reason, decision.reason_detail,
            _format_keys(decision.trigger_link_keys), decision.affected_qubits,
        )
        log.info(
            "step %d evaluated %d selective candidates; objective %.2f -> %.2f "
            "(communication %.2f -> %.2f, deferred load %.1f -> %.1f, migration %.2f)",
            step, outcome.candidates_evaluated, outcome.before.objective,
            outcome.after.objective, outcome.before.communication_cost,
            outcome.after.communication_cost, outcome.before.deferred_load,
            outcome.after.deferred_load, outcome.after.migration_cost,
        )

        if not outcome.accepted:
            log.info("step %d repartition rejected: %s", step, outcome.rejection_reason)
            return outcome

        before_map = qubit_to_qpu_map(self.partition)

        self.partition = {qpu: list(qubits) for qpu, qubits in outcome.new_partition.items()}
        self.trigger_policy.record_repartition(step)
        self.partition_history.append((step, _format_partition(self.partition)))

        after_map = qubit_to_qpu_map(self.partition)
        moved = sorted(q for q in after_map if after_map[q] != before_map[q])

        for move in outcome.moves:
            log.info(
                "step %d accepted %s of qubits %s: QPU %s -> QPU %s",
                step, move["type"], move["qubits"], move["from"], move["to"],
            )

        for key in outcome.trigger_link_keys:
            log.info(
                "step %d link %d-%d projected peak utilization %.2f -> %.2f",
                step, key[0], key[1], outcome.before.peak_link_utilization[key],
                outcome.after.peak_link_utilization[key],
            )

        log.info("step %d moved qubits %s; new partition %s", step, moved,
                 _format_partition(self.partition))

        return outcome

    def _record_link_states(self, step):

        for link in self.topology.all_links():
            self.link_state_history.append({
                "step": step,
                "link": f"{link.key[0]}-{link.key[1]}",
                "latency": link.latency,
                "fidelity": link.fidelity,
                "capacity": link.capacity,
                "active_load": link.active_load,
                "utilization": round(link.utilization, 6),
                "state": link.state,
                "available_bell_pairs": round(link.available_bell_pairs, 4),
                "reserved_bell_pairs": round(link.reserved_bell_pairs, 4),
                "consecutive_congested_windows": link.consecutive_congested_windows,
            })
