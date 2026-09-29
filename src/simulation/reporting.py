import csv
from collections import Counter
from dataclasses import asdict, fields, is_dataclass


def write_rows(path, rows, field_names=None):

    rows = [asdict(row) if is_dataclass(row) else dict(row) for row in rows]

    if field_names is None:
        field_names = list(rows[0].keys()) if rows else []

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_step_records(path, records):

    from simulation.records import StepRecord

    write_rows(path, records, [field.name for field in fields(StepRecord)])


def reroute_rows(records):
    return [asdict(record) for record in records]


def block_rows(blocks):

    rows = []

    for block in blocks:

        details = block.qaoa_details or {}

        rows.append({
            "step": block.step,
            "block_index": block.block_index,
            "communications": len(block.communication_ids),
            "communication_ids": "|".join(block.communication_ids),
            "qubo_variables": block.num_variables,
            "capacity_conflict_pairs": block.capacity_conflict_pairs,
            "classical_method": block.classical_method,
            "classical_energy": block.classical_energy,
            "classical_evaluations": block.classical_evaluations,
            "qaoa_attempted": block.qaoa_attempted,
            "qaoa_feasible": block.qaoa_feasible,
            "qaoa_energy": block.qaoa_energy,
            "qaoa_matches_classical_optimum": (
                block.qaoa_feasible
                and block.classical_method == "classical_exhaustive"
                and abs(block.qaoa_energy - block.classical_energy) < 1e-9
            ),
            "qaoa_evaluations": block.qaoa_evaluations,
            "qaoa_run_seed": details.get("run_seed"),
            "qaoa_feasible_sample_fraction": details.get("feasible_sample_fraction"),
            "qaoa_onehot_sample_fraction": details.get("onehot_sample_fraction"),
            "qaoa_optimized_expectation": details.get("optimized_expectation"),
            "executed_method": block.executed_method,
            "executed_energy": block.executed_energy,
            "executed_raw_cost": block.executed_raw_cost,
            "keep_current_energy": block.keep_current_energy,
            "routes_changed": block.changed_routes,
        })

    return rows


def repartition_rows(outcomes):

    rows = []

    for index, outcome in enumerate(outcomes, start=1):
        rows.append({
            "event_id": index,
            "step": outcome.step,
            "reason": outcome.reason,
            "reason_detail": outcome.reason_detail,
            "trigger_links": "|".join(f"{a}-{b}" for a, b in outcome.trigger_link_keys),
            "affected_qubits": " ".join(str(q) for q in outcome.affected_qubits),
            "accepted": outcome.accepted,
            "rejection_reason": outcome.rejection_reason,
            "moves": "; ".join(
                f"{m['type']} {m['qubits']} {m['from']}->{m['to']}" for m in outcome.moves
            ) if outcome.accepted else "",
            "qubits_moved": " ".join(str(q) for q in outcome.moved_qubits)
            if outcome.accepted else "",
            "unaffected_qubits_moved": len(outcome.unaffected_qubits_moved),
            "candidates_evaluated": outcome.candidates_evaluated,
            "objective_before": outcome.before.objective,
            "objective_after": outcome.after.objective,
            "communication_cost_before": outcome.before.communication_cost,
            "communication_cost_after": outcome.after.communication_cost,
            "deferred_load_before": outcome.before.deferred_load,
            "deferred_load_after": outcome.after.deferred_load,
            "migration_cost": outcome.after.migration_cost,
            "trigger_link_peak_utilization_before": max(
                (outcome.before.peak_link_utilization[k] for k in outcome.trigger_link_keys),
                default=0.0,
            ),
            "trigger_link_peak_utilization_after": max(
                (outcome.after.peak_link_utilization[k] for k in outcome.trigger_link_keys),
                default=0.0,
            ),
            "old_partition": str(outcome.old_partition),
            "new_partition": str(outcome.new_partition) if outcome.accepted else "",
        })

    return rows


def link_summary_rows(topology):

    return [
        {
            "link": f"{link.key[0]}-{link.key[1]}",
            "baseline_latency": link.baseline.latency,
            "final_latency": link.latency,
            "baseline_fidelity": link.baseline.fidelity,
            "final_fidelity": link.fidelity,
            "baseline_capacity": link.baseline.capacity,
            "final_capacity": link.capacity,
            "bell_pair_pool": link.bell_pair_pool,
            "bell_pair_regeneration": link.bell_pair_regeneration,
            "final_available_bell_pairs": round(link.available_bell_pairs, 4),
            "bell_pairs_consumed": round(link.consumed_bell_pairs, 4),
            "final_active_load": link.active_load,
            "peak_utilization": round(link.peak_utilization, 6),
            "congested_windows": link.congested_windows,
            "saturated_windows": link.saturated_windows,
            "rejected_reservations": link.rejected_allocations,
            "congestion_threshold": link.congestion_threshold,
        }
        for link in topology.all_links()
    ]


def summarise(simulator):

    records = simulator.step_records

    def total(name):
        return sum(getattr(record, name) for record in records)

    def mean(name):
        values = [getattr(record, name) for record in records]
        return sum(values) / len(values) if values else 0.0

    outcomes = simulator.repartition_outcomes
    accepted = [outcome for outcome in outcomes if outcome.accepted]
    blocks = simulator.block_decisions
    rerouted_steps = [record for record in records if record.affected_communications]

    return {
        "steps": len(records),
        "remote_demand_instances": total("remote_demand"),
        "fresh_interactions": total("fresh_interactions"),
        "local_interactions": total("local_interactions"),
        "admitted": total("admitted"),
        "admitted_on_primary_route": total("admitted_on_primary_route"),
        "diverted_at_admission": total("diverted_at_admission"),
        "deferral_events": total("deferred"),
        "dropped_from_queue": total("dropped_from_queue"),
        "queue_length_final": records[-1].queue_length_after if records else 0,
        "completed_communications": total("completed_communications"),
        "in_flight_at_end": len(simulator.admission.allocations),
        "prevented_capacity_violations": total("prevented_capacity_violations"),
        "rejected_for_capacity": total("rejected_for_capacity"),
        "rejected_for_bell_pairs": total("rejected_for_bell_pairs"),
        "congestion_windows": sum(1 for r in records if r.congested_links_before_reroute),
        "congested_link_instances_before_reroute": total("congested_links_before_reroute"),
        "congested_link_instances_after_reroute": total("congested_links_after_reroute"),
        "saturated_link_instances_after_reroute": total("saturated_links_after_reroute"),
        "affected_communications": total("affected_communications"),
        "unaffected_communications_on_congestion_windows": sum(
            r.unaffected_communications for r in rerouted_steps
        ),
        "selective_reroutes": total("selective_reroutes"),
        "affected_kept_route": total("affected_kept_route"),
        "affected_cost_before_reroute": total("affected_cost_before_reroute"),
        "affected_cost_after_reroute": total("affected_cost_after_reroute"),
        "max_link_utilization": max(
            (r.max_utilization_before_reroute for r in records), default=0.0
        ),
        "mean_max_utilization_before_reroute_on_congestion_windows": (
            sum(r.max_utilization_before_reroute for r in rerouted_steps) / len(rerouted_steps)
            if rerouted_steps else 0.0
        ),
        "mean_max_utilization_after_reroute_on_congestion_windows": (
            sum(r.max_utilization_after_reroute for r in rerouted_steps) / len(rerouted_steps)
            if rerouted_steps else 0.0
        ),
        "avg_link_utilization": mean("avg_link_utilization"),
        "mean_window_communication_cost": mean("window_communication_cost"),
        "total_window_communication_cost": total("window_communication_cost"),
        "repartition_triggers": len(outcomes),
        "repartition_triggers_by_reason": dict(Counter(o.reason_detail for o in outcomes)),
        "repartition_suppressed_by_cooldown": sum(
            1 for r in records if r.repartition_suppressed_by_cooldown
        ),
        "repartitions_accepted": len(accepted),
        "repartition_rejections_by_reason": dict(
            Counter(o.rejection_reason for o in outcomes if not o.accepted)
        ),
        "logical_qubits_moved": sum(len(o.moved_qubits) for o in accepted),
        "reroute_blocks": len(blocks),
        "qaoa_blocks_attempted": sum(1 for b in blocks if b.qaoa_attempted),
        "qaoa_blocks_feasible": sum(1 for b in blocks if b.qaoa_feasible),
        "qaoa_blocks_executed": sum(1 for b in blocks if b.executed_method == "qaoa"),
        "capacity_reductions_blocked_by_reservations": (
            simulator.evolution.capacity_reductions_blocked
        ),
        "capacity_invariant_violations": sum(1 for r in records if not r.capacity_invariant_ok),
        "load_ledger_violations": sum(1 for r in records if not r.load_ledger_ok),
        "unaffected_route_violations": total("unaffected_route_violations"),
        "reroute_admission_failures": total("reroute_admission_failures"),
    }


def qaoa_summary(blocks, policy):

    attempted = [b for b in blocks if b.qaoa_attempted]
    feasible = [b for b in attempted if b.qaoa_feasible]
    exact = [b for b in feasible if b.classical_method == "classical_exhaustive"]

    gaps = [b.qaoa_energy - b.classical_energy for b in exact]

    return {
        "selection_policy": policy,
        "decision_variables": "x[c,r] = 1 if affected communication c uses candidate route r",
        "reroute_blocks": len(blocks),
        "qaoa_attempted": len(attempted),
        "qaoa_not_attempted_too_large_or_disabled": len(blocks) - len(attempted),
        "qaoa_feasible": len(feasible),
        "qaoa_infeasible": len(attempted) - len(feasible),
        "qaoa_executed": sum(1 for b in blocks if b.executed_method == "qaoa"),
        "qaoa_matched_exact_classical_optimum": sum(1 for g in gaps if abs(g) < 1e-9),
        "qaoa_compared_with_exact_classical": len(exact),
        "mean_energy_gap_to_exact_classical": sum(gaps) / len(gaps) if gaps else None,
        "max_energy_gap_to_exact_classical": max(gaps) if gaps else None,
        "mean_feasible_sample_fraction": (
            sum(b.qaoa_details.get("feasible_sample_fraction", 0.0) for b in attempted)
            / len(attempted) if attempted else None
        ),
        "note": (
            "Energies are evaluated on the same QUBO for both solvers; executed_energy in "
            "qaoa_results.csv is the energy of the route selection actually admitted."
        ),
    }
