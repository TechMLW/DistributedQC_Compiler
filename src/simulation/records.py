from dataclasses import dataclass


@dataclass
class StepRecord:
    step: int
    window_index: int

    fresh_interactions: int
    local_interactions: int
    remote_demand: int
    carried_from_queue: int
    became_local_after_repartition: int

    admitted: int
    admitted_on_primary_route: int
    diverted_at_admission: int
    deferred: int
    deferred_for_capacity: int
    deferred_for_bell_pairs: int
    dropped_from_queue: int
    queue_length_after: int

    prevented_capacity_violations: int
    rejected_for_capacity: int
    rejected_for_bell_pairs: int
    rejected_for_no_route: int

    in_flight_communications: int
    in_flight_load: float

    congested_links_before_reroute: int
    saturated_links_before_reroute: int
    congested_link_keys: str
    affected_communications: int
    unaffected_communications: int
    in_flight_on_congested_links: int
    selective_reroutes: int
    affected_kept_route: int
    reroute_blocks: int
    qaoa_blocks: int
    qaoa_executed_blocks: int
    affected_cost_before_reroute: float
    affected_cost_after_reroute: float

    congested_links_after_reroute: int
    saturated_links_after_reroute: int
    max_utilization_before_reroute: float
    max_utilization_after_reroute: float
    avg_link_utilization: float
    total_active_load: float

    window_communication_cost: float
    completed_communications: int

    bell_pairs_available: float
    bell_pairs_consumed_total: float
    bell_pairs_regenerated: float

    repartition_trigger_reason: str
    repartition_triggered: bool
    repartition_suppressed_by_cooldown: bool
    repartition_accepted: bool
    logical_qubits_moved: int

    capacity_invariant_ok: bool
    load_ledger_ok: bool
    unaffected_route_violations: int
    reroute_admission_failures: int
