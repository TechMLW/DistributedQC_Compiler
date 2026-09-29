import hashlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone

import qiskit

import config
from compiler.admission import AdmissionController
from compiler.circuits import (
    RANDOMIZED_BENCHMARKS,
    circuit_summary,
    circuit_to_qasm,
    generate_benchmark,
    load_circuit_file,
    normalize_circuit,
    two_qubit_interactions,
)
from compiler.communication_cost import RouteCostModel, RouteCostWeights
from compiler.communication_graph import CommunicationGraph, communication_graph_to_dict
from compiler.initial_partitioner import CapacityAwareKLPartitioner, validate_partition
from compiler.partition_objective import static_partition_metrics
from compiler.quantum_optimizer import QAOARouteSolver
from compiler.repartition_policy import RepartitionTriggerPolicy
from compiler.route_optimization import ClassicalRouteSolver
from compiler.router import Router
from compiler.selective_repartitioner import SelectiveRepartitioner
from compiler.selective_router import SelectiveCongestionRouter
from compiler.workload import DemandStream, build_execution_windows
from experiment.output import open_run_loggers
from network.evolution import EvolutionParameters, NetworkEvolution
from network.topology_generator import STOCHASTIC_TOPOLOGIES, generate_topology
from simulation.adaptive_simulator import AdaptiveSimulator
from simulation import reporting
from utils.seeding import derived_rng
from utils.visualizer import Visualizer
from validation.invariants import validate_output_location, validate_simulation


@dataclass
class ExperimentResult:
    seed_plan: object
    settings: object
    directory: object
    circuit_summary: dict
    interactions: list
    communication_graph: dict
    initial_partition: dict
    topology_initial: dict
    simulator: object
    topology: object
    summary: dict
    baseline_simulator: object = None
    baseline_topology: object = None
    baseline_summary: dict = None
    validation: list = field(default_factory=list)
    fingerprint: dict = field(default_factory=dict)

    @property
    def final_partition(self):
        return self.simulator.partition

    @property
    def passed_validation(self):
        return all(check["passed"] for check in self.validation)


def build_circuit(settings, seed_plan):

    if settings.circuit_file:
        circuit = load_circuit_file(settings.circuit_file)
        source = f"file:{settings.circuit_file}"
    else:
        circuit = generate_benchmark(
            settings.benchmark,
            settings.num_logical_qubits,
            settings.layers,
            derived_rng(seed_plan.seed_for("circuit"), "benchmark"),
        )
        source = f"benchmark:{settings.benchmark}"

    circuit = normalize_circuit(circuit, seed_plan.seed_for("circuit") % (2 ** 31))

    if circuit.num_qubits != settings.num_logical_qubits:
        raise ValueError(
            f"The circuit has {circuit.num_qubits} qubits but --qubits is "
            f"{settings.num_logical_qubits}"
        )

    return circuit, source


def cost_model_from_config():

    return RouteCostModel(RouteCostWeights(
        latency=config.ROUTE_COST_LATENCY_WEIGHT,
        hop=config.ROUTE_COST_HOP_WEIGHT,
        infidelity=config.ROUTE_COST_INFIDELITY_WEIGHT,
        utilization=config.ROUTE_COST_UTILIZATION_WEIGHT,
        congestion=config.ROUTE_COST_CONGESTION_WEIGHT,
        bell_scarcity=config.ROUTE_COST_BELL_SCARCITY_WEIGHT,
    ))


def build_topology(settings, seed_plan):

    return generate_topology(
        model=settings.topology,
        qpu_capacities=settings.qpu_capacities,
        ranges=settings.link_ranges,
        congestion_threshold=settings.congestion_threshold,
        topology_seed=seed_plan.seed_for("topology"),
        extra_link_probability=settings.extra_link_probability,
    )


def build_simulator(settings, seed_plan, topology, initial_partition, windows,
                    use_quantum, loggers):

    router = Router(config.CANDIDATE_ROUTES_PER_PAIR, config.MAX_EXTRA_HOPS)
    cost_model = cost_model_from_config()

    admission = AdmissionController(
        topology, config.BELL_PAIRS_PER_UNIT_LOAD, config.WINDOW_DURATION
    )

    quantum_solver = QAOARouteSolver(
        reps=config.QAOA_REPS,
        max_qubits=config.QAOA_MAX_QUBITS,
        maxiter=config.QAOA_MAXITER,
        shots=config.QAOA_SHOTS,
    ) if use_quantum else None

    selective_router = SelectiveCongestionRouter(
        router=router,
        admission=admission,
        cost_model=cost_model,
        classical_solver=ClassicalRouteSolver(config.CLASSICAL_EXHAUSTIVE_LIMIT),
        quantum_solver=quantum_solver,
        use_quantum=use_quantum,
        selection_policy=config.QAOA_SELECTION_POLICY,
        block_max_variables=config.QAOA_MAX_QUBITS,
        qaoa_seed=seed_plan.seed_for("qaoa"),
        qubo_parameters={
            "contention_weight": config.QUBO_CONTENTION_WEIGHT,
            "onehot_penalty_scale": config.QUBO_ONEHOT_PENALTY_SCALE,
            "capacity_penalty_scale": config.QUBO_CAPACITY_PENALTY_SCALE,
        },
    )

    trigger_policy = RepartitionTriggerPolicy(
        enabled_reasons=config.REPARTITION_TRIGGER_REASONS,
        congestion_persistence_windows=config.CONGESTION_PERSISTENCE_WINDOWS,
        resource_persistence_windows=config.RESOURCE_DEFERRAL_PERSISTENCE_WINDOWS,
        latency_limit=config.TRIGGER_LATENCY_LIMIT,
        fidelity_limit=config.TRIGGER_FIDELITY_LIMIT,
        cooldown_windows=config.REPARTITION_COOLDOWN_WINDOWS,
    )

    repartitioner = SelectiveRepartitioner(
        router=router,
        cost_model=cost_model,
        qpu_capacities=settings.qpu_capacities,
        lookahead_windows=config.REPARTITION_LOOKAHEAD_WINDOWS,
        max_moves=config.REPARTITION_MAX_MOVES_PER_EVENT,
        improvement_margin=config.REPARTITION_IMPROVEMENT_MARGIN,
        deferral_penalty=config.REPARTITION_DEFERRAL_PENALTY,
        migration_weight=config.REPARTITION_MIGRATION_WEIGHT,
        bell_pairs_per_unit_load=config.BELL_PAIRS_PER_UNIT_LOAD,
    )

    evolution = NetworkEvolution(
        seed_plan.seed_for("network"),
        EvolutionParameters(
            latency_step_fraction=config.EVOLUTION_LATENCY_STEP_FRACTION,
            latency_bounds_fraction=config.EVOLUTION_LATENCY_BOUNDS_FRACTION,
            fidelity_step=config.EVOLUTION_FIDELITY_STEP,
            fidelity_max_degradation=config.EVOLUTION_FIDELITY_MAX_DEGRADATION,
            fidelity_max_improvement=config.EVOLUTION_FIDELITY_MAX_IMPROVEMENT,
            capacity_change_probability=config.EVOLUTION_CAPACITY_CHANGE_PROBABILITY,
            capacity_max_deviation=config.EVOLUTION_CAPACITY_MAX_DEVIATION,
            mean_reversion=config.EVOLUTION_MEAN_REVERSION,
            regeneration_jitter=config.EVOLUTION_REGENERATION_JITTER,
        ),
    )

    return AdaptiveSimulator(
        topology=topology,
        initial_partition=initial_partition,
        demand_stream=DemandStream(windows, config.LOAD_PER_REMOTE_INTERACTION),
        evolution=evolution,
        router=router,
        cost_model=cost_model,
        admission=admission,
        selective_router=selective_router,
        trigger_policy=trigger_policy,
        repartitioner=repartitioner,
        max_queue_age=config.MAX_QUEUE_AGE_WINDOWS,
        loggers=loggers,
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def fingerprint_result(result):

    simulator = result.simulator

    return {
        "circuit": _digest(result.interactions),
        "communication_graph": _digest(result.communication_graph),
        "topology_links": _digest(result.topology_initial["summary"]["physical_links"]),
        "link_properties": _digest([
            {k: v for k, v in link.items() if k not in {"active_load", "available_bell_pairs"}}
            for link in result.topology_initial["links"]
        ]),
        "network_evolution": _digest([
            (row["step"], row["link"], row["latency"], row["fidelity"], row["capacity"])
            for row in simulator.link_state_history
        ]),
        "initial_partition": _digest(result.initial_partition),
        "final_partition": _digest(simulator.partition),
        "metrics": _digest(result.summary),
        "reroutes": _digest([r.__dict__ for r in simulator.reroute_records]),
        "route_decisions": _digest(reporting.block_rows(simulator.block_decisions)),
    }


def _log_startup(log, settings, seed_plan, summary, topology, initial_partition, windows):

    log.info("simulation seed %d", seed_plan.master_seed)

    for name, value in seed_plan.derived.items():
        log.info("derived %s = %d", name, value)

    for name, value in settings.controls().items():
        log.info("control %s = %s", name, value)

    log.info("circuit source %s", summary["source"])
    log.info(
        "circuit: %d logical qubits, %d gates, %d two-qubit gates, %d distinct interaction pairs, "
        "%d execution windows",
        summary["num_qubits"], summary["gate_count"], summary["two_qubit_gate_count"],
        summary["distinct_interaction_pairs"], len(windows),
    )
    log.info("QPUs %d with capacities %s", len(settings.qpu_capacities), settings.qpu_capacities)
    log.info("topology model %s with %d physical links: %s",
             topology.model, len(topology.links),
             ", ".join(f"{a}-{b}" for a, b in sorted(topology.links)))

    for link in topology.all_links():
        log.info("physical %r", link)

    log.info("initial partition %s", initial_partition)


def run_experiment(settings, seed_plan, directory, console=print, make_graphs=True):

    loggers = open_run_loggers(directory, f"simulation_{seed_plan.master_seed}")

    try:
        return _run(settings, seed_plan, directory, loggers, console, make_graphs)
    finally:
        loggers.close()


def _run(settings, seed_plan, directory, loggers, console, make_graphs):

    started = datetime.now(timezone.utc)

    circuit, source = build_circuit(settings, seed_plan)
    summary_of_circuit = circuit_summary(circuit, source)
    interactions = two_qubit_interactions(circuit)

    graph = CommunicationGraph().build(circuit)
    graph_data = communication_graph_to_dict(graph)

    partitioner = CapacityAwareKLPartitioner(
        derived_rng(seed_plan.seed_for("partition"), "kernighan-lin"),
        config.PARTITION_KL_RESTARTS,
        config.PARTITION_REFINEMENT_PASSES,
    )
    initial_partition = partitioner.partition(graph, settings.qpu_capacities)

    problems = validate_partition(
        initial_partition, settings.num_logical_qubits, settings.qpu_capacities
    )

    if problems:
        raise RuntimeError("Initial partition invalid: " + "; ".join(problems))

    topology = build_topology(settings, seed_plan)
    topology_initial = topology.to_dict()
    windows = build_execution_windows(circuit)

    _log_startup(loggers.simulation, settings, seed_plan, summary_of_circuit,
                 topology, initial_partition, windows)

    console(f"[seed {seed_plan.master_seed}] circuit: {summary_of_circuit['num_qubits']} qubits, "
            f"{summary_of_circuit['two_qubit_gate_count']} two-qubit gates; topology "
            f"{topology.model} with {len(topology.links)} links; running {settings.steps} windows"
            + (" with QAOA" if settings.use_quantum else " (classical)"))

    simulator = build_simulator(settings, seed_plan, topology, initial_partition, windows,
                                settings.use_quantum, loggers)
    simulator.run(settings.steps)

    loggers.simulation.info("final partition %s", simulator.partition)

    summary = reporting.summarise(simulator)

    probe_router = Router(config.CANDIDATE_ROUTES_PER_PAIR, config.MAX_EXTRA_HOPS)
    probe_topology = build_topology(settings, seed_plan)
    cost_model = cost_model_from_config()

    summary["initial_partition_static"] = static_partition_metrics(
        graph, initial_partition, probe_topology, probe_router, cost_model
    )
    summary["final_partition_static"] = static_partition_metrics(
        graph, simulator.partition, probe_topology, probe_router, cost_model
    )

    initial_map = {q: qpu for qpu, qubits in initial_partition.items() for q in qubits}
    final_map = {q: qpu for qpu, qubits in simulator.partition.items() for q in qubits}
    summary["qubits_relocated_from_initial"] = sorted(
        q for q in final_map if final_map[q] != initial_map[q]
    )

    result = ExperimentResult(
        seed_plan=seed_plan,
        settings=settings,
        directory=directory,
        circuit_summary=summary_of_circuit,
        interactions=interactions,
        communication_graph=graph_data,
        initial_partition=initial_partition,
        topology_initial=topology_initial,
        simulator=simulator,
        topology=topology,
        summary=summary,
    )

    if settings.run_baseline:

        baseline_directory = directory.child("baseline_classical")
        baseline_loggers = open_run_loggers(
            baseline_directory, f"classical_baseline_{seed_plan.master_seed}"
        )

        try:
            baseline_topology = build_topology(settings, seed_plan)
            baseline = build_simulator(settings, seed_plan, baseline_topology,
                                       initial_partition, windows, False, baseline_loggers)
            baseline.run(settings.steps)
        finally:
            baseline_loggers.close()

        result.baseline_simulator = baseline
        result.baseline_topology = baseline_topology
        result.baseline_summary = reporting.summarise(baseline)

        _write_simulation_metrics(baseline_directory, baseline, baseline_topology,
                                  result.baseline_summary)

    result.validation = validate_simulation(
        simulator, topology, settings.num_logical_qubits, settings.qpu_capacities,
        "qaoa_hybrid" if settings.use_quantum else "classical",
    )

    if result.baseline_simulator is not None:
        result.validation += validate_simulation(
            result.baseline_simulator, result.baseline_topology,
            settings.num_logical_qubits, settings.qpu_capacities, "classical_baseline",
        )

    result.fingerprint = fingerprint_result(result)

    _write_artifacts(result, circuit, graph, windows, started, make_graphs)

    result.validation += validate_output_location(directory)
    directory.write_json({"all_passed": result.passed_validation, "checks": result.validation},
                         "metrics", "validation_report.json")

    return result


def _write_simulation_metrics(directory, simulator, topology, summary):

    reporting.write_step_records(directory.path("metrics", "step_records.csv"),
                                 simulator.step_records)
    reporting.write_rows(directory.path("metrics", "link_state_history.csv"),
                         simulator.link_state_history)
    reporting.write_rows(directory.path("metrics", "link_summary.csv"),
                         reporting.link_summary_rows(topology))
    reporting.write_rows(directory.path("metrics", "reroute_log.csv"),
                         reporting.reroute_rows(simulator.reroute_records),
                         [
                             "step", "block_index", "communication_id", "qubit_u", "qubit_v",
                             "source_qpu", "destination_qpu", "old_path", "new_path",
                             "old_hops", "new_hops", "old_latency", "new_latency",
                             "old_fidelity", "new_fidelity", "old_cost", "new_cost", "method",
                         ])
    reporting.write_rows(directory.path("metrics", "affected_link_utilization.csv"),
                         simulator.affected_link_utilization,
                         ["step", "link", "utilization_before", "utilization_after"])
    reporting.write_rows(directory.path("metrics", "repartition_events.csv"),
                         reporting.repartition_rows(simulator.repartition_outcomes))
    reporting.write_rows(directory.path("metrics", "route_decision_blocks.csv"),
                         reporting.block_rows(simulator.block_decisions))
    directory.write_json(summary, "metrics", "summary.json")


def _write_artifacts(result, circuit, graph, windows, started, make_graphs):

    directory = result.directory
    simulator = result.simulator
    settings = result.settings
    topology = result.topology

    directory.write_text(circuit_to_qasm(circuit), "circuit", "circuit.qasm")
    directory.write_json(
        dict(result.circuit_summary, execution_windows=len(windows),
             randomized_structure=(settings.circuit_file is None
                                   and settings.benchmark in RANDOMIZED_BENCHMARKS)),
        "circuit", "circuit_summary.json",
    )
    directory.write_json(result.communication_graph, "circuit", "communication_graph.json")

    directory.write_json(result.topology_initial, "topology", "topology_initial.json")
    directory.write_json(topology.to_dict(), "topology", "topology_final.json")

    _write_simulation_metrics(directory, simulator, topology, result.summary)

    reporting.write_rows(directory.path("quantum", "qaoa_results.csv"),
                         reporting.block_rows(simulator.block_decisions))
    directory.write_json(
        reporting.qaoa_summary(simulator.block_decisions, config.QAOA_SELECTION_POLICY)
        if settings.use_quantum else {"qaoa_enabled": False},
        "quantum", "qaoa_summary.json",
    )

    if result.baseline_summary is not None:
        rows = []
        for key in sorted(result.summary):
            value = result.summary[key]
            if isinstance(value, (int, float)):
                rows.append({"metric": key, "qaoa_hybrid": value,
                             "classical_only": result.baseline_summary.get(key)})
        reporting.write_rows(directory.path("metrics", "baseline_comparison.csv"), rows,
                             ["metric", "qaoa_hybrid", "classical_only"])

    manifest = {
        "simulation_id": f"simulation_{result.seed_plan.master_seed}",
        "created_utc": started.isoformat(),
        "code_environment": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
        },
        **result.seed_plan.to_dict(),
        "experiment": settings.to_dict(),
        "stochastic_components": {
            "circuit_structure": settings.circuit_file is None
            and settings.benchmark in RANDOMIZED_BENCHMARKS,
            "topology_edges": settings.topology in STOCHASTIC_TOPOLOGIES,
            "link_properties": True,
            "network_evolution": True,
            "partition_tie_breaking": True,
            "qaoa": settings.use_quantum,
        },
        "circuit": dict(result.circuit_summary, execution_windows=len(windows)),
        "communication_graph": {
            k: v for k, v in result.communication_graph.items() if k != "edges"
        },
        "topology": result.topology_initial["summary"],
        "link_properties_initial": result.topology_initial["links"],
        "initial_partition": {str(k): v for k, v in result.initial_partition.items()},
        "final_partition": {str(k): v for k, v in simulator.partition.items()},
        "fingerprint": result.fingerprint,
        "summary": {
            key: result.summary[key] for key in (
                "congestion_windows", "affected_communications", "selective_reroutes",
                "repartition_triggers", "repartitions_accepted", "logical_qubits_moved",
                "prevented_capacity_violations", "max_link_utilization",
            )
        },
    }
    directory.write_json(manifest, "manifest.json")

    if not make_graphs:
        return

    visualizer = Visualizer()
    graphs = lambda name: directory.path("graphs", name)

    visualizer.communication_graph(
        graph, graphs("communication_graph.png"),
        f"{graph.number_of_nodes()} logical qubits, {graph.number_of_edges()} interacting pairs, "
        f"{result.circuit_summary['two_qubit_gate_count']} two-qubit gates "
        f"(edge weight = gate count)",
    )

    static_initial = result.summary["initial_partition_static"]
    static_final = result.summary["final_partition_static"]

    visualizer.partition(
        graph, result.initial_partition, topology, graphs("initial_partition.png"),
        "Initial partition (capacity-aware recursive Kernighan-Lin)",
        [f"cut weight {static_initial['cut_weight']} of "
         f"{static_initial['cut_weight'] + static_initial['local_weight']}",
         f"static communication cost {static_initial['static_communication_cost']:.1f}"],
    )

    moved_overall = sorted(
        q for q, qpu in {q: p for p, qs in simulator.partition.items() for q in qs}.items()
        if qpu != {q: p for p, qs in result.initial_partition.items() for q in qs}[q]
    )

    visualizer.partition(
        graph, simulator.partition, topology, graphs("final_partition.png"),
        f"Final partition after {settings.steps} windows",
        [f"cut weight {static_final['cut_weight']}",
         f"static communication cost {static_final['static_communication_cost']:.1f}",
         f"accepted selective repartitions {result.summary['repartitions_accepted']}",
         f"qubits whose QPU differs from initial: {moved_overall}"],
        highlight_qubits=moved_overall, previous_partition=result.initial_partition,
    )

    initial_topology = build_topology(settings, result.seed_plan)
    visualizer.topology(initial_topology, graphs("topology_initial.png"),
                        f"Physical QPU topology ({topology.model}, generated from topology seed)")
    visualizer.topology(
        topology, graphs("topology_final.png"),
        f"Physical QPU topology after {settings.steps} windows (same links, evolved properties)",
        congestion_counts={link.key: link.congested_windows for link in topology.all_links()},
    )

    visualizer.utilization_heatmap(simulator.link_state_history, settings.congestion_threshold,
                                   graphs("utilization_timeline.png"))

    accepted_steps = [o.step for o in simulator.repartition_outcomes if o.accepted]

    visualizer.congestion_timeline(simulator.step_records, graphs("congestion_timeline.png"),
                                   accepted_steps)
    visualizer.rerouting_timeline(simulator.step_records, graphs("rerouting_timeline.png"))
    visualizer.affected_link_utilization(
        simulator.affected_link_utilization, settings.congestion_threshold,
        graphs("affected_link_utilization.png"),
    )

    accepted = [o for o in simulator.repartition_outcomes if o.accepted]

    for index, outcome in enumerate(accepted, start=1):
        visualizer.partition(
            graph, outcome.new_partition, topology,
            graphs(f"repartition_{index:02d}_step_{outcome.step:03d}.png"),
            f"Selective repartition #{index} at step {outcome.step} (reason: {outcome.reason})",
            [f"trigger links: {', '.join(f'{a}-{b}' for a, b in outcome.trigger_link_keys)}",
             f"affected qubits (movable): {outcome.affected_qubits}",
             f"moved qubits: {outcome.moved_qubits}",
             f"objective {outcome.before.objective:.1f} -> {outcome.after.objective:.1f}",
             f"projected deferred load {outcome.before.deferred_load:.0f} -> "
             f"{outcome.after.deferred_load:.0f}"],
            highlight_qubits=outcome.moved_qubits, previous_partition=outcome.old_partition,
        )

    if settings.use_quantum:
        visualizer.qaoa_vs_classical(simulator.block_decisions,
                                     directory.path("quantum", "qaoa_vs_classical.png"))
