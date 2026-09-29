import argparse
import csv
import json
import os
import sys

import config
from compiler.circuits import SUPPORTED_BENCHMARKS
from experiment.output import RunDirectoryConflict, prepare_seed_directory
from experiment.reproducibility import verify_reproducibility
from experiment.runner import run_experiment
from experiment.seeds import SeedPlan, generate_master_seed
from experiment.settings import build_settings
from network.topology_generator import SUPPORTED_TOPOLOGIES


SWEEP_COLUMNS = [
    "congestion_windows", "affected_communications", "selective_reroutes",
    "prevented_capacity_violations", "deferral_events", "dropped_from_queue",
    "max_link_utilization", "avg_link_utilization", "mean_window_communication_cost",
    "repartition_triggers", "repartitions_accepted", "logical_qubits_moved",
    "qaoa_blocks_attempted", "qaoa_blocks_executed",
]


def parse_arguments(argv=None):

    parser = argparse.ArgumentParser(
        description="Selective Congestion-Triggered Qubit Routing: variable-network simulator"
    )

    controls = parser.add_argument_group("experimental controls")
    controls.add_argument("--seed", type=int, default=None,
                          help="master seed; omitted = fresh random seed")
    controls.add_argument("--qubits", type=int, default=None,
                          help=f"logical qubits (default {config.DEFAULT_NUM_LOGICAL_QUBITS})")
    controls.add_argument("--qpus", type=int, default=config.DEFAULT_NUM_QPUS)
    controls.add_argument("--qpu-capacity", type=int, default=config.DEFAULT_QPU_CAPACITY,
                          help="maximum logical qubits per QPU (uniform)")
    controls.add_argument("--qpu-capacities", type=str, default=None,
                          help="comma-separated per-QPU capacities, overrides --qpu-capacity")
    controls.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS,
                          default=config.DEFAULT_BENCHMARK)
    controls.add_argument("--layers", type=int, default=config.DEFAULT_BENCHMARK_LAYERS,
                          help="layers for layered_random, random and hardware_efficient")
    controls.add_argument("--circuit-file", type=str, default=None,
                          help="OpenQASM file used instead of a generated benchmark")
    controls.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES,
                          default=config.DEFAULT_TOPOLOGY)
    controls.add_argument("--steps", type=int, default=config.DEFAULT_SIMULATION_STEPS)
    controls.add_argument("--congestion-threshold", type=float,
                          default=config.DEFAULT_CONGESTION_THRESHOLD)

    physical = parser.add_argument_group("physical-link property ranges (min,max)")
    physical.add_argument("--link-latency-range", type=str, default=None)
    physical.add_argument("--link-fidelity-range", type=str, default=None)
    physical.add_argument("--link-capacity-range", type=str, default=None)
    physical.add_argument("--link-bell-pairs-range", type=str, default=None)
    physical.add_argument("--link-regeneration-range", type=str, default=None)
    physical.add_argument("--extra-link-probability", type=float,
                          default=config.TOPOLOGY_EXTRA_LINK_PROBABILITY,
                          help="random_connected: probability of each non-tree link")

    modes = parser.add_argument_group("modes")
    modes.add_argument("--no-quantum", action="store_true",
                       help="classical route optimisation only")
    modes.add_argument("--no-baseline", action="store_true",
                       help="skip the classical-only baseline replay")
    modes.add_argument("--no-graphs", action="store_true")
    modes.add_argument("--overwrite", action="store_true",
                       help="replace an existing seed directory holding different controls")
    modes.add_argument("--seeds", type=str, default=None,
                       help="seed sweep: comma-separated master seeds, same controls")
    modes.add_argument("--num-seeds", type=int, default=None,
                       help="seed sweep: this many fresh seeds, same controls")
    modes.add_argument("--verify-reproducibility", action="store_true",
                       help="run the seed twice plus seed+1 in temporary directories and compare")
    modes.add_argument("--results-dir", type=str, default=config.RESULTS_DIRECTORY)

    return parser.parse_args(argv)


def run_single(settings, master_seed, arguments):

    seed_plan = SeedPlan.from_master(master_seed)

    directory = prepare_seed_directory(
        arguments.results_dir, master_seed, settings.controls(), arguments.overwrite
    )

    print(f"Simulation seed {master_seed} -> {directory.root}")

    result = run_experiment(settings, seed_plan, directory, make_graphs=not arguments.no_graphs)

    print_summary(result)

    return result


def print_summary(result):

    summary = result.summary
    settings = result.settings

    print(f"\n=== simulation_{result.seed_plan.master_seed} ===")
    print(f"derived seeds           : {result.seed_plan.derived}")
    print(f"controls                : {settings.num_logical_qubits} qubits, {settings.num_qpus} QPUs "
          f"(capacities {list(settings.qpu_capacities.values())}), topology {settings.topology}, "
          f"{settings.steps} windows, threshold {settings.congestion_threshold:.2f}")
    print(f"circuit                 : {result.circuit_summary['source']}, "
          f"{result.circuit_summary['gate_count']} gates, "
          f"{result.circuit_summary['two_qubit_gate_count']} two-qubit gates")
    print(f"physical links          : {', '.join(result.topology_initial['summary']['physical_links'])}")
    print(f"initial partition       : {result.initial_partition}")
    print(f"final partition         : {result.final_partition}")
    print(f"remote demand / admitted: {summary['remote_demand_instances']} / {summary['admitted']} "
          f"(deferral events {summary['deferral_events']}, dropped {summary['dropped_from_queue']})")
    print(f"congestion windows      : {summary['congestion_windows']} of {summary['steps']}")
    print(f"affected / rerouted     : {summary['affected_communications']} / "
          f"{summary['selective_reroutes']}")
    print(f"prevented violations    : {summary['prevented_capacity_violations']}")
    print(f"max link utilization    : {summary['max_link_utilization'] * 100:.1f}%")
    print(f"repartition triggers    : {summary['repartition_triggers']} "
          f"{summary['repartition_triggers_by_reason']}, accepted "
          f"{summary['repartitions_accepted']}, qubit moves {summary['logical_qubits_moved']} "
          f"(distinct qubits now off their initial QPU: "
          f"{len(summary['qubits_relocated_from_initial'])})")
    print(f"static comm. cost       : {summary['initial_partition_static']['static_communication_cost']:.1f}"
          f" -> {summary['final_partition_static']['static_communication_cost']:.1f}")

    if settings.use_quantum:
        print(f"QAOA blocks             : attempted {summary['qaoa_blocks_attempted']}, feasible "
              f"{summary['qaoa_blocks_feasible']}, executed {summary['qaoa_blocks_executed']} "
              f"of {summary['reroute_blocks']}")

    if result.baseline_summary:
        baseline = result.baseline_summary
        print(f"classical baseline      : reroutes {baseline['selective_reroutes']}, mean window "
              f"cost {baseline['mean_window_communication_cost']:.1f} vs "
              f"{summary['mean_window_communication_cost']:.1f} (hybrid)")

    failed = [check for check in result.validation if not check["passed"]]
    print(f"validation              : {len(result.validation) - len(failed)}/"
          f"{len(result.validation)} checks passed")

    for check in failed:
        print(f"  FAILED: {check['check']} {check['detail']}")

    print(f"artifacts               : {result.directory.root}")


def run_sweep(arguments, seeds):

    results = []

    for master_seed in seeds:
        settings = build_settings(arguments)
        results.append(run_single(settings, master_seed, arguments))

    sweep_root = os.path.join(arguments.results_dir, f"sweep_{seeds[0]}_{len(seeds)}_seeds")
    os.makedirs(sweep_root, exist_ok=True)

    with open(os.path.join(sweep_root, "sweep_summary.csv"), "w", newline="") as handle:

        writer = csv.DictWriter(handle, fieldnames=["master_seed", "run_directory"] + SWEEP_COLUMNS)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "master_seed": result.seed_plan.master_seed,
                "run_directory": result.directory.root,
                **{column: result.summary[column] for column in SWEEP_COLUMNS},
            })

    with open(os.path.join(sweep_root, "sweep_controls.json"), "w") as handle:
        json.dump({"seeds": seeds, "controls": results[0].settings.controls()}, handle, indent=2)

    print(f"\nSweep of {len(seeds)} seeds written to {sweep_root}/sweep_summary.csv")


def main(argv=None):

    arguments = parse_arguments(argv)

    try:
        settings = build_settings(arguments)
    except (ValueError, FileNotFoundError) as error:
        print(f"Invalid experiment: {error}", file=sys.stderr)
        return 2

    if arguments.verify_reproducibility:
        master_seed = arguments.seed if arguments.seed is not None else generate_master_seed(
            arguments.results_dir, config.MASTER_SEED_RANGE
        )
        report = verify_reproducibility(settings, master_seed)
        return 0 if report["reproducible"] and report["seed_sensitive"] else 1

    if arguments.seeds or arguments.num_seeds:

        if arguments.seeds:
            seeds = [int(value) for value in arguments.seeds.split(",")]
        else:
            seeds = []
            while len(seeds) < arguments.num_seeds:
                candidate = generate_master_seed(arguments.results_dir, config.MASTER_SEED_RANGE)
                if candidate not in seeds:
                    seeds.append(candidate)

        run_sweep(arguments, seeds)
        return 0

    master_seed = arguments.seed if arguments.seed is not None else generate_master_seed(
        arguments.results_dir, config.MASTER_SEED_RANGE
    )

    try:
        result = run_single(settings, master_seed, arguments)
    except RunDirectoryConflict as error:
        print(str(error), file=sys.stderr)
        return 2

    return 0 if result.passed_validation else 1


if __name__ == "__main__":
    sys.exit(main())
