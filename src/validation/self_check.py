import argparse
import os
import random
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compiler.circuits import generate_benchmark, normalize_circuit
from compiler.communication_graph import CommunicationGraph
from compiler.initial_partitioner import CapacityAwareKLPartitioner, validate_partition
from compiler.router import Router
from experiment.output import prepare_seed_directory
from experiment.reproducibility import verify_reproducibility
from experiment.runner import run_experiment
from experiment.seeds import SeedPlan
from main import parse_arguments
from experiment.settings import build_settings
from network.topology_generator import SUPPORTED_TOPOLOGIES, generate_physical_edges
from network.topology import NonexistentPhysicalLinkError, Topology
from qiskit import QuantumCircuit


class CheckRecorder:

    def __init__(self):
        self.results = []

    def record(self, name, passed, detail=""):
        self.results.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))

    @property
    def failed(self):
        return [result for result in self.results if not result[1]]


def check_communication_weights(recorder):

    circuit = QuantumCircuit(5)
    circuit.cx(0, 4)
    circuit.cx(0, 4)
    circuit.cx(4, 0)
    circuit.cx(2, 4)
    circuit.barrier()

    graph = CommunicationGraph().build(circuit)

    recorder.record(
        "communication weight = number of two-qubit gates per pair",
        graph[0][4]["weight"] == 3 and graph[2][4]["weight"] == 1
        and graph.number_of_edges() == 2 and graph.number_of_nodes() == 5,
        f"w(0,4)={graph[0][4]['weight']}, w(2,4)={graph[2][4]['weight']}",
    )

    qft = normalize_circuit(generate_benchmark("qft", 6, 0, random.Random(1)), 1)
    qft_graph = CommunicationGraph().build(qft)

    recorder.record(
        "multi-qubit benchmarks are decomposed before building the graph",
        qft_graph.number_of_edges() == 15, f"QFT(6) interaction edges {qft_graph.number_of_edges()}",
    )


def check_topologies(recorder):

    import networkx as nx

    disconnected = []

    for model in SUPPORTED_TOPOLOGIES:
        for num_qpus in (1, 2, 3, 5, 7, 8, 12):
            for seed in range(25):

                edges = generate_physical_edges(model, num_qpus, random.Random(seed), 0.2)
                graph = nx.Graph()
                graph.add_nodes_from(range(num_qpus))
                graph.add_edges_from(edges)

                if num_qpus > 1 and not nx.is_connected(graph):
                    disconnected.append(f"{model}/{num_qpus}/{seed}")

    recorder.record("every topology model is connected for many sizes and seeds",
                    not disconnected, ", ".join(disconnected[:5]))

    realisations = {
        tuple(generate_physical_edges("random_connected", 8, random.Random(seed), 0.25))
        for seed in range(20)
    }

    recorder.record("random_connected edges vary with the seed", len(realisations) > 10,
                    f"{len(realisations)} distinct edge sets from 20 seeds")

    topology = Topology("line")

    for qpu in range(3):
        topology.add_qpu(qpu, 2)

    topology.connect(0, 1, latency=10, fidelity=0.99, capacity=4, bell_pair_pool=10,
                     bell_pair_regeneration=2, congestion_threshold=0.6)
    topology.connect(1, 2, latency=10, fidelity=0.99, capacity=4, bell_pair_pool=10,
                     bell_pair_regeneration=2, congestion_threshold=0.6)

    try:
        topology.links_on_path((0, 2))
        invented = True
    except NonexistentPhysicalLinkError:
        invented = False

    routes = Router(4, 2).candidate_routes(topology, 0, 2)

    recorder.record(
        "routing never invents a physical edge",
        not invented and [route.path for route in routes] == [(0, 1, 2)],
        f"routes 0->2: {[route.label() for route in routes]}",
    )


def check_partitioner(recorder):

    problems = []

    for seed in range(15):

        rng = random.Random(seed)
        num_qubits = rng.randint(6, 30)
        capacities = {qpu: rng.randint(2, 7) for qpu in range(rng.randint(2, 7))}

        if sum(capacities.values()) < num_qubits:
            capacities[0] += num_qubits - sum(capacities.values())

        circuit = generate_benchmark("layered_random", num_qubits, 8, rng)
        graph = CommunicationGraph().build(circuit)
        partition = CapacityAwareKLPartitioner(random.Random(seed), 4, 2).partition(
            graph, capacities
        )
        problems += validate_partition(partition, num_qubits, capacities)

    recorder.record("initial partition respects heterogeneous capacities, one QPU per qubit",
                    not problems, "; ".join(problems[:3]))


def small_arguments(extra):
    return parse_arguments(
        ["--qubits", "12", "--qpus", "4", "--qpu-capacity", "4", "--steps", "15",
         "--no-quantum", "--no-graphs"] + extra
    )


def check_runs(recorder, root):

    for topology in SUPPORTED_TOPOLOGIES:

        arguments = small_arguments(["--topology", topology])
        settings = build_settings(arguments)
        directory = prepare_seed_directory(root, 424242, settings.controls(), True)
        result = run_experiment(settings, SeedPlan.from_master(424242), directory,
                                console=lambda message: None, make_graphs=False)

        failed = [check["check"] for check in result.validation if not check["passed"]]

        recorder.record(
            f"main pipeline invariants hold on '{topology}' topology", not failed,
            "; ".join(failed) or f"{len(result.validation)} runtime checks",
        )

        recorder.record(
            f"results for '{topology}' are inside Results/simulation_424242",
            os.path.basename(result.directory.root) == "simulation_424242"
            and os.path.exists(os.path.join(result.directory.root, "manifest.json")),
        )


def check_heterogeneous_and_benchmarks(recorder, root):

    for extra in (
        ["--qpu-capacities", "5,3,4,2", "--qubits", "13"],
        ["--benchmark", "ghz"],
        ["--benchmark", "qft", "--qubits", "8"],
        ["--benchmark", "hardware_efficient", "--layers", "3"],
        ["--benchmark", "random", "--layers", "10"],
    ):

        arguments = small_arguments(extra)
        settings = build_settings(arguments)
        directory = prepare_seed_directory(root, 515151, settings.controls(), True)
        result = run_experiment(settings, SeedPlan.from_master(515151), directory,
                                console=lambda message: None, make_graphs=False)

        failed = [check["check"] for check in result.validation if not check["passed"]]

        recorder.record(f"pipeline invariants hold for {' '.join(extra)}", not failed,
                        "; ".join(failed))


def check_reproducibility(recorder):

    settings = build_settings(small_arguments(["--topology", "random_connected"]))
    report = verify_reproducibility(settings, 777001, console=lambda message: None)

    recorder.record("same master seed reproduces circuit, topology, links, evolution, "
                    "partitions and metrics", report["reproducible"],
                    str({k: v for k, v in report["same_seed_components_identical"].items() if not v}))
    recorder.record("a different master seed changes every stochastic component",
                    report["seed_sensitive"],
                    str(report["different_seed_stochastic_components_differ"]))

    quantum_settings = build_settings(parse_arguments(
        ["--qubits", "12", "--qpus", "3", "--qpu-capacity", "4", "--steps", "10",
         "--topology", "ring", "--link-capacity-range", "4,5", "--no-graphs", "--no-baseline"]
    ))
    quantum_settings.run_baseline = False
    quantum_report = verify_reproducibility(quantum_settings, 777002,
                                            console=lambda message: None)

    recorder.record("QAOA runs are reproducible from the derived QAOA seed",
                    quantum_report["reproducible"] and quantum_report["qaoa_blocks_attempted"] > 0,
                    f"{quantum_report['qaoa_blocks_attempted']} QAOA blocks executed per run")


def main():

    parser = argparse.ArgumentParser(description="Structural self-check of the simulator")
    parser.add_argument("--skip-quantum", action="store_true")
    arguments = parser.parse_args()

    recorder = CheckRecorder()
    root = tempfile.mkdtemp(prefix="self_check_")

    try:
        print("communication graph")
        check_communication_weights(recorder)
        print("physical topology and routing")
        check_topologies(recorder)
        print("initial partition")
        check_partitioner(recorder)
        print("end-to-end runs through the real runner")
        check_runs(recorder, root)
        check_heterogeneous_and_benchmarks(recorder, root)
        print("reproducibility")
        if arguments.skip_quantum:
            settings = build_settings(small_arguments([]))
            report = verify_reproducibility(settings, 777001, console=lambda message: None)
            recorder.record("same seed reproduces the experiment", report["reproducible"])
            recorder.record("different seed changes stochastic components",
                            report["seed_sensitive"])
        else:
            check_reproducibility(recorder)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{len(recorder.results) - len(recorder.failed)}/{len(recorder.results)} checks passed")

    return 1 if recorder.failed else 0


if __name__ == "__main__":
    sys.exit(main())
