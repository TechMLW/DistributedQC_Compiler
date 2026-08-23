from qiskit import QuantumCircuit
from compiler.analyzer import CircuitAnalyzer
from compiler.partitioner import Partitioner
from compiler.communication_graph import CommunicationGraph
from compiler.communication_cost import CommunicationCost
from utils.visualizer import Visualizer
from compiler.benchmarks import Benchmarks
from compiler.optimizer import Optimizer
from network.topology import Topology
from compiler.router import Router
from compiler.kl_partitioner import KLPartitioner
from network.monitor import NetworkMonitor
from compiler.repartition_trigger import RepartitionTrigger
# Create a sample circuit
# qc = QuantumCircuit(5)

# qc.h(0)
# qc.cx(0, 1)
# qc.cx(1, 2)
# qc.cx(2, 3)
# qc.cx(3, 4)

bench = Benchmarks()
# qc = bench.ghz(8)
# qc = bench.qft(8)
qc = bench.random(8, depth=30)
# qc = bench.hardware_efficient(8)

print("=== Quantum Circuit ===")
print(qc)

print("\n=== Analysis ===")

analysis = CircuitAnalyzer().analyze(qc)

for key, value in analysis.items():
    print(f"{key}: {value}")


graph = CommunicationGraph().build(qc)

print("\n=== Communication Graph ===")

for u, v, data in graph.edges(data=True):
    print(f"Qubit {u} <--> Qubit {v} | Weight = {data['weight']}")
    
# partitioner = Partitioner()

# partitions = partitioner.partition(graph, 2)

# partitions = {
#     0: [0, 1, 2],
#     1: [3, 4]
# }

from compiler.kl_partitioner import KLPartitioner

partitioner = KLPartitioner()
partitions = partitioner.partition(graph)

print("\n=== KL Partition ===")
for qpu, qubits in partitions.items():
    print(f"QPU {qpu}: {qubits}")

print("\n=== Initial Partition ===")

for qpu, qubits in partitions.items():

    print(f"QPU {qpu}: {qubits}")
    
monitor = NetworkMonitor()

monitor.register_link(
    0,
    1,
    latency=15,
    fidelity=0.98,
    bell_pairs=20,
    congestion=10
)

topology = Topology()

topology.add_qpu(0,4)
topology.add_qpu(1,4)

topology.connect(
    0,
    1,
    latency=15,
    fidelity=0.97,
    bell_pairs=20
)

# for i in range(5):
#     print(f"Iteration {i + 1}:")
#     monitor.update()
#     monitor.print_state()

cost_calculator = CommunicationCost(monitor)

cost = cost_calculator.calculate(graph, partitions)

print("\n=== Communication Cost ===")
print(f"Communication Cost = {cost}")

router = Router()


optimizer = Optimizer(monitor)
trigger = RepartitionTrigger()

# best_partition, score = optimizer.optimize(
#     graph,
#     partitions
# )

# print("\n===== Optimized Partition =====")

# for qpu, qubits in best_partition.items():
#     print(f"QPU {qpu}: {sorted(qubits)}")

# print(f"\nObjective Score = {score}")

for i in range(5):

    print(f"\nIteration {i + 1}:")

    monitor.update()


    for (source, destination), state in monitor.links.items():
            topology.update_link(
            source,
            destination,
            latency=state["latency"],
            fidelity=state["fidelity"],
            bell_pairs=state["bell_pairs"]
        )
    path = router.route(
        topology.graph,
        0,
        1
    )
    
    print(path)
    monitor.print_state()

    
    if trigger.should_repartition(monitor):

        print("\nNetwork degraded.")
        print("Running adaptive optimizer...")

        partitions, score = optimizer.adaptive_optimize(
            graph,
            partitions,
            cost,
            threshold=10
        )

    else:

        print("\nNetwork healthy.")
        print("Keeping current partition.")

        score = optimizer.objective.evaluate(cost, partitions)
    print(f"\nCommunication Cost = {cost}")

    print("\n===== Current Partition =====")

    for qpu, qubits in partitions.items():
        print(f"QPU {qpu}: {sorted(qubits)}")

    print(f"\nObjective Score = {score:.3f}")



topology.print_topology()
    
Visualizer().draw(graph)

