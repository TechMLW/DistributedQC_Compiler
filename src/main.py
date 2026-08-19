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
    
cost_calculator = CommunicationCost()

cost = cost_calculator.calculate(graph, partitions)

print("\n=== Communication Cost ===")
print(f"Communication Cost = {cost}")



optimizer = Optimizer()

best_partition, score = optimizer.optimize(
    graph,
    partitions
)

print("\n===== Optimized Partition =====")

for qpu, qubits in best_partition.items():
    print(f"QPU {qpu}: {sorted(qubits)}")

print(f"\nObjective Score = {score}")

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

topology.print_topology()

router = Router()

path = router.route(
    topology.graph,
    0,
    1
)

print(path)

Visualizer().draw(graph)

