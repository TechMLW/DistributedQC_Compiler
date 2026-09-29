from dataclasses import dataclass, field

import config
from compiler.circuits import LAYERED_BENCHMARKS, SUPPORTED_BENCHMARKS, load_circuit_file
from network.topology_generator import LinkPropertyRanges, SUPPORTED_TOPOLOGIES


MODEL_PARAMETER_NAMES = [
    name for name in dir(config)
    if name.isupper() and not name.startswith("DEFAULT_")
    and name not in {"RESULTS_DIRECTORY", "MASTER_SEED_RANGE"}
]


@dataclass
class ExperimentSettings:
    num_logical_qubits: int
    num_qpus: int
    qpu_capacities: dict
    benchmark: str
    layers: int
    circuit_file: str
    topology: str
    steps: int
    congestion_threshold: float
    link_ranges: LinkPropertyRanges
    extra_link_probability: float
    use_quantum: bool
    run_baseline: bool
    model_parameters: dict = field(default_factory=dict)

    def validate(self):

        problems = []

        if self.num_qpus < 1:
            problems.append("--qpus must be at least 1")

        if len(self.qpu_capacities) != self.num_qpus:
            problems.append("the number of QPU capacities must equal --qpus")

        if any(capacity < 1 for capacity in self.qpu_capacities.values()):
            problems.append("every QPU capacity must be at least 1")

        if self.circuit_file is None and self.benchmark not in SUPPORTED_BENCHMARKS:
            problems.append(f"--benchmark must be one of {', '.join(SUPPORTED_BENCHMARKS)}")

        if self.circuit_file is None and self.benchmark in LAYERED_BENCHMARKS and self.layers < 1:
            problems.append("--layers must be at least 1 for this benchmark")

        if self.topology not in SUPPORTED_TOPOLOGIES:
            problems.append(f"--topology must be one of {', '.join(SUPPORTED_TOPOLOGIES)}")

        if self.steps < 1:
            problems.append("--steps must be at least 1")

        if not 0.0 < self.congestion_threshold < 1.0:
            problems.append("--congestion-threshold must lie strictly between 0 and 1")

        if self.num_logical_qubits is not None and self.num_logical_qubits > sum(
            self.qpu_capacities.values()
        ):
            problems.append(
                f"{self.num_logical_qubits} logical qubits exceed total QPU capacity "
                f"{sum(self.qpu_capacities.values())}"
            )

        for name, (low, high) in self.link_ranges.to_dict().items():
            if low > high:
                problems.append(f"link {name} range has min > max")

        if self.link_ranges.capacity[0] < 1:
            problems.append("link capacity range must start at 1 or more")

        if not (0.0 < self.link_ranges.fidelity[0] <= self.link_ranges.fidelity[1] <= 1.0):
            problems.append("link fidelity range must lie in (0, 1]")

        if problems:
            raise ValueError("; ".join(problems))

    @property
    def benchmark_uses_layers(self):
        return self.circuit_file is None and self.benchmark in LAYERED_BENCHMARKS

    def controls(self):

        return {
            "num_logical_qubits": self.num_logical_qubits,
            "num_qpus": self.num_qpus,
            "qpu_capacities": {str(qpu): cap for qpu, cap in sorted(self.qpu_capacities.items())},
            "benchmark": None if self.circuit_file else self.benchmark,
            "layers": self.layers if self.benchmark_uses_layers else None,
            "circuit_file": self.circuit_file,
            "topology": self.topology,
            "steps": self.steps,
            "congestion_threshold": self.congestion_threshold,
            "link_property_ranges": self.link_ranges.to_dict(),
            "extra_link_probability": (
                self.extra_link_probability if self.topology == "random_connected" else None
            ),
            "use_quantum": self.use_quantum,
            "run_baseline": self.run_baseline,
        }

    def to_dict(self):
        return {"controls": self.controls(), "model_parameters": self.model_parameters}


def snapshot_model_parameters():

    snapshot = {}

    for name in MODEL_PARAMETER_NAMES:

        value = getattr(config, name)

        if isinstance(value, tuple):
            value = list(value)

        snapshot[name] = value

    return snapshot


def build_settings(arguments):

    num_qpus = arguments.qpus

    if arguments.qpu_capacities:
        capacities = [int(value) for value in arguments.qpu_capacities.split(",")]
        if len(capacities) != num_qpus:
            raise ValueError(
                f"--qpu-capacities lists {len(capacities)} values but --qpus is {num_qpus}"
            )
    else:
        capacities = [arguments.qpu_capacity] * num_qpus

    def parse_range(text, default, cast):
        if text is None:
            return tuple(default)
        low, high = (cast(value) for value in text.split(","))
        return (low, high)

    ranges = LinkPropertyRanges(
        latency=parse_range(arguments.link_latency_range, config.LINK_LATENCY_RANGE, float),
        fidelity=parse_range(arguments.link_fidelity_range, config.LINK_FIDELITY_RANGE, float),
        capacity=parse_range(arguments.link_capacity_range, config.LINK_CAPACITY_RANGE, int),
        bell_pair_pool=parse_range(
            arguments.link_bell_pairs_range, config.LINK_BELL_PAIR_POOL_RANGE, int
        ),
        bell_pair_regeneration=parse_range(
            arguments.link_regeneration_range, config.LINK_BELL_PAIR_REGENERATION_RANGE, float
        ),
    )

    num_logical_qubits = arguments.qubits

    if num_logical_qubits is None:
        num_logical_qubits = (
            load_circuit_file(arguments.circuit_file).num_qubits
            if arguments.circuit_file else config.DEFAULT_NUM_LOGICAL_QUBITS
        )

    settings = ExperimentSettings(
        num_logical_qubits=num_logical_qubits,
        num_qpus=num_qpus,
        qpu_capacities={qpu: capacities[qpu] for qpu in range(num_qpus)},
        benchmark=arguments.benchmark,
        layers=arguments.layers,
        circuit_file=arguments.circuit_file,
        topology=arguments.topology,
        steps=arguments.steps,
        congestion_threshold=arguments.congestion_threshold,
        link_ranges=ranges,
        extra_link_probability=arguments.extra_link_probability,
        use_quantum=not arguments.no_quantum,
        run_baseline=not arguments.no_baseline and not arguments.no_quantum,
        model_parameters=snapshot_model_parameters(),
    )

    settings.validate()

    return settings
